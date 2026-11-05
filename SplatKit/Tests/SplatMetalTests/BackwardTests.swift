import Foundation
import Metal
import Testing
import SplatCore
@testable import SplatMetal

/// GPU backward pass against the CPU reference and against finite differences on the GPU.
/// The golden comparison lives in GoldenBackwardTests.swift.
@Suite(.enabled(if: TestEnvironment.hasMetal))
struct BackwardTests {

    // MARK: - Helpers

    /// Renders through `render(camera:settings:to:)`, which keeps the frame state `backward` needs,
    /// and reads the RGB image back.
    private func renderFrame(_ renderer: SplatRenderer, context: MetalContext,
                             camera: Camera, settings: RasterSettings)
        throws -> (image: [SIMD3<Float>], stats: FrameStats) {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: camera.width, height: camera.height, mipmapped: false)
        descriptor.usage = [.shaderWrite, .shaderRead]
        descriptor.storageMode = .shared
        guard let texture = context.device.makeTexture(descriptor: descriptor) else {
            throw MetalError.bufferAllocation
        }
        let stats = try renderer.render(camera: camera, settings: settings, to: texture)

        let pixelCount = camera.width * camera.height
        var raw = [Float](repeating: 0, count: pixelCount * 4)
        raw.withUnsafeMutableBytes { bytes in
            texture.getBytes(bytes.baseAddress!,
                             bytesPerRow: camera.width * 4 * MemoryLayout<Float>.size,
                             from: MTLRegionMake2D(0, 0, camera.width, camera.height),
                             mipmapLevel: 0)
        }
        var image: [SIMD3<Float>] = []
        image.reserveCapacity(pixelCount)
        for i in 0..<pixelCount {
            image.append(SIMD3(raw[4 * i], raw[4 * i + 1], raw[4 * i + 2]))
        }
        return (image, stats)
    }

    private func randomImageGradient(count: Int, seed: UInt64) -> [SIMD3<Float>] {
        var rng = SplitMix64(seed: seed)
        var gradient: [SIMD3<Float>] = []
        gradient.reserveCapacity(count)
        for _ in 0..<count {
            let r = Float.random(in: -1...1, using: &rng)
            let g = Float.random(in: -1...1, using: &rng)
            let b = Float.random(in: -1...1, using: &rng)
            gradient.append(SIMD3(r, g, b))
        }
        return gradient
    }

    private func axisAngle(_ axis: SIMD3<Float>, _ angle: Float) -> SIMD4<Float> {
        let half = angle / 2
        return SIMD4(cos(half), sin(half) * axis.x, sin(half) * axis.y, sin(half) * axis.z)
    }

    /// Pinhole camera with R = Rx(0.35)·Ry(−0.25) and t = (0.3, −0.2, 0.4).
    private func rotatedCamera(width: Int, height: Int) -> Camera {
        let rx = Matrix3.rotation(quaternion: axisAngle(SIMD3(1, 0, 0), 0.35))
        let ry = Matrix3.rotation(quaternion: axisAngle(SIMD3(0, 1, 0), -0.25))
        return Camera(width: width, height: height,
                      fx: Float(height), fy: Float(height),
                      cx: Float(width) / 2, cy: Float(height) / 2,
                      rotation: rx * ry, translation: SIMD3(0.3, -0.2, 0.4))
    }

    /// Every entry within 1e-4 + 1e-3·|cpu|.
    private func expectMatch(_ gpu: [Float], _ cpu: [Float], _ name: String) {
        #expect(gpu.count == cpu.count, "\(name): count \(gpu.count) vs \(cpu.count)")
        guard gpu.count == cpu.count else { return }
        var failures = 0
        var firstFailure: Int?
        for i in gpu.indices {
            let tolerance: Float = 1e-4 + 1e-3 * abs(cpu[i])
            if !(abs(gpu[i] - cpu[i]) <= tolerance) {
                failures += 1
                if firstFailure == nil { firstFailure = i }
            }
        }
        let detail = firstFailure.map { "first at \($0): gpu \(gpu[$0]), cpu \(cpu[$0])" } ?? ""
        #expect(failures == 0, "\(name): \(failures) of \(gpu.count) entries outside tolerance; \(detail)")
    }

    /// Runs the GPU backward for the frame already rendered by `renderer` and compares it with
    /// `CPURasterizer.backward`.
    private func expectBackwardMatchesCPU(renderer: SplatRenderer, cloud: GaussianCloud, camera: Camera,
                                          settings: RasterSettings, seed: UInt64) throws {
        let imageGradient = randomImageGradient(count: camera.width * camera.height, seed: seed)
        let gpu = try renderer.backward(imageGradient: imageGradient)
        let cpu = CPURasterizer.backward(cloud, camera: camera, settings: settings,
                                         imageGradient: imageGradient)
        expectMatch(gpu.positions, cpu.positions, "positions")
        expectMatch(gpu.logScales, cpu.logScales, "logScales")
        expectMatch(gpu.rotations, cpu.rotations, "rotations")
        expectMatch(gpu.opacityLogits, cpu.opacityLogits, "opacityLogits")
        expectMatch(gpu.sh, cpu.sh, "sh")
    }

    private func parameterGroups(_ positions: [Float], _ logScales: [Float], _ rotations: [Float],
                                 _ opacity: [Float], _ sh: [Float]) -> [[Float]] {
        [positions, logScales, rotations, opacity, sh]
    }

    /// Copy of `cloud` with one scalar replaced. Group order as in `parameterGroups`.
    private func perturbed(_ cloud: GaussianCloud, group: Int, index: Int, to value: Float) throws -> GaussianCloud {
        var positions = cloud.positions
        var logScales = cloud.logScales
        var rotations = cloud.rotations
        var opacity = cloud.opacityLogits
        var sh = cloud.sh
        switch group {
        case 0: positions[index] = value
        case 1: logScales[index] = value
        case 2: rotations[index] = value
        case 3: opacity[index] = value
        default: sh[index] = value
        }
        return try GaussianCloud(shDegree: cloud.shDegree, positions: positions, logScales: logScales,
                                 rotations: rotations, opacityLogits: opacity, sh: sh)
    }

    /// L = Σ imageGradient · image, accumulated in Double.
    private func loss(_ image: [SIMD3<Float>], _ gradient: [SIMD3<Float>]) -> Double {
        var total = 0.0
        for i in image.indices {
            total += Double(image[i].x) * Double(gradient[i].x)
            total += Double(image[i].y) * Double(gradient[i].y)
            total += Double(image[i].z) * Double(gradient[i].z)
        }
        return total
    }

    // MARK: - Tests

    @Test func backwardMatchesCPU() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 200)
        var settings = RasterSettings()
        settings.background = SIMD3(0.1, 0.2, 0.3)

        let identity = makeTestCamera(width: 64, height: 64)
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        _ = try renderFrame(renderer, context: context, camera: identity, settings: settings)
        try expectBackwardMatchesCPU(renderer: renderer, cloud: cloud, camera: identity,
                                     settings: settings, seed: 11)

        let rotated = rotatedCamera(width: 64, height: 64)
        _ = try renderFrame(renderer, context: context, camera: rotated, settings: settings)
        try expectBackwardMatchesCPU(renderer: renderer, cloud: cloud, camera: rotated,
                                     settings: settings, seed: 12)
    }

    @Test func backwardMatchesCPUWithMultipleBatches() throws {
        let context = try makeTestContext()
        // All Gaussians project into the single 16×16 tile, so the tile holds far more than
        // 256 samples and the backward pass walks several batches.
        let cloud = try makeRandomCloud(count: 600, seed: 9, scale: 0.05...0.5)
        let camera = makeTestCamera(width: 16, height: 16)
        let settings = RasterSettings()

        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let (_, stats) = try renderFrame(renderer, context: context, camera: camera, settings: settings)
        #expect(stats.tileIntersections > 512, "tile holds \(stats.tileIntersections) samples")

        try expectBackwardMatchesCPU(renderer: renderer, cloud: cloud, camera: camera,
                                     settings: settings, seed: 13)
    }

    @Test func finiteDifferencesOnGPU() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 60, seed: 5)
        let camera = makeTestCamera(width: 32, height: 32)
        var settings = RasterSettings()
        settings.background = SIMD3(0.1, 0.2, 0.3)
        let imageGradient = randomImageGradient(count: 32 * 32, seed: 7)

        let base = try SplatRenderer(context: context, cloud: cloud)
        let (_, baseStats) = try renderFrame(base, context: context, camera: camera, settings: settings)
        let analytic = try base.backward(imageGradient: imageGradient)
        let analyticGroups = parameterGroups(analytic.positions, analytic.logScales, analytic.rotations,
                                             analytic.opacityLogits, analytic.sh)
        let cloudGroups = parameterGroups(cloud.positions, cloud.logScales, cloud.rotations,
                                          cloud.opacityLogits, cloud.sh)

        // Only parameters whose analytic gradient is large enough to be checked by a difference.
        var candidates: [(group: Int, index: Int)] = []
        for (group, values) in analyticGroups.enumerated() {
            for (index, value) in values.enumerated() where abs(value) > 1e-2 {
                candidates.append((group: group, index: index))
            }
        }
        #expect(candidates.count >= 20, "only \(candidates.count) parameters with |gradient| > 1e-2")
        var rng = SplitMix64(seed: 7)
        candidates.shuffle(using: &rng)

        let h: Float = 1e-3
        var checked = 0
        var skippedFootprint = 0
        var failures: [String] = []
        for candidate in candidates {
            if checked == 20 { break }
            let value = cloudGroups[candidate.group][candidate.index]
            let plusValue = value + h
            let minusValue = value - h
            let plusCloud = try perturbed(cloud, group: candidate.group, index: candidate.index, to: plusValue)
            let minusCloud = try perturbed(cloud, group: candidate.group, index: candidate.index, to: minusValue)
            let plusRenderer = try SplatRenderer(context: context, cloud: plusCloud)
            let minusRenderer = try SplatRenderer(context: context, cloud: minusCloud)
            let (plusImage, plusStats) = try renderFrame(plusRenderer, context: context,
                                                         camera: camera, settings: settings)
            let (minusImage, minusStats) = try renderFrame(minusRenderer, context: context,
                                                           camera: camera, settings: settings)

            // A perturbation that changes the tile footprint crosses a discontinuity: skip it.
            guard plusStats.tileIntersections == baseStats.tileIntersections,
                  minusStats.tileIntersections == baseStats.tileIntersections else {
                skippedFootprint += 1
                continue
            }

            let step = Double(plusValue) - Double(minusValue)
            let numeric = (loss(plusImage, imageGradient) - loss(minusImage, imageGradient)) / step
            let exact = Double(analyticGroups[candidate.group][candidate.index])
            let relative = abs(numeric - exact) / abs(exact)
            checked += 1
            if !(relative <= 5e-2) {
                failures.append("group \(candidate.group)[\(candidate.index)]: analytic \(exact), numeric \(numeric)")
            }
        }
        #expect(checked == 20, "checked \(checked) parameters, \(skippedFootprint) skipped for footprint changes")
        #expect(failures.isEmpty, "relative error above 5e-2: \(failures)")
    }

    @Test func noVisibleGaussiansGiveZeroGradients() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 50, depth: (-8)...(-2))
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let camera = makeTestCamera(width: 40, height: 30)
        let settings = RasterSettings()
        let (_, stats) = try renderFrame(renderer, context: context, camera: camera, settings: settings)
        #expect(stats.tileIntersections == 0)

        let gradients = try renderer.backward(imageGradient: randomImageGradient(count: 40 * 30, seed: 3))
        #expect(gradients.positions.count == 150)
        #expect(gradients.positions.allSatisfy { $0 == 0 })
        #expect(gradients.logScales.allSatisfy { $0 == 0 })
        #expect(gradients.rotations.allSatisfy { $0 == 0 })
        #expect(gradients.opacityLogits.allSatisfy { $0 == 0 })
        #expect(gradients.sh.allSatisfy { $0 == 0 })
    }

    @Test func backwardRequiresRender() throws {
        let context = try makeTestContext()
        let renderer = try SplatRenderer(context: context, cloud: try makeRandomCloud(count: 10))
        #expect(throws: MetalError.self) {
            _ = try renderer.backward(imageGradient: [])
        }
    }
}
