import Foundation
import Metal
import Testing
import SplatCore
@testable import SplatMetal

/// Phase 4 Task 14: rendered depth on the Metal forward and backward pass.
/// Draw = Σ αᵢ Tᵢ zᵢ and A = 1 − T_final per pixel; the backward input is (∂L/∂Draw, ∂L/∂A) per pixel.
/// References: `CPURasterizer.renderWithDepth` / `backwardWithScreenSpace(depthGradient:)` (CPU) and the
/// PyTorch goldens `golden_depth_single` / `golden_depth_occluded`.
@Suite(.enabled(if: TestEnvironment.hasMetal))
struct DepthRenderTests {

    /// FNV-1a 64-bit hashes of the image bit patterns, recorded from the Phase 3 code before Task 14.
    /// The forward pass uses no atomics, so with depth off these must not change.
    private static let phase3ImageHashes: [(name: String, hash: UInt64)] = [
        (name: "single", hash: 0xd4ff_ee3a_2420_c89c),
        (name: "multi", hash: 0xb688_89f1_bd12_6bd0),
        (name: "rotated", hash: 0xa2ec_b991_0f9c_fba2),
        (name: "occluded", hash: 0x09f8_d72b_caf1_0208),
    ]

    // MARK: - Helpers

    private struct Forward {
        var image: [SIMD3<Float>]
        var depth: DepthRender?
        var tileIntersections: Int
    }

    private static func fnv1a(_ words: [UInt32]) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for word in words {
            hash ^= UInt64(word)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return hash
    }

    private static func imageHash(_ image: [SIMD3<Float>]) -> UInt64 {
        var words: [UInt32] = []
        words.reserveCapacity(3 * image.count)
        for p in image {
            words.append(p.x.bitPattern)
            words.append(p.y.bitPattern)
            words.append(p.z.bitPattern)
        }
        return fnv1a(words)
    }

    /// Forward pass through `encodeForward`. The depth buffer is bound only when `withDepth`.
    /// Reads the image texture and, when requested, the per-pixel (Draw, A) buffer back.
    private func forward(_ renderer: SplatRenderer, context: MetalContext, camera: Camera,
                         settings: RasterSettings, withDepth: Bool) throws -> Forward {
        let width = camera.width
        let height = camera.height
        let pixelCount = width * height
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderWrite, .shaderRead]
        descriptor.storageMode = .shared
        guard let texture = context.device.makeTexture(descriptor: descriptor) else {
            throw MetalError.bufferAllocation
        }
        var depthBuffer: MTLBuffer? = nil
        if withDepth {
            depthBuffer = try context.makeBuffer(length: pixelCount * 8)
        }
        let pass = try renderer.encodeForward(camera: camera, settings: settings, to: texture, depth: depthBuffer)
        pass.commandBuffer.commit()
        pass.commandBuffer.waitUntilCompleted()
        guard pass.commandBuffer.status != .error else {
            throw MetalError.commandFailed(pass.commandBuffer.error?.localizedDescription ?? "unknown GPU error")
        }

        var raw = [Float](repeating: 0, count: pixelCount * 4)
        raw.withUnsafeMutableBytes { bytes in
            texture.getBytes(bytes.baseAddress!,
                             bytesPerRow: width * 4 * MemoryLayout<Float>.size,
                             from: MTLRegionMake2D(0, 0, width, height),
                             mipmapLevel: 0)
        }
        var image: [SIMD3<Float>] = []
        image.reserveCapacity(pixelCount)
        for i in 0..<pixelCount {
            image.append(SIMD3(raw[4 * i], raw[4 * i + 1], raw[4 * i + 2]))
        }

        var depth: DepthRender? = nil
        if let depthBuffer {
            let values = depthBuffer.contents().bindMemory(to: Float.self, capacity: 2 * pixelCount)
            var draw = [Float](repeating: 0, count: pixelCount)
            var alpha = [Float](repeating: 0, count: pixelCount)
            for i in 0..<pixelCount {
                draw[i] = values[2 * i]
                alpha[i] = values[2 * i + 1]
            }
            depth = DepthRender(raw: draw, alpha: alpha)
        }
        return Forward(image: image, depth: depth, tileIntersections: pass.tileIntersections)
    }

    /// Encodes one backward pass with `encodeBackward` into `gradients`, commits it and reads the result.
    private func encodedBackward(_ renderer: SplatRenderer, context: MetalContext, camera: Camera,
                                 imageGradient: [SIMD3<Float>], depthGradient: [SIMD2<Float>]?,
                                 gradients: GradientBuffers, accumulate: Bool,
                                 count: Int, shDegree: Int) throws -> GaussianGradients {
        let pixelCount = camera.width * camera.height
        let imageBuffer = try context.makeBuffer(length: pixelCount * 4 * MemoryLayout<Float>.size)
        let imagePointer = imageBuffer.contents().bindMemory(to: Float.self, capacity: pixelCount * 4)
        for i in 0..<pixelCount {
            imagePointer[4 * i] = imageGradient[i].x
            imagePointer[4 * i + 1] = imageGradient[i].y
            imagePointer[4 * i + 2] = imageGradient[i].z
            imagePointer[4 * i + 3] = 0
        }

        var depthBuffer: MTLBuffer? = nil
        if let depthGradient {
            let buffer = try context.makeBuffer(length: pixelCount * 8)
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: pixelCount * 2)
            for i in 0..<pixelCount {
                pointer[2 * i] = depthGradient[i].x
                pointer[2 * i + 1] = depthGradient[i].y
            }
            depthBuffer = buffer
        }

        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.commandFailed("cannot create command buffer")
        }
        try renderer.encodeBackward(imageGradient: imageBuffer, depthGradient: depthBuffer, into: gradients,
                                    statistics: nil, accumulate: accumulate, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status != .error else {
            throw MetalError.commandFailed(commandBuffer.error?.localizedDescription ?? "unknown GPU error")
        }
        return gradients.toGaussianGradients(count: count, shDegree: shDegree)
    }

    private func randomPixels(count: Int, seed: UInt64) -> [SIMD3<Float>] {
        var rng = SplitMix64(seed: seed)
        var values: [SIMD3<Float>] = []
        values.reserveCapacity(count)
        for _ in 0..<count {
            let x = Float.random(in: -1...1, using: &rng)
            let y = Float.random(in: -1...1, using: &rng)
            let z = Float.random(in: -1...1, using: &rng)
            values.append(SIMD3(x, y, z))
        }
        return values
    }

    private func randomDepthGradient(count: Int, seed: UInt64) -> [SIMD2<Float>] {
        var rng = SplitMix64(seed: seed)
        var values: [SIMD2<Float>] = []
        values.reserveCapacity(count)
        for _ in 0..<count {
            let gd = Float.random(in: -1...1, using: &rng)
            let ga = Float.random(in: -1...1, using: &rng)
            values.append(SIMD2(gd, ga))
        }
        return values
    }

    /// Interleaved (x, y) pairs from a flat row-major W·H·2 array.
    private func pairs(_ flat: [Float]) -> [SIMD2<Float>] {
        var out: [SIMD2<Float>] = []
        out.reserveCapacity(flat.count / 2)
        var i = 0
        while i + 1 < flat.count {
            out.append(SIMD2(flat[i], flat[i + 1]))
            i += 2
        }
        return out
    }

    /// Every entry within `absolute + relative·|expected|`.
    private func expectClose(_ actual: [Float], _ expected: [Float], absolute: Float, relative: Float,
                             _ name: String) {
        #expect(actual.count == expected.count, "\(name): count \(actual.count) vs \(expected.count)")
        guard actual.count == expected.count else { return }
        var failures = 0
        var first: Int? = nil
        for i in actual.indices where !(abs(actual[i] - expected[i]) <= absolute + relative * abs(expected[i])) {
            failures += 1
            if first == nil { first = i }
        }
        let detail = first.map { "first at \($0): \(actual[$0]) vs \(expected[$0])" } ?? ""
        #expect(failures == 0, "\(name): \(failures) of \(actual.count) entries outside tolerance; \(detail)")
    }

    private func expectGradients(_ actual: GaussianGradients, _ expected: GaussianGradients,
                                 absolute: Float, relative: Float, _ label: String) {
        expectClose(actual.positions, expected.positions, absolute: absolute, relative: relative, "\(label).positions")
        expectClose(actual.logScales, expected.logScales, absolute: absolute, relative: relative, "\(label).logScales")
        expectClose(actual.rotations, expected.rotations, absolute: absolute, relative: relative, "\(label).rotations")
        expectClose(actual.opacityLogits, expected.opacityLogits, absolute: absolute, relative: relative,
                    "\(label).opacityLogits")
        expectClose(actual.sh, expected.sh, absolute: absolute, relative: relative, "\(label).sh")
    }

    private func sum(_ a: GaussianGradients, _ b: GaussianGradients) -> GaussianGradients {
        func add(_ x: [Float], _ y: [Float]) -> [Float] {
            var out = x
            for i in out.indices { out[i] += y[i] }
            return out
        }
        return GaussianGradients(positions: add(a.positions, b.positions),
                                 logScales: add(a.logScales, b.logScales),
                                 rotations: add(a.rotations, b.rotations),
                                 opacityLogits: add(a.opacityLogits, b.opacityLogits),
                                 sh: add(a.sh, b.sh))
    }

    /// Copy of `cloud` with one scalar replaced. Group 0 = positions, 3 = opacity logits.
    private func perturbed(_ cloud: GaussianCloud, group: Int, index: Int, to value: Float) throws -> GaussianCloud {
        var positions = cloud.positions
        var opacity = cloud.opacityLogits
        if group == 0 {
            positions[index] = value
        } else {
            opacity[index] = value
        }
        return try GaussianCloud(shDegree: cloud.shDegree, positions: positions, logScales: cloud.logScales,
                                 rotations: cloud.rotations, opacityLogits: opacity, sh: cloud.sh)
    }

    /// L = Σ gD·Draw + gA·A, accumulated in Double.
    private func depthLoss(_ depth: DepthRender, _ gradient: [SIMD2<Float>]) -> Double {
        var total = 0.0
        for i in gradient.indices {
            total += Double(gradient[i].x) * Double(depth.raw[i])
            total += Double(gradient[i].y) * Double(depth.alpha[i])
        }
        return total
    }

    // MARK: - Forward

    @Test func depthOffIsBitIdenticalToPhase3() throws {
        let context = try makeTestContext()
        for entry in Self.phase3ImageHashes {
            let golden = try GoldenCase.load(entry.name)
            let renderer = try SplatRenderer(context: context, cloud: try golden.cloud())
            let image = try renderer.renderImage(camera: golden.camera(), settings: golden.settings())
            #expect(Self.imageHash(image.pixels) == entry.hash,
                    "\(entry.name): depth-off image changed from the Phase 3 bit pattern")
        }
    }

    @Test func depthOnLeavesImageUnchanged() throws {
        let context = try makeTestContext()
        let golden = try GoldenCase.load("depth_occluded")
        let renderer = try SplatRenderer(context: context, cloud: try golden.cloud())
        let off = try forward(renderer, context: context, camera: golden.camera(),
                              settings: golden.settings(), withDepth: false)
        let on = try forward(renderer, context: context, camera: golden.camera(),
                             settings: golden.settings(), withDepth: true)
        #expect(Self.imageHash(on.image) == Self.imageHash(off.image),
                "depth-on image differs from depth-off image")
    }

    @Test func depthMatchesCPUReference() throws {
        let context = try makeTestContext()
        let golden = try GoldenCase.load("depth_occluded")
        let cloud = try golden.cloud()
        let camera = golden.camera()
        let settings = golden.settings()
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let gpu = try forward(renderer, context: context, camera: camera, settings: settings, withDepth: true)
        let cpu = CPURasterizer.renderWithDepth(cloud, camera: camera, settings: settings)

        expectClose(gpu.depth!.raw, cpu.depth.raw, absolute: 1e-4, relative: 1e-4, "occluded.raw")
        expectClose(gpu.depth!.alpha, cpu.depth.alpha, absolute: 1e-4, relative: 1e-4, "occluded.alpha")
        var imageFlat: [Float] = []
        var cpuFlat: [Float] = []
        for i in gpu.image.indices {
            imageFlat += [gpu.image[i].x, gpu.image[i].y, gpu.image[i].z]
            cpuFlat += [cpu.image.pixels[i].x, cpu.image.pixels[i].y, cpu.image.pixels[i].z]
        }
        expectClose(imageFlat, cpuFlat, absolute: 1e-4, relative: 1e-4, "occluded.image")
    }

    @Test func renderDepthMatchesCPUReference() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 200, seed: 21)
        let camera = makeTestCamera(width: 64, height: 64)
        var settings = RasterSettings()
        settings.background = SIMD3(0.1, 0.2, 0.3)
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let gpu = try renderer.renderDepth(camera: camera, settings: settings)
        let cpu = CPURasterizer.renderWithDepth(cloud, camera: camera, settings: settings).depth
        expectClose(gpu.raw, cpu.raw, absolute: 1e-4, relative: 1e-4, "random.raw")
        expectClose(gpu.alpha, cpu.alpha, absolute: 1e-4, relative: 1e-4, "random.alpha")
    }

    @Test(arguments: ["depth_single", "depth_occluded"])
    func depthMatchesGoldens(name: String) throws {
        let golden = try GoldenCase.load(name)
        let context = try makeTestContext()
        let renderer = try SplatRenderer(context: context, cloud: try golden.cloud())
        let gpu = try forward(renderer, context: context, camera: golden.camera(),
                              settings: golden.settings(), withDepth: true)
        expectClose(gpu.depth!.raw, golden.renderedDepth!, absolute: 1e-4, relative: 1e-4, "\(name).raw")
        expectClose(gpu.depth!.alpha, golden.renderedAlpha!, absolute: 1e-4, relative: 1e-4, "\(name).alpha")
    }

    // MARK: - Backward

    @Test func depthBackwardMatchesCPUReference() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 200, seed: 31)
        let camera = rotatedCamera(width: 64, height: 64)
        var settings = RasterSettings()
        settings.background = SIMD3(0.1, 0.2, 0.3)
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let depthGradient = randomDepthGradient(count: 64 * 64, seed: 32)
        let zeroImage = [SIMD3<Float>](repeating: .zero, count: 64 * 64)

        _ = try renderer.renderDepth(camera: camera, settings: settings)
        let gpuDepthOnly = try renderer.backward(imageGradient: zeroImage, depthGradient: depthGradient)
        let cpuDepthOnly = CPURasterizer.backwardWithScreenSpace(cloud, camera: camera, settings: settings,
                                                                 imageGradient: zeroImage,
                                                                 depthGradient: depthGradient).gradients
        expectGradients(gpuDepthOnly, cpuDepthOnly, absolute: 1e-5, relative: 1e-3, "depthOnly")

        let imageGradient = randomPixels(count: 64 * 64, seed: 33)
        let gpuCombined = try renderer.backward(imageGradient: imageGradient, depthGradient: depthGradient)
        let cpuCombined = CPURasterizer.backwardWithScreenSpace(cloud, camera: camera, settings: settings,
                                                                imageGradient: imageGradient,
                                                                depthGradient: depthGradient).gradients
        expectGradients(gpuCombined, cpuCombined, absolute: 1e-5, relative: 1e-3, "combined")
    }

    @Test(arguments: ["depth_single", "depth_occluded"])
    func depthBackwardMatchesGoldens(name: String) throws {
        let golden = try GoldenCase.load(name)
        let context = try makeTestContext()
        let renderer = try SplatRenderer(context: context, cloud: try golden.cloud())
        _ = try renderer.renderDepth(camera: golden.camera(), settings: golden.settings())
        let gradients = try renderer.backward(imageGradient: golden.imageGradientPixels,
                                              depthGradient: pairs(golden.depthGradient!))
        let expected = GaussianGradients(positions: golden.gradients.positions,
                                         logScales: golden.gradients.logScales,
                                         rotations: golden.gradients.rotations,
                                         opacityLogits: golden.gradients.opacityLogits,
                                         sh: golden.gradients.sh)
        expectGradients(gradients, expected, absolute: 1e-4, relative: 1e-3, name)
    }

    @Test func depthBackwardMatchesFiniteDifferencesOnGPU() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 60, seed: 5)
        let camera = makeTestCamera(width: 32, height: 32)
        var settings = RasterSettings()
        settings.background = SIMD3(0.1, 0.2, 0.3)
        let depthGradient = randomDepthGradient(count: 32 * 32, seed: 41)
        let zeroImage = [SIMD3<Float>](repeating: .zero, count: 32 * 32)

        let base = try SplatRenderer(context: context, cloud: cloud)
        let baseForward = try forward(base, context: context, camera: camera, settings: settings, withDepth: true)
        let analytic = try base.backward(imageGradient: zeroImage, depthGradient: depthGradient)

        // Candidates: positions (group 0) and opacity logits (group 3) with a large analytic gradient.
        var candidates: [(group: Int, index: Int)] = []
        for (index, value) in analytic.positions.enumerated() where abs(value) > 1e-2 {
            candidates.append((group: 0, index: index))
        }
        for (index, value) in analytic.opacityLogits.enumerated() where abs(value) > 1e-2 {
            candidates.append((group: 3, index: index))
        }
        #expect(candidates.count >= 4, "only \(candidates.count) parameters with |gradient| > 1e-2")
        var rng = SplitMix64(seed: 43)
        candidates.shuffle(using: &rng)

        let h: Float = 1e-3
        var checked = 0
        var skippedFootprint = 0
        var failures: [String] = []
        for candidate in candidates {
            if checked == 4 { break }
            let value: Float
            let exact: Float
            if candidate.group == 0 {
                value = cloud.positions[candidate.index]
                exact = analytic.positions[candidate.index]
            } else {
                value = cloud.opacityLogits[candidate.index]
                exact = analytic.opacityLogits[candidate.index]
            }
            let plusCloud = try perturbed(cloud, group: candidate.group, index: candidate.index, to: value + h)
            let minusCloud = try perturbed(cloud, group: candidate.group, index: candidate.index, to: value - h)
            let plusRenderer = try SplatRenderer(context: context, cloud: plusCloud)
            let minusRenderer = try SplatRenderer(context: context, cloud: minusCloud)
            let plus = try forward(plusRenderer, context: context, camera: camera, settings: settings, withDepth: true)
            let minus = try forward(minusRenderer, context: context, camera: camera, settings: settings, withDepth: true)

            // A perturbation that changes the tile footprint crosses a discontinuity: skip it.
            guard plus.tileIntersections == baseForward.tileIntersections,
                  minus.tileIntersections == baseForward.tileIntersections else {
                skippedFootprint += 1
                continue
            }

            let step = Double(value + h) - Double(value - h)
            let numeric = (depthLoss(plus.depth!, depthGradient) - depthLoss(minus.depth!, depthGradient)) / step
            checked += 1
            let relative = abs(numeric - Double(exact)) / max(abs(Double(exact)), 1e-12)
            if !(relative <= 5e-3) {
                failures.append("group \(candidate.group)[\(candidate.index)]: analytic \(exact), numeric \(numeric)")
            }
        }
        #expect(checked == 4, "checked \(checked) parameters, \(skippedFootprint) skipped for footprint changes")
        #expect(failures.isEmpty, "relative error above 5e-3: \(failures)")
    }

    @Test func accumulateAddsToExistingGradients() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 200, seed: 51)
        let camera = rotatedCamera(width: 64, height: 64)
        var settings = RasterSettings()
        settings.background = SIMD3(0.1, 0.2, 0.3)
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        _ = try forward(renderer, context: context, camera: camera, settings: settings, withDepth: true)

        let imageA = randomPixels(count: 64 * 64, seed: 52)
        let depthA = randomDepthGradient(count: 64 * 64, seed: 53)
        let imageB = randomPixels(count: 64 * 64, seed: 54)
        let depthB = randomDepthGradient(count: 64 * 64, seed: 55)

        let gradients = try GradientBuffers(context: context, count: cloud.count, shDegree: cloud.shDegree)
        _ = try encodedBackward(renderer, context: context, camera: camera, imageGradient: imageA,
                                depthGradient: depthA, gradients: gradients, accumulate: false,
                                count: cloud.count, shDegree: cloud.shDegree)
        let accumulated = try encodedBackward(renderer, context: context, camera: camera, imageGradient: imageB,
                                              depthGradient: depthB, gradients: gradients, accumulate: true,
                                              count: cloud.count, shDegree: cloud.shDegree)

        let cpuA = CPURasterizer.backwardWithScreenSpace(cloud, camera: camera, settings: settings,
                                                         imageGradient: imageA, depthGradient: depthA).gradients
        let cpuB = CPURasterizer.backwardWithScreenSpace(cloud, camera: camera, settings: settings,
                                                         imageGradient: imageB, depthGradient: depthB).gradients
        expectGradients(accumulated, sum(cpuA, cpuB), absolute: 1e-5, relative: 1e-3, "accumulate")
    }

    @Test func accumulateLeavesCulledGaussiansUntouched() throws {
        let context = try makeTestContext()
        // 100 visible Gaussians in front of the camera, then 50 behind it (culled by the near plane).
        let visible = try makeRandomCloud(count: 100, seed: 61)
        let behind = try makeRandomCloud(count: 50, seed: 62, depth: (-8)...(-2))
        let cloud = try GaussianCloud(shDegree: visible.shDegree,
                                      positions: visible.positions + behind.positions,
                                      logScales: visible.logScales + behind.logScales,
                                      rotations: visible.rotations + behind.rotations,
                                      opacityLogits: visible.opacityLogits + behind.opacityLogits,
                                      sh: visible.sh + behind.sh)
        let camera = makeTestCamera(width: 64, height: 64)
        var settings = RasterSettings()
        settings.background = SIMD3(0.1, 0.2, 0.3)
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        _ = try forward(renderer, context: context, camera: camera, settings: settings, withDepth: true)

        let imageGradient = randomPixels(count: 64 * 64, seed: 63)
        let depthGradient = randomDepthGradient(count: 64 * 64, seed: 64)
        let sentinel: Float = 7.0
        let gradients = try GradientBuffers(context: context, count: cloud.count, shDegree: cloud.shDegree)
        fill(gradients.positions, count: 3 * cloud.count, value: sentinel)
        fill(gradients.logScales, count: 3 * cloud.count, value: sentinel)
        fill(gradients.rotations, count: 4 * cloud.count, value: sentinel)
        fill(gradients.opacityLogits, count: cloud.count, value: sentinel)
        fill(gradients.sh, count: cloud.count * GaussianCloud.coefficientCount(degree: cloud.shDegree) * 3,
             value: sentinel)

        let result = try encodedBackward(renderer, context: context, camera: camera, imageGradient: imageGradient,
                                         depthGradient: depthGradient, gradients: gradients, accumulate: true,
                                         count: cloud.count, shDegree: cloud.shDegree)
        let cpu = CPURasterizer.backwardWithScreenSpace(cloud, camera: camera, settings: settings,
                                                        imageGradient: imageGradient,
                                                        depthGradient: depthGradient).gradients

        // Culled Gaussians (indices 100..<150) keep the sentinel exactly; visible ones get sentinel + CPU.
        // Per-Gaussian strides: positions 3, logScales 3, rotations 4, opacity 1, SH K·3.
        let shStride = GaussianCloud.coefficientCount(degree: cloud.shDegree) * 3
        var culledTouched = 0
        for gaussian in 100..<150 {
            for c in 0..<3 where result.positions[3 * gaussian + c] != sentinel { culledTouched += 1 }
            for c in 0..<3 where result.logScales[3 * gaussian + c] != sentinel { culledTouched += 1 }
            for c in 0..<4 where result.rotations[4 * gaussian + c] != sentinel { culledTouched += 1 }
            if result.opacityLogits[gaussian] != sentinel { culledTouched += 1 }
            for c in 0..<shStride where result.sh[shStride * gaussian + c] != sentinel { culledTouched += 1 }
        }
        #expect(culledTouched == 0, "culled Gaussians were written: \(culledTouched) entries")

        let shifted = GaussianGradients(
            positions: cpu.positions.map { $0 + sentinel },
            logScales: cpu.logScales.map { $0 + sentinel },
            rotations: cpu.rotations.map { $0 + sentinel },
            opacityLogits: cpu.opacityLogits.map { $0 + sentinel },
            sh: cpu.sh.map { $0 + sentinel })
        expectGradients(result, shifted, absolute: 1e-5, relative: 1e-3, "visibleAccumulate")
    }

    private func fill(_ buffer: MTLBuffer, count: Int, value: Float) {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            pointer[i] = value
        }
    }

    /// Opacity logits near 8 give a raw α of about 0.99966 at the Gaussian centre, above the 0.999 clamp. For those
    /// samples the colour and depth terms are still computed, but opacity, conic and mean2D get no gradient.
    /// The depth slot (9) must still accumulate. The random clouds never reach the clamp (logits in -2...4).
    @Test func clampedSamplesKeepDepthGradient() throws {
        let context = try makeTestContext()
        let base = try makeRandomCloud(count: 40, seed: 81)
        let coefficients = GaussianCloud.coefficientCount(degree: base.shDegree) * 3
        var positions = base.positions
        var logScales = base.logScales
        var rotations = base.rotations
        var opacity = base.opacityLogits
        var sh = base.sh
        for i in 0..<10 {
            opacity[i] = 8
        }
        // A small Gaussian centred exactly on the centre of pixel (32, 32): x = (32.5 − 32)·z/fx with z = 4, fx = 64.
        // Its footprint is about 10 px, so only the centre pixel has raw α > 0.999 (dx = dy = 0 there).
        positions += [0.03125, 0.03125, 4]
        logScales += [log(0.2), log(0.2), log(0.2)]
        rotations += [1, 0, 0, 0]
        opacity += [8]
        sh += [Float](repeating: 0, count: coefficients)
        let cloud = try GaussianCloud(shDegree: base.shDegree, positions: positions, logScales: logScales,
                                      rotations: rotations, opacityLogits: opacity, sh: sh)
        #expect(1 / (1 + exp(-Float(8))) > 0.999, "the centre Gaussian must reach the alpha clamp")

        let camera = makeTestCamera(width: 64, height: 64)
        var settings = RasterSettings()
        settings.background = SIMD3(0.1, 0.2, 0.3)
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        _ = try renderer.renderDepth(camera: camera, settings: settings)

        let zeroImage = [SIMD3<Float>](repeating: .zero, count: 64 * 64)
        let depthGradient = randomDepthGradient(count: 64 * 64, seed: 82)
        let gpuDepthOnly = try renderer.backward(imageGradient: zeroImage, depthGradient: depthGradient)
        let cpuDepthOnly = CPURasterizer.backwardWithScreenSpace(cloud, camera: camera, settings: settings,
                                                                 imageGradient: zeroImage,
                                                                 depthGradient: depthGradient).gradients
        expectGradients(gpuDepthOnly, cpuDepthOnly, absolute: 1e-5, relative: 1e-3, "clamped.depthOnly")
        // The centre Gaussian's own depth gradient is non-zero, so slot 9 was not skipped by the clamp guard.
        let centre = cloud.count - 1
        #expect(gpuDepthOnly.positions[3 * centre + 2] != 0, "centre Gaussian got no depth gradient")

        let imageGradient = randomPixels(count: 64 * 64, seed: 83)
        let gpuCombined = try renderer.backward(imageGradient: imageGradient, depthGradient: depthGradient)
        let cpuCombined = CPURasterizer.backwardWithScreenSpace(cloud, camera: camera, settings: settings,
                                                                imageGradient: imageGradient,
                                                                depthGradient: depthGradient).gradients
        expectGradients(gpuCombined, cpuCombined, absolute: 1e-5, relative: 1e-3, "clamped.combined")
    }

    /// Six stacked Gaussians at the same spot with α ≈ 0.98 near the centre: T is about 4e-4 after two samples and
    /// 1.5e-4 after three, and the fourth sample would take it below 1e-4, so the forward loop stops after three
    /// blended samples. The centre pixel must record `lastContributor == 3` and `finalT < 1e-3` (read back from the
    /// renderer's buffers; `lastContributor` is relative to the tile start, which is 0 for the single 16×16 tile). Depth forward and depth
    /// backward are then compared with the CPU reference, which terminates at the same sample.
    @Test func earlyTerminationMatchesCPUReference() throws {
        let context = try makeTestContext()
        let count = 6
        var positions: [Float] = []
        var logScales: [Float] = []
        var rotations: [Float] = []
        var opacity: [Float] = []
        var sh: [Float] = []
        for i in 0..<count {
            positions += [0, 0, 3 + 0.1 * Float(i)]
            logScales += [log(0.5), log(0.5), log(0.5)]
            rotations += [1, 0, 0, 0]
            opacity += [4]
            sh += [0.8, 0.4, 0.2]
        }
        let cloud = try GaussianCloud(shDegree: 0, positions: positions, logScales: logScales,
                                      rotations: rotations, opacityLogits: opacity, sh: sh)
        // One 16×16 tile, so the sorted index of the tile starts at 0 and lastContributor counts blended samples.
        let camera = makeTestCamera(width: 16, height: 16)
        var settings = RasterSettings()
        settings.background = SIMD3(0.1, 0.2, 0.3)
        let renderer = try SplatRenderer(context: context, cloud: cloud)

        let gpuDepth = try renderer.renderDepth(camera: camera, settings: settings)
        let cpu = CPURasterizer.renderWithDepth(cloud, camera: camera, settings: settings)

        // Read back the transmittance and last-contributor buffers of the frame just rendered.
        let pixelCount = 16 * 16
        let centre = 8 * 16 + 8
        let finalT = renderer.finalTransmittance!.contents().bindMemory(to: Float.self, capacity: pixelCount)
        let lastContributor = renderer.lastContributor!.contents().bindMemory(to: UInt32.self, capacity: pixelCount)
        var terminatedPixels = 0
        for pixel in 0..<pixelCount where lastContributor[pixel] < 6 && finalT[pixel] < 1e-3 {
            terminatedPixels += 1
        }
        #expect(terminatedPixels > 0, "no pixel terminated early: the test does not exercise the early exit")
        #expect(lastContributor[centre] == 3, "the centre pixel should stop after three blended samples of six")
        #expect(finalT[centre] < 1e-3, "the centre pixel's transmittance should be saturated")

        expectClose(gpuDepth.raw, cpu.depth.raw, absolute: 1e-4, relative: 1e-4, "early.raw")
        expectClose(gpuDepth.alpha, cpu.depth.alpha, absolute: 1e-4, relative: 1e-4, "early.alpha")

        let zeroImage = [SIMD3<Float>](repeating: .zero, count: pixelCount)
        let depthGradient = randomDepthGradient(count: pixelCount, seed: 91)
        let gpuDepthOnly = try renderer.backward(imageGradient: zeroImage, depthGradient: depthGradient)
        let cpuDepthOnly = CPURasterizer.backwardWithScreenSpace(cloud, camera: camera, settings: settings,
                                                                 imageGradient: zeroImage,
                                                                 depthGradient: depthGradient).gradients
        expectGradients(gpuDepthOnly, cpuDepthOnly, absolute: 1e-5, relative: 1e-3, "early.depthOnly")

        let imageGradient = randomPixels(count: pixelCount, seed: 92)
        let gpuCombined = try renderer.backward(imageGradient: imageGradient, depthGradient: depthGradient)
        let cpuCombined = CPURasterizer.backwardWithScreenSpace(cloud, camera: camera, settings: settings,
                                                                imageGradient: imageGradient,
                                                                depthGradient: depthGradient).gradients
        expectGradients(gpuCombined, cpuCombined, absolute: 1e-5, relative: 1e-3, "early.combined")
    }

    /// Logs the GPU time of the forward and backward pass at 64×64, depth off and on (best of 5 after a warm-up).
    /// The numbers go into the commit notes; the test only asserts that the timings are finite and positive.
    @Test func logsGPUTimingsAt64x64() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 2_000, seed: 71)
        let camera = makeTestCamera(width: 64, height: 64)
        let settings = RasterSettings()
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let imageGradient = randomPixels(count: 64 * 64, seed: 72)
        let depthGradient = randomDepthGradient(count: 64 * 64, seed: 73)
        let gradients = try GradientBuffers(context: context, count: cloud.count, shDegree: cloud.shDegree)

        var timings: [String] = []
        for withDepth in [false, true] {
            var bestForward = Double.infinity
            var bestBackward = Double.infinity
            for run in 0..<6 {
                let texture = try makeTexture(context: context, width: 64, height: 64)
                var depthBuffer: MTLBuffer? = nil
                if withDepth {
                    depthBuffer = try context.makeBuffer(length: 64 * 64 * 8)
                }
                let pass = try renderer.encodeForward(camera: camera, settings: settings, to: texture, depth: depthBuffer)
                pass.commandBuffer.commit()
                pass.commandBuffer.waitUntilCompleted()
                let forwardMs = (pass.projectionSeconds + pass.commandBuffer.gpuEndTime - pass.commandBuffer.gpuStartTime) * 1000

                let imageBuffer = try context.makeBuffer(length: 64 * 64 * 16)
                let imagePointer = imageBuffer.contents().bindMemory(to: Float.self, capacity: 64 * 64 * 4)
                for i in 0..<(64 * 64) {
                    imagePointer[4 * i] = imageGradient[i].x
                    imagePointer[4 * i + 1] = imageGradient[i].y
                    imagePointer[4 * i + 2] = imageGradient[i].z
                    imagePointer[4 * i + 3] = 0
                }
                var depthGradientBuffer: MTLBuffer? = nil
                if withDepth {
                    let buffer = try context.makeBuffer(length: 64 * 64 * 8)
                    let pointer = buffer.contents().bindMemory(to: Float.self, capacity: 64 * 64 * 2)
                    for i in 0..<(64 * 64) {
                        pointer[2 * i] = depthGradient[i].x
                        pointer[2 * i + 1] = depthGradient[i].y
                    }
                    depthGradientBuffer = buffer
                }
                guard let backwardBuffer = context.queue.makeCommandBuffer() else {
                    throw MetalError.commandFailed("cannot create command buffer")
                }
                try renderer.encodeBackward(imageGradient: imageBuffer, depthGradient: depthGradientBuffer,
                                            into: gradients, commandBuffer: backwardBuffer)
                backwardBuffer.commit()
                backwardBuffer.waitUntilCompleted()
                let backwardMs = (backwardBuffer.gpuEndTime - backwardBuffer.gpuStartTime) * 1000

                if run > 0 {
                    bestForward = min(bestForward, forwardMs)
                    bestBackward = min(bestBackward, backwardMs)
                }
            }
            #expect(bestForward.isFinite && bestForward > 0 && bestBackward.isFinite && bestBackward > 0)
            timings.append("depth=\(withDepth ? "on" : "off") forward=\(String(format: "%.3f", bestForward)) ms backward=\(String(format: "%.3f", bestBackward)) ms")
        }
        print("DEPTH_TIMING 64x64 gaussians=\(cloud.count) best-of-5: " + timings.joined(separator: " | "))
    }

    private func makeTexture(context: MetalContext, width: Int, height: Int) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderWrite, .shaderRead]
        descriptor.storageMode = .shared
        guard let texture = context.device.makeTexture(descriptor: descriptor) else {
            throw MetalError.bufferAllocation
        }
        return texture
    }

    /// Pinhole camera with R = Rx(0.35)·Ry(−0.25) and t = (0.3, −0.2, 0.4).
    private func rotatedCamera(width: Int, height: Int) -> Camera {
        func axisAngle(_ axis: SIMD3<Float>, _ angle: Float) -> SIMD4<Float> {
            let half = angle / 2
            return SIMD4(cos(half), sin(half) * axis.x, sin(half) * axis.y, sin(half) * axis.z)
        }
        let rx = Matrix3.rotation(quaternion: axisAngle(SIMD3(1, 0, 0), 0.35))
        let ry = Matrix3.rotation(quaternion: axisAngle(SIMD3(0, 1, 0), -0.25))
        return Camera(width: width, height: height,
                      fx: Float(height), fy: Float(height),
                      cx: Float(width) / 2, cy: Float(height) / 2,
                      rotation: rx * ry, translation: SIMD3(0.3, -0.2, 0.4))
    }
}
