import Foundation
import Metal
import Testing
import SplatCore
@testable import SplatMetal

/// Experimental DropGaussian dropout (`RasterSettings.dropRate`). A pass with drop rate p must equal the CPU reference
/// on the reduced cloud: the dropped Gaussians removed and the kept ones at opacity σ' = σ / (1 − p). Every gradient
/// except the opacity logit is parametrisation-free and must match directly; the logit gradient differs only by the
/// chain factor k·σ(1 − σ) / (σ'(1 − σ')) with k = 1 / (1 − p). Classic and mip paths. Same tolerances as the other
/// Metal-vs-CPU tests (1e-4 + 1e-3·|cpu|).
@Suite(.enabled(if: TestEnvironment.hasMetal))
struct DropoutMetalTests {

    /// Swift mirror of `dropGaussian` in Common.metal.
    private func isDropped(_ gid: Int, rate: Float, seed: UInt32) -> Bool {
        guard rate > 0 else { return false }
        var h: UInt32 = UInt32(gid) &* 747_796_405 &+ (seed & 0xFF_FFFF) &* 2_891_336_453 &+ 1
        h = ((h >> ((h >> 28) + 4)) ^ h) &* 277_803_737
        h = (h >> 22) ^ h
        return Float(h >> 8) * (1.0 / 16_777_216.0) < rate
    }

    private func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }

    private func renderFrame(_ renderer: SplatRenderer, context: MetalContext,
                             camera: Camera, settings: RasterSettings) throws -> [SIMD3<Float>] {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: camera.width, height: camera.height, mipmapped: false)
        descriptor.usage = [.shaderWrite, .shaderRead]
        descriptor.storageMode = .shared
        guard let texture = context.device.makeTexture(descriptor: descriptor) else {
            throw MetalError.bufferAllocation
        }
        try renderer.render(camera: camera, settings: settings, to: texture)
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
        return image
    }

    private func expectMatch(_ gpu: [Float], _ cpu: [Float], _ name: String) {
        #expect(gpu.count == cpu.count, "\(name): count \(gpu.count) vs \(cpu.count)")
        guard gpu.count == cpu.count else { return }
        var failures = 0
        var firstFailure: Int?
        for i in gpu.indices where !(abs(gpu[i] - cpu[i]) <= 1e-4 + 1e-3 * abs(cpu[i])) {
            failures += 1
            if firstFailure == nil { firstFailure = i }
        }
        let detail = firstFailure.map { "first at \($0): gpu \(gpu[$0]), cpu \(cpu[$0])" } ?? ""
        #expect(failures == 0, "\(name): \(failures) of \(gpu.count) entries outside tolerance; \(detail)")
    }

    /// Entries `[i·stride, (i+1)·stride)` of every kept Gaussian i.
    private func kept(_ values: [Float], stride: Int, keep: [Int]) -> [Float] {
        var out: [Float] = []
        out.reserveCapacity(keep.count * stride)
        for i in keep {
            out.append(contentsOf: values[(i * stride)..<((i + 1) * stride)])
        }
        return out
    }

    private func check(mip: Bool) throws {
        let context = try makeTestContext()
        let rate: Float = 0.3
        let seed: UInt32 = 12_345
        let base = try makeRandomCloud(count: 240, seed: 61, scale: 0.02...0.3)
        // Opacities at most 0.6, so σ / (1 − p) stays a valid opacity for the reference cloud.
        let capLogit: Float = log(0.6 / 0.4)
        let cloud = try GaussianCloud(shDegree: base.shDegree, positions: base.positions, logScales: base.logScales,
                                      rotations: base.rotations, opacityLogits: base.opacityLogits.map { min($0, capLogit) },
                                      sh: base.sh)
        let camera = makeTestCamera(width: 64, height: 64)
        let rates: [Float] = SamplingRate.compute(positions: cloud.positions, cameras: [camera], near: 0.01)

        var keep: [Int] = []
        for i in 0..<cloud.count where !isDropped(i, rate: rate, seed: seed) {
            keep.append(i)
        }
        #expect(keep.count > cloud.count / 2 && keep.count < cloud.count, "kept \(keep.count) of \(cloud.count)")
        let scale: Float = 1 / (1 - rate)
        var reducedLogits: [Float] = []
        for i in keep {
            let used: Float = sigmoid(cloud.opacityLogits[i]) * scale
            reducedLogits.append(log(used / (1 - used)))
        }
        let k = GaussianCloud.coefficientCount(degree: cloud.shDegree) * 3
        let reduced = try GaussianCloud(shDegree: cloud.shDegree,
                                        positions: kept(cloud.positions, stride: 3, keep: keep),
                                        logScales: kept(cloud.logScales, stride: 3, keep: keep),
                                        rotations: kept(cloud.rotations, stride: 4, keep: keep),
                                        opacityLogits: reducedLogits,
                                        sh: kept(cloud.sh, stride: k, keep: keep))

        var settings = RasterSettings()
        settings.background = SIMD3(0.1, 0.2, 0.3)
        var cpuSettings = settings
        if mip {
            settings.mipFilter = MipFilter(isEnabled: true, filter2DVariance: 0.1, filter3DVariance: 1.0,
                                           samplingRates: rates)
            cpuSettings.mipFilter = MipFilter(isEnabled: true, filter2DVariance: 0.1, filter3DVariance: 1.0,
                                              samplingRates: keep.map { rates[$0] })
        }
        settings.dropRate = rate
        settings.dropSeed = seed

        let renderer = try SplatRenderer(context: context, cloud: cloud)
        if mip {
            let buffer = try context.makeBuffer(length: cloud.count * MemoryLayout<Float>.size)
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: cloud.count)
            for i in 0..<cloud.count { pointer[i] = rates[i] }
            renderer.samplingRateBuffer = buffer
        }
        let gpuImage = try renderFrame(renderer, context: context, camera: camera, settings: settings)
        let cpuImage = CPURasterizer.render(reduced, camera: camera, settings: cpuSettings).pixels
        expectMatch(gpuImage.flatMap { [$0.x, $0.y, $0.z] }, cpuImage.flatMap { [$0.x, $0.y, $0.z] }, "image")

        var rng = SplitMix64(seed: 67)
        let imageGradient: [SIMD3<Float>] = (0..<(64 * 64)).map { _ in
            SIMD3(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                  Float.random(in: -1...1, using: &rng))
        }
        let gpu = try renderer.backward(imageGradient: imageGradient)
        let cpu = CPURasterizer.backward(reduced, camera: camera, settings: cpuSettings, imageGradient: imageGradient)

        // Dropped Gaussians get no gradient at all.
        let keepSet = Set(keep)
        var droppedNonZero = 0
        for i in 0..<cloud.count where !keepSet.contains(i) {
            if gpu.opacityLogits[i] != 0 || gpu.positions[3 * i] != 0 || gpu.logScales[3 * i] != 0 {
                droppedNonZero += 1
            }
        }
        #expect(droppedNonZero == 0, "\(droppedNonZero) dropped Gaussians have a gradient")

        expectMatch(kept(gpu.positions, stride: 3, keep: keep), cpu.positions, "positions")
        expectMatch(kept(gpu.logScales, stride: 3, keep: keep), cpu.logScales, "logScales")
        expectMatch(kept(gpu.rotations, stride: 4, keep: keep), cpu.rotations, "rotations")
        expectMatch(kept(gpu.sh, stride: k, keep: keep), cpu.sh, "sh")
        var expectedLogit: [Float] = []
        for (j, i) in keep.enumerated() {
            let s: Float = sigmoid(cloud.opacityLogits[i])
            let used: Float = sigmoid(reducedLogits[j])
            expectedLogit.append(cpu.opacityLogits[j] * scale * s * (1 - s) / (used * (1 - used)))
        }
        expectMatch(kept(gpu.opacityLogits, stride: 1, keep: keep), expectedLogit, "opacityLogits")
    }

    @Test func classicDropoutMatchesReducedCloud() throws {
        try check(mip: false)
    }

    @Test func mipDropoutMatchesReducedCloud() throws {
        try check(mip: true)
    }

    @Test func zeroRateIsBitIdentical() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 150, seed: 71)
        let camera = makeTestCamera(width: 48, height: 48)
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let plain = try renderFrame(renderer, context: context, camera: camera, settings: RasterSettings())
        var settings = RasterSettings()
        settings.dropSeed = 99
        let seeded = try renderFrame(renderer, context: context, camera: camera, settings: settings)
        #expect(plain == seeded)
    }

    @Test func rateRampsLinearlyAndIsOffByDefault() throws {
        var configuration = TrainingConfiguration()
        #expect(configuration.dropout == nil)
        var dropout = GaussianDropoutSettings()
        #expect(!dropout.isEnabled)
        #expect(dropout.rate(iteration: 500, iterations: 1000) == 0)
        dropout.finalRate = 0.2
        #expect(dropout.rate(iteration: 0, iterations: 1000) == 0)
        #expect(abs(dropout.rate(iteration: 500, iterations: 1000) - 0.1) < 1e-6)
        #expect(abs(dropout.rate(iteration: 2000, iterations: 1000) - 0.2) < 1e-6)
        // A configuration without the key decodes to nil, and nil is not encoded (manifest and hash unchanged).
        let data = try JSONEncoder().encode(configuration)
        #expect(!(String(decoding: data, as: UTF8.self).contains("dropout")))
        configuration.dropout = dropout
        let decoded = try JSONDecoder().decode(TrainingConfiguration.self, from: JSONEncoder().encode(configuration))
        #expect(decoded.dropout == dropout)
    }
}
