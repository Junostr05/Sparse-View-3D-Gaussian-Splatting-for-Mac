import Foundation
import Metal
import Testing
import SplatCore
@testable import SplatMetal

/// GPU L1 + D-SSIM loss and its gradient against `ImageLoss.evaluate`, and the fused path
/// into `SplatRenderer.encodeBackward`.
@Suite(.enabled(if: TestEnvironment.hasMetal))
struct GPUImageLossTests {

    // MARK: - Helpers

    private func makeRandomImage(width: Int, height: Int, seed: UInt64) -> RGBImage {
        var rng = SplitMix64(seed: seed)
        var pixels: [SIMD3<Float>] = []
        pixels.reserveCapacity(width * height)
        for _ in 0..<(width * height) {
            let r = Float.random(in: 0...1, using: &rng)
            let g = Float.random(in: 0...1, using: &rng)
            let b = Float.random(in: 0...1, using: &rng)
            pixels.append(SIMD3(r, g, b))
        }
        return RGBImage(width: width, height: height, pixels: pixels)
    }

    /// Target correlated with `rendered` (so SSIM is far from zero): a blend with independent noise,
    /// clamped to [0, 1].
    private func makeCorrelatedTarget(of rendered: RGBImage, seed: UInt64) -> RGBImage {
        let noise = makeRandomImage(width: rendered.width, height: rendered.height, seed: seed)
        var pixels: [SIMD3<Float>] = []
        pixels.reserveCapacity(rendered.pixels.count)
        for i in rendered.pixels.indices {
            let mixed = 0.6 * rendered.pixels[i] + 0.4 * noise.pixels[i]
            pixels.append(pointwiseMin(pointwiseMax(mixed, SIMD3(repeating: 0)), SIMD3(repeating: 1)))
        }
        return RGBImage(width: rendered.width, height: rendered.height, pixels: pixels)
    }

    private func makeRenderTexture(_ context: MetalContext, width: Int, height: Int) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderWrite, .shaderRead]
        descriptor.storageMode = .shared
        guard let texture = context.device.makeTexture(descriptor: descriptor) else {
            throw MetalError.bufferAllocation
        }
        return texture
    }

    /// Uploads an RGB image into an rgba32Float texture (alpha = 1; the loss ignores it).
    private func makeTexture(_ context: MetalContext, image: RGBImage) throws -> MTLTexture {
        let texture = try makeRenderTexture(context, width: image.width, height: image.height)
        var raw: [Float] = []
        raw.reserveCapacity(image.pixels.count * 4)
        for p in image.pixels {
            raw += [p.x, p.y, p.z, 1]
        }
        raw.withUnsafeBytes { bytes in
            texture.replace(region: MTLRegionMake2D(0, 0, image.width, image.height),
                            mipmapLevel: 0,
                            withBytes: bytes.baseAddress!,
                            bytesPerRow: image.width * 4 * MemoryLayout<Float>.size)
        }
        return texture
    }

    private struct GPUResult {
        var value: ImageLossValue
        var gradient: [SIMD3<Float>]
    }

    /// Runs the GPU loss in its own command buffer and reads back value and gradient.
    private func runGPULoss(context: MetalContext, rendered: RGBImage, target: RGBImage,
                            lambda: Float) throws -> GPUResult {
        let loss = try GPUImageLoss(context: context)
        let texture = try makeTexture(context, image: rendered)
        let targetBuffer = try loss.makeTarget(target)
        let count = rendered.width * rendered.height
        let gradientBuffer = try context.makeBuffer(length: count * 4 * MemoryLayout<Float>.size)
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.commandFailed("cannot create command buffer")
        }
        try loss.encode(rendered: texture, target: targetBuffer, width: rendered.width,
                        height: rendered.height, lambda: lambda, imageGradient: gradientBuffer,
                        commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status != .error, "loss command buffer failed")

        let pointer = gradientBuffer.contents().bindMemory(to: Float.self, capacity: count * 4)
        var gradient: [SIMD3<Float>] = []
        gradient.reserveCapacity(count)
        for i in 0..<count {
            gradient.append(SIMD3(pointer[4 * i], pointer[4 * i + 1], pointer[4 * i + 2]))
        }
        // The alpha component is documented as 0.
        for i in 0..<count {
            #expect(pointer[4 * i + 3] == 0, "alpha of gradient pixel \(i) is not zero")
        }
        return GPUResult(value: loss.readValue(), gradient: gradient)
    }

    /// Scalar within 1e-4 relative (with a 1e-6 absolute floor for values near zero).
    private func expectScalar(_ gpu: Double, _ cpu: Double, _ name: String) {
        let tolerance = 1e-4 * abs(cpu) + 1e-6
        #expect(abs(gpu - cpu) <= tolerance, "\(name): gpu \(gpu), cpu \(cpu)")
    }

    /// Every component within 1e-6 + 1e-3·|cpu|.
    private func expectGradient(_ gpu: [SIMD3<Float>], _ cpu: [SIMD3<Float>]) {
        #expect(gpu.count == cpu.count, "gradient count \(gpu.count) vs \(cpu.count)")
        guard gpu.count == cpu.count else { return }
        var failures = 0
        var first: String?
        for i in gpu.indices {
            for c in 0..<3 {
                let tolerance: Float = 1e-6 + 1e-3 * abs(cpu[i][c])
                if !(abs(gpu[i][c] - cpu[i][c]) <= tolerance) {
                    failures += 1
                    if first == nil {
                        first = "pixel \(i) channel \(c): gpu \(gpu[i][c]), cpu \(cpu[i][c])"
                    }
                }
            }
        }
        #expect(failures == 0, "\(failures) gradient components outside tolerance; first: \(first ?? "")")
    }

    private func expectParameterMatch(_ gpu: [Float], _ cpu: [Float], _ name: String) {
        #expect(gpu.count == cpu.count, "\(name): count \(gpu.count) vs \(cpu.count)")
        guard gpu.count == cpu.count else { return }
        var failures = 0
        for i in gpu.indices {
            let tolerance: Float = 1e-4 + 1e-3 * abs(cpu[i])
            if !(abs(gpu[i] - cpu[i]) <= tolerance) {
                failures += 1
            }
        }
        #expect(failures == 0, "\(name): \(failures) of \(gpu.count) entries outside tolerance")
    }

    private func expectMatchesCPU(width: Int, height: Int, seed: UInt64) throws {
        let context = try makeTestContext()
        let rendered = makeRandomImage(width: width, height: height, seed: seed)
        let target = makeCorrelatedTarget(of: rendered, seed: seed &+ 1000)
        let lambda: Float = 0.2

        let cpu = ImageLoss.evaluate(rendered: rendered, target: target, lambda: lambda)
        let gpu = try runGPULoss(context: context, rendered: rendered, target: target, lambda: lambda)

        expectScalar(gpu.value.loss, cpu.loss, "loss")
        expectScalar(gpu.value.l1, cpu.l1, "l1")
        expectScalar(gpu.value.ssim, cpu.ssim, "ssim")
        expectGradient(gpu.gradient, cpu.gradient)
    }

    // MARK: - Tests

    @Test func matchesCPULossOnOddSize() throws {
        try expectMatchesCPU(width: 37, height: 29, seed: 9)
    }

    @Test func matchesCPULossOnAnotherOddSize() throws {
        try expectMatchesCPU(width: 37, height: 23, seed: 17)
    }

    @Test func matchesCPULossOnSinglePixel() throws {
        try expectMatchesCPU(width: 1, height: 1, seed: 5)
    }

    @Test func identicalImagesGiveZeroLoss() throws {
        let context = try makeTestContext()
        let image = makeRandomImage(width: 37, height: 23, seed: 3)
        let gpu = try runGPULoss(context: context, rendered: image, target: image, lambda: 0.2)

        #expect(abs(gpu.value.loss) <= 1e-5, "loss \(gpu.value.loss) is not zero")
        #expect(gpu.value.l1 == 0, "l1 \(gpu.value.l1) is not zero")
        #expect(abs(gpu.value.ssim - 1) <= 1e-5, "ssim \(gpu.value.ssim) is not one")
        let largest = gpu.gradient.flatMap { [$0.x, $0.y, $0.z] }.map { abs($0) }.max() ?? 0
        #expect(largest <= 1e-6, "gradient magnitude \(largest) is not zero")
    }

    /// Regression (real-photo `nonFiniteLoss`): where render and target are anticorrelated, a₂ = 2σxy + C₂ crosses
    /// zero. The backward used to divide by a₂ (in `2s/a₂` and `s·2μy/a₂`, which cancel analytically), so a window
    /// whose a₂ rounded to exactly 0 wrote NaN into the SSIM maps, and the 11×11 blur spread it over 121 pixels of
    /// the gradient. Here each 24-row block has its own levels: render μx + r·d + jitter and target μy − d/r for a ±a
    /// checkerboard d, with a² swept across the columns through C₂/2, so σxy ≈ −a² meets −C₂/2 within float rounding in
    /// many windows (on an M-series GPU, 13 windows round a₂ to exactly 0, which failed this test before the fix).
    @Test func gradientIsFiniteWhereSSIMCovarianceTermCrossesZero() throws {
        let context = try makeTestContext()
        let width = 512
        let height = 384
        var rng = SplitMix64(seed: 77)
        let halfC2: Double = Double(ImageLoss.c2) / 2
        var renderedPixels: [SIMD3<Float>] = []
        var targetPixels: [SIMD3<Float>] = []
        renderedPixels.reserveCapacity(width * height)
        targetPixels.reserveCapacity(width * height)
        let blockRows = 24
        let blocks = height / blockRows
        var levels: [[SIMD3<Double>]] = []
        for _ in 0..<blocks {
            var block: [SIMD3<Double>] = []
            for _ in 0..<3 {
                let meanX: Double = Double.random(in: 0.1...0.9, using: &rng)
                let meanY: Double = Double.random(in: 0.1...0.9, using: &rng)
                let ratio: Double = Double.random(in: 0.3...3, using: &rng)
                block.append(SIMD3<Double>(meanX, meanY, ratio))
            }
            levels.append(block)
        }
        for y in 0..<height {
            for x in 0..<width {
                let sweep: Double = (Double(x) / Double(width - 1) - 0.5) * 2e-3
                let sign: Double = (x + y) % 2 == 0 ? 1 : -1
                var r = SIMD3<Float>(repeating: 0)
                var t = SIMD3<Float>(repeating: 0)
                for c in 0..<3 {
                    let level: SIMD3<Double> = levels[min(y / blockRows, blocks - 1)][c]
                    let amplitude: Double = (halfC2 * (1 + sweep)).squareRoot()
                    let d: Double = sign * amplitude
                    let jitter: Double = Double.random(in: -1e-4...1e-4, using: &rng)
                    r[c] = Float(level.x + d * level.z + jitter)
                    t[c] = Float(level.y - d / level.z)
                }
                renderedPixels.append(r)
                targetPixels.append(t)
            }
        }
        let rendered = RGBImage(width: width, height: height, pixels: renderedPixels)
        let target = RGBImage(width: width, height: height, pixels: targetPixels)
        let lambda: Float = 0.2

        let gpu = try runGPULoss(context: context, rendered: rendered, target: target, lambda: lambda)
        var nonFinite = 0
        for g in gpu.gradient {
            for c in 0..<3 where !g[c].isFinite {
                nonFinite += 1
            }
        }
        #expect(gpu.value.loss.isFinite, "loss \(gpu.value.loss) is not finite")
        #expect(nonFinite == 0, "\(nonFinite) non-finite gradient components")

        let cpu = ImageLoss.evaluate(rendered: rendered, target: target, lambda: lambda)
        #expect(cpu.gradient.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite },
                "CPU reference gradient is not finite")
        expectScalar(gpu.value.loss, cpu.loss, "loss")
        expectGradient(gpu.gradient, cpu.gradient)
    }

    /// Forward render, loss and backward in one command buffer. The parameter gradients must match
    /// `CPURasterizer.backward` fed with `ImageLoss.evaluate`'s gradient.
    @Test func gradientFeedsEncodeBackward() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 60, seed: 21, scale: 0.05...0.4)
        let camera = makeTestCamera(width: 32, height: 32)
        let settings = RasterSettings()
        let lambda: Float = 0.2

        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let target = makeRandomImage(width: 32, height: 32, seed: 22)
        let loss = try GPUImageLoss(context: context)
        let targetBuffer = try loss.makeTarget(target)
        let texture = try makeRenderTexture(context, width: 32, height: 32)
        let imageGradient = try context.makeBuffer(length: 32 * 32 * 4 * MemoryLayout<Float>.size)
        let gradients = try GradientBuffers(context: context, count: cloud.count, shDegree: cloud.shDegree)

        // Fused: forward, loss and backward share one command buffer.
        let pass = try renderer.encodeForward(camera: camera, settings: settings, to: texture)
        try loss.encode(rendered: texture, target: targetBuffer, width: 32, height: 32, lambda: lambda,
                        imageGradient: imageGradient, commandBuffer: pass.commandBuffer)
        try renderer.encodeBackward(imageGradient: imageGradient, into: gradients,
                                    commandBuffer: pass.commandBuffer)
        pass.commandBuffer.commit()
        pass.commandBuffer.waitUntilCompleted()
        #expect(pass.commandBuffer.status != .error, "fused command buffer failed")
        let fused = gradients.toGaussianGradients(count: cloud.count, shDegree: cloud.shDegree)

        // Reference: CPU render, CPU loss gradient, CPU backward.
        let rendered = CPURasterizer.render(cloud, camera: camera, settings: settings)
        let cpuLoss = ImageLoss.evaluate(rendered: rendered, target: target, lambda: lambda)
        let cpu = CPURasterizer.backward(cloud, camera: camera, settings: settings,
                                         imageGradient: cpuLoss.gradient)
        expectParameterMatch(fused.positions, cpu.positions, "positions")
        expectParameterMatch(fused.logScales, cpu.logScales, "logScales")
        expectParameterMatch(fused.rotations, cpu.rotations, "rotations")
        expectParameterMatch(fused.opacityLogits, cpu.opacityLogits, "opacityLogits")
        expectParameterMatch(fused.sh, cpu.sh, "sh")
    }
}
