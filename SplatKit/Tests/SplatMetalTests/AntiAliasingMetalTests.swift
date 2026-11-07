import Foundation
import Metal
import Testing
import SplatCore
@testable import SplatMetal

/// Mip-Splatting anti-aliasing on the GPU against the CPU reference: sampling rates, the forward image and the
/// backward gradients. Same tolerances as the classic Metal-vs-CPU tests (1e-4 + 1e-3·|cpu|).
@Suite(.enabled(if: TestEnvironment.hasMetal))
struct AntiAliasingMetalTests {

    // MARK: - Helpers

    /// Pinhole camera with R = Rx(0.35)·Ry(−0.25) and t = (0.3, −0.2, 0.4).
    private func rotatedCamera(width: Int, height: Int) -> Camera {
        let rx = Matrix3.rotation(quaternion: SIMD4(cos(0.175), sin(0.175), 0, 0))
        let ry = Matrix3.rotation(quaternion: SIMD4(cos(-0.125), 0, sin(-0.125), 0))
        return Camera(width: width, height: height,
                      fx: Float(height), fy: Float(height),
                      cx: Float(width) / 2, cy: Float(height) / 2,
                      rotation: rx * ry, translation: SIMD3(0.3, -0.2, 0.4))
    }

    /// Mip settings for `cloud` and `cameras`: the GPU buffer gets the same ν as the CPU array.
    private func mipSettings(v3: Float, samplingRates: [Float]) -> RasterSettings {
        var settings = RasterSettings()
        settings.background = SIMD3(0.1, 0.2, 0.3)
        settings.mipFilter = MipFilter(isEnabled: true, filter2DVariance: 0.1, filter3DVariance: v3,
                                       samplingRates: samplingRates)
        return settings
    }

    /// Computes ν on the GPU into the renderer's sampling-rate buffer and returns it as an array.
    private func gpuSamplingRates(renderer: SplatRenderer, context: MetalContext,
                                  cameras: [Camera]) throws -> [Float] {
        let count = renderer.gaussians.count
        let kernel = try GPUSamplingRate(context: context)
        let buffer = try context.makeBuffer(length: max(1, count) * MemoryLayout<Float>.size)
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.bufferAllocation
        }
        try kernel.encode(positions: renderer.gaussians.positions, count: count, cameras: cameras,
                          near: 0.01, into: buffer, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try #require(commandBuffer.status != .error)
        renderer.samplingRateBuffer = buffer
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    /// Renders through `render(camera:settings:to:)` (keeps the frame state for `backward`) and reads RGB back.
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

    private func randomImageGradient(count: Int, seed: UInt64) -> [SIMD3<Float>] {
        var rng = SplitMix64(seed: seed)
        return (0..<count).map { _ in
            SIMD3(Float.random(in: -1...1, using: &rng),
                  Float.random(in: -1...1, using: &rng),
                  Float.random(in: -1...1, using: &rng))
        }
    }

    /// Every entry within 1e-4 + 1e-3·|cpu|, reported as a count of failures.
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

    // MARK: - Tests

    @Test func samplingRateKernelMatchesCPU() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 300, seed: 21, depth: 1...10)
        let cameras = [makeTestCamera(width: 64, height: 64), rotatedCamera(width: 48, height: 40)]
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let gpu = try gpuSamplingRates(renderer: renderer, context: context, cameras: cameras)
        let cpu = SamplingRate.compute(positions: cloud.positions, cameras: cameras, near: 0.01)
        #expect(gpu.count == cpu.count)
        var maxRelative: Float = 0
        for i in gpu.indices {
            maxRelative = max(maxRelative, abs(gpu[i] - cpu[i]) / max(abs(cpu[i]), 1e-6))
        }
        #expect(maxRelative <= 1e-4, "largest relative error \(maxRelative)")
    }

    @Test func mipForwardMatchesCPU() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 200, seed: 23, scale: 0.02...0.3)
        let camera = makeTestCamera(width: 64, height: 64)
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let rates = try gpuSamplingRates(renderer: renderer, context: context, cameras: [camera])
        let settings = mipSettings(v3: 0.2, samplingRates: rates)

        let gpu = try renderFrame(renderer, context: context, camera: camera, settings: settings)
        let cpu = CPURasterizer.render(cloud, camera: camera, settings: settings).pixels
        // Guard against a silently inactive filter: the mip image must differ from the classic one.
        let classic = CPURasterizer.render(cloud, camera: camera, settings: RasterSettings(background: settings.background)).pixels
        var mipEffect: Float = 0
        for i in cpu.indices {
            mipEffect = max(mipEffect, abs(cpu[i].x - classic[i].x), abs(cpu[i].y - classic[i].y),
                            abs(cpu[i].z - classic[i].z))
        }
        #expect(mipEffect > 1e-3, "mip filter changes the image by only \(mipEffect)")
        var failures = 0
        for i in gpu.indices {
            let tolerance: Float = 2e-4 + 1e-3 * max(abs(cpu[i].x), abs(cpu[i].y), abs(cpu[i].z))
            if !(abs(gpu[i].x - cpu[i].x) <= tolerance && abs(gpu[i].y - cpu[i].y) <= tolerance
                 && abs(gpu[i].z - cpu[i].z) <= tolerance) {
                failures += 1
            }
        }
        #expect(failures == 0, "\(failures) of \(gpu.count) pixels outside tolerance")
    }

    @Test func mipBackwardMatchesCPU() throws {
        let context = try makeTestContext()
        // A large 3D variance makes the filter visible in the gradients.
        let cloud = try makeRandomCloud(count: 200, seed: 29, scale: 0.02...0.3)
        let cameras = [makeTestCamera(width: 64, height: 64), rotatedCamera(width: 64, height: 64)]
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        for (index, camera) in cameras.enumerated() {
            let rates = try gpuSamplingRates(renderer: renderer, context: context, cameras: cameras)
            let settings = mipSettings(v3: 1.0, samplingRates: rates)
            _ = try renderFrame(renderer, context: context, camera: camera, settings: settings)
            let imageGradient = randomImageGradient(count: 64 * 64, seed: UInt64(31 + index))
            let gpu = try renderer.backward(imageGradient: imageGradient)
            let cpu = CPURasterizer.backward(cloud, camera: camera, settings: settings,
                                             imageGradient: imageGradient)
            expectMatch(gpu.positions, cpu.positions, "positions (camera \(index))")
            expectMatch(gpu.logScales, cpu.logScales, "logScales (camera \(index))")
            expectMatch(gpu.rotations, cpu.rotations, "rotations (camera \(index))")
            expectMatch(gpu.opacityLogits, cpu.opacityLogits, "opacityLogits (camera \(index))")
            expectMatch(gpu.sh, cpu.sh, "sh (camera \(index))")
        }
    }

    @Test func mipOffLeavesMetalPathUnchanged() throws {
        // Without the mip filter no sampling-rate buffer is needed, and the forward image is the classic one.
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 100, seed: 37)
        let camera = makeTestCamera(width: 48, height: 48)
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let settings = RasterSettings()
        let gpu = try renderFrame(renderer, context: context, camera: camera, settings: settings)
        let cpu = CPURasterizer.render(cloud, camera: camera, settings: settings).pixels
        var maxDifference: Float = 0
        for i in gpu.indices {
            maxDifference = max(maxDifference, abs(gpu[i].x - cpu[i].x), abs(gpu[i].y - cpu[i].y),
                                abs(gpu[i].z - cpu[i].z))
        }
        #expect(maxDifference < 2e-4, "classic image differs by \(maxDifference)")
    }
}
