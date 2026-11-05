import Foundation
import Metal
import Testing
import SplatCore
@testable import SplatMetal

/// Capacity-based Gaussian store and the renderer generalization built on it.
@Suite(.enabled(if: TestEnvironment.hasMetal))
struct GaussianBuffersTests {

    // MARK: - Helpers

    /// Exact comparison of two clouds (`GaussianCloud` is not Equatable).
    private func expectSameCloud(_ a: GaussianCloud, _ b: GaussianCloud) {
        #expect(a.shDegree == b.shDegree)
        #expect(a.count == b.count)
        #expect(a.positions == b.positions)
        #expect(a.logScales == b.logScales)
        #expect(a.rotations == b.rotations)
        #expect(a.opacityLogits == b.opacityLogits)
        #expect(a.sh == b.sh)
    }

    /// The first `n` Gaussians of `cloud`.
    private func prefix(_ cloud: GaussianCloud, count n: Int) throws -> GaussianCloud {
        let k = GaussianCloud.coefficientCount(degree: cloud.shDegree)
        return try GaussianCloud(shDegree: cloud.shDegree,
                                 positions: Array(cloud.positions[0..<3 * n]),
                                 logScales: Array(cloud.logScales[0..<3 * n]),
                                 rotations: Array(cloud.rotations[0..<4 * n]),
                                 opacityLogits: Array(cloud.opacityLogits[0..<n]),
                                 sh: Array(cloud.sh[0..<n * k * 3]))
    }

    private func maxDifference(_ a: RGBImage, _ b: RGBImage) -> Float {
        guard a.pixels.count == b.pixels.count else { return .infinity }
        var worst: Float = 0
        for (p, q) in zip(a.pixels, b.pixels) {
            for c in 0..<3 {
                let error = abs(p[c] - q[c])
                if !(error <= worst) { worst = error.isNaN ? .infinity : error }
            }
        }
        return worst
    }

    private func settings() -> RasterSettings {
        var settings = RasterSettings()
        settings.background = SIMD3(0.1, 0.2, 0.3)
        return settings
    }

    private func image(of renderer: SplatRenderer, width: Int, height: Int) throws -> RGBImage {
        try renderer.renderImage(camera: makeTestCamera(width: width, height: height), settings: settings())
    }

    private func makeRenderTarget(_ context: MetalContext, width: Int, height: Int) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderWrite, .shaderRead]
        descriptor.storageMode = .shared
        guard let texture = context.device.makeTexture(descriptor: descriptor) else {
            throw MetalError.bufferAllocation
        }
        return texture
    }

    private func randomImageGradient(count: Int, seed: UInt64) -> [SIMD3<Float>] {
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

    /// RGBA float buffer holding `values` in the layout `encodeBackward` expects (A = 0).
    private func makeGradientImage(_ context: MetalContext, values: [SIMD3<Float>]) throws -> MTLBuffer {
        let buffer = try context.makeBuffer(length: values.count * 4 * MemoryLayout<Float>.size)
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: values.count * 4)
        for (i, value) in values.enumerated() {
            pointer[4 * i] = value.x
            pointer[4 * i + 1] = value.y
            pointer[4 * i + 2] = value.z
            pointer[4 * i + 3] = 0
        }
        return buffer
    }

    // MARK: - Store

    @Test func uploadSnapshotRoundTrip() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 37, seed: 3, shDegree: 1)
        let store = try GaussianBuffers(context: context, capacity: 100, shDegree: 1)
        #expect(store.count == 0)
        store.upload(cloud)
        #expect(store.count == 37)
        #expect(store.capacity == 100)
        expectSameCloud(store.snapshot(), cloud)
    }

    @Test func growPreservesContents() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 37, seed: 4, shDegree: 1)
        let store = try GaussianBuffers(context: context, cloud: cloud, capacity: 100)
        try store.grow(toCapacity: 250)
        #expect(store.capacity == 250)
        #expect(store.count == 37)
        expectSameCloud(store.snapshot(), cloud)

        // Spare capacity after the live Gaussians is zero-filled.
        let positions = store.positions.contents().bindMemory(to: Float.self, capacity: 3 * 250)
        for i in (3 * 37)..<(3 * 250) {
            #expect(positions[i] == 0)
        }
    }

    // MARK: - Renderer

    @Test func rendererOverLargerCapacityMatchesCloudRenderer() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 120, seed: 21)
        let store = try GaussianBuffers(context: context, cloud: cloud, capacity: 3 * cloud.count)
        let renderer = try SplatRenderer(context: context, gaussians: store)
        let reference = try SplatRenderer(context: context, cloud: cloud)

        let actual = try image(of: renderer, width: 64, height: 64)
        let expected = try image(of: reference, width: 64, height: 64)
        #expect(actual.pixels.count == 64 * 64)
        #expect(maxDifference(actual, expected) == 0)
    }

    @Test func countChangeIsVisibleToRenderer() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 120, seed: 22)
        let store = try GaussianBuffers(context: context, cloud: cloud, capacity: 3 * cloud.count)
        let renderer = try SplatRenderer(context: context, gaussians: store)

        store.count = cloud.count / 2
        let actual = try image(of: renderer, width: 64, height: 64)
        let halfCloud = try prefix(cloud, count: cloud.count / 2)
        let expected = try image(of: SplatRenderer(context: context, cloud: halfCloud), width: 64, height: 64)
        #expect(maxDifference(actual, expected) == 0)
    }

    @Test func encodeForwardThenBackwardMatchesBackward() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 80, seed: 31, scale: 0.05...0.4)
        let camera = makeTestCamera(width: 32, height: 32)
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let texture = try makeRenderTarget(context, width: 32, height: 32)
        let imageGradient = randomImageGradient(count: 32 * 32, seed: 4)

        // Fused: forward and backward share one command buffer.
        let pass = try renderer.encodeForward(camera: camera, settings: settings(), to: texture)
        let gradientImage = try makeGradientImage(context, values: imageGradient)
        let gradients = try GradientBuffers(context: context, count: cloud.count, shDegree: cloud.shDegree)
        try renderer.encodeBackward(imageGradient: gradientImage, into: gradients, commandBuffer: pass.commandBuffer)
        pass.commandBuffer.commit()
        pass.commandBuffer.waitUntilCompleted()
        #expect(pass.commandBuffer.status != .error)
        let fused = gradients.toGaussianGradients(count: cloud.count, shDegree: cloud.shDegree)

        // Reference: the same frame through the synchronous path.
        let reference = try renderer.backward(imageGradient: imageGradient)

        func expectClose(_ a: [Float], _ b: [Float], _ name: String) {
            #expect(a.count == b.count, "\(name) length")
            guard a.count == b.count else { return }
            var worst: Float = 0
            for (x, y) in zip(a, b) {
                let scale = max(1, abs(y))
                worst = max(worst, abs(x - y) / scale)
            }
            // Float atomics sum per-pixel contributions in a scheduling-dependent order, so two runs
            // agree only to a few ulps of the accumulated values (observed up to ~1.2e-6).
            #expect(worst <= 2e-5, "\(name) relative error \(worst)")
        }
        expectClose(fused.positions, reference.positions, "positions")
        expectClose(fused.logScales, reference.logScales, "logScales")
        expectClose(fused.rotations, reference.rotations, "rotations")
        expectClose(fused.opacityLogits, reference.opacityLogits, "opacityLogits")
        expectClose(fused.sh, reference.sh, "sh")
    }

    @Test func backwardRejectsCountChange() throws {
        let context = try makeTestContext()
        let cloud = try makeRandomCloud(count: 60, seed: 51)
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let texture = try makeRenderTarget(context, width: 32, height: 32)
        try renderer.render(camera: makeTestCamera(width: 32, height: 32), settings: settings(), to: texture)

        renderer.gaussians.count = 30
        let imageGradient = randomImageGradient(count: 32 * 32, seed: 5)
        #expect(throws: MetalError.self) {
            _ = try renderer.backward(imageGradient: imageGradient)
        }
    }

    @Test func setCloudReplacesScene() throws {
        let context = try makeTestContext()
        let first = try makeRandomCloud(count: 50, seed: 41, shDegree: 3)
        let second = try makeRandomCloud(count: 80, seed: 42, shDegree: 1)
        let renderer = try SplatRenderer(context: context, cloud: first)

        // Different SH degree: the store is replaced.
        try renderer.setCloud(second)
        #expect(renderer.gaussians.count == 80)
        #expect(renderer.gaussians.shDegree == 1)
        let actual = try image(of: renderer, width: 48, height: 40)
        let expected = try image(of: SplatRenderer(context: context, cloud: second), width: 48, height: 40)
        #expect(maxDifference(actual, expected) == 0)

        // Same degree and fits the capacity: the store is updated in place.
        let store = renderer.gaussians
        let third = try makeRandomCloud(count: 40, seed: 43, shDegree: 1)
        try renderer.setCloud(third)
        #expect(renderer.gaussians === store)
        #expect(renderer.gaussians.count == 40)
        expectSameCloud(renderer.gaussians.snapshot(), third)
    }
}
