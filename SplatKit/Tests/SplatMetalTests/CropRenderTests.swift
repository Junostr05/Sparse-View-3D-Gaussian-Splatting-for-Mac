import Foundation
import Testing
import SplatCore
@testable import SplatMetal

/// The viewer crop clip on the forward pass: a box that holds every Gaussian changes nothing, a box that
/// leaves one out removes exactly that splat, and clearing the crop restores the unclipped image.
@Suite(.enabled(if: TestEnvironment.hasMetal))
struct CropRenderTests {
    /// Two bright degree-0 splats at the points-test positions: pixel (44, 32) and pixel (19, 32) of a 64 × 64 view.
    private func makeCloud() throws -> GaussianCloud {
        let positions: [Float] = [0.9765625, 0.0390625, 5, -0.9765625, 0.0390625, 5]
        return try GaussianCloud(shDegree: 0,
                                 positions: positions,
                                 logScales: [Float](repeating: log(0.05), count: 6),
                                 rotations: [1, 0, 0, 0, 1, 0, 0, 0],
                                 opacityLogits: [5, 5],
                                 sh: [2, 2, 2, 2, 2, 2])
    }

    private func pixel(_ image: RGBImage, _ x: Int, _ y: Int) -> SIMD3<Float> {
        image.pixels[y * image.width + x]
    }

    private func brightness(_ value: SIMD3<Float>) -> Float {
        value.x + value.y + value.z
    }

    @Test func boxHoldingEverythingLeavesTheImageUnchanged() throws {
        let context = try makeTestContext()
        let renderer = try SplatRenderer(context: context, cloud: try makeCloud())
        let camera = makeTestCamera(width: 64, height: 64)
        let unclipped = try renderer.renderImage(camera: camera, settings: RasterSettings())
        let everything = CropBox(center: .zero, halfExtents: SIMD3<Float>(100, 100, 100))
        renderer.setCrop(everything)
        let clipped = try renderer.renderImage(camera: camera, settings: RasterSettings())
        #expect(clipped.pixels == unclipped.pixels)
    }

    @Test func boxLeavingOneSplatOutRemovesOnlyThatSplat() throws {
        let context = try makeTestContext()
        let renderer = try SplatRenderer(context: context, cloud: try makeCloud())
        let camera = makeTestCamera(width: 64, height: 64)
        let unclipped = try renderer.renderImage(camera: camera, settings: RasterSettings())
        #expect(brightness(pixel(unclipped, 44, 32)) > 0.1)
        #expect(brightness(pixel(unclipped, 19, 32)) > 0.1)

        // Holds only the left splat (x = -0.98); the right splat at x = +0.98 is outside.
        renderer.setCrop(CropBox(center: SIMD3<Float>(-0.9765625, 0.0390625, 5),
                                 halfExtents: SIMD3<Float>(0.5, 0.5, 1)))
        let clipped = try renderer.renderImage(camera: camera, settings: RasterSettings())
        #expect(brightness(pixel(clipped, 44, 32)) < 1e-4)
        #expect(brightness(pixel(clipped, 19, 32)) > 0.1)
        #expect(abs(brightness(pixel(clipped, 19, 32)) - brightness(pixel(unclipped, 19, 32))) < 1e-4)

        renderer.setCrop(nil)
        let restored = try renderer.renderImage(camera: camera, settings: RasterSettings())
        #expect(restored.pixels == unclipped.pixels)
    }
}
