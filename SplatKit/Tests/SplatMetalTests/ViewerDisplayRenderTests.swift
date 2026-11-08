import Foundation
import Metal
import Testing
import SplatCore
@testable import SplatMetal

/// Viewer display modes on the Metal forward pass: points, shapes, heatmap colours, the depth view, and the
/// guarantee that the splat mode is unchanged by any earlier display.
@Suite(.enabled(if: TestEnvironment.hasMetal))
struct ViewerDisplayRenderTests {
    /// Degree-0 cloud from (position, log scale, opacity logit) triples. Colours come from the SH override in the tests.
    private func makeCloud(_ gaussians: [(position: SIMD3<Float>, logScale: Float, logit: Float)]) throws -> GaussianCloud {
        var positions: [Float] = []
        var logScales: [Float] = []
        var rotations: [Float] = []
        var logits: [Float] = []
        var sh: [Float] = []
        for gaussian in gaussians {
            positions += [gaussian.position.x, gaussian.position.y, gaussian.position.z]
            logScales += [gaussian.logScale, gaussian.logScale, gaussian.logScale]
            rotations += [1, 0, 0, 0]
            logits.append(gaussian.logit)
            sh += [0, 0, 0]
        }
        return try GaussianCloud(shDegree: 0, positions: positions, logScales: logScales,
                                 rotations: rotations, opacityLogits: logits, sh: sh)
    }

    private func pixel(_ image: RGBImage, _ x: Int, _ y: Int) -> SIMD3<Float> {
        image.pixels[y * image.width + x]
    }

    private func near(_ a: SIMD3<Float>, _ b: SIMD3<Float>, tolerance: Float) -> Bool {
        abs(a.x - b.x) <= tolerance && abs(a.y - b.y) <= tolerance && abs(a.z - b.z) <= tolerance
    }

    /// Reads back an rgba32Float texture as RGB.
    private func readTexture(_ texture: MTLTexture, width: Int, height: Int) -> [SIMD3<Float>] {
        var raw = [Float](repeating: 0, count: width * height * 4)
        raw.withUnsafeMutableBytes { bytes in
            texture.getBytes(bytes.baseAddress!,
                             bytesPerRow: width * 4 * MemoryLayout<Float>.size,
                             from: MTLRegionMake2D(0, 0, width, height),
                             mipmapLevel: 0)
        }
        var pixels: [SIMD3<Float>] = []
        pixels.reserveCapacity(width * height)
        for i in 0..<(width * height) {
            pixels.append(SIMD3<Float>(raw[4 * i], raw[4 * i + 1], raw[4 * i + 2]))
        }
        return pixels
    }

    private func makeRGBATexture(context: MetalContext, width: Int, height: Int) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderWrite, .shaderRead]
        descriptor.storageMode = .shared
        guard let texture = context.device.makeTexture(descriptor: descriptor) else {
            throw MetalError.bufferAllocation
        }
        return texture
    }

    @Test func pointsDrawRoundDiscsInTheOverrideColour() throws {
        // Camera: fx = fy = 64, principal point (32, 32). The points project to the pixel centres (44.5, 32.5) and
        // (19.5, 32.5), so the sampled pixels sit exactly on the points (the alpha falls off away from the centre).
        let cloud = try makeCloud([
            (SIMD3<Float>(0.9765625, 0.0390625, 5), log(0.05), 5),
            (SIMD3<Float>(-0.9765625, 0.0390625, 5), log(0.05), 5),
        ])
        let context = try makeTestContext()
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        try renderer.setDisplay(ViewerDisplay(mode: .points, pointRadiusPixels: 4,
                                              colorOverride: [SIMD4<Float>(1, 0, 0, 1), SIMD4<Float>(0, 0, 1, 1)]))
        let image = try renderer.renderImage(camera: makeTestCamera(width: 64, height: 64), settings: RasterSettings())
        #expect(near(pixel(image, 44, 32), SIMD3<Float>(1, 0, 0), tolerance: 0.01))
        #expect(near(pixel(image, 19, 32), SIMD3<Float>(0, 0, 1), tolerance: 0.01))
        // Between the two points and well outside the disc of radius 4: background (black).
        #expect(near(pixel(image, 32, 32), .zero, tolerance: 0.01))
        #expect(near(pixel(image, 52, 32), .zero, tolerance: 0.01))
        #expect(near(pixel(image, 0, 0), .zero, tolerance: 0.01))
    }

    @Test func shapesBlendWithTheOpacityScale() throws {
        // Opacity sigmoid(2) ≈ 0.881, scaled by 0.5 in the shapes mode: the centre pixel is about 0.44 · white.
        let cloud = try makeCloud([(SIMD3<Float>(0, 0, 5), log(0.5), 2)])
        let context = try makeTestContext()
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        try renderer.setDisplay(ViewerDisplay(mode: .shapes, opacityScale: 0.5,
                                              colorOverride: [SIMD4<Float>(1, 1, 1, 1)]))
        let image = try renderer.renderImage(camera: makeTestCamera(width: 64, height: 64), settings: RasterSettings())
        let centre: SIMD3<Float> = pixel(image, 32, 32)
        #expect(centre.x > 0.40 && centre.x < 0.47)
        #expect(abs(centre.x - centre.y) < 1e-5 && abs(centre.y - centre.z) < 1e-5)
    }

    @Test func heatmapShowsTheOverrideColourAtFullOpacity() throws {
        // Opacity 0.5 would give about half the colour; the heatmap draws opacity 1 (the ellipse falls off only slightly).
        let cloud = try makeCloud([(SIMD3<Float>(0, 0, 5), log(0.5), 0)])
        let context = try makeTestContext()
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        try renderer.setDisplay(ViewerDisplay(mode: .heatmap, colorOverride: [SIMD4<Float>(0.2, 0.9, 0.1, 1)]))
        let image = try renderer.renderImage(camera: makeTestCamera(width: 64, height: 64), settings: RasterSettings())
        #expect(near(pixel(image, 32, 32), SIMD3<Float>(0.2, 0.9, 0.1) * 0.99, tolerance: 0.02))
    }

    @Test func splatModeIsUnchangedByEarlierDisplays() throws {
        let cloud = try makeRandomCloud(count: 200)
        let camera = makeTestCamera(width: 64, height: 48)
        let context = try makeTestContext()
        let fresh = try SplatRenderer(context: context, cloud: cloud)
        let reference = try fresh.renderImage(camera: camera, settings: RasterSettings())

        let used = try SplatRenderer(context: context, cloud: cloud)
        let overrides: [SIMD4<Float>] = (0..<cloud.count).map { i in
            SIMD4<Float>(Float(i % 7) / 7, 0.5, 0.25, 1)
        }
        try used.setDisplay(ViewerDisplay(mode: .points, pointRadiusPixels: 3, colorOverride: overrides))
        _ = try used.renderImage(camera: camera, settings: RasterSettings())
        try used.setDisplay(ViewerDisplay(mode: .heatmap, colorOverride: overrides))
        _ = try used.renderImage(camera: camera, settings: RasterSettings())
        try used.setDisplay(.splat)
        let after = try used.renderImage(camera: camera, settings: RasterSettings())

        #expect(after.pixels == reference.pixels)
    }

    @Test func backwardRefusesNonSplatDisplays() throws {
        let cloud = try makeRandomCloud(count: 50)
        let context = try makeTestContext()
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        try renderer.setDisplay(ViewerDisplay(mode: .points,
                                              colorOverride: [SIMD4<Float>](repeating: SIMD4<Float>(1, 1, 1, 1),
                                                                            count: cloud.count)))
        let camera = makeTestCamera(width: 32, height: 32)
        _ = try renderer.render(camera: camera, settings: RasterSettings(), to: try makeRGBATexture(
            context: context, width: 32, height: 32))
        let gradient = [SIMD3<Float>](repeating: .zero, count: 32 * 32)
        var refused = false
        do {
            _ = try renderer.backward(imageGradient: gradient)
        } catch {
            refused = true
        }
        #expect(refused)
    }

    @Test func overrideShorterThanTheCloudIsRejected() throws {
        let cloud = try makeRandomCloud(count: 10)
        let context = try makeTestContext()
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        var rejected = false
        do {
            try renderer.setDisplay(ViewerDisplay(mode: .points,
                                                  colorOverride: [SIMD4<Float>](repeating: .zero, count: 4)))
        } catch {
            rejected = true
        }
        #expect(rejected)
    }

    @Test func depthViewIsFiniteAndTheNearObjectIsWarmer() throws {
        // Near Gaussian in the centre at depth 3, far one at depth 8 to the right (projected to x = 48).
        let cloud = try makeCloud([
            (SIMD3<Float>(0, 0, 3), log(0.4), 6),
            (SIMD3<Float>(2, 0, 8), log(0.8), 6),
        ])
        let context = try makeTestContext()
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let width = 64
        let height = 64
        let texture = try makeRGBATexture(context: context, width: width, height: height)
        var settings = RasterSettings()
        settings.background = SIMD3<Float>(0.1, 0.1, 0.1)
        try renderer.renderDepthColorized(camera: makeTestCamera(width: width, height: height),
                                          settings: settings, to: texture)
        let pixels = readTexture(texture, width: width, height: height)

        for value in pixels {
            #expect(value.x.isFinite && value.y.isFinite && value.z.isFinite)
            #expect(value.x >= 0 && value.x <= 1 && value.y >= 0 && value.y <= 1 && value.z >= 0 && value.z <= 1)
        }
        let nearPixel = pixels[32 * width + 32]
        let farPixel = pixels[32 * width + 48]
        // Warm end is red, cold end blue: the near object has more red than blue, the far one less.
        #expect(nearPixel.x - nearPixel.z > farPixel.x - farPixel.z)
        #expect(nearPixel.x > nearPixel.z)
        // No coverage in the top-left corner: the background colour.
        #expect(near(pixels[0], SIMD3<Float>(0.1, 0.1, 0.1), tolerance: 1e-4))
    }
}
