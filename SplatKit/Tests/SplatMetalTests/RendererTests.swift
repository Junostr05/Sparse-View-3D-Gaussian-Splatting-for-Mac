import Metal
import Testing
import SplatCore
@testable import SplatMetal

@Suite(.enabled(if: TestEnvironment.hasMetal))
struct RendererTests {
    /// >= 99.9 % of all channels within 1e-4 of the reference, all within 1e-2.
    private func expectMatchesReference(_ gpu: RGBImage, _ cpu: RGBImage) {
        #expect(gpu.width == cpu.width)
        #expect(gpu.height == cpu.height)
        #expect(gpu.pixels.count == cpu.pixels.count)
        guard gpu.pixels.count == cpu.pixels.count else { return }
        var channels = 0
        var close = 0
        var maxError: Float = 0
        for (a, b) in zip(gpu.pixels, cpu.pixels) {
            for c in 0..<3 {
                let error = abs(a[c] - b[c])
                channels += 1
                if error <= 1e-4 { close += 1 }
                if !(error <= maxError) { maxError = error.isNaN ? .infinity : error }
            }
        }
        let fraction = channels == 0 ? 1.0 : Double(close) / Double(channels)
        #expect(fraction >= 0.999, "only \(fraction * 100) % of channels within 1e-4")
        #expect(maxError <= 1e-2, "max channel error \(maxError)")
    }

    private func compare(width: Int, height: Int, cloud: GaussianCloud) throws {
        let context = try makeTestContext()
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let camera = makeTestCamera(width: width, height: height)
        var settings = RasterSettings()
        settings.background = SIMD3(0.1, 0.2, 0.3)
        let gpu = try renderer.renderImage(camera: camera, settings: settings)
        let cpu = CPURasterizer.render(cloud, camera: camera, settings: settings)
        expectMatchesReference(gpu, cpu)
    }

    @Test func matchesCPUReferenceOn64x64() throws {
        let cloud = try makeRandomCloud(count: 200)
        try compare(width: 64, height: 64, cloud: cloud)
    }

    @Test func matchesCPUReferenceOnOddSize() throws {
        let cloud = try makeRandomCloud(count: 200)
        try compare(width: 37, height: 23, cloud: cloud)
        try compare(width: 1, height: 1, cloud: cloud)
    }

    @Test func noVisibleGaussiansRendersBackground() throws {
        let cloud = try makeRandomCloud(count: 50, depth: (-8)...(-2))
        let context = try makeTestContext()
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        var settings = RasterSettings()
        settings.background = SIMD3(0.2, 0.3, 0.4)
        let camera = makeTestCamera(width: 40, height: 30)

        let image = try renderer.renderImage(camera: camera, settings: settings)
        #expect(image.pixels.count == 40 * 30)
        for pixel in image.pixels {
            #expect(abs(pixel.x - 0.2) <= 1e-6)
            #expect(abs(pixel.y - 0.3) <= 1e-6)
            #expect(abs(pixel.z - 0.4) <= 1e-6)
        }

        // Also through the texture API: no visible Gaussians, no intersections.
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: 40, height: 30, mipmapped: false)
        descriptor.usage = [.shaderWrite, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(context.device.makeTexture(descriptor: descriptor))
        let stats = try renderer.render(camera: camera, settings: settings, to: texture)
        #expect(stats.visibleGaussians == 0)
        #expect(stats.tileIntersections == 0)
    }

    @Test func benchmarkOneMillion() throws {
        let cloud = try makeRandomCloud(count: 1_000_000, depth: 2...8, scale: 0.01...0.1)
        let context = try makeTestContext()
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let camera = makeTestCamera(width: 1920, height: 1080)
        let settings = RasterSettings()

        var times: [Double] = []
        var last: FrameStats?
        for _ in 0..<5 {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba32Float, width: 1920, height: 1080, mipmapped: false)
            descriptor.usage = [.shaderWrite, .shaderRead]
            descriptor.storageMode = .shared
            let texture = try #require(context.device.makeTexture(descriptor: descriptor))
            let stats = try renderer.render(camera: camera, settings: settings, to: texture)
            times.append(stats.gpuTimeMilliseconds)
            last = stats
        }
        let median = times.sorted()[times.count / 2]
        print("benchmarkOneMillion: median gpuTimeMilliseconds = \(median) ms, "
              + "visible = \(last?.visibleGaussians ?? 0), intersections = \(last?.tileIntersections ?? 0)")
    }
}
