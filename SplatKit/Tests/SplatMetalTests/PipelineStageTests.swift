import Metal
import Testing
import SplatCore
@testable import SplatMetal

/// Stage-by-stage checks of the forward pipeline against the CPU reference. When the final image
/// differs, these tests pinpoint which stage (projection, key sort, tile ranges) is wrong.
@Suite(.enabled(if: TestEnvironment.hasMetal))
struct PipelineStageTests {
    private func maxAbs<V: SIMD>(_ v: V) -> Float where V.Scalar == Float {
        var m: Float = 0
        for i in v.indices { m = max(m, Swift.abs(v[i])) }
        return m
    }

    private func readBack<T>(_ buffer: MTLBuffer?, as type: T.Type, count: Int) -> [T] {
        guard let buffer, count > 0 else { return [] }
        let pointer = buffer.contents().bindMemory(to: T.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func renderScene(width: Int, height: Int) throws
        -> (renderer: SplatRenderer, cloud: GaussianCloud, camera: Camera, settings: RasterSettings, stats: FrameStats) {
        let cloud = try makeRandomCloud(count: 200)
        let context = try makeTestContext()
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let camera = makeTestCamera(width: width, height: height)
        let settings = RasterSettings()
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderWrite, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try #require(context.device.makeTexture(descriptor: descriptor))
        let stats = try renderer.render(camera: camera, settings: settings, to: texture)
        return (renderer, cloud, camera, settings, stats)
    }

    @Test func projectionMatchesCPU() throws {
        let (renderer, cloud, camera, settings, _) = try renderScene(width: 64, height: 64)
        let n = cloud.count
        let meanDepth = readBack(renderer.meanDepth, as: SIMD4<Float>.self, count: n)
        let conic = readBack(renderer.conicOpacity, as: SIMD4<Float>.self, count: n)
        let color = readBack(renderer.colors, as: SIMD4<Float>.self, count: n)
        let rect = readBack(renderer.tileRects, as: SIMD4<Int32>.self, count: n)
        let touched = readBack(renderer.tilesTouched, as: UInt32.self, count: n)

        var mismatches: [String] = []
        for i in 0..<n {
            guard let p = Projection.project(cloud, index: i, camera: camera, settings: settings) else {
                if touched[i] != 0 { mismatches.append("#\(i): CPU culled, GPU touched \(touched[i])") }
                continue
            }
            let area = UInt32((p.tileMax.x - p.tileMin.x) * (p.tileMax.y - p.tileMin.y))
            if touched[i] != area { mismatches.append("#\(i): touched GPU \(touched[i]) CPU \(area)") }
            let gpuRect = SIMD4(p.tileMin.x, p.tileMin.y, p.tileMax.x, p.tileMax.y)
            if rect[i] != gpuRect { mismatches.append("#\(i): rect GPU \(rect[i]) CPU \(gpuRect)") }
            if abs(meanDepth[i].x - p.mean2D.x) > 1e-3 || abs(meanDepth[i].y - p.mean2D.y) > 1e-3
                || abs(meanDepth[i].z - p.depth) > 1e-4 {
                mismatches.append("#\(i): meanDepth GPU \(meanDepth[i]) CPU \(p.mean2D) \(p.depth)")
            }
            let cpuConic = SIMD4(p.conic.x, p.conic.y, p.conic.z, p.opacity)
            let conicError = maxAbs(conic[i] - cpuConic)
            if conicError > 1e-3 * max(1, maxAbs(cpuConic)) {
                mismatches.append("#\(i): conic GPU \(conic[i]) CPU \(cpuConic)")
            }
            let colorError = maxAbs(SIMD3(color[i].x, color[i].y, color[i].z) - p.color)
            if colorError > 1e-4 { mismatches.append("#\(i): color GPU \(color[i]) CPU \(p.color)") }
        }
        #expect(mismatches.isEmpty, "\(mismatches.count) projection mismatches, first: \(mismatches.prefix(5))")
    }

    @Test func sortedKeysAndTileRangesAreConsistent() throws {
        let (renderer, cloud, _, _, stats) = try renderScene(width: 64, height: 64)
        let total = stats.tileIntersections
        try #require(total > 0)
        let keys = readBack(renderer.sortKeys, as: UInt64.self, count: total)
        let values = readBack(renderer.sortValues, as: UInt32.self, count: total)
        let tileCount = 4 * 4
        let ranges = readBack(renderer.tileRangeBuffer, as: UInt32.self, count: 2 * tileCount)

        // Keys must be ascending; equal (tile, depth) keys keep ascending Gaussian index.
        var orderErrors = 0
        for i in 1..<total {
            if keys[i - 1] > keys[i] || (keys[i - 1] == keys[i] && values[i - 1] > values[i]) {
                orderErrors += 1
            }
        }
        #expect(orderErrors == 0, "\(orderErrors) out-of-order keys")
        #expect(values.allSatisfy { Int($0) < cloud.count }, "value out of range")

        // Expected ranges straight from the sorted keys.
        var expected = [UInt32](repeating: 0, count: 2 * tileCount)
        for i in 0..<total {
            let tile = Int(keys[i] >> 32)
            guard tile < tileCount else { continue }
            if i == 0 || Int(keys[i - 1] >> 32) != tile { expected[2 * tile] = UInt32(i) }
            if i == total - 1 || Int(keys[i + 1] >> 32) != tile { expected[2 * tile + 1] = UInt32(i + 1) }
        }
        #expect(ranges == expected, "tile ranges GPU \(ranges) expected \(expected)")
    }
}
