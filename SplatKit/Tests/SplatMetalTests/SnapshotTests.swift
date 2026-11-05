import CoreGraphics
import Foundation
import ImageIO
import Metal
import Testing
import SplatCore
@testable import SplatMetal

/// Snapshot output directory, read from `SNAPSHOT_DIR`. Unset means the suite is disabled.
enum SnapshotEnvironment {
    static let directory: URL? = {
        guard let raw = ProcessInfo.processInfo.environment["SNAPSHOT_DIR"], !raw.isEmpty else {
            return nil
        }
        return URL(fileURLWithPath: raw, isDirectory: true)
    }()
}

enum SnapshotTestError: Error {
    case imageCreation(String)
}

/// Renders deterministic synthetic scenes and writes them as PNGs into `SNAPSHOT_DIR`
/// so CI can upload them as artifacts. Run with:
/// `SNAPSHOT_DIR=/some/dir swift test --filter SnapshotTests`
@Suite(.enabled(if: TestEnvironment.hasMetal && SnapshotEnvironment.directory != nil))
struct SnapshotTests {
    private static let background = SIMD3<Float>(0.08, 0.08, 0.09)

    // MARK: - Scenes

    @Test func writesRandomSH3Scene() throws {
        let cloud = try makeRandomCloud(count: 3000, seed: 7, shDegree: 3)
        let camera = makeTestCamera(width: 640, height: 400)
        let image = try render(cloud, camera: camera)
        try writePNG(image, name: "scene_random_sh3.png")
    }

    @Test func writesOrbitRingScene() throws {
        let cloud = try makeRing(count: 4000)
        let camera = orbitCamera(for: cloud, width: 640, height: 400)
        let image = try render(cloud, camera: camera)
        try writePNG(image, name: "scene_orbit_ring.png")
    }

    @Test func writesDepthOrderScene() throws {
        let cloud = try makeDepthOrder()
        let camera = makeTestCamera(width: 640, height: 400)
        let image = try render(cloud, camera: camera)
        try writePNG(image, name: "scene_depth_order.png")
    }

    /// Renders the orbit ring at 160x100 on both the GPU and the CPU reference, writes
    /// both images, and prints the largest per-channel difference.
    @Test func writesOrbitRingCPUReference() throws {
        let cloud = try makeRing(count: 4000)
        let camera = orbitCamera(for: cloud, width: 160, height: 100)

        var settings = RasterSettings()
        settings.background = Self.background
        let cpu = CPURasterizer.render(cloud, camera: camera, settings: settings)
        let gpu = try render(cloud, camera: camera)

        let maxDifference = maxAbsDifference(gpu, cpu)
        print("SnapshotTests: max abs GPU-CPU difference (orbit ring, 160x100) = \(maxDifference)")

        try writePNG(cpu, name: "scene_orbit_ring_cpu.png")
        try writePNG(gpu, name: "scene_orbit_ring_gpu_160x100.png")
    }

    // MARK: - Scene builders

    /// Torus of `count` Gaussians around the Y axis, colored by the ring index.
    private func makeRing(count: Int) throws -> GaussianCloud {
        let majorRadius: Float = 1.0
        let tubeRadius: Float = 0.35
        let ringCount = 10
        let perRing = count / ringCount
        let total = perRing * ringCount
        let c0: Float = 0.28209479177387814
        let logScale: Float = Float(log(Double(0.04)))

        var positions: [Float] = []
        var logScales: [Float] = []
        var rotations: [Float] = []
        var opacity: [Float] = []
        var sh: [Float] = []
        positions.reserveCapacity(3 * total)
        logScales.reserveCapacity(3 * total)
        rotations.reserveCapacity(4 * total)
        opacity.reserveCapacity(total)
        sh.reserveCapacity(3 * total)

        for index in 0..<total {
            let ring = index / perRing
            let step = index % perRing
            let v: Float = 2 * Float.pi * Float(ring) / Float(ringCount)
            let u: Float = 2 * Float.pi * Float(step) / Float(perRing)

            let radial: Float = majorRadius + tubeRadius * cos(u)
            let x: Float = radial * cos(v)
            let y: Float = tubeRadius * sin(u)
            let z: Float = radial * sin(v)
            positions += [x, y, z]
            logScales += [logScale, logScale, logScale]
            rotations += [1, 0, 0, 0]
            opacity.append(1.5)

            let shade: Float = 0.6 + 0.4 * cos(u)
            let red: Float = (0.5 + 0.5 * cos(v)) * shade
            let green: Float = (0.5 + 0.5 * cos(v - 2.0944)) * shade
            let blue: Float = (0.5 + 0.5 * cos(v + 2.0944)) * shade
            sh.append((red - 0.5) / c0)
            sh.append((green - 0.5) / c0)
            sh.append((blue - 0.5) / c0)
        }

        return try GaussianCloud(shDegree: 0, positions: positions, logScales: logScales,
                                 rotations: rotations, opacityLogits: opacity, sh: sh)
    }

    /// Five large, half-transparent Gaussians at different depths, drawn back to front
    /// by the sorter so the blending order is visible.
    private func makeDepthOrder() throws -> GaussianCloud {
        let centers: [SIMD3<Float>] = [
            SIMD3<Float>(-0.9, -0.2, 6.0),
            SIMD3<Float>(-0.3, 0.2, 5.0),
            SIMD3<Float>(0.3, -0.1, 4.0),
            SIMD3<Float>(0.9, 0.25, 3.0),
            SIMD3<Float>(0.0, 0.0, 7.0),
        ]
        let colors: [SIMD3<Float>] = [
            SIMD3<Float>(0.9, 0.2, 0.2),
            SIMD3<Float>(0.2, 0.9, 0.2),
            SIMD3<Float>(0.2, 0.3, 0.95),
            SIMD3<Float>(0.95, 0.85, 0.2),
            SIMD3<Float>(0.9, 0.3, 0.9),
        ]
        let c0: Float = 0.28209479177387814
        let logScale: Float = Float(log(Double(0.7)))

        var positions: [Float] = []
        var logScales: [Float] = []
        var rotations: [Float] = []
        var opacity: [Float] = []
        var sh: [Float] = []
        for index in 0..<centers.count {
            let center: SIMD3<Float> = centers[index]
            let color: SIMD3<Float> = colors[index]
            positions += [center.x, center.y, center.z]
            logScales += [logScale, logScale, logScale]
            rotations += [1, 0, 0, 0]
            opacity.append(0.0)
            sh.append((color.x - 0.5) / c0)
            sh.append((color.y - 0.5) / c0)
            sh.append((color.z - 0.5) / c0)
        }

        return try GaussianCloud(shDegree: 0, positions: positions, logScales: logScales,
                                 rotations: rotations, opacityLogits: opacity, sh: sh)
    }

    /// Framing camera for the whole cloud, looking at it from the default orbit pitch.
    private func orbitCamera(for cloud: GaussianCloud, width: Int, height: Int) -> Camera {
        let bounds = cloud.robustBounds()
        return OrbitCamera.framing(bounds, upAxis: .positiveY).camera(width: width, height: height)
    }

    // MARK: - Rendering and output

    private func render(_ cloud: GaussianCloud, camera: Camera) throws -> RGBImage {
        let context = try makeTestContext()
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        var settings = RasterSettings()
        settings.background = Self.background
        return try renderer.renderImage(camera: camera, settings: settings)
    }

    private func maxAbsDifference(_ a: RGBImage, _ b: RGBImage) -> Float {
        var maxError: Float = 0
        for (p, q) in zip(a.pixels, b.pixels) {
            for channel in 0..<3 {
                let error: Float = abs(p[channel] - q[channel])
                if error > maxError { maxError = error }
            }
        }
        return maxError
    }

    private func outputURL(_ name: String) throws -> URL {
        let directory = try #require(SnapshotEnvironment.directory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appendingPathComponent(name)
    }

    /// Writes an sRGB PNG. Values are clamped to 0...1 and quantized to 8 bits.
    private func writePNG(_ image: RGBImage, name: String) throws {
        let url = try outputURL(name)
        let pixelCount = image.width * image.height
        var bytes = [UInt8](repeating: 255, count: pixelCount * 4)
        for index in 0..<pixelCount {
            let pixel: SIMD3<Float> = image.pixels[index]
            for channel in 0..<3 {
                let clamped: Float = min(max(pixel[channel], 0), 1)
                let scaled: Float = (clamped * 255).rounded()
                bytes[index * 4 + channel] = UInt8(scaled)
            }
        }

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let provider = CGDataProvider(data: Data(bytes) as CFData),
              let cgImage = CGImage(
                width: image.width,
                height: image.height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: image.width * 4,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
              ) else {
            throw SnapshotTestError.imageCreation(name)
        }

        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
            throw SnapshotTestError.imageCreation(name)
        }
        CGImageDestinationAddImage(destination, cgImage, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw SnapshotTestError.imageCreation(name)
        }
    }
}
