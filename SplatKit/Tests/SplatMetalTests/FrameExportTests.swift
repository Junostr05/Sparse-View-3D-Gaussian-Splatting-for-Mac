import AVFoundation
import Foundation
import ImageIO
import Metal
import Testing
import SplatCore
@testable import SplatMetal

/// The export of rendered frames: one rendered frame written as PNG, and a short H.264 video of rendered frames.
/// Both files are read back with the system readers. The app's own export sheets cannot be driven from here.
@Suite(.enabled(if: TestEnvironment.hasMetal))
struct FrameExportTests {
    private func exportDirectory() throws -> URL {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("FrameExportTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    /// Renders one frame of `cloud` into a readback texture and returns the texture after the GPU has finished.
    private func renderFrame(context: MetalContext, renderer: SplatRenderer, width: Int, height: Int) throws -> MTLTexture {
        let texture = try FrameExport.makeReadbackTexture(device: context.device, width: width, height: height)
        var settings = RasterSettings()
        settings.background = SIMD3(0.1, 0.2, 0.3)
        try renderer.render(camera: makeTestCamera(width: width, height: height), settings: settings, to: texture)
        return texture
    }

    @Test func rendersAPNGThatImageIOReadsBackAtTheRequestedSize() async throws {
        let context = try makeTestContext()
        let renderer = try SplatRenderer(context: context, cloud: try makeRandomCloud(count: 400, shDegree: 0))
        let width = 320
        let height = 180
        let texture = try renderFrame(context: context, renderer: renderer, width: width, height: height)

        var bgra = [UInt8](repeating: 0, count: width * height * 4)
        bgra.withUnsafeMutableBytes { bytes in
            FrameExport.copyBGRA(texture, into: bytes.baseAddress!, bytesPerRow: width * 4)
        }
        let url = try exportDirectory().appendingPathComponent("frame.png")
        try FrameExport.writePNG(bgra: Data(bgra), width: width, height: height, to: url)

        #expect(FileManager.default.fileExists(atPath: url.path))
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        #expect(CGImageSourceGetType(source) as String? == "public.png")
        let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        #expect(image.width == width)
        #expect(image.height == height)
        #expect(image.colorSpace?.name == CGColorSpace.sRGB)
    }

    @Test func writesATenFrameH264VideoWithTheRequestedGeometry() async throws {
        let context = try makeTestContext()
        let renderer = try SplatRenderer(context: context, cloud: try makeRandomCloud(count: 400, shDegree: 0))
        let width = 320
        let height = 180
        let frameRate = 30
        let url = try exportDirectory().appendingPathComponent("flight.mp4")
        let writer = try H264VideoWriter(url: url, width: width, height: height, frameRate: frameRate)
        for index in 0..<10 {
            let texture = try renderFrame(context: context, renderer: renderer, width: width, height: height)
            try await writer.waitUntilReadyForFrame()
            try writer.append(texture: texture, frameIndex: index)
        }
        try await writer.finish()

        #expect(FileManager.default.fileExists(atPath: url.path))
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        #expect(abs(duration - 10.0 / Double(frameRate)) < 0.05, "duration \(duration) s")

        let tracks = try await asset.loadTracks(withMediaType: .video)
        let track = try #require(tracks.first)
        let size = try await track.load(.naturalSize)
        #expect(Int(size.width) == width)
        #expect(Int(size.height) == height)

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        reader.add(output)
        #expect(reader.startReading())
        var frames = 0
        while let sample = output.copyNextSampleBuffer() {
            frames += 1
            _ = sample
        }
        #expect(frames == 10, "decoded \(frames) frames")
    }
}
