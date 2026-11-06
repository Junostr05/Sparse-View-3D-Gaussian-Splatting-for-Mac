#if canImport(AVFoundation) && canImport(ImageIO) && canImport(CoreGraphics)
import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import Testing
import SplatCore
@testable import SplatIO

private enum CaptureTestError: Error {
    case io
}

private func makeTempDirectory() throws -> URL {
    let url: URL = FileManager.default.temporaryDirectory
        .appendingPathComponent("CaptureImportTests-" + UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func isPNG(_ data: Data) -> Bool {
    let signature: Data = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
    return Data(data.prefix(8)) == signature
}

// MARK: - Synthetic video

/// Sharp checkerboard frames whose phase moves per frame, except the listed blurry frames, which are
/// smooth gradients (almost zero Laplacian variance).
private func makeFrame(width: Int, height: Int, index: Int, blurry: Bool) throws -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    let status: CVReturn = CVPixelBufferCreate(kCFAllocatorDefault,
                                               width,
                                               height,
                                               kCVPixelFormatType_32BGRA,
                                               nil,
                                               &buffer)
    guard status == kCVReturnSuccess, let pixelBuffer = buffer else {
        throw CaptureTestError.io
    }
    CVPixelBufferLockBaseAddress(pixelBuffer, [])
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
    guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
        throw CaptureTestError.io
    }
    let bytesPerRow: Int = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let pixels: UnsafeMutablePointer<UInt8> = base.assumingMemoryBound(to: UInt8.self)
    let phase: Int = index * 5
    let gradientScale: Int = width + height
    for y in 0..<height {
        for x in 0..<width {
            var value: UInt8 = 0
            if blurry {
                let sum: Int = (x + y) * 255 / gradientScale
                value = UInt8(sum)
            } else {
                let cellX: Int = (x + phase) / 8
                let cellY: Int = y / 8
                let parity: Int = (cellX + cellY) % 2
                value = parity == 0 ? 255 : 0
            }
            let offset: Int = y * bytesPerRow + x * 4
            pixels[offset] = value
            pixels[offset + 1] = value
            pixels[offset + 2] = value
            pixels[offset + 3] = 255
        }
    }
    return pixelBuffer
}

/// Writes a 4 fps, 12-frame (3 s) H.264 MP4. Frames 3 and 7 are blurry.
private func writeTestVideo(to url: URL) async throws {
    let width: Int = 320
    let height: Int = 240
    let frameCount: Int = 12
    let fps: Int32 = 4
    let blurryFrames: Set<Int> = [3, 7]

    let writer: AVAssetWriter = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let settings: [String: Any] = [
        AVVideoCodecKey: AVVideoCodecType.h264.rawValue,
        AVVideoWidthKey: width,
        AVVideoHeightKey: height,
    ]
    let input: AVAssetWriterInput = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
    let adaptor: AVAssetWriterInputPixelBufferAdaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: input,
        sourcePixelBufferAttributes: nil)
    writer.add(input)
    guard writer.startWriting() else {
        throw CaptureTestError.io
    }
    writer.startSession(atSourceTime: .zero)

    for index in 0..<frameCount {
        while !input.isReadyForMoreMediaData {
            try await Task.sleep(nanoseconds: 2_000_000)
        }
        let buffer: CVPixelBuffer = try makeFrame(width: width,
                                                  height: height,
                                                  index: index,
                                                  blurry: blurryFrames.contains(index))
        let time: CMTime = CMTime(value: Int64(index), timescale: fps)
        guard adaptor.append(buffer, withPresentationTime: time) else {
            throw CaptureTestError.io
        }
    }
    input.markAsFinished()
    await writer.finishWriting()
    guard writer.status == .completed else {
        throw CaptureTestError.io
    }
}

// MARK: - Synthetic photos

private func makeSolidImage(width: Int, height: Int) throws -> CGImage {
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
        throw CaptureTestError.io
    }
    guard let context = CGContext(data: nil,
                                  width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: 0,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
        throw CaptureTestError.io
    }
    context.setFillColor(red: 0.2, green: 0.5, blue: 0.8, alpha: 1)
    context.fill(CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))
    guard let image = context.makeImage() else {
        throw CaptureTestError.io
    }
    return image
}

/// Writes a JPEG with the given EXIF orientation and a GPS block, to test orientation and stripping.
private func writePhoto(to url: URL, width: Int, height: Int, orientation: Int) throws {
    let image: CGImage = try makeSolidImage(width: width, height: height)
    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.jpeg" as CFString, 1, nil) else {
        throw CaptureTestError.io
    }
    let gps: [String: Any] = [
        kCGImagePropertyGPSLatitude as String: 52.52,
        kCGImagePropertyGPSLatitudeRef as String: "N",
        kCGImagePropertyGPSLongitude as String: 13.405,
        kCGImagePropertyGPSLongitudeRef as String: "E",
    ]
    let properties: [String: Any] = [
        kCGImagePropertyOrientation as String: NSNumber(value: orientation),
        kCGImagePropertyGPSDictionary as String: gps,
    ]
    CGImageDestinationAddImage(destination, image, properties as CFDictionary)
    guard CGImageDestinationFinalize(destination) else {
        throw CaptureTestError.io
    }
}

/// Writes a JPEG whose EXIF dictionary holds the given value for `FocalLenIn35mmFilm`, plus a GPS block.
/// Pass `nil` to write no focal length at all.
private func writePhotoWithFocalLength(to url: URL, focalLength: Any?) throws {
    let image: CGImage = try makeSolidImage(width: 320, height: 240)
    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.jpeg" as CFString, 1, nil) else {
        throw CaptureTestError.io
    }
    let gps: [String: Any] = [
        kCGImagePropertyGPSLatitude as String: 52.52,
        kCGImagePropertyGPSLatitudeRef as String: "N",
        kCGImagePropertyGPSLongitude as String: 13.405,
        kCGImagePropertyGPSLongitudeRef as String: "E",
    ]
    var properties: [String: Any] = [
        kCGImagePropertyGPSDictionary as String: gps,
    ]
    if let focalLength {
        let exif: [String: Any] = [
            kCGImagePropertyExifFocalLenIn35mmFilm as String: focalLength,
        ]
        properties[kCGImagePropertyExifDictionary as String] = exif
    }
    CGImageDestinationAddImage(destination, image, properties as CFDictionary)
    guard CGImageDestinationFinalize(destination) else {
        throw CaptureTestError.io
    }
}

// MARK: - Tests

@Suite struct CaptureImportTests {
    @Test func videoCountSelectsSharpFramesAndAllKeepsEveryCandidate() async throws {
        let directory: URL = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url: URL = directory.appendingPathComponent("clip.mp4")
        try await writeTestVideo(to: url)

        let selected: [CapturedFrame] = try await VideoFrameExtractor.extract(
            from: url,
            mode: .count(4),
            sampleInterval: 0.25)
        #expect(selected.count == 4)
        let times: [Double] = selected.compactMap { $0.timestamp }
        #expect(times.count == 4)
        #expect(times == times.sorted())
        // Blurry frames 3 and 7 cover [0.75, 1.0) and [1.75, 2.0) (PTS and sample times both fall there).
        for time in times {
            #expect(!(time >= 0.75 && time < 1.0))
            #expect(!(time >= 1.75 && time < 2.0))
        }
        for frame in selected {
            #expect(isPNG(frame.data))
            #expect(frame.sourceName == "clip.mp4")
        }
        #expect(selected.map { $0.name } == ["frame_0001.png", "frame_0002.png", "frame_0003.png", "frame_0004.png"])

        let all: [CapturedFrame] = try await VideoFrameExtractor.extract(
            from: url,
            mode: .all,
            sampleInterval: 0.25)
        #expect(all.count == 12)
        #expect(all.allSatisfy { isPNG($0.data) })
    }

    @Test func photoFolderAppliesOrientationSortsNamesAndStripsMetadata() throws {
        let directory: URL = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // Name order is numeric-aware: IMG_1, IMG_2, IMG_10.
        try writePhoto(to: directory.appendingPathComponent("IMG_10.jpg"), width: 320, height: 240, orientation: 6)
        try writePhoto(to: directory.appendingPathComponent("IMG_2.jpg"), width: 320, height: 240, orientation: 6)
        try writePhoto(to: directory.appendingPathComponent("IMG_1.jpg"), width: 320, height: 240, orientation: 6)
        try Data("not an image".utf8).write(to: directory.appendingPathComponent("notes.txt"))
        try writePhoto(to: directory.appendingPathComponent(".hidden.jpg"), width: 320, height: 240, orientation: 1)

        let frames: [CapturedFrame] = try PhotoFolderImporter.load(folder: directory)
        #expect(frames.count == 3)
        #expect(frames.map { $0.sourceName } == ["IMG_1.jpg", "IMG_2.jpg", "IMG_10.jpg"])
        #expect(frames.map { $0.name } == ["photo_0001.png", "photo_0002.png", "photo_0003.png"])
        #expect(frames.allSatisfy { $0.timestamp == nil })

        for frame in frames {
            #expect(isPNG(frame.data))
            guard let source = CGImageSourceCreateWithData(frame.data as CFData, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                Issue.record("output is not decodable: \(frame.name)")
                continue
            }
            // Stored pixels are 320x240; orientation 6 rotates them to 240x320.
            #expect(image.width == 240)
            #expect(image.height == 320)
            let properties: [String: Any] = (CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any]) ?? [:]
            #expect(properties[kCGImagePropertyGPSDictionary as String] == nil)
        }
    }

    @Test func photoFolderReadsFocalLengthFromExifAndStripsMetadata() throws {
        let directory: URL = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writePhotoWithFocalLength(to: directory.appendingPathComponent("a.jpg"), focalLength: NSNumber(value: 35))

        let frames: [CapturedFrame] = try PhotoFolderImporter.load(folder: directory)
        #expect(frames.count == 1)
        #expect(frames[0].focalLength35mm == 35)

        let source: CGImageSource? = CGImageSourceCreateWithData(frames[0].data as CFData, nil)
        let properties: [String: Any] = (source.flatMap { CGImageSourceCopyPropertiesAtIndex($0, 0, nil) } as? [String: Any]) ?? [:]
        // ImageIO may write its own pixel-size EXIF block, but the source focal length must not survive.
        let exif: [String: Any] = (properties[kCGImagePropertyExifDictionary as String] as? [String: Any]) ?? [:]
        #expect(exif[kCGImagePropertyExifFocalLenIn35mmFilm as String] == nil)
        #expect(properties[kCGImagePropertyGPSDictionary as String] == nil)
    }

    @Test func photoWithoutFocalLengthGivesNil() throws {
        let directory: URL = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writePhotoWithFocalLength(to: directory.appendingPathComponent("a.jpg"), focalLength: nil)

        let frames: [CapturedFrame] = try PhotoFolderImporter.load(folder: directory)
        #expect(frames.count == 1)
        #expect(frames[0].focalLength35mm == nil)
    }

    @Test func photoWithImplausibleFocalLengthGivesNil() throws {
        let directory: URL = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writePhotoWithFocalLength(to: directory.appendingPathComponent("zero.jpg"), focalLength: NSNumber(value: 0))
        try writePhotoWithFocalLength(to: directory.appendingPathComponent("huge.jpg"), focalLength: NSNumber(value: 5000))

        let frames: [CapturedFrame] = try PhotoFolderImporter.load(folder: directory)
        #expect(frames.map { $0.sourceName } == ["huge.jpg", "zero.jpg"])
        #expect(frames.allSatisfy { $0.focalLength35mm == nil })
    }

    @Test func emptyFolderThrows() throws {
        let directory: URL = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("text".utf8).write(to: directory.appendingPathComponent("readme.txt"))

        #expect(throws: FrameExtractionError.emptyFolder(directory.lastPathComponent)) {
            _ = try PhotoFolderImporter.load(folder: directory)
        }
    }
}
#endif
