import AVFoundation
import CoreVideo
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers

/// Errors of the frame export: PNG and H.264 files written from rendered frames.
public enum FrameExportError: Error, LocalizedError {
    case cannotCreateImage
    case cannotWriteImage
    case cannotCreateWriter(String)
    case cannotAppendFrame(String)
    case cannotFinish(String)

    public var errorDescription: String? {
        switch self {
        case .cannotCreateImage: "The image could not be created."
        case .cannotWriteImage: "The image file could not be written."
        case .cannotCreateWriter(let reason): "The video writer could not be created: \(reason)"
        case .cannotAppendFrame(let reason): "A video frame could not be written: \(reason)"
        case .cannotFinish(let reason): "The video could not be finished: \(reason)"
        }
    }
}

/// Offscreen frames for export. A frame is a BGRA8 texture that `SplatRenderer` draws into and the CPU reads back
/// once the GPU has finished. Pure Metal and ImageIO, no viewer state.
public enum FrameExport {
    /// A `bgra8Unorm` texture in shared storage, which both the renderer and the CPU can access.
    public static func makeReadbackTexture(device: MTLDevice, width: Int, height: Int) throws(MetalError) -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderWrite, .shaderRead]
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw .bufferAllocation
        }
        return texture
    }

    /// Copies the pixels of `texture` (BGRA8) into `destination`, whose rows are `bytesPerRow` apart. Call only after
    /// the GPU has finished writing the texture.
    public static func copyBGRA(_ texture: MTLTexture, into destination: UnsafeMutableRawPointer, bytesPerRow: Int) {
        texture.getBytes(destination,
                         bytesPerRow: bytesPerRow,
                         from: MTLRegionMake2D(0, 0, texture.width, texture.height),
                         mipmapLevel: 0)
    }

    /// Writes `bgra` (`width × height × 4` bytes, rows packed, alpha ignored) as an sRGB PNG at `url`.
    public static func writePNG(bgra: Data, width: Int, height: Int, to url: URL) throws(FrameExportError) {
        guard width > 0, height > 0, bgra.count == width * height * 4 else {
            throw .cannotCreateImage
        }
        guard let provider = CGDataProvider(data: bgra as CFData),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bitsPerPixel: 32,
                                  bytesPerRow: width * 4,
                                  space: colorSpace,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Little.rawValue
                                      | CGImageAlphaInfo.noneSkipFirst.rawValue),
                                  provider: provider,
                                  decode: nil,
                                  shouldInterpolate: false,
                                  intent: .defaultIntent) else {
            throw .cannotCreateImage
        }
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw .cannotWriteImage
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw .cannotWriteImage
        }
    }
}

/// Writes frames as an H.264 video in an MP4 file, at the frame rate given. Frames are BGRA textures; the writer copies
/// each one into a pooled pixel buffer.
///
/// `@unchecked Sendable`: AVFoundation's writer objects are not Sendable, but this writer is used by one caller at a
/// time (the viewer's export job, on the main actor) and only moves between isolation regions with that caller.
public final class H264VideoWriter: @unchecked Sendable {
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private let frameRate: Int
    private var isFinished = false

    /// Starts writing a video of `width × height` pixels to `url`. Both sides must be even. The file must not exist.
    public init(url: URL, width: Int, height: Int, frameRate: Int) throws(FrameExportError) {
        guard width > 0, height > 0, width % 2 == 0, height % 2 == 0, frameRate > 0 else {
            throw .cannotCreateWriter("the frame size must be positive and even")
        }
        self.frameRate = frameRate
        do {
            writer = try AVAssetWriter(url: url, fileType: .mp4)
        } catch {
            throw .cannotCreateWriter(error.localizedDescription)
        }
        let bitRate: Int = Int(Double(width * height * frameRate) * 0.15)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitRate,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
                AVVideoMaxKeyFrameIntervalKey: frameRate,
            ],
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
            ],
        ]
        guard writer.canApply(outputSettings: settings, forMediaType: .video) else {
            throw .cannotCreateWriter("H.264 at \(width)×\(height) is not supported")
        }
        input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
        ]
        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: attributes)
        guard writer.canAdd(input) else {
            throw .cannotCreateWriter("the video input cannot be added")
        }
        writer.add(input)
        guard writer.startWriting() else {
            throw .cannotCreateWriter(writer.error?.localizedDescription ?? "startWriting failed")
        }
        writer.startSession(atSourceTime: .zero)
    }

    /// Waits until the encoder can take another frame. Throws `CancellationError` when the calling task is cancelled.
    /// Call it before each `append`. It is separate so that no texture has to cross the suspension point.
    public func waitUntilReadyForFrame() async throws {
        while !input.isReadyForMoreMediaData {
            try await Task.sleep(for: .milliseconds(4))
        }
    }

    /// Appends the frame `frameIndex` (presented at `frameIndex / frameRate` seconds). The texture must hold the
    /// finished frame. Copies the pixels synchronously, so the caller may reuse the texture afterwards.
    public func append(texture: MTLTexture, frameIndex: Int) throws(FrameExportError) {
        guard let pool = adaptor.pixelBufferPool else {
            throw FrameExportError.cannotAppendFrame("no pixel buffer pool")
        }
        var pixelBuffer: CVPixelBuffer?
        let status: CVReturn = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixelBuffer)
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw FrameExportError.cannotAppendFrame("pixel buffer \(status)")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            CVPixelBufferUnlockBaseAddress(buffer, [])
            throw FrameExportError.cannotAppendFrame("no base address")
        }
        FrameExport.copyBGRA(texture, into: base, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer))
        CVPixelBufferUnlockBaseAddress(buffer, [])

        let time = CMTime(value: CMTimeValue(frameIndex), timescale: CMTimeScale(frameRate))
        guard adaptor.append(buffer, withPresentationTime: time) else {
            throw FrameExportError.cannotAppendFrame(writer.error?.localizedDescription ?? "append failed")
        }
    }

    /// Ends the video and closes the file.
    public func finish() async throws {
        guard !isFinished else { return }
        isFinished = true
        input.markAsFinished()
        await writer.finishWriting()
        guard writer.status == .completed else {
            throw FrameExportError.cannotFinish(writer.error?.localizedDescription ?? "status \(writer.status.rawValue)")
        }
    }

    /// Stops writing and removes the partial file. Safe to call after `finish`, which then does nothing.
    public func cancel() {
        guard !isFinished else { return }
        isFinished = true
        writer.cancelWriting()
    }
}
