import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import SplatCore

/// Why the companion's depth model could not be loaded or run.
nonisolated enum CompanionDepthError: Error, Equatable {
    case modelNotFound(String)
    case unexpectedModelInterface(String)
    case predictionFailed(String)
    case imageBuffer(String)
}

// Image preparation and depth conversion, copied from SplatKit/Sources/SplatDepth/MonocularDepthEstimator.swift
// (MonocularDepthImageSupport), renamed and with the companion's error type. The SplatDepth module is macOS-only,
// so the companion carries its own copy until the shared part moves to a platform-neutral module.
nonisolated enum CompanionImageSupport {
    /// Rotates an image 90° clockwise. The result is `height` wide and `width` high.
    static func rotatedClockwise(_ image: RGBImage) -> RGBImage {
        let outputWidth: Int = image.height
        let outputHeight: Int = image.width
        var pixels: [SIMD3<Float>] = [SIMD3<Float>](repeating: SIMD3<Float>(0, 0, 0), count: outputWidth * outputHeight)
        for row in 0..<image.height {
            for column in 0..<image.width {
                // Clockwise: source (x, y) lands at output (H - 1 - y, x).
                let outputColumn: Int = image.height - 1 - row
                let outputRow: Int = column
                pixels[outputRow * outputWidth + outputColumn] = image.pixels[row * image.width + column]
            }
        }
        return RGBImage(width: outputWidth, height: outputHeight, pixels: pixels)
    }

    /// Converts an sRGB image (values 0...1) to a 32BGRA pixel buffer of `width` × `height`, stretching
    /// it with Core Graphics. Both sides use the sRGB colour space, so the values are not altered.
    static func inputBuffer(from image: RGBImage, width: Int, height: Int) throws(CompanionDepthError) -> CVPixelBuffer {
        guard image.width > 0, image.height > 0, width > 0, height > 0,
              image.pixels.count == image.width * image.height else {
            throw .predictionFailed("the view image is empty or inconsistent")
        }
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw .predictionFailed("the sRGB colour space is unavailable")
        }

        // Source pixels as 8-bit RGBX, top row first.
        let sourceBytesPerRow: Int = image.width * 4
        var sourceBytes: [UInt8] = [UInt8](repeating: 255, count: sourceBytesPerRow * image.height)
        for index in 0..<(image.width * image.height) {
            let pixel: SIMD3<Float> = image.pixels[index]
            sourceBytes[index * 4] = byte(pixel.x)
            sourceBytes[index * 4 + 1] = byte(pixel.y)
            sourceBytes[index * 4 + 2] = byte(pixel.z)
        }
        guard let provider = CGDataProvider(data: Data(sourceBytes) as CFData),
              let source = CGImage(width: image.width,
                                   height: image.height,
                                   bitsPerComponent: 8,
                                   bitsPerPixel: 32,
                                   bytesPerRow: sourceBytesPerRow,
                                   space: colorSpace,
                                   bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                   provider: provider,
                                   decode: nil,
                                   shouldInterpolate: false,
                                   intent: .defaultIntent) else {
            throw .predictionFailed("could not create the source image")
        }

        var created: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        let status: CVReturn = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                                   kCVPixelFormatType_32BGRA,
                                                   attributes as CFDictionary, &created)
        guard status == kCVReturnSuccess, let buffer = created else {
            throw .predictionFailed("could not allocate the input buffer (status \(status))")
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw .predictionFailed("the input buffer has no memory")
        }
        // 32-bit little endian with no alpha: bytes are B, G, R, X, which is 32BGRA.
        let bitmapInfo: UInt32 = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue
        guard let context = CGContext(data: base,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                      space: colorSpace,
                                      bitmapInfo: bitmapInfo) else {
            throw .predictionFailed("could not create the drawing context")
        }
        context.interpolationQuality = .high
        context.draw(source, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }

    /// Reads a one-component `Grayscale16Half` or `OneComponent32Float` output into a depth map.
    /// Values that are not finite or are ≤ 0 become 0 (invalid).
    static func depthMap(from buffer: CVPixelBuffer) throws(CompanionDepthError) -> DepthMap {
        let format: OSType = CVPixelBufferGetPixelFormatType(buffer)
        let isHalf: Bool = format == kCVPixelFormatType_OneComponent16Half
        guard isHalf || format == kCVPixelFormatType_OneComponent32Float else {
            throw .unexpectedModelInterface("unsupported output pixel format \(format)")
        }
        let width: Int = CVPixelBufferGetWidth(buffer)
        let height: Int = CVPixelBufferGetHeight(buffer)

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw .predictionFailed("the output buffer has no memory")
        }
        let bytesPerRow: Int = CVPixelBufferGetBytesPerRow(buffer)
        var values: [Float] = [Float](repeating: 0, count: width * height)
        for row in 0..<height {
            let rowPointer: UnsafeRawPointer = UnsafeRawPointer(base + row * bytesPerRow)
            for column in 0..<width {
                let value: Float
                if isHalf {
                    let bits: UInt16 = rowPointer.load(fromByteOffset: column * 2, as: UInt16.self)
                    value = Float(Float16(bitPattern: bits))
                } else {
                    value = rowPointer.load(fromByteOffset: column * 4, as: Float.self)
                }
                values[row * width + column] = (value.isFinite && value > 0) ? value : 0
            }
        }
        return DepthMap(width: width, height: height, kind: .relativeInverseDepth,
                        values: values, confidence: nil, isEstimated: true)
    }

    /// An sRGB value in 0...1 as an 8-bit byte.
    private static func byte(_ value: Float) -> UInt8 {
        let clamped: Float = min(max(value, 0), 1)
        return UInt8((clamped * 255).rounded())
    }
}
