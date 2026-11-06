#if canImport(AVFoundation) && canImport(ImageIO) && canImport(CoreGraphics)
import CoreGraphics
import Foundation
import ImageIO
import SplatCore

/// Image helpers shared by the video and photo frame sources. macOS only.
enum CaptureImageSupport {
    /// Converts `image` to an sRGB `RGBImage` (components in `0...1`), scaled down so the longer side
    /// is at most `maxSide` (never upscaled).
    static func rgbImage(_ image: CGImage, maxSide: Int, sourceName: String) throws -> RGBImage {
        let sourceWidth: Int = image.width
        let sourceHeight: Int = image.height
        let longSide: Int = max(sourceWidth, sourceHeight)
        var width: Int = sourceWidth
        var height: Int = sourceHeight
        if longSide > maxSide && longSide > 0 {
            let scale: Double = Double(maxSide) / Double(longSide)
            let scaledWidth: Double = (Double(sourceWidth) * scale).rounded()
            let scaledHeight: Double = (Double(sourceHeight) * scale).rounded()
            width = max(1, Int(scaledWidth))
            height = max(1, Int(scaledHeight))
        }
        guard width > 0, height > 0 else {
            throw FrameExtractionError.processingFailed(sourceName)
        }
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw FrameExtractionError.processingFailed(sourceName)
        }

        let bytesPerRow: Int = width * 4
        var buffer: [UInt8] = [UInt8](repeating: 0, count: bytesPerRow * height)
        let bitmapInfo: UInt32 = CGImageAlphaInfo.noneSkipLast.rawValue
        let drawn: Bool = buffer.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) -> Bool in
            guard let context = CGContext(data: raw.baseAddress,
                                          width: width,
                                          height: height,
                                          bitsPerComponent: 8,
                                          bytesPerRow: bytesPerRow,
                                          space: colorSpace,
                                          bitmapInfo: bitmapInfo) else {
                return false
            }
            context.interpolationQuality = .high
            let target: CGRect = CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height))
            context.draw(image, in: target)
            return true
        }
        guard drawn else {
            throw FrameExtractionError.processingFailed(sourceName)
        }

        let pixelCount: Int = width * height
        var pixels: [SIMD3<Float>] = []
        pixels.reserveCapacity(pixelCount)
        let scale: Float = 1.0 / 255.0
        for index in 0..<pixelCount {
            let base: Int = index * 4
            let red: Float = Float(buffer[base]) * scale
            let green: Float = Float(buffer[base + 1]) * scale
            let blue: Float = Float(buffer[base + 2]) * scale
            pixels.append(SIMD3<Float>(red, green, blue))
        }
        return RGBImage(width: width, height: height, pixels: pixels)
    }

    /// Encodes `image` as PNG. The output carries no metadata beyond what the CGImage holds, so
    /// EXIF and GPS from the source file are never written.
    static func pngData(_ image: CGImage, sourceName: String) throws -> Data {
        let storage: NSMutableData = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(storage, "public.png" as CFString, 1, nil) else {
            throw FrameExtractionError.encodingFailed(sourceName)
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw FrameExtractionError.encodingFailed(sourceName)
        }
        return storage as Data
    }
}
#endif
