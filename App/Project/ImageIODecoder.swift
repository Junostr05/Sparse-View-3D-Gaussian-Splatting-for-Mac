import CoreGraphics
import Foundation
import ImageIO
import SplatCore

/// Decodes image files into `RGBImage` values with sRGB components in `0...1`.
///
/// The image is drawn into an 8-bit RGBX sRGB bitmap, so embedded color profiles are converted to
/// sRGB and any alpha is composited over black. `nonisolated` so the decode runs on the caller's
/// executor (the training preparation runs it off the main actor).
nonisolated enum ImageIODecoder {
    enum DecodeError: Error, Equatable {
        case unreadableImage
        case emptyImage
        case bitmapCreationFailed
    }

    static func decode(_ data: Data) throws -> RGBImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw DecodeError.unreadableImage
        }
        let width: Int = image.width
        let height: Int = image.height
        guard width > 0, height > 0 else {
            throw DecodeError.emptyImage
        }
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw DecodeError.bitmapCreationFailed
        }

        let bytesPerRow: Int = width * 4
        var buffer = [UInt8](repeating: 0, count: bytesPerRow * height)
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
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else {
            throw DecodeError.bitmapCreationFailed
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
}
