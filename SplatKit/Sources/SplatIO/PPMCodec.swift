import Foundation
import SplatCore

/// Errors thrown by `PPMCodec.decode`.
public enum PPMError: Error, Equatable {
    /// The data is not a binary `P6` PPM, or its width, height or maxval is malformed.
    case invalidHeader
    /// The maxval is not 255. The payload is the declared value.
    case unsupportedMaxValue(Int)
    /// The pixel data ends before `width × height × 3` bytes.
    case truncated
}

/// Binary `P6` PPM codec with maxval 255. Used by tests and by tools that avoid ImageIO.
/// Pixel values map `byte / 255` to `0…1` on decode and back on encode.
public enum PPMCodec {
    public static func decode(_ data: Data) throws -> RGBImage {
        let bytes = [UInt8](data)
        var cursor = 0

        guard let magic = nextToken(bytes, &cursor), magic == "P6" else {
            throw PPMError.invalidHeader
        }
        let width = try positiveInt(nextToken(bytes, &cursor))
        let height = try positiveInt(nextToken(bytes, &cursor))
        guard let maxValueToken = nextToken(bytes, &cursor), let maxValue = Int(maxValueToken) else {
            throw PPMError.invalidHeader
        }
        guard maxValue == 255 else {
            throw PPMError.unsupportedMaxValue(maxValue)
        }
        // Exactly one whitespace byte separates the header from the raster.
        guard cursor < bytes.count, isWhitespace(bytes[cursor]) else {
            throw PPMError.invalidHeader
        }
        cursor += 1

        let (count, overflow) = width.multipliedReportingOverflow(by: height)
        guard !overflow, count <= (Int.max / 3) else {
            throw PPMError.invalidHeader
        }
        let byteCount = count * 3
        guard bytes.count - cursor >= byteCount else {
            throw PPMError.truncated
        }

        var pixels = [SIMD3<Float>](repeating: .zero, count: count)
        for index in 0..<count {
            let base = cursor + index * 3
            pixels[index] = SIMD3(Float(bytes[base]) / 255,
                                  Float(bytes[base + 1]) / 255,
                                  Float(bytes[base + 2]) / 255)
        }
        return RGBImage(width: width, height: height, pixels: pixels)
    }

    public static func encode(_ image: RGBImage) -> Data {
        precondition(image.pixels.count == image.width * image.height,
                     "pixel count must equal width × height")
        var bytes = Array("P6\n\(image.width) \(image.height)\n255\n".utf8)
        bytes.reserveCapacity(bytes.count + image.pixels.count * 3)
        for pixel in image.pixels {
            bytes.append(quantize(pixel.x))
            bytes.append(quantize(pixel.y))
            bytes.append(quantize(pixel.z))
        }
        return Data(bytes)
    }

    // MARK: - Helpers

    private static func quantize(_ value: Float) -> UInt8 {
        let clamped = min(max(value, 0), 1)
        return UInt8((clamped * 255).rounded())
    }

    private static func isWhitespace(_ byte: UInt8) -> Bool {
        byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D || byte == 0x0B || byte == 0x0C
    }

    /// Next whitespace-delimited header token, skipping `#` comments. Leaves the cursor on the
    /// whitespace byte that ends the token. Returns `nil` at end of input.
    private static func nextToken(_ bytes: [UInt8], _ cursor: inout Int) -> String? {
        while cursor < bytes.count {
            if bytes[cursor] == 0x23 {  // '#': comment runs to end of line
                while cursor < bytes.count, bytes[cursor] != 0x0A {
                    cursor += 1
                }
            } else if isWhitespace(bytes[cursor]) {
                cursor += 1
            } else {
                break
            }
        }
        guard cursor < bytes.count else { return nil }
        let start = cursor
        while cursor < bytes.count, !isWhitespace(bytes[cursor]) {
            cursor += 1
        }
        return String(decoding: bytes[start..<cursor], as: UTF8.self)
    }

    private static func positiveInt(_ token: String?) throws -> Int {
        guard let token, let value = Int(token), value > 0 else {
            throw PPMError.invalidHeader
        }
        return value
    }
}
