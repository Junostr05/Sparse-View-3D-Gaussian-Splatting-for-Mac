import Foundation
#if canImport(Compression)
import Compression
#endif

/// Gzip container (RFC 1952) around raw deflate (RFC 1951), as used by SPZ version 2 files.
///
/// Writing: on Apple platforms (`Compression` available) the deflate body is produced with
/// `COMPRESSION_ZLIB`, which emits raw deflate. Elsewhere the body is a sequence of stored
/// (uncompressed) deflate blocks: a valid gzip stream, but not compressed.
///
/// Reading: on Apple platforms every deflate stream is inflated with `Compression`. Elsewhere only
/// stored blocks are accepted; compressed blocks throw `SplatIOError.unsupportedCompression`.
/// On every platform the CRC-32 and ISIZE trailer are verified.
enum Gzip {
    /// Largest expansion deflate can produce (about 1032:1). Bounds the size declared by ISIZE
    /// before any output buffer is allocated.
    private static let maxDeflateExpansion = 1032

    // MARK: - Writing

    /// Wraps `input` in a gzip stream. Uses `Compression` where it is available, stored blocks otherwise.
    static func compress(_ input: [UInt8]) -> [UInt8] {
        #if canImport(Compression)
        let body = deflateWithCompression(input) ?? storedDeflate(input)
        #else
        let body = storedDeflate(input)
        #endif
        return wrap(body: body, input: input)
    }

    /// Wraps `input` in a gzip stream whose deflate body is made of stored blocks on every platform.
    static func compressStored(_ input: [UInt8]) -> [UInt8] {
        wrap(body: storedDeflate(input), input: input)
    }

    private static func wrap(body: [UInt8], input: [UInt8]) -> [UInt8] {
        var out: [UInt8] = [0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff]
        out.reserveCapacity(out.count + body.count + 8)
        out.append(contentsOf: body)
        appendLittleEndian(CRC32.checksum(input), to: &out)
        appendLittleEndian(UInt32(truncatingIfNeeded: input.count), to: &out)
        return out
    }

    /// Deflate body made of stored blocks (BTYPE = 00), at most 65 535 bytes each.
    static func storedDeflate(_ input: [UInt8]) -> [UInt8] {
        var out: [UInt8] = []
        out.reserveCapacity(input.count + 5 * (input.count / 65_535 + 1))
        var offset = 0
        repeat {
            let length = min(65_535, input.count - offset)
            let isFinal = offset + length == input.count
            out.append(isFinal ? 0x01 : 0x00)
            let nlength = (~length) & 0xffff
            out.append(UInt8(length & 0xff))
            out.append(UInt8(length >> 8))
            out.append(UInt8(nlength & 0xff))
            out.append(UInt8(nlength >> 8))
            out.append(contentsOf: input[offset..<offset + length])
            offset += length
        } while offset < input.count
        return out
    }

    // MARK: - Reading

    /// Returns the uncompressed payload of a gzip stream, after checking CRC-32 and ISIZE.
    static func decompress(_ data: Data) throws(SplatIOError) -> [UInt8] {
        let bytes = [UInt8](data)
        guard bytes.count >= 18, bytes[0] == 0x1f, bytes[1] == 0x8b else {
            throw .corrupt("missing gzip header")
        }
        guard bytes[2] == 0x08 else {
            throw .corrupt("unsupported gzip compression method \(bytes[2])")
        }
        let flags = bytes[3]
        var position = 10
        let trailerStart = bytes.count - 8

        if flags & 0x04 != 0 {
            guard position + 2 <= trailerStart else { throw .corrupt("truncated gzip header") }
            let extraLength = Int(bytes[position]) | Int(bytes[position + 1]) << 8
            position += 2 + extraLength
        }
        for bit: UInt8 in [0x08, 0x10] where flags & bit != 0 {
            // FNAME and FCOMMENT are zero-terminated strings.
            while position < trailerStart && bytes[position] != 0 { position += 1 }
            position += 1
        }
        if flags & 0x02 != 0 {
            position += 2
        }
        guard position <= trailerStart else { throw .corrupt("truncated gzip header") }

        let expectedCRC = readLittleEndian32(bytes, at: trailerStart)
        let expectedSize = readLittleEndian32(bytes, at: trailerStart + 4)
        let body = Array(bytes[position..<trailerStart])
        guard UInt64(expectedSize) <= UInt64(body.count) * UInt64(maxDeflateExpansion) + 16 else {
            throw .corrupt("ISIZE \(expectedSize) is impossible for a \(body.count)-byte body")
        }

        #if canImport(Compression)
        let output: [UInt8]
        if expectedSize == 0 {
            output = []
        } else {
            guard let inflated = inflateWithCompression(body, outputSize: Int(expectedSize)) else {
                throw .corrupt("deflate stream does not decode to \(expectedSize) bytes")
            }
            output = inflated
        }
        #else
        let output = try inflateStored(body, sizeLimit: Int(expectedSize))
        #endif

        guard UInt32(truncatingIfNeeded: output.count) == expectedSize else {
            throw .corrupt("uncompressed size mismatch")
        }
        guard CRC32.checksum(output) == expectedCRC else {
            throw .corrupt("CRC-32 mismatch")
        }
        return output
    }

    /// Inflates a deflate body that contains only stored blocks.
    private static func inflateStored(_ body: [UInt8], sizeLimit: Int) throws(SplatIOError) -> [UInt8] {
        var output: [UInt8] = []
        output.reserveCapacity(sizeLimit)
        var position = 0
        while true {
            guard position < body.count else { throw .corrupt("truncated deflate stream") }
            let header = body[position]
            position += 1
            let isFinal = header & 0x01 != 0
            let blockType = (header >> 1) & 0x03
            guard blockType == 0 else {
                throw .unsupportedCompression
            }
            guard position + 4 <= body.count else { throw .corrupt("truncated stored block") }
            let length = Int(body[position]) | Int(body[position + 1]) << 8
            let nlength = Int(body[position + 2]) | Int(body[position + 3]) << 8
            guard length ^ 0xffff == nlength else { throw .corrupt("stored block length check failed") }
            position += 4
            guard position + length <= body.count else { throw .corrupt("truncated stored block") }
            guard output.count + length <= sizeLimit else { throw .corrupt("stored data exceeds ISIZE") }
            output.append(contentsOf: body[position..<position + length])
            position += length
            if isFinal { break }
        }
        guard position == body.count else { throw .corrupt("trailing bytes after deflate stream") }
        return output
    }

    // MARK: - Apple Compression framework

    #if canImport(Compression)
    private static func deflateWithCompression(_ input: [UInt8]) -> [UInt8]? {
        guard !input.isEmpty else { return nil }
        var output = [UInt8](repeating: 0, count: input.count * 2 + 1024)
        let scratchSize = compression_encode_scratch_buffer_size(COMPRESSION_ZLIB)
        let scratch = UnsafeMutableRawPointer.allocate(byteCount: max(scratchSize, 1), alignment: 16)
        defer { scratch.deallocate() }
        let written: Int = input.withUnsafeBufferPointer { source in
            output.withUnsafeMutableBufferPointer { destination in
                compression_encode_buffer(destination.baseAddress!, destination.count,
                                          source.baseAddress!, source.count,
                                          scratch, COMPRESSION_ZLIB)
            }
        }
        guard written > 0 else { return nil }
        return Array(output[0..<written])
    }

    private static func inflateWithCompression(_ body: [UInt8], outputSize: Int) -> [UInt8]? {
        guard !body.isEmpty else { return nil }
        var output = [UInt8](repeating: 0, count: outputSize)
        let scratchSize = compression_decode_scratch_buffer_size(COMPRESSION_ZLIB)
        let scratch = UnsafeMutableRawPointer.allocate(byteCount: max(scratchSize, 1), alignment: 16)
        defer { scratch.deallocate() }
        let produced: Int = body.withUnsafeBufferPointer { source in
            output.withUnsafeMutableBufferPointer { destination in
                compression_decode_buffer(destination.baseAddress!, destination.count,
                                          source.baseAddress!, source.count,
                                          scratch, COMPRESSION_ZLIB)
            }
        }
        guard produced == outputSize else { return nil }
        return output
    }
    #endif

    // MARK: - Byte helpers

    private static func appendLittleEndian(_ value: UInt32, to out: inout [UInt8]) {
        out.append(UInt8(value & 0xff))
        out.append(UInt8((value >> 8) & 0xff))
        out.append(UInt8((value >> 16) & 0xff))
        out.append(UInt8(value >> 24))
    }

    private static func readLittleEndian32(_ bytes: [UInt8], at offset: Int) -> UInt32 {
        UInt32(bytes[offset])
            | UInt32(bytes[offset + 1]) << 8
            | UInt32(bytes[offset + 2]) << 16
            | UInt32(bytes[offset + 3]) << 24
    }
}

/// CRC-32 as used by gzip (polynomial 0xEDB88320, reflected).
enum CRC32 {
    private static let table: [UInt32] = {
        var result = [UInt32](repeating: 0, count: 256)
        for index in 0..<256 {
            var value = UInt32(index)
            for _ in 0..<8 {
                if value & 1 != 0 {
                    value = 0xEDB8_8320 ^ (value >> 1)
                } else {
                    value >>= 1
                }
            }
            result[index] = value
        }
        return result
    }()

    static func checksum(_ bytes: [UInt8]) -> UInt32 {
        var crc: UInt32 = 0xffff_ffff
        for byte in bytes {
            let index = Int((crc ^ UInt32(byte)) & 0xff)
            crc = table[index] ^ (crc >> 8)
        }
        return crc ^ 0xffff_ffff
    }
}
