import Foundation
import SplatCore

/// Why a `.sdpt` depth file cannot be decoded.
public enum DepthMapError: Error, Equatable {
    case badMagic
    case unsupportedVersion(Int)
    case unknownKind(UInt8)
    case invalidSize(width: Int, height: Int)
    /// The byte count does not match the header. Also reported for trailing bytes.
    case truncated(expected: Int, actual: Int)
}

/// Reads and writes the `.sdpt` depth format (Phase 4 P2). Stored in a project as
/// `depth/<image name>.sdpt`.
///
/// All integers are little-endian. Layout:
///
/// | Offset | Size | Field |
/// |---|---|---|
/// | 0 | 4 | magic `"SDPT"` |
/// | 4 | 2 | `UInt16` version, currently 1 |
/// | 6 | 1 | `UInt8` kind: 0 metric, 1 relative depth, 2 relative inverse depth |
/// | 7 | 1 | `UInt8` flags: bit 0 confidence plane present, bit 1 estimated by a model |
/// | 8 | 4 | `UInt32` width, 1...16384 |
/// | 12 | 4 | `UInt32` height, 1...16384 |
/// | 16 | 8 | reserved, zero |
/// | 24 | 4·w·h | `Float32` values, row-major; non-finite or ≤ 0 is invalid |
/// | after values | w·h | optional `UInt8` confidence (ARKit 0/1/2) |
///
/// The file length must match the header exactly.
public enum DepthMapCodec {
    public static let fileExtension = "sdpt"

    static let magic: [UInt8] = [0x53, 0x44, 0x50, 0x54]   // "SDPT"
    static let formatVersion: UInt16 = 1
    static let headerSize: Int = 24
    static let maximumSide: Int = 16384
    static let confidenceFlag: UInt8 = 1 << 0
    static let estimatedFlag: UInt8 = 1 << 1

    /// Serialises `map`. A map whose value count does not match its size is a programming error.
    public static func encode(_ map: DepthMap) -> Data {
        let count: Int = map.width * map.height
        precondition(map.values.count == count, "DepthMap values do not match its size")
        if let confidence = map.confidence {
            precondition(confidence.count == count, "DepthMap confidence does not match its size")
        }

        var flags: UInt8 = 0
        if map.confidence != nil {
            flags |= confidenceFlag
        }
        if map.isEstimated {
            flags |= estimatedFlag
        }

        var bytes: [UInt8] = []
        let payloadCount: Int = map.confidence == nil ? count * 4 : count * 5
        bytes.reserveCapacity(headerSize + payloadCount)
        bytes.append(contentsOf: magic)
        appendUInt16(formatVersion, to: &bytes)
        bytes.append(map.kind.rawValue)
        bytes.append(flags)
        appendUInt32(UInt32(map.width), to: &bytes)
        appendUInt32(UInt32(map.height), to: &bytes)
        bytes.append(contentsOf: [UInt8](repeating: 0, count: 8))

        for value in map.values {
            appendUInt32(value.bitPattern, to: &bytes)
        }
        if let confidence = map.confidence {
            bytes.append(contentsOf: confidence)
        }
        return Data(bytes)
    }

    /// Parses a `.sdpt` file. Checks, in order: header length, magic, version, kind, size, and
    /// the exact file length.
    public static func decode(_ data: Data) throws(DepthMapError) -> DepthMap {
        let bytes: [UInt8] = [UInt8](data)
        guard bytes.count >= headerSize else {
            throw .truncated(expected: headerSize, actual: bytes.count)
        }
        guard Array(bytes[0..<4]) == magic else {
            throw .badMagic
        }

        let version: Int = Int(readUInt16(bytes, at: 4))
        guard version == Int(formatVersion) else {
            throw .unsupportedVersion(version)
        }

        let kindByte: UInt8 = bytes[6]
        guard let kind = DepthKind(rawValue: kindByte) else {
            throw .unknownKind(kindByte)
        }

        let flags: UInt8 = bytes[7]
        let hasConfidence: Bool = flags & confidenceFlag != 0
        let isEstimated: Bool = flags & estimatedFlag != 0

        let width: Int = Int(readUInt32(bytes, at: 8))
        let height: Int = Int(readUInt32(bytes, at: 12))
        guard (1...maximumSide).contains(width), (1...maximumSide).contains(height) else {
            throw .invalidSize(width: width, height: height)
        }

        let count: Int = width * height
        let expectedLength: Int = headerSize + count * 4 + (hasConfidence ? count : 0)
        guard bytes.count == expectedLength else {
            throw .truncated(expected: expectedLength, actual: bytes.count)
        }

        var values: [Float] = []
        values.reserveCapacity(count)
        for index in 0..<count {
            let offset: Int = headerSize + index * 4
            values.append(Float(bitPattern: readUInt32(bytes, at: offset)))
        }

        var confidence: [UInt8]? = nil
        if hasConfidence {
            let start: Int = headerSize + count * 4
            confidence = Array(bytes[start..<(start + count)])
        }

        return DepthMap(width: width, height: height, kind: kind, values: values,
                        confidence: confidence, isEstimated: isEstimated)
    }

    // MARK: - Byte helpers

    private static func appendUInt16(_ value: UInt16, to bytes: inout [UInt8]) {
        for shift in 0..<2 {
            let byte: UInt16 = (value >> UInt16(8 * shift)) & 0xFF
            bytes.append(UInt8(byte))
        }
    }

    private static func appendUInt32(_ value: UInt32, to bytes: inout [UInt8]) {
        for shift in 0..<4 {
            let byte: UInt32 = (value >> UInt32(8 * shift)) & 0xFF
            bytes.append(UInt8(byte))
        }
    }

    private static func readUInt16(_ bytes: [UInt8], at offset: Int) -> UInt16 {
        var value: UInt16 = 0
        for index in 0..<2 {
            let byte: UInt16 = UInt16(bytes[offset + index])
            value |= byte << UInt16(8 * index)
        }
        return value
    }

    private static func readUInt32(_ bytes: [UInt8], at offset: Int) -> UInt32 {
        var value: UInt32 = 0
        for index in 0..<4 {
            let byte: UInt32 = UInt32(bytes[offset + index])
            value |= byte << UInt32(8 * index)
        }
        return value
    }
}
