import Foundation

/// Errors from decoding a deletion list.
public enum DeletionListError: Error, Equatable {
    /// The data ends in the middle of a varint.
    case truncated
    /// A varint does not fit in 32 bits, or a running index sum exceeds `UInt32.max`.
    case overflow
    /// A delta after the first entry is zero, so indices are not strictly ascending.
    case notAscending
    /// An index is at or beyond `maxIndex`. Carries the index.
    case indexOutOfRange(UInt32)
}

/// Compact, persistent form of a sorted set of Gaussian indices.
///
/// Layout: the first index, then the difference to each previous index, every value as an unsigned
/// LEB128 varint (7 bits per byte, least significant group first, high bit = continuation). The first
/// value is the first index itself. Deltas after it are at least 1, so the list is strictly ascending.
public enum DeletionListCodec {
    /// Encodes strictly ascending indices. An empty list encodes to empty data.
    ///
    /// Precondition: `sortedIndices` is strictly ascending. Unsorted input is a programming error.
    public static func encode(_ sortedIndices: [UInt32]) -> Data {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(sortedIndices.count * 2)
        var previous: UInt32 = 0
        for (position, index) in sortedIndices.enumerated() {
            let delta: UInt32
            if position == 0 {
                delta = index
            } else {
                precondition(index > previous, "deletion indices must be strictly ascending")
                delta = index - previous
            }
            appendVarint(delta, to: &bytes)
            previous = index
        }
        return Data(bytes)
    }

    /// Decodes a list written by `encode`. Every index must be below `maxIndex`.
    ///
    /// Throws `DeletionListError` for truncated varints, overflow, non-ascending entries and indices at
    /// or beyond `maxIndex`. It never traps.
    public static func decode(_ data: Data, maxIndex: Int) throws(DeletionListError) -> [UInt32] {
        let bytes = [UInt8](data)
        let limit = UInt64(max(0, maxIndex))
        var indices: [UInt32] = []
        var position = 0
        var previous: UInt64 = 0
        while position < bytes.count {
            let delta = try readVarint(bytes, at: &position)
            if !indices.isEmpty && delta == 0 {
                throw .notAscending
            }
            let sum: UInt64 = previous + UInt64(delta)
            guard sum <= UInt64(UInt32.max) else {
                throw .overflow
            }
            let index = UInt32(sum)
            guard sum < limit else {
                throw .indexOutOfRange(index)
            }
            indices.append(index)
            previous = sum
        }
        return indices
    }

    // MARK: - Varint

    private static func appendVarint(_ value: UInt32, to bytes: inout [UInt8]) {
        var remaining = value
        while remaining >= 0x80 {
            bytes.append(UInt8(remaining & 0x7f) | 0x80)
            remaining >>= 7
        }
        bytes.append(UInt8(remaining))
    }

    /// Reads one unsigned LEB128 value of at most 32 bits.
    private static func readVarint(_ bytes: [UInt8], at position: inout Int) throws(DeletionListError) -> UInt32 {
        var value: UInt64 = 0
        var shift: UInt64 = 0
        while true {
            guard position < bytes.count else {
                throw .truncated
            }
            // Five bytes cover 35 bits. A sixth byte can only overflow 32 bits.
            guard shift < 35 else {
                throw .overflow
            }
            let byte = bytes[position]
            position += 1
            value |= UInt64(byte & 0x7f) << shift
            if byte & 0x80 == 0 {
                break
            }
            shift += 7
        }
        guard value <= UInt64(UInt32.max) else {
            throw .overflow
        }
        return UInt32(value)
    }
}
