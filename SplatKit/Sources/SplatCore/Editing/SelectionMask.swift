/// Bit set over `count` Gaussians, one bit per Gaussian index.
///
/// Invariant: bits at positions `>= count` in the last storage word are always zero, so
/// equality, population counts and inversion never see phantom selections.
public struct SelectionMask: Sendable, Equatable {
    public let count: Int
    private var words: [UInt64]

    /// Empty selection (no bits set) over `count` Gaussians.
    public init(count: Int) {
        precondition(count >= 0, "SelectionMask count must be non-negative, got \(count)")
        self.count = count
        self.words = [UInt64](repeating: 0, count: Self.wordCount(for: count))
    }

    /// Selection over `count` Gaussians with exactly the given indices set.
    /// Each index must lie in `0..<count`; an out-of-range index is a precondition failure.
    public init(count: Int, indices: [Int]) {
        self.init(count: count)
        for index in indices {
            precondition(index >= 0 && index < count,
                         "SelectionMask index \(index) is out of range 0..<\(count)")
            words[index >> 6] |= UInt64(1) << UInt64(index & 63)
        }
    }

    /// Selection with every Gaussian selected.
    public static func all(count: Int) -> SelectionMask {
        var mask = SelectionMask(count: count)
        for wordIndex in mask.words.indices {
            mask.words[wordIndex] = ~UInt64(0)
        }
        mask.clearUnusedTailBits()
        return mask
    }

    /// Number of selected Gaussians.
    public var selectedCount: Int {
        var total = 0
        for word in words {
            total += word.nonzeroBitCount
        }
        return total
    }

    /// True when no Gaussian is selected.
    public var isEmpty: Bool {
        words.allSatisfy { $0 == 0 }
    }

    public subscript(index: Int) -> Bool {
        get {
            precondition(index >= 0 && index < count,
                         "SelectionMask index \(index) is out of range 0..<\(count)")
            return words[index >> 6] & (UInt64(1) << UInt64(index & 63)) != 0
        }
        set {
            precondition(index >= 0 && index < count,
                         "SelectionMask index \(index) is out of range 0..<\(count)")
            let bit = UInt64(1) << UInt64(index & 63)
            if newValue {
                words[index >> 6] |= bit
            } else {
                words[index >> 6] &= ~bit
            }
        }
    }

    /// Adds every Gaussian selected in `other`. Both masks must have the same count.
    public mutating func formUnion(_ other: SelectionMask) {
        precondition(count == other.count,
                     "SelectionMask counts differ: \(count) and \(other.count)")
        for wordIndex in words.indices {
            words[wordIndex] |= other.words[wordIndex]
        }
    }

    /// Removes every Gaussian selected in `other`. Both masks must have the same count.
    public mutating func formSubtraction(_ other: SelectionMask) {
        precondition(count == other.count,
                     "SelectionMask counts differ: \(count) and \(other.count)")
        for wordIndex in words.indices {
            words[wordIndex] &= ~other.words[wordIndex]
        }
    }

    /// Flips every bit within `0..<count`.
    public mutating func invert() {
        for wordIndex in words.indices {
            words[wordIndex] = ~words[wordIndex]
        }
        clearUnusedTailBits()
    }

    /// Indices of the selected Gaussians, ascending.
    public func selectedIndices() -> [Int] {
        var result: [Int] = []
        result.reserveCapacity(selectedCount)
        for (wordIndex, word) in words.enumerated() {
            var remaining = word
            while remaining != 0 {
                let bit = remaining.trailingZeroBitCount
                result.append(wordIndex * 64 + bit)
                remaining &= remaining - 1
            }
        }
        return result
    }

    /// Indices of the Gaussians that are not selected, ascending.
    public func unselectedIndices() -> [Int] {
        var result: [Int] = []
        result.reserveCapacity(count - selectedCount)
        for (wordIndex, word) in words.enumerated() {
            var remaining = ~word
            while remaining != 0 {
                let bit = remaining.trailingZeroBitCount
                let index = wordIndex * 64 + bit
                if index < count {
                    result.append(index)
                }
                remaining &= remaining - 1
            }
        }
        return result
    }

    private static func wordCount(for count: Int) -> Int {
        (count + 63) >> 6
    }

    /// Restores the invariant after a whole-word operation that may have set bits past `count`.
    private mutating func clearUnusedTailBits() {
        let used = count & 63
        if used != 0, let last = words.indices.last {
            words[last] &= (UInt64(1) << UInt64(used)) - 1
        }
    }
}
