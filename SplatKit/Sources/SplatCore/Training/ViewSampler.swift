import Foundation

/// Visits every view once per epoch. Each epoch is a fresh seeded Fisher–Yates shuffle drawn
/// from one `SeededGenerator`, so the whole sequence is a deterministic function of the seed.
///
/// Shared by `CPUTrainer` and the Metal trainer (`SplatTraining`), which must visit views in the
/// same order for their results to be comparable.
public struct ViewSampler: Sendable {
    private var generator: SeededGenerator
    private var order: [Int]
    private var cursor: Int

    public init(count: Int, seed: UInt64) {
        generator = SeededGenerator(seed: seed)
        order = Array(0..<count)
        cursor = count   // forces a shuffle on the first call
    }

    public mutating func next() -> Int {
        if cursor >= order.count {
            shuffle()
            cursor = 0
        }
        let index = order[cursor]
        cursor += 1
        return index
    }

    private mutating func shuffle() {
        var i = order.count - 1
        while i > 0 {
            let j = Int.random(in: 0...i, using: &generator)
            order.swapAt(i, j)
            i -= 1
        }
    }
}
