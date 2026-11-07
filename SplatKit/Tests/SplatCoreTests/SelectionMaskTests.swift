import Foundation
import Testing
import SplatCore

@Suite struct SelectionMaskTests {
    /// Counts that stress the 64-bit word boundary.
    private let boundaryCounts: [Int] = [1, 63, 64, 65, 130]

    @Test func newMaskIsEmpty() {
        for count in boundaryCounts {
            let mask = SelectionMask(count: count)
            #expect(mask.count == count)
            #expect(mask.selectedCount == 0)
            #expect(mask.isEmpty)
            #expect(mask.selectedIndices().isEmpty)
            #expect(mask.unselectedIndices() == Array(0..<count))
        }
    }

    @Test func allSelectsExactlyTheCount() {
        for count in boundaryCounts {
            let mask = SelectionMask.all(count: count)
            #expect(mask.selectedCount == count, "count \(count)")
            #expect(!mask.isEmpty)
            #expect(mask.selectedIndices() == Array(0..<count))
            #expect(mask.unselectedIndices().isEmpty)
        }
    }

    @Test func invertNeverSetsBitsBeyondCount() {
        for count in boundaryCounts {
            var mask = SelectionMask.all(count: count)
            mask.invert()
            #expect(mask.selectedCount == 0, "count \(count)")
            #expect(mask.isEmpty)
            #expect(mask == SelectionMask(count: count))

            mask.invert()
            #expect(mask == SelectionMask.all(count: count), "double invert, count \(count)")
        }
    }

    @Test func invertOfPartialMaskFlipsEveryBit() {
        var mask = SelectionMask(count: 65, indices: [0, 63, 64])
        mask.invert()
        #expect(mask.selectedCount == 62)
        #expect(!mask[0])
        #expect(!mask[63])
        #expect(!mask[64])
        #expect(mask[1])
        #expect(mask.selectedIndices().last == 62)
    }

    @Test func subscriptGetAndSetAcrossWords() {
        var mask = SelectionMask(count: 130)
        for index in [0, 1, 63, 64, 65, 127, 128, 129] {
            mask[index] = true
            #expect(mask[index])
        }
        #expect(mask.selectedCount == 8)
        mask[64] = false
        #expect(!mask[64])
        #expect(mask.selectedCount == 7)
        #expect(mask.selectedIndices() == [0, 1, 63, 65, 127, 128, 129])
    }

    @Test func indicesRoundTripThroughSelectedIndices() {
        var rng = SeededGenerator(seed: 11)
        for count in boundaryCounts {
            var chosen: [Int] = []
            for index in 0..<count where Int.random(in: 0..<3, using: &rng) == 0 {
                chosen.append(index)
            }
            let mask = SelectionMask(count: count, indices: chosen)
            #expect(mask.selectedIndices() == chosen, "count \(count)")
            #expect(mask.selectedCount == chosen.count)
            let rebuilt = SelectionMask(count: count, indices: mask.selectedIndices())
            #expect(rebuilt == mask)
            let complement = Set(0..<count).subtracting(chosen).sorted()
            #expect(mask.unselectedIndices() == complement)
        }
    }

    @Test func unionAndSubtractionMatchHandComputedSets() {
        let a = SelectionMask(count: 70, indices: [0, 1, 2, 64])
        let b = SelectionMask(count: 70, indices: [2, 3, 69])

        var union = a
        union.formUnion(b)
        #expect(union.selectedIndices() == [0, 1, 2, 3, 64, 69])

        var difference = a
        difference.formSubtraction(b)
        #expect(difference.selectedIndices() == [0, 1, 64])

        var reversed = b
        reversed.formSubtraction(a)
        #expect(reversed.selectedIndices() == [3, 69])
    }

    @Test func subtractingItselfGivesEmpty() {
        var mask = SelectionMask.all(count: 65)
        let copy = mask
        mask.formSubtraction(copy)
        #expect(mask.isEmpty)
    }

    @Test func unionWithEmptyIsIdentity() {
        let mask = SelectionMask(count: 3, indices: [1])
        var result = mask
        result.formUnion(SelectionMask(count: 3))
        #expect(result == mask)
    }

    @Test func equalityComparesCountAndBits() {
        #expect(SelectionMask(count: 5, indices: [1]) == SelectionMask(count: 5, indices: [1]))
        #expect(SelectionMask(count: 5, indices: [1]) != SelectionMask(count: 5, indices: [2]))
        #expect(SelectionMask(count: 5) != SelectionMask(count: 6))
    }
}
