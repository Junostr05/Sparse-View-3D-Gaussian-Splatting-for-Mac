import Foundation
import Testing
import SplatCore

@Suite struct EditHistoryTests {
    @Test func startsWithNothingDeleted() {
        let history = EditHistory(baseCount: 6)
        #expect(history.deleted.count == 6)
        #expect(history.deleted.isEmpty)
        #expect(history.survivorCount == 6)
        #expect(!history.canUndo)
        #expect(!history.canRedo)
        #expect(history.baseIndices() == [0, 1, 2, 3, 4, 5])
    }

    @Test func deleteUndoRedoSequence() {
        var history = EditHistory(baseCount: 10)
        history.delete(SelectionMask(count: 10, indices: [1, 2]))
        #expect(history.deleted.selectedIndices() == [1, 2])
        #expect(history.survivorCount == 8)
        #expect(history.canUndo)
        #expect(!history.canRedo)

        history.undo()
        #expect(history.deleted.isEmpty)
        #expect(!history.canUndo)
        #expect(history.canRedo)

        history.redo()
        #expect(history.deleted.selectedIndices() == [1, 2])
        #expect(history.canUndo)
        #expect(!history.canRedo)
    }

    @Test func multipleStepsUndoInReverseOrder() {
        var history = EditHistory(baseCount: 8)
        history.delete(SelectionMask(count: 8, indices: [0]))
        history.delete(SelectionMask(count: 8, indices: [5]))
        history.delete(SelectionMask(count: 8, indices: [7]))
        #expect(history.deleted.selectedIndices() == [0, 5, 7])

        history.undo()
        #expect(history.deleted.selectedIndices() == [0, 5])
        history.undo()
        #expect(history.deleted.selectedIndices() == [0])
        history.redo()
        #expect(history.deleted.selectedIndices() == [0, 5])
        history.undo()
        history.undo()
        #expect(history.deleted.isEmpty)
        #expect(!history.canUndo)
    }

    @Test func newDeletionClearsRedoStack() {
        var history = EditHistory(baseCount: 4)
        history.delete(SelectionMask(count: 4, indices: [0]))
        history.undo()
        #expect(history.canRedo)

        history.delete(SelectionMask(count: 4, indices: [3]))
        #expect(!history.canRedo)
        history.redo()
        #expect(history.deleted.selectedIndices() == [3])
    }

    @Test func deletingOnlyAlreadyDeletedRecordsNothingAndKeepsRedo() {
        var history = EditHistory(baseCount: 4)
        history.delete(SelectionMask(count: 4, indices: [1]))
        history.delete(SelectionMask(count: 4, indices: [1]))
        history.undo()
        #expect(history.deleted.isEmpty, "the repeated delete must not add an undo step")
        #expect(!history.canUndo)

        var redoHistory = EditHistory(baseCount: 4)
        redoHistory.delete(SelectionMask(count: 4, indices: [2]))
        redoHistory.undo()
        #expect(redoHistory.canRedo)
        redoHistory.delete(SelectionMask(count: 4))
        #expect(redoHistory.canRedo, "deleting an empty selection must keep the redo stack")
        redoHistory.redo()
        #expect(redoHistory.deleted.selectedIndices() == [2])
    }

    @Test func partlyNewDeletionRecordsOnlyNewBits() {
        var history = EditHistory(baseCount: 5)
        history.delete(SelectionMask(count: 5, indices: [1]))
        history.delete(SelectionMask(count: 5, indices: [1, 2]))
        #expect(history.deleted.selectedIndices() == [1, 2])
        history.undo()
        #expect(history.deleted.selectedIndices() == [1])
    }

    @Test func undoAndRedoWithEmptyStacksAreNoOps() {
        var history = EditHistory(baseCount: 3)
        history.undo()
        history.redo()
        #expect(history.deleted.isEmpty)
        #expect(history.survivorCount == 3)
    }

    @Test func baseIndicesMapDisplayIndexToBaseIndex() {
        var history = EditHistory(baseCount: 6)
        history.delete(SelectionMask(count: 6, indices: [0, 3]))
        #expect(history.baseIndices() == [1, 2, 4, 5])
        #expect(history.survivorCount == 4)
        #expect(history.baseIndices().count == history.survivorCount)
    }

    @Test func restoringFromSavedMaskHasEmptyUndoStack() {
        let saved = SelectionMask(count: 6, indices: [0, 3])
        var history = EditHistory(baseCount: 6, deleted: saved)
        #expect(history.deleted == saved)
        #expect(history.survivorCount == 4)
        #expect(history.baseIndices() == [1, 2, 4, 5])
        #expect(!history.canUndo)
        #expect(!history.canRedo)

        history.delete(SelectionMask(count: 6, indices: [5]))
        history.undo()
        #expect(history.deleted == saved, "undo must stop at the restored state")
        #expect(!history.canUndo)
    }

    @Test func resetRestoresEverythingAndClearsStacks() {
        var history = EditHistory(baseCount: 5)
        history.delete(SelectionMask(count: 5, indices: [0, 1]))
        history.delete(SelectionMask(count: 5, indices: [4]))
        history.undo()
        history.reset()
        #expect(history.deleted.isEmpty)
        #expect(history.survivorCount == 5)
        #expect(!history.canUndo)
        #expect(!history.canRedo)
    }

    @Test func undoStackIsCappedAtOneHundredSteps() {
        // 150 single-Gaussian deletions in a shuffled order; only the last 100 can be undone.
        var rng = SeededGenerator(seed: 2026)
        var order: [Int] = Array(0..<200)
        order.shuffle(using: &rng)

        var history = EditHistory(baseCount: 200)
        for index in order.prefix(150) {
            history.delete(SelectionMask(count: 200, indices: [index]))
        }
        #expect(history.deleted.selectedCount == 150)

        var undone = 0
        while history.canUndo {
            history.undo()
            undone += 1
        }
        #expect(undone == 100)
        let oldestKept = Set(order.prefix(50))
        #expect(history.deleted.selectedIndices() == oldestKept.sorted())
    }

    @Test func randomOperationsMatchSnapshotModel() {
        // Reference model: full snapshots of the deleted flags, with the same cap and redo rules.
        let count: Int = 400
        var rng = SeededGenerator(seed: 77)
        var history = EditHistory(baseCount: count)
        var current: [Bool] = [Bool](repeating: false, count: count)
        var undoSnapshots: [[Bool]] = []
        var redoSnapshots: [[Bool]] = []

        for _ in 0..<3_000 {
            let roll: Int = Int.random(in: 0..<10, using: &rng)
            if roll < 6 {
                var indices: [Int] = []
                for _ in 0..<Int.random(in: 0...6, using: &rng) {
                    indices.append(Int.random(in: 0..<count, using: &rng))
                }
                let selection = SelectionMask(count: count, indices: indices)
                let hasNew = indices.contains { !current[$0] }
                if hasNew {
                    undoSnapshots.append(current)
                    if undoSnapshots.count > 100 { undoSnapshots.removeFirst() }
                    for index in indices { current[index] = true }
                    redoSnapshots.removeAll()
                }
                history.delete(selection)
            } else if roll < 8 {
                if let previous = undoSnapshots.popLast() {
                    redoSnapshots.append(current)
                    current = previous
                }
                history.undo()
            } else {
                if let next = redoSnapshots.popLast() {
                    undoSnapshots.append(current)
                    if undoSnapshots.count > 100 { undoSnapshots.removeFirst() }
                    current = next
                }
                history.redo()
            }

            let expectedSelected: [Int] = current.indices.filter { current[$0] }
            #expect(history.deleted.selectedIndices() == expectedSelected)
            #expect(history.canUndo == !undoSnapshots.isEmpty)
            #expect(history.canRedo == !redoSnapshots.isEmpty)
            if history.deleted.selectedIndices() != expectedSelected { return }
        }
    }
}
