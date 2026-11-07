/// Non-destructive deletion of Gaussians from a fixed base cloud, with undo and redo.
///
/// All selections are in base indices. Each undo step stores only the Gaussians that step newly
/// deleted, so undoing a step restores exactly the state before it.
public struct EditHistory: Sendable, Equatable {
    /// Maximum number of undo steps kept; the oldest step is dropped beyond this.
    public static let maximumUndoSteps: Int = 100

    public let baseCount: Int
    /// Cumulative deletions, in base indices.
    public private(set) var deleted: SelectionMask
    /// Newly deleted Gaussians of each undoable step, oldest first.
    private var undoSteps: [SelectionMask]
    /// Steps that were undone and can be redone, most recently undone last.
    private var redoSteps: [SelectionMask]

    /// Fresh history over a base cloud of `baseCount` Gaussians with nothing deleted.
    public init(baseCount: Int) {
        precondition(baseCount >= 0, "EditHistory baseCount must be non-negative, got \(baseCount)")
        self.baseCount = baseCount
        self.deleted = SelectionMask(count: baseCount)
        self.undoSteps = []
        self.redoSteps = []
    }

    /// Restores a saved project: `deleted` is the cumulative deletion. The undo and redo stacks
    /// start empty, so the restored deletions cannot be undone.
    public init(baseCount: Int, deleted: SelectionMask) {
        precondition(baseCount >= 0, "EditHistory baseCount must be non-negative, got \(baseCount)")
        precondition(deleted.count == baseCount,
                     "EditHistory deleted mask count \(deleted.count) differs from baseCount \(baseCount)")
        self.baseCount = baseCount
        self.deleted = deleted
        self.undoSteps = []
        self.redoSteps = []
    }

    public var canUndo: Bool { !undoSteps.isEmpty }

    public var canRedo: Bool { !redoSteps.isEmpty }

    /// Number of Gaussians that are not deleted.
    public var survivorCount: Int {
        baseCount - deleted.selectedCount
    }

    /// Base indices of the surviving Gaussians, ascending. Element `i` is the base index of the
    /// survivor displayed at index `i`.
    public func baseIndices() -> [Int] {
        deleted.unselectedIndices()
    }

    /// Deletes the Gaussians in `selection` (base indices). Only Gaussians not already deleted are
    /// recorded, as one undo step. If nothing new is deleted, nothing is recorded and the redo
    /// stack is kept. A real deletion clears the redo stack.
    public mutating func delete(_ selection: SelectionMask) {
        precondition(selection.count == baseCount,
                     "EditHistory selection count \(selection.count) differs from baseCount \(baseCount)")
        var newlyDeleted: SelectionMask = selection
        newlyDeleted.formSubtraction(deleted)
        guard !newlyDeleted.isEmpty else { return }

        deleted.formUnion(newlyDeleted)
        appendUndoStep(newlyDeleted)
        redoSteps.removeAll()
    }

    /// Restores the Gaussians deleted by the most recent step. No-op when nothing can be undone.
    public mutating func undo() {
        guard let step = undoSteps.popLast() else { return }
        deleted.formSubtraction(step)
        redoSteps.append(step)
    }

    /// Re-applies the most recently undone step. No-op when nothing can be redone.
    public mutating func redo() {
        guard let step = redoSteps.popLast() else { return }
        deleted.formUnion(step)
        appendUndoStep(step)
    }

    /// Restores every Gaussian and clears both stacks.
    public mutating func reset() {
        deleted = SelectionMask(count: baseCount)
        undoSteps.removeAll()
        redoSteps.removeAll()
    }

    private mutating func appendUndoStep(_ step: SelectionMask) {
        undoSteps.append(step)
        if undoSteps.count > Self.maximumUndoSteps {
            undoSteps.removeFirst(undoSteps.count - Self.maximumUndoSteps)
        }
    }
}
