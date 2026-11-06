import Foundation

/// Pure selection state for the import UI: which frames (video candidates or photos) go into a project.
///
/// The automatic selection comes from `FrameSelector`. Manual edits (`toggle`, `swap`) are kept until
/// `setMode` is called again, which recomputes the automatic selection and discards them.
public struct FrameSelectionPlan: Sendable, Equatable {
    public struct Item: Sendable, Equatable, Identifiable {
        /// Caller's identifier: unique, stable.
        public var id: Int
        /// Seconds in the video, `nil` for photos.
        public var time: Double?
        /// Higher = sharper. `nil` for photos.
        public var sharpness: Double?

        public init(id: Int, time: Double?, sharpness: Double?) {
            self.id = id
            self.time = time
            self.sharpness = sharpness
        }
    }

    public enum Warning: Sendable, Equatable {
        /// Fewer than `minimumRecommendedViews` selected.
        case tooFew(selected: Int)
        /// More than `slowTrainingViewCount` selected: slow and memory-heavy.
        case manyViews(selected: Int)
        case noneSelected
    }

    public static let minimumRecommendedViews: Int = 3
    public static let slowTrainingViewCount: Int = 100

    /// Items in ascending time order. Items without a time keep their input order, after the timed ones.
    public private(set) var items: [Item]
    public private(set) var selectedIDs: Set<Int>
    public private(set) var mode: FrameSelectionMode
    /// `true` after `toggle` or `swap` until the mode is set again.
    public private(set) var isManuallyEdited: Bool

    /// Applies `mode` immediately. Duplicate ids: keep the first.
    public init(items: [Item], mode: FrameSelectionMode, minimumSpacing: Double = 0) {
        var seen: Set<Int> = []
        var unique: [Item] = []
        for item in items where !seen.contains(item.id) {
            seen.insert(item.id)
            unique.append(item)
        }
        self.items = FrameSelectionPlan.sortedByTime(unique)
        self.mode = mode
        self.isManuallyEdited = false
        self.selectedIDs = FrameSelectionPlan.automaticSelection(items: self.items,
                                                                 mode: mode,
                                                                 minimumSpacing: minimumSpacing)
    }

    /// Recomputes the automatic selection and clears `isManuallyEdited`; manual edits are discarded on purpose.
    public mutating func setMode(_ mode: FrameSelectionMode, minimumSpacing: Double = 0) {
        self.mode = mode
        self.selectedIDs = FrameSelectionPlan.automaticSelection(items: items,
                                                                 mode: mode,
                                                                 minimumSpacing: minimumSpacing)
        self.isManuallyEdited = false
    }

    /// Flips one item. Unknown ids are a full no-op (the flag stays as it was). Sets `isManuallyEdited`.
    public mutating func toggle(_ id: Int) {
        guard items.contains(where: { $0.id == id }) else { return }
        if selectedIDs.contains(id) {
            selectedIDs.remove(id)
        } else {
            selectedIDs.insert(id)
        }
        isManuallyEdited = true
    }

    /// Deselects `id` and selects `replacement` in one step. Both must exist, `id` must be selected and
    /// `replacement` must not be; otherwise nothing changes and the flag stays as it was.
    public mutating func swap(_ id: Int, with replacement: Int) {
        guard selectedIDs.contains(id),
              !selectedIDs.contains(replacement),
              items.contains(where: { $0.id == replacement }) else { return }
        selectedIDs.remove(id)
        selectedIDs.insert(replacement)
        isManuallyEdited = true
    }

    /// Selected ids in item order.
    public var selectedInOrder: [Int] {
        items.compactMap { selectedIDs.contains($0.id) ? $0.id : nil }
    }

    public var selectedCount: Int {
        selectedIDs.count
    }

    /// `nil` when the selection is fine.
    public var warning: Warning? {
        let count: Int = selectedCount
        if count == 0 { return .noneSelected }
        if count < FrameSelectionPlan.minimumRecommendedViews { return .tooFew(selected: count) }
        if count > FrameSelectionPlan.slowTrainingViewCount { return .manyViews(selected: count) }
        return nil
    }

    // MARK: - Private helpers

    /// Timed items ascending by time (ties keep input order), then untimed items in input order.
    private static func sortedByTime(_ items: [Item]) -> [Item] {
        var timed: [(offset: Int, item: Item)] = []
        var untimed: [Item] = []
        for (offset, item) in items.enumerated() {
            if item.time != nil {
                timed.append((offset: offset, item: item))
            } else {
                untimed.append(item)
            }
        }
        timed.sort { a, b in
            let ta: Double = a.item.time!
            let tb: Double = b.item.time!
            if ta != tb { return ta < tb }
            return a.offset < b.offset
        }
        var result: [Item] = []
        for entry in timed {
            result.append(entry.item)
        }
        result.append(contentsOf: untimed)
        return result
    }

    /// Runs `FrameSelector` over the items. Items without a time use their position in the sorted list,
    /// so photos spread evenly; items without sharpness use 1.0.
    private static func automaticSelection(items: [Item],
                                           mode: FrameSelectionMode,
                                           minimumSpacing: Double) -> Set<Int> {
        var candidates: [FrameCandidate] = []
        for (position, item) in items.enumerated() {
            let time: Double = item.time ?? Double(position)
            let sharpness: Double = item.sharpness ?? 1.0
            candidates.append(FrameCandidate(index: item.id, time: time, sharpness: sharpness))
        }
        let chosen: [Int] = FrameSelector.select(candidates, mode: mode, minimumSpacing: minimumSpacing)
        return Set(chosen)
    }
}
