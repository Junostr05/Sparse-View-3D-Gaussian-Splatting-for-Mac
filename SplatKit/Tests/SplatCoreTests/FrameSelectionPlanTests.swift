import Testing
import SplatCore

// MARK: - Helpers

/// Video-style items: id == position, time == id seconds, sharpness varies deterministically.
private func videoItems(count: Int) -> [FrameSelectionPlan.Item] {
    var items: [FrameSelectionPlan.Item] = []
    for i in 0..<count {
        let time: Double = Double(i)
        let sharpness: Double = Double((i * 37) % 11) / 10.0
        items.append(FrameSelectionPlan.Item(id: i, time: time, sharpness: sharpness))
    }
    return items
}

/// Photo-style items: no time and no sharpness.
private func photoItems(count: Int) -> [FrameSelectionPlan.Item] {
    var items: [FrameSelectionPlan.Item] = []
    for i in 0..<count {
        items.append(FrameSelectionPlan.Item(id: i, time: nil, sharpness: nil))
    }
    return items
}

// MARK: - Automatic selection

@Test func defaultModePicksSixSpreadFramesFromVideo() {
    let plan: FrameSelectionPlan = FrameSelectionPlan(items: videoItems(count: 20),
                                                      mode: .count(FrameSelectionMode.defaultCount))
    let selected: [Int] = plan.selectedInOrder

    #expect(selected.count == 6)
    #expect(plan.selectedCount == 6)
    #expect(selected == selected.sorted())
    #expect(Set(selected).count == selected.count)
    // Span 0...19 s split into 6 bins: the first bin starts at id 0, the last bin ends at id 19.
    #expect(selected.min()! <= 3)
    #expect(selected.max()! >= 16)
}

@Test func allModeSelectsEveryItem() {
    let plan: FrameSelectionPlan = FrameSelectionPlan(items: videoItems(count: 8), mode: .all)
    #expect(plan.selectedInOrder == [0, 1, 2, 3, 4, 5, 6, 7])
    #expect(plan.selectedCount == 8)
}

@Test func photosWithoutTimeAreSpreadEvenly() {
    let plan: FrameSelectionPlan = FrameSelectionPlan(items: photoItems(count: 9), mode: .count(3))
    let selected: [Int] = plan.selectedInOrder

    #expect(selected.count == 3)
    // No two adjacent photos are picked.
    for i in 1..<selected.count {
        #expect(selected[i] - selected[i - 1] >= 2)
    }
    // One pick in the first third (ids 0...2) and one in the last third (ids 6...8).
    #expect(selected.filter { $0 <= 2 }.count == 1)
    #expect(selected.filter { $0 >= 6 }.count == 1)
}

@Test func sharperFrameWinsWithinBin() {
    // Times 0...5, count 2 -> bins [0,2.5) holding ids 0...2 and [2.5,5] holding ids 3...5.
    let sharpness: [Double] = [0.1, 0.9, 0.5, 0.2, 0.8, 0.3]
    var items: [FrameSelectionPlan.Item] = []
    for i in 0..<6 {
        items.append(FrameSelectionPlan.Item(id: i, time: Double(i), sharpness: sharpness[i]))
    }
    let plan: FrameSelectionPlan = FrameSelectionPlan(items: items, mode: .count(2))
    #expect(plan.selectedInOrder == [1, 4])
}

// MARK: - Manual edits

@Test func toggleFlipsSelectionAndMarksManual() {
    var plan: FrameSelectionPlan = FrameSelectionPlan(items: videoItems(count: 6), mode: .count(2))
    #expect(plan.isManuallyEdited == false)

    let selectedBefore: Set<Int> = plan.selectedIDs
    let unselected: Int = (0..<6).first { !selectedBefore.contains($0) }!
    let selected: Int = selectedBefore.first!

    plan.toggle(unselected)
    #expect(plan.selectedIDs.contains(unselected))
    #expect(plan.isManuallyEdited == true)

    plan.toggle(selected)
    #expect(plan.selectedIDs.contains(selected) == false)
    #expect(plan.isManuallyEdited == true)
}

@Test func toggleUnknownIdIsNoOp() {
    var plan: FrameSelectionPlan = FrameSelectionPlan(items: videoItems(count: 6), mode: .count(2))
    let before: FrameSelectionPlan = plan

    plan.toggle(999)
    #expect(plan == before)
    #expect(plan.isManuallyEdited == false)

    plan.toggle(0)
    plan.toggle(999)
    #expect(plan.isManuallyEdited == true)
}

@Test func setModeDiscardsManualEdits() {
    let fresh: FrameSelectionPlan = FrameSelectionPlan(items: videoItems(count: 6), mode: .count(2))
    var plan: FrameSelectionPlan = fresh

    let unselected: Int = (0..<6).first { !fresh.selectedIDs.contains($0) }!
    plan.toggle(unselected)
    #expect(plan.isManuallyEdited == true)
    #expect(plan.selectedIDs != fresh.selectedIDs)

    plan.setMode(.count(2))
    #expect(plan.isManuallyEdited == false)
    #expect(plan.selectedIDs == fresh.selectedIDs)
    #expect(plan.mode == .count(2))

    plan.setMode(.all)
    #expect(plan.mode == .all)
    #expect(plan.selectedCount == 6)
    #expect(plan.isManuallyEdited == false)
}

@Test func validSwapReplacesSelectedFrame() {
    var plan: FrameSelectionPlan = FrameSelectionPlan(items: videoItems(count: 6), mode: .count(2))
    let selectedBefore: Set<Int> = plan.selectedIDs
    let id: Int = selectedBefore.sorted().first!
    let replacement: Int = (0..<6).first { !selectedBefore.contains($0) }!

    plan.swap(id, with: replacement)

    var expected: Set<Int> = selectedBefore
    expected.remove(id)
    expected.insert(replacement)
    #expect(plan.selectedIDs == expected)
    #expect(plan.selectedCount == 2)
    #expect(plan.isManuallyEdited == true)
}

@Test func invalidSwapLeavesStateUnchanged() {
    let fresh: FrameSelectionPlan = FrameSelectionPlan(items: videoItems(count: 6), mode: .count(2))
    var plan: FrameSelectionPlan = fresh

    let selectedIDs: [Int] = fresh.selectedIDs.sorted()
    let unselected: Int = (0..<6).first { !fresh.selectedIDs.contains($0) }!

    // `id` not selected.
    plan.swap(unselected, with: selectedIDs[0])
    #expect(plan == fresh)

    // `replacement` already selected.
    plan.swap(selectedIDs[0], with: selectedIDs[1])
    #expect(plan == fresh)

    // Unknown replacement.
    plan.swap(selectedIDs[0], with: 999)
    #expect(plan == fresh)
    #expect(plan.isManuallyEdited == false)
}

// MARK: - Warnings

@Test func warningsFollowSelectedCount() {
    // Zero selected: noneSelected (also covers count <= 0).
    let none: FrameSelectionPlan = FrameSelectionPlan(items: photoItems(count: 9), mode: .count(0))
    #expect(none.warning == .noneSelected)

    // Two selected: below the recommended minimum of 3.
    let two: FrameSelectionPlan = FrameSelectionPlan(items: photoItems(count: 2), mode: .all)
    #expect(two.warning == .tooFew(selected: 2))

    let three: FrameSelectionPlan = FrameSelectionPlan(items: photoItems(count: 3), mode: .all)
    #expect(three.warning == nil)

    let hundred: FrameSelectionPlan = FrameSelectionPlan(items: photoItems(count: 100), mode: .all)
    #expect(hundred.warning == nil)

    let hundredOne: FrameSelectionPlan = FrameSelectionPlan(items: photoItems(count: 101), mode: .all)
    #expect(hundredOne.warning == .manyViews(selected: 101))
}

@Test func emptyItemsWarnNoneSelected() {
    let plan: FrameSelectionPlan = FrameSelectionPlan(items: [], mode: .all)
    #expect(plan.selectedInOrder.isEmpty)
    #expect(plan.warning == .noneSelected)
}

// MARK: - Input normalisation

@Test func duplicateIdsKeepFirst() {
    let items: [FrameSelectionPlan.Item] = [
        FrameSelectionPlan.Item(id: 1, time: 0.0, sharpness: 0.2),
        FrameSelectionPlan.Item(id: 1, time: 5.0, sharpness: 0.9),
        FrameSelectionPlan.Item(id: 2, time: 3.0, sharpness: 0.5),
    ]
    let plan: FrameSelectionPlan = FrameSelectionPlan(items: items, mode: .all)

    #expect(plan.items.count == 2)
    #expect(plan.items.map { $0.id } == [1, 2])
    #expect(plan.items[0].time == 0.0)
    #expect(plan.items[0].sharpness == 0.2)
}

@Test func outOfOrderItemsAreSortedByTime() {
    let items: [FrameSelectionPlan.Item] = [
        FrameSelectionPlan.Item(id: 10, time: 5.0, sharpness: 0.5),
        FrameSelectionPlan.Item(id: 11, time: 1.0, sharpness: 0.5),
        FrameSelectionPlan.Item(id: 12, time: 3.0, sharpness: 0.5),
    ]
    let plan: FrameSelectionPlan = FrameSelectionPlan(items: items, mode: .all)

    #expect(plan.items.map { $0.id } == [11, 12, 10])
    #expect(plan.selectedInOrder == [11, 12, 10])
}

@Test func minimumSpacingIsHonouredByAutomaticSelection() {
    // Times 0...6, one bin per item. Spacing 2.5 keeps 0, 3 and 6 only.
    var items: [FrameSelectionPlan.Item] = []
    for i in 0..<7 {
        items.append(FrameSelectionPlan.Item(id: i, time: Double(i), sharpness: 1.0))
    }
    let plan: FrameSelectionPlan = FrameSelectionPlan(items: items, mode: .all, minimumSpacing: 2.5)
    #expect(plan.selectedInOrder == [0, 3, 6])
}

@Test func isDeterministicForEqualInput() {
    let first: FrameSelectionPlan = FrameSelectionPlan(items: videoItems(count: 20),
                                                       mode: .count(6),
                                                       minimumSpacing: 1.0)
    let second: FrameSelectionPlan = FrameSelectionPlan(items: videoItems(count: 20),
                                                        mode: .count(6),
                                                        minimumSpacing: 1.0)
    #expect(first == second)
    #expect(first.selectedInOrder == second.selectedInOrder)
}
