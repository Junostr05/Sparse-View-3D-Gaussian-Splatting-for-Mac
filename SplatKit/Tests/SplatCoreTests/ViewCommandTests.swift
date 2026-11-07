import Testing
import SplatCore

@Suite struct ViewCommandTests {
    private func map(_ character: Character, control: Bool = false, shift: Bool = false,
                     keypad: Bool = false) -> ViewCommand? {
        ViewCommand.command(character: character, control: control, shift: shift, keypad: keypad)
    }

    @Test func numpadViewsWithAndWithoutControl() {
        #expect(map("1") == .preset(.front))
        #expect(map("1", control: true) == .preset(.back))
        #expect(map("3") == .preset(.right))
        #expect(map("3", control: true) == .preset(.left))
        #expect(map("7") == .preset(.top))
        #expect(map("7", control: true) == .preset(.bottom))
    }

    @Test func shiftIsIgnoredForViews() {
        #expect(map("1", shift: true) == .preset(.front))
        #expect(map("1", control: true, shift: true) == .preset(.back))
        #expect(map("7", shift: true) == .preset(.top))
    }

    @Test func arrowKeysOrbitWithoutControlAndPanWithControl() {
        #expect(map("4") == .stepOrbit(yaw: -1, pitch: 0))
        #expect(map("6") == .stepOrbit(yaw: 1, pitch: 0))
        #expect(map("8") == .stepOrbit(yaw: 0, pitch: -1))
        #expect(map("2") == .stepOrbit(yaw: 0, pitch: 1))

        #expect(map("4", control: true) == .stepPan(x: -1, y: 0))
        #expect(map("6", control: true) == .stepPan(x: 1, y: 0))
        #expect(map("8", control: true) == .stepPan(x: 0, y: -1))
        #expect(map("2", control: true) == .stepPan(x: 0, y: 1))
    }

    @Test func keypadAndTopRowAreTheSame() {
        for character: Character in ["1", "2", "3", "4", "6", "7", "8"] {
            #expect(map(character, keypad: true) == map(character, keypad: false))
        }
    }

    @Test func otherCommands() {
        #expect(map("0") == .cameraView)
        #expect(map("[") == .previousCamera)
        #expect(map("]") == .nextCamera)
        #expect(map(".") == .frameAll)
        #expect(map(" ") == .togglePlayback)
    }

    @Test func unmappedKeysReturnNil() {
        for character: Character in ["5", "9", "a", "x", "-", "+", "/", "*", "\n", "\t", "`"] {
            #expect(map(character) == nil, "\(character) should be unmapped")
            #expect(map(character, control: true) == nil, "ctrl+\(character) should be unmapped")
        }
    }
}
