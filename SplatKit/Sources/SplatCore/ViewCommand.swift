/// A viewer action triggered by a key. The app maps its own keys (Home, arrows, etc.) itself.
public enum ViewCommand: Sendable, Equatable {
    case preset(ViewPreset)
    /// Orbit in units of 15°; the caller applies ±15° per unit.
    case stepOrbit(yaw: Int, pitch: Int)
    /// Pan in units of one step; the caller converts these to pixels.
    case stepPan(x: Int, y: Int)
    /// Blender numpad 0: the camera view.
    case cameraView
    case previousCamera
    case nextCamera
    /// Blender Home / numpad period: frame the whole scene.
    case frameAll
    case togglePlayback

    /// Maps a key to a command. `character` is the lowercase character of the key.
    ///
    /// Digits from the top row and the keypad are treated the same. `keypad` is informational
    /// and ignored. `shift` is ignored. `control` selects the pan or the back and left variants
    /// where they exist. For keys without a control variant, `control` is ignored. Returns nil
    /// for unmapped keys.
    public static func command(character: Character, control: Bool, shift: Bool, keypad: Bool) -> ViewCommand? {
        switch character {
        case "1": return control ? .preset(.back) : .preset(.front)
        case "3": return control ? .preset(.left) : .preset(.right)
        case "7": return control ? .preset(.bottom) : .preset(.top)
        case "4": return control ? .stepPan(x: -1, y: 0) : .stepOrbit(yaw: -1, pitch: 0)
        case "6": return control ? .stepPan(x: 1, y: 0) : .stepOrbit(yaw: 1, pitch: 0)
        case "8": return control ? .stepPan(x: 0, y: -1) : .stepOrbit(yaw: 0, pitch: -1)
        case "2": return control ? .stepPan(x: 0, y: 1) : .stepOrbit(yaw: 0, pitch: 1)
        case "0": return .cameraView
        case "[": return .previousCamera
        case "]": return .nextCamera
        case ".": return .frameAll
        case " ": return .togglePlayback
        default: return nil
        }
    }
}
