import SwiftUI

/// The View menu item "Command Palette" (Command K). It runs the action that the focused window published with
/// `commandPalette(_:)`, and is disabled when no such window has focus.
///
/// The app adds it with `.commands { PaletteCommands() }` on its scene.
struct PaletteCommands: Commands {
    @FocusedValue(\.showCommandPalette) private var showPalette: ShowCommandPaletteAction?

    var body: some Commands {
        CommandGroup(after: .toolbar) {
            Button("Command Palette\u{2026}") {
                showPalette?.perform()
            }
            .keyboardShortcut("k", modifiers: .command)
            .disabled(showPalette == nil)
        }
    }
}
