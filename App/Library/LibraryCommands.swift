import AppKit
import SwiftUI

/// Identifier of the library window scene.
enum LibraryWindow {
    static let id: String = "library"
}

/// Menu commands of the library: New Project replaces the document New, and Show Library sits in the Window menu.
/// Open… stays with the document scenes, which provide it for projects and scenes alike.
struct LibraryCommands: Commands {
    let store: LibraryStore

    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("New Project\u{2026}") {
                store.isNewProjectRequested = true
                openWindow(id: LibraryWindow.id)
            }
            .keyboardShortcut("n", modifiers: .command)
        }
        CommandGroup(before: .windowList) {
            Button("Show Library") {
                openWindow(id: LibraryWindow.id)
            }
            .keyboardShortcut("l", modifiers: [.command, .shift])
        }
    }
}

/// Closing the last project window keeps the app running: the library stays available.
final class SparseSplatAppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }
}
