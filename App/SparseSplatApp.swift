import SwiftUI

@main
struct SparseSplatApp: App {
    @NSApplicationDelegateAdaptor(SparseSplatAppDelegate.self) private var appDelegate
    /// The library folder, its projects and the cover cache. Owned here so every scene and command shares them.
    @State private var folder: LibraryFolder = LibraryFolder()
    @State private var library: LibraryStore = LibraryStore()
    @State private var covers: LibraryCovers = LibraryCovers()

    init() {
        // Launching opens the library, never an untitled document. Opening a document is explicit.
        UserDefaults.standard.register(defaults: ["NSShowAppCentricOpenPanelInsteadOfUntitledFile": false])
    }

    var body: some Scene {
        // The library is the start screen. It is the only window shown at launch.
        Window("Library", id: LibraryWindow.id) {
            LibraryView(folder: folder, store: library, covers: covers)
        }
        .defaultLaunchBehavior(.presented)
        .defaultSize(width: 1040, height: 720)
        .commands {
            LibraryCommands(store: library)
        }

        // Documents open from the library, Finder or Open. Launching the app never opens them.
        DocumentGroup(newDocument: ProjectDocument()) { file in
            ProjectWindow(document: file.$document,
                          name: file.fileURL?.deletingPathExtension().lastPathComponent ?? "Scene")
        }
        .defaultLaunchBehavior(.suppressed)

        DocumentGroup(viewing: SplatDocument.self) { file in
            SceneWindow(scene: file.document.scene,
                        name: file.fileURL?.deletingPathExtension().lastPathComponent ?? "Scene")
        }
        .defaultLaunchBehavior(.suppressed)

        Settings {
            LibrarySettingsView(folder: folder)
        }
    }
}
