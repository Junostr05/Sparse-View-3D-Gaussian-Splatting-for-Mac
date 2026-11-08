import AppIntents

/// The shortcuts and Spotlight phrases of SparseSplat. Every phrase is English.
struct SparseSplatShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(intent: CreateSplatIntent(),
                    phrases: [
                        "Create a splat in \(.applicationName)",
                        "Make a 3D scene in \(.applicationName)",
                    ],
                    shortTitle: "Create Splat",
                    systemImageName: "cube.transparent")
        AppShortcut(intent: AddToQueueIntent(),
                    phrases: [
                        "Add a project to the training queue in \(.applicationName)",
                    ],
                    shortTitle: "Add to Queue",
                    systemImageName: "list.number")
        AppShortcut(intent: ExportSplatIntent(),
                    phrases: [
                        "Export a splat from \(.applicationName)",
                    ],
                    shortTitle: "Export Splat",
                    systemImageName: "square.and.arrow.up")
    }
}
