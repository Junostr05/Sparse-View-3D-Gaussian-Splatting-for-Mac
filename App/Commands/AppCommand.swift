import SplatCore
import SplatIO
import SwiftUI

/// One entry of the command palette. The registry builds the list from the live state of a window on every
/// evaluation, so `isEnabled` and the titles follow the model.
nonisolated struct AppCommand: CommandSearchable, Identifiable {
    let id: String
    let title: String
    /// The group shown beside the title, such as "View", "Edit" or "Project".
    let category: String
    /// The shortcut as text for the trailing edge, or nil when the command has none.
    let shortcut: String?
    let keywords: [String]
    let isEnabled: Bool
    let action: @MainActor () -> Void
}

/// Builds the palette commands from one window's models. Each command calls the same `ViewerModel` or
/// `ProjectModel` action as the matching control, so no logic is duplicated here.
enum CommandRegistry {
    /// Commands of a viewer window, with the window commands.
    static func sceneWindowCommands(viewer: ViewerModel, openWindow: OpenWindowAction) -> [AppCommand] {
        viewCommands(viewer) + editCommands(viewer) + windowCommands(openWindow)
    }

    /// Commands of a project window. The viewer commands are present once the project has a viewer.
    static func projectWindowCommands(viewer: ViewerModel?,
                                      project: ProjectModel,
                                      document: Binding<ProjectDocument>,
                                      openWindow: OpenWindowAction) -> [AppCommand] {
        var commands: [AppCommand] = []
        if let viewer {
            commands += viewCommands(viewer) + editCommands(viewer)
        }
        commands += projectCommands(project, document: document)
        commands += windowCommands(openWindow)
        return commands
    }

    // MARK: - View

    private static func viewCommands(_ viewer: ViewerModel) -> [AppCommand] {
        var commands: [AppCommand] = []
        commands.append(command("view.frameAll", "Frame All", category: "View", shortcut: "Home",
                                keywords: ["fit", "scene", "reset"]) {
            viewer.apply(.frameAll)
        })
        let presets: [(ViewPreset, String, String)] = [
            (.front, "Front View", "1"),
            (.back, "Back View", "\u{2303}1"),
            (.right, "Right View", "3"),
            (.left, "Left View", "\u{2303}3"),
            (.top, "Top View", "7"),
            (.bottom, "Bottom View", "\u{2303}7"),
        ]
        for (preset, title, shortcut) in presets {
            commands.append(command("view.preset.\(title)", title, category: "View", shortcut: shortcut,
                                    keywords: ["camera", "numpad", "axis"]) {
                viewer.apply(.preset(preset))
            })
        }
        for mode in ViewerDisplayMode.allCases {
            commands.append(command("view.display.\(mode.rawValue)", "Display Mode: \(mode.title)",
                                    category: "View", keywords: ["render", "mode", "show"]) {
                viewer.displayMode = mode
            })
        }
        commands.append(command("view.cycleDisplay", "Next Display Mode", category: "View", shortcut: "Z",
                                keywords: ["cycle", "switch"]) {
            viewer.apply(.cycleDisplayMode)
        })
        commands.append(overlayCommand("Grid", isOn: viewer.showsGrid, id: "grid") {
            viewer.showsGrid.toggle()
        })
        commands.append(overlayCommand("Axes", isOn: viewer.showsAxes, id: "axes") {
            viewer.showsAxes.toggle()
        })
        if !viewer.cameras.isEmpty {
            commands.append(overlayCommand("Cameras", isOn: viewer.showsCameras, id: "cameras") {
                viewer.showsCameras.toggle()
            })
            commands.append(overlayCommand("Camera Names", isOn: viewer.showsCameraNames, id: "cameraNames",
                                           isEnabled: viewer.showsCameras) {
                viewer.showsCameraNames.toggle()
            })
        }
        commands.append(command("view.fly", viewer.isFlying ? "Leave Fly Mode" : "Enter Fly Mode",
                                category: "View", shortcut: "\u{21E7}`", keywords: ["walk", "navigate"]) {
            viewer.toggleFlying()
        })
        commands.append(command("view.saveImage", "Save Image\u{2026}", category: "Export",
                                keywords: ["png", "picture", "screenshot"],
                                enabled: viewer.videoExport?.state != .running) {
            viewer.presentedExport = .image
        })
        commands.append(command("view.exportWeb", "Export for Web\u{2026}", category: "Export",
                                keywords: ["html", "web", "page", "share", "browser", "embed"],
                                enabled: viewer.videoExport?.state != .running) {
            viewer.exportRequest = .webPage
        })
        commands.append(command("view.exportVideo", "Export Flight Video\u{2026}", category: "Export",
                                keywords: ["movie", "mp4", "animation", "turntable"],
                                enabled: viewer.canEdit) {
            viewer.presentedExport = .video
        })
        commands.append(command("view.playFlight", viewer.isPlaying ? "Pause Flight" : "Play Flight",
                                category: "View", shortcut: "Space", keywords: ["camera", "animate", "playback"],
                                enabled: viewer.cameras.count >= 2) {
            viewer.togglePlayback()
        })
        return commands
    }

    private static func overlayCommand(_ name: String, isOn: Bool, id: String, isEnabled: Bool = true,
                                       action: @escaping @MainActor () -> Void) -> AppCommand {
        command("view.overlay.\(id)", isOn ? "Hide \(name)" : "Show \(name)", category: "View",
                keywords: ["overlay", name.lowercased()], enabled: isEnabled, action: action)
    }

    // MARK: - Edit

    /// Edit commands, present only while editing is allowed (no training preview, no video export).
    private static func editCommands(_ viewer: ViewerModel) -> [AppCommand] {
        guard viewer.canEdit else { return [] }
        return [
            command("edit.toggle", viewer.isEditing ? "Exit Edit Mode" : "Enter Edit Mode", category: "Edit",
                    keywords: ["brush", "paint", "select"]) {
                viewer.isEditing.toggle()
            },
            command("edit.selectAll", "Select All", category: "Edit", shortcut: "A",
                    keywords: ["everything"]) {
                viewer.performEdit(.selectAll)
            },
            command("edit.invert", "Invert Selection", category: "Edit", shortcut: "\u{2303}I",
                    keywords: ["flip", "complement"]) {
                viewer.performEdit(.invertSelection)
            },
            command("edit.delete", "Delete Selection", category: "Edit", shortcut: "Delete",
                    keywords: ["remove", "erase"], enabled: viewer.selectedCount > 0) {
                viewer.performEdit(.deleteSelection)
            },
            command("edit.undo", "Undo", category: "Edit", shortcut: "\u{2318}Z",
                    keywords: ["revert"], enabled: viewer.canUndo) {
                viewer.performEdit(.undo)
            },
            command("edit.redo", "Redo", category: "Edit", shortcut: "\u{21E7}\u{2318}Z",
                    keywords: ["again"], enabled: viewer.canRedo) {
                viewer.performEdit(.redo)
            },
            command("edit.cleanUp", "Clean Up Assistant\u{2026}", category: "Edit",
                    keywords: ["floaters", "artifacts", "noise", "outliers", "transparent", "huge", "needles"]) {
                viewer.presentedCleanup = true
            },
        ]
    }

    // MARK: - Project

    /// Training commands, as the project inspector offers them: Pause or Resume and Stop while a run is active,
    /// otherwise Continue Training with a stored checkpoint, and Train or Retrain from Scratch.
    private static func projectCommands(_ project: ProjectModel, document: Binding<ProjectDocument>) -> [AppCommand] {
        let contents: ProjectContents = document.wrappedValue.contents
        if project.isTraining {
            return [
                command("project.pause", project.isPaused ? "Resume Training" : "Pause Training",
                        category: "Project", keywords: ["train", "suspend"]) {
                    project.togglePause()
                },
                command("project.stop", "Stop Training", category: "Project",
                        keywords: ["cancel", "end", "train"]) {
                    project.stop()
                },
            ]
        }
        let hasImages: Bool = !contents.manifest.imageFiles.isEmpty
        let hasCheckpoint: Bool = contents.files[ProjectLayout.latestCheckpoint] != nil
        var commands: [AppCommand] = []
        commands.append(command("project.continue", "Continue Training", category: "Project",
                                keywords: ["train", "more", "rounds"],
                                enabled: project.canContinue(contents)) {
            project.continueTraining(document: document)
        })
        commands.append(command(hasCheckpoint ? "project.retrain" : "project.train",
                                hasCheckpoint ? "Retrain from Scratch" : "Train", category: "Project",
                                keywords: ["train", "reconstruct", "run", "start"],
                                enabled: hasImages) {
            project.startTraining(document: document)
        })
        return commands
    }

    // MARK: - Window

    private static func windowCommands(_ openWindow: OpenWindowAction) -> [AppCommand] {
        [
            command("window.library", "Show Library", category: "Window", keywords: ["projects", "gallery"]) {
                openWindow(id: "library")
            },
        ]
    }

    // MARK: - Construction

    private static func command(_ id: String,
                                _ title: String,
                                category: String,
                                shortcut: String? = nil,
                                keywords: [String] = [],
                                enabled: Bool = true,
                                action: @escaping @MainActor () -> Void) -> AppCommand {
        AppCommand(id: id,
                   title: title,
                   category: category,
                   shortcut: shortcut,
                   keywords: keywords,
                   isEnabled: enabled,
                   action: action)
    }
}
