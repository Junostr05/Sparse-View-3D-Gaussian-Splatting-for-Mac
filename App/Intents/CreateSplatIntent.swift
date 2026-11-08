import AppIntents
import Foundation
import SplatCore
import SplatIO
import UniformTypeIdentifiers

/// The training presets offered to Shortcuts.
enum SplatPresetOption: String, AppEnum {
    case fast
    case standard
    case quality

    static let typeDisplayRepresentation: TypeDisplayRepresentation = "Training Preset"
    static let caseDisplayRepresentations: [SplatPresetOption: DisplayRepresentation] = [
        .fast: "Fast",
        .standard: "Standard",
        .quality: "Quality",
    ]

    var preset: TrainingPreset {
        TrainingPreset(rawValue: rawValue) ?? .standard
    }
}

/// Creates a project from a folder of photos or a video, estimates its poses, and queues it or trains it at once.
struct CreateSplatIntent: AppIntent {
    static let title: LocalizedStringResource = "Create Splat"
    static let description = IntentDescription("Creates a SparseSplat project in the library from a folder of photos or a video, estimates its camera poses, and adds it to the training queue or starts training.")
    /// Runs in the background: the app does not need a window.
    static let openAppWhenRun: Bool = false

    @Parameter(title: "Photos Folder or Video", supportedContentTypes: [.folder, .movie])
    var source: IntentFile

    @Parameter(title: "Project Name")
    var name: String

    @Parameter(title: "Preset", default: .standard)
    var preset: SplatPresetOption

    @Parameter(title: "Train Immediately", default: false)
    var trainImmediately: Bool

    static var parameterSummary: some ParameterSummary {
        Summary("Create a splat named \(\.$name) from \(\.$source)") {
            \.$preset
            \.$trainImmediately
        }
    }

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        let folder: URL = try IntentWork.libraryFolder()
        let package: URL
        do {
            package = try await LibraryFiles.createProject(named: name, in: folder)
        } catch {
            throw SparseSplatIntentError(LibraryStore.describe(error))
        }
        let summary: String
        do {
            let sourceURL: URL = try IntentWork.localURL(of: source)
            let imported = try await IntentWork.frames(from: sourceURL)
            summary = try await IntentWork.estimatePoses(package: package,
                                                         frames: imported.frames,
                                                         inputKind: imported.inputKind,
                                                         preset: preset.preset)
        } catch {
            // The package was created by this intent and holds nothing usable: remove it again.
            try? FileManager.default.removeItem(at: package)
            if let intentError = error as? SparseSplatIntentError {
                throw intentError
            }
            throw SparseSplatIntentError("The project could not be created: \(error.localizedDescription)")
        }

        let queue: TrainingQueue = TrainingQueue.shared
        queue.add(package: package)
        if trainImmediately {
            queue.runQueue(modelStore: PipelineModel.makeModelStore())
        }
        let next: String = trainImmediately ? "Training has started." : "It is in the training queue."
        return .result(dialog: IntentDialog(stringLiteral: "Created \(name). \(summary). \(next)"))
    }
}
