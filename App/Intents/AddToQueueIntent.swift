import AppIntents
import Foundation

/// Adds a project of the library to the training queue.
struct AddToQueueIntent: AppIntent {
    static let title: LocalizedStringResource = "Add Project to Training Queue"
    static let description = IntentDescription("Adds a project of the library to the SparseSplat training queue. The queue trains it with the rest, in order.")
    static let openAppWhenRun: Bool = false

    @Parameter(title: "Project")
    var project: ProjectEntity

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        guard FileManager.default.fileExists(atPath: project.id) else {
            throw SparseSplatIntentError("\(project.name) is no longer in the library.")
        }
        let queue: TrainingQueue = TrainingQueue.shared
        guard queue.add(package: project.url) else {
            let reason: String = queue.message ?? "\(project.name) could not be queued."
            queue.message = nil
            throw SparseSplatIntentError(reason)
        }
        return .result(dialog: IntentDialog(stringLiteral: "Added \(project.name) to the training queue."))
    }
}
