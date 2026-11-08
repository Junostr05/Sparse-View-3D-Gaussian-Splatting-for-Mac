import Foundation
import SplatCore
import SplatIO
import SplatTraining

/// Trains one project package without a window: the queue's path through the app. It reads the package from disk,
/// runs `HeadlessTraining` (the same core as the `sparsesplat` command-line tool) and writes the result back with
/// `ProjectPackageFiles.write`, which replaces the package atomically. Blocking work runs off the main actor.
nonisolated enum ProjectTrainingRunner {
    @concurrent
    static func run(package: URL,
                    mode: HeadlessTrainingMode,
                    preset: TrainingPreset?,
                    modelStore: ModelStore?,
                    onProgress: @escaping @Sendable (TrainingProgress) -> Void) async throws -> HeadlessTrainingResult {
        let contents: ProjectContents = try ProjectPackageFiles.read(package: package)
        let options: HeadlessTrainingOptions = HeadlessTrainingOptions(mode: mode,
                                                                       preset: preset,
                                                                       modelStore: modelStore)
        let result: HeadlessTrainingResult = try await HeadlessTraining.run(contents: contents,
                                                                            options: options,
                                                                            onProgress: onProgress)
        // A failed or unfinished run never reaches this line, so the package is left as it was.
        try ProjectPackageFiles.write(result.contents, to: package)
        return result
    }
}
