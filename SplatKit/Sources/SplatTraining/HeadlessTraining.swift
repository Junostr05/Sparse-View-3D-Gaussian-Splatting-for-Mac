import Foundation
import SplatCore
import SplatDepth
import SplatIO

/// How a headless run trains: from scratch on the original poses, or one more round from the stored checkpoint.
public enum HeadlessTrainingMode: String, Sendable, Equatable {
    case fromScratch
    case continueTraining
}

/// Options of one headless run. Used by the app's training queue and by the `sparsesplat` command-line tool.
public struct HeadlessTrainingOptions: Sendable {
    public var mode: HeadlessTrainingMode
    /// Replaces the project's training preset (image size, iterations and schedules) before the run. The
    /// settings are written back into the package with the checkpoint. `nil` keeps the stored settings.
    public var preset: TrainingPreset?
    /// Iterations of a run from scratch, or of the single round of a continuation. `nil` keeps the stored count
    /// (from scratch) or `continuation.roundIterations` (continuation).
    public var iterations: Int?
    /// Growth and round settings of a continuation. Its `isEndless` flag is ignored: a headless run is one round.
    public var continuation: ContinuationSettings
    /// Store that holds the monocular depth model. `nil` disables model inference (supplied and cached maps still count).
    public var modelStore: ModelStore?

    public init(mode: HeadlessTrainingMode = .fromScratch,
                preset: TrainingPreset? = nil,
                iterations: Int? = nil,
                continuation: ContinuationSettings = ContinuationSettings(),
                modelStore: ModelStore? = nil) {
        self.mode = mode
        self.preset = preset
        self.iterations = iterations
        self.continuation = continuation
        self.modelStore = modelStore
    }
}

/// The outcome of a headless run. `contents` already holds the stored checkpoint when `state` is `finished` or `cancelled`.
public struct HeadlessTrainingResult: Sendable {
    public var contents: ProjectContents
    /// `.finished`, or `.cancelled` when the run was stopped (its last completed model is still stored).
    public var state: TrainingProgress.State
    /// Cumulative iterations: the stored checkpoint's iterations plus this run's completed iterations.
    public var iteration: Int
    public var gaussianCount: Int
    public var psnr: Double?
    /// Number of checkpoints in the package after the run.
    public var checkpointCount: Int
    /// Non-fatal depth problems of the run, such as a depth model that is not installed. Empty when there were none.
    public var warnings: [String]
}

public enum HeadlessTrainingError: Error, Equatable, Sendable, CustomStringConvertible {
    /// The package has no images to train on.
    case noImages
    /// A continuation was requested, but the package has no stored checkpoint.
    case noCheckpoint
    /// The stored checkpoint could not be read or the session failed. The message is for display.
    case failed(String)

    public var description: String {
        switch self {
        case .noImages:
            return "the project has no images"
        case .noCheckpoint:
            return "the project has no checkpoint to continue from"
        case .failed(let message):
            return message
        }
    }
}

/// Headless training of a `.gsproj` package: the work `ProjectModel` does for a window, without a document or a
/// viewer. The caller loads the package contents and stores the result (see `ProjectPackageIO`).
public enum HeadlessTraining {
    /// Runs one training session on `contents` and returns the updated contents. Cancelling the calling task stops
    /// the run; a cancelled run still stores its last completed model. Every session event is passed to `onProgress`.
    public static func run(contents input: ProjectContents,
                           options: HeadlessTrainingOptions,
                           onProgress: @escaping @Sendable (TrainingProgress) -> Void = { _ in }) async throws(HeadlessTrainingError) -> HeadlessTrainingResult {
        guard !input.manifest.imageFiles.isEmpty else {
            throw .noImages
        }
        var contents: ProjectContents = input
        let isContinuation: Bool = options.mode == .continueTraining
        if isContinuation {
            guard contents.files[ProjectLayout.latestCheckpoint] != nil else {
                throw .noCheckpoint
            }
        } else {
            // A run from scratch trains on the original poses, as `ProjectModel.startTraining` does.
            contents = ProjectBundle.withoutRefinedPoses(contents)
        }
        if let preset = options.preset {
            preset.apply(to: &contents.manifest.training)
        }
        if !isContinuation, let iterations = options.iterations {
            contents.manifest.training.iterations = max(1, iterations)
        }

        let depth: DepthPreparedRun
        do {
            depth = try await DepthPipeline.prepare(contents: contents,
                                                    store: options.modelStore,
                                                    useRefinedPoses: isContinuation)
        } catch {
            throw .failed("Depth preparation failed: \(String(describing: error))")
        }
        if !depth.estimatedMaps.isEmpty {
            // Cache the inferred maps under their estimator, as the app does.
            contents = ProjectBundle.withEstimatedDepthMaps(contents, depth.estimatedMaps, estimator: depth.estimator)
        }
        let viewNames: [String] = depth.dataset.views.map(\.name)

        let session: TrainingSession
        var offset: Int = 0
        if isContinuation {
            let start: GaussianCloud
            do {
                start = try continuationStart(contents)
            } catch {
                throw .failed("The stored checkpoint could not be read: \(String(describing: error))")
            }
            offset = contents.manifest.checkpoints.last?.iteration ?? 0
            var settings: ContinuationSettings = options.continuation
            settings.isEndless = false
            if let iterations = options.iterations {
                settings.roundIterations = max(1, iterations)
            }
            let roundConfiguration: TrainingConfiguration = depth.configuration.continued(
                from: start.count,
                shDegree: start.shDegree,
                round: 0,
                settings: settings
            )
            session = TrainingSession(dataset: depth.dataset, initialCloud: start, configuration: roundConfiguration)
        } else {
            session = TrainingSession(dataset: depth.dataset,
                                      configuration: depth.configuration,
                                      initializer: depth.initializer)
        }

        let outcome: (terminal: TrainingProgress?, result: TrainingOutcome?, heldOutView: String?) = await consume(
            session, onProgress: onProgress
        )
        guard let terminal = outcome.terminal else {
            throw .failed("The training session ended without a result.")
        }
        switch terminal.state {
        case .failed(let message):
            throw .failed(message)
        case .running, .paused:
            throw .failed("The training session ended while still running.")
        case .finished, .cancelled:
            break
        }
        guard let cloud = terminal.preview else {
            throw .failed("The training session produced no model.")
        }
        let stored: StoredRun = storingRun(into: contents,
                                           cloud: cloud,
                                           event: terminal,
                                           iterationOffset: offset,
                                           heldOutView: outcome.heldOutView,
                                           outcome: outcome.result,
                                           viewNames: viewNames,
                                           date: Date())
        return HeadlessTrainingResult(contents: stored.contents,
                                      state: terminal.state,
                                      iteration: offset + terminal.iteration,
                                      gaussianCount: cloud.count,
                                      psnr: terminal.psnr,
                                      checkpointCount: stored.contents.manifest.checkpoints.count,
                                      warnings: depth.warnings)
    }

    /// Runs `session` with its progress forwarded to `onProgress`, and returns the last terminal event, the outcome
    /// of the run and the held-out view name (leave-one-out only). The progress stream ends with the run.
    private static func consume(_ session: TrainingSession,
                                onProgress: @escaping @Sendable (TrainingProgress) -> Void) async -> (terminal: TrainingProgress?, result: TrainingOutcome?, heldOutView: String?) {
        // A child task: cancelling the caller cancels the run, and the run's stream ends with it.
        async let runOutcome: TrainingOutcome? = {
            try? await session.runWithOutcome()
        }()
        var terminal: TrainingProgress?
        var heldOutView: String?
        for await event in session.progress {
            if case .selection(let name) = event.phase {
                heldOutView = name
            }
            onProgress(event)
            switch event.state {
            case .finished, .cancelled, .failed:
                terminal = event
            case .running, .paused:
                break
            }
        }
        let result: TrainingOutcome? = await runOutcome
        return (terminal, result, heldOutView)
    }

    // MARK: - Shared package logic

    /// The model a continuation starts from: the stored latest checkpoint without its stored deletions. The deletions
    /// only apply while they belong to that checkpoint.
    public static func continuationStart(_ contents: ProjectContents) throws -> GaussianCloud {
        guard let cloud = try ProjectBundle.checkpointWithoutDeletions(contents) else {
            throw ProjectError.corruptCheckpoint("the latest checkpoint is missing")
        }
        return cloud
    }

    /// The package contents after a run: the checkpoint, its record, and the refined poses when the run produced one
    /// for each dataset view. Also returns those poses by name, so a viewer can show the cameras the model used.
    public struct StoredRun: Sendable {
        public var contents: ProjectContents
        /// The refined cameras stored with the checkpoint, or `nil` when the run refined none.
        public var refinedCameras: [NamedCamera]?
    }

    /// Stores the checkpoint of a finished or cancelled run. `iterationOffset` is the iteration count before the run
    /// (zero for a run from scratch). `heldOutView` names the leave-one-out view, if the run had one.
    public static func storingRun(into contents: ProjectContents,
                                  cloud: GaussianCloud,
                                  event: TrainingProgress,
                                  iterationOffset: Int,
                                  heldOutView: String?,
                                  outcome: TrainingOutcome?,
                                  viewNames: [String],
                                  date: Date) -> StoredRun {
        // Cumulative over the stored checkpoint and the finished rounds; equal to `event.iteration` for a run from scratch.
        let record = CheckpointRecord(iteration: iterationOffset + event.iteration,
                                      gaussianCount: cloud.count,
                                      psnr: event.psnr,
                                      date: date,
                                      heldOutView: heldOutView,
                                      heldOutPSNR: event.heldOut?.psnr,
                                      heldOutSSIM: event.heldOut?.ssim,
                                      bestIteration: event.bestIteration)
        // The optimizer state of the run is stored with its checkpoint, so the next continuation resumes exactly.
        var result: ProjectContents = ProjectBundle.withCheckpoint(contents, cloud: cloud, record: record,
                                                                  optimizer: outcome?.optimizerState)
        var named: [NamedCamera]?
        if let refined = outcome?.refinedCameras, refined.count == viewNames.count {
            let cameras: [NamedCamera] = zip(viewNames, refined).map { NamedCamera(name: $0.0, camera: $0.1) }
            result = ProjectBundle.withRefinedPoses(result, cameras: cameras)
            named = cameras
        }
        return StoredRun(contents: result, refinedCameras: named)
    }
}
