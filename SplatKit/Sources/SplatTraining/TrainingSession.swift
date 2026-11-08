import os
import SplatCore
import SplatIO
import SplatMetal

// Headless training run (Phase 3 Task 15, leave-one-out Phase 4 Task 19). The actor owns the Metal
// context and the GPUTrainer of the current run; no Metal object crosses an isolation boundary.
// Progress leaves the actor only as `TrainingProgress` values (Sendable, previews are snapshots).
//
// Interpretations, where the plan is literal:
// - A session runs once. A second `run()`, concurrent or after the first has finished, throws
//   `.alreadyRunning`, because the progress stream is finished when the first run ends.
// - Pause is a 50 ms sleep loop checked between iterations. A step in progress is never interrupted.
// - Cancellation is checked between iterations; densification is atomic with respect to it. A
//   cancelled run returns the snapshot at the last completed iteration.
// - `.failed` carries the last good preview, so a caller can still show the last usable model.
// - Leave-one-out (P11) runs only when early stopping is enabled, has an evaluation interval, and the
//   dataset has at least `minimumViews` views. Otherwise the run is `.single`, exactly the Phase 3 loop.
// - Run 1 (`.selection`) trains without the held-out view; the initializer sees the reduced dataset.
//   Its trainer is released before run 2 starts. Run 2 (`.final`) starts from `initializer(full dataset)`
//   with a fresh trainer and the loop limit `finalIterations(viewCount:)`. Its trainer keeps the nominal
//   configuration, so the learning-rate and densification schedules are unchanged.
// - The stream is finished only by the terminal event of the last run (or by a cancelled run 1).
//   Run 1 hands over to run 2 without a terminal event.

/// Result of a `runWithOutcome()`. The held-out fields are nil for a single run, and `heldOutView` is
/// also nil for a run that stopped without a held-out view.
public struct TrainingOutcome: Sendable {
    /// Final cloud, or the cloud at the last completed iteration when cancelled.
    public var cloud: GaussianCloud
    /// The held-out view of leave-one-out, or nil for a single run.
    public var heldOutView: String?
    /// Best held-out iteration (leave-one-out only).
    public var bestIteration: Int?
    /// Best held-out metrics (leave-one-out only).
    public var heldOut: HeldOutMetrics?
    /// Completed iterations of the run that produced `cloud`.
    public var iterations: Int
    /// Training cameras (dataset view order) with the learned pose corrections applied, from the run that produced
    /// `cloud`. Nil when pose refinement is off.
    public var refinedCameras: [Camera]?
    /// Optimizer state at the end of the run that produced `cloud` (v1.5). Nil for a leave-one-out selection run that
    /// was cancelled, which stores no state.
    public var optimizerState: OptimizerState?

    public init(cloud: GaussianCloud, heldOutView: String? = nil, bestIteration: Int? = nil,
                heldOut: HeldOutMetrics? = nil, iterations: Int, refinedCameras: [Camera]? = nil,
                optimizerState: OptimizerState? = nil) {
        self.cloud = cloud
        self.heldOutView = heldOutView
        self.bestIteration = bestIteration
        self.heldOut = heldOut
        self.iterations = iterations
        self.refinedCameras = refinedCameras
        self.optimizerState = optimizerState
    }
}

/// A snapshot of an interrupted run, handed to an `AutosaveHandler` every `configuration.autosaveInterval` iterations.
public struct TrainingAutosave: Sendable {
    /// Completed iterations of the session's trainer (the round's own count, not the project's cumulative one).
    public var iteration: Int
    /// The cloud at that iteration, as `GPUTrainer.snapshot()` returns it.
    public var cloud: GaussianCloud
    /// The optimizer state at that iteration. Its Adam step is cumulative over the lineage.
    public var optimizerState: OptimizerState

    public init(iteration: Int, cloud: GaussianCloud, optimizerState: OptimizerState) {
        self.iteration = iteration
        self.cloud = cloud
        self.optimizerState = optimizerState
    }
}

/// Receives autosaves. Called on the session's actor, synchronously, between two iterations: a handler must only hand
/// the snapshot off (for example to a background writer) and return. It is not a progress stream, so no autosave is
/// dropped by a slow consumer.
public typealias AutosaveHandler = @Sendable (TrainingAutosave) -> Void

/// Runs one `GPUTrainer` (two in leave-one-out) and reports its progress as an `AsyncStream`.
public actor TrainingSession {
    /// Progress events, `.bufferingNewest(4)`: a slow consumer loses old events, never the terminal one
    /// (the newest). Finished when `run()` returns or throws, and when the session is deinitialized.
    public nonisolated let progress: AsyncStream<TrainingProgress>
    private let continuation: AsyncStream<TrainingProgress>.Continuation

    private let dataset: TrainingDataset
    private let configuration: TrainingConfiguration
    /// Builds the initial cloud of a run from the dataset of that run.
    private let initializer: @Sendable (TrainingDataset) -> GaussianCloud
    /// Receives autosaves (v1.5). Nil: no autosave.
    private let autosaveHandler: AutosaveHandler?
    /// The optimizer state of a continuation, taken by the single run. Cleared once used or ignored.
    private var pendingOptimizerState: OptimizerState?
    private let logger = Logger(subsystem: "io.github.junostr05.SparseSplat", category: "training")

    private var hasStarted = false
    private var isPaused = false
    /// The trainer of the current run, for `currentIteration`. Owned by this actor.
    private var currentTrainer: GPUTrainer?
    private var lastProgress: TrainingProgress?
    /// Last snapshot known to be finite. Set to the initial cloud when a run starts, so `.failed` carries a model.
    private var lastGoodPreview: GaussianCloud?
    private var lastLoss: Double?
    private var lastDepthLoss: Double?
    private var lastPSNR: Double?

    /// Phase of the run in progress.
    private var currentPhase: TrainingPhase = .single
    /// Run 1 only: the held-out training view, which is not in run 1's dataset.
    private var heldOutTrainingView: TrainingView?
    /// Run 1 onwards: the scheduler, kept after run 1 for the best iteration and metrics.
    private var scheduler: LeaveOneOutScheduler?
    /// Run 1 only: the latest held-out evaluation.
    private var latestHeldOut: HeldOutMetrics?

    /// `autosave` receives a snapshot every `configuration.autosaveInterval` iterations (not in the held-out selection
    /// run, and never at the last iteration, whose result is the run's checkpoint).
    public init(dataset: TrainingDataset, configuration: TrainingConfiguration,
                initializer: @escaping @Sendable (TrainingDataset) -> GaussianCloud,
                autosave: AutosaveHandler? = nil) {
        let (stream, continuation) = AsyncStream.makeStream(of: TrainingProgress.self,
                                                            bufferingPolicy: .bufferingNewest(4))
        progress = stream
        self.continuation = continuation
        self.dataset = dataset
        self.configuration = configuration
        self.initializer = initializer
        self.autosaveHandler = autosave
        self.pendingOptimizerState = nil
    }

    /// Single-cloud form: every run starts from `initialCloud`, whatever its dataset. Early stopping
    /// with this form still holds the view out of training, but the cloud was built from all views.
    /// `optimizerState` (v1.5) continues the moments of a previous run, as a continuation round: it is used only when it
    /// fits `initialCloud` and the configuration's switches (count, SH degree, views, appearance, pose refinement).
    /// Otherwise it is ignored with a logged reason and the run starts with fresh Adam state.
    public init(dataset: TrainingDataset, initialCloud: GaussianCloud, configuration: TrainingConfiguration,
                optimizerState: OptimizerState? = nil, autosave: AutosaveHandler? = nil) {
        let (stream, continuation) = AsyncStream.makeStream(of: TrainingProgress.self,
                                                            bufferingPolicy: .bufferingNewest(4))
        progress = stream
        self.continuation = continuation
        self.dataset = dataset
        self.configuration = configuration
        self.initializer = { _ in initialCloud }
        self.autosaveHandler = autosave
        self.pendingOptimizerState = optimizerState
    }

    deinit {
        // Consumers of `progress` must never hang, even when `run()` was never called.
        continuation.finish()
    }

    /// Completed iterations of the current run (0 before `run()` has created the trainer).
    public var currentIteration: Int {
        currentTrainer?.iteration ?? 0
    }

    /// Trains until the run's limit is reached or the task is cancelled. Returns the final cloud, or
    /// the cloud at the last completed iteration when cancelled.
    public func run() async throws(TrainingError) -> GaussianCloud {
        try await runWithOutcome().cloud
    }

    /// Like `run()`, and also reports the leave-one-out result.
    public func runWithOutcome() async throws(TrainingError) -> TrainingOutcome {
        guard !hasStarted else {
            throw .alreadyRunning
        }
        hasStarted = true
        do {
            if let index = heldOutIndex() {
                return try await runLeaveOneOut(heldOutIndex: index)
            }
            return try await runSingle()
        } catch {
            emitFailure(error)
            throw error
        }
    }

    /// Requests a pause. Takes effect at the next iteration boundary.
    public func pause() {
        isPaused = true
    }

    public func resume() {
        isPaused = false
    }

    // MARK: - Runs

    /// The held-out view index, or nil when leave-one-out does not run (P11).
    private func heldOutIndex() -> Int? {
        let policy = configuration.earlyStopping
        guard policy.isEnabled, policy.evalInterval > 0, dataset.views.count >= policy.minimumViews else {
            return nil
        }
        return HeldOutSelection.select(names: dataset.views.map(\.name),
                                       cameraCenters: dataset.cameraCenters,
                                       override: configuration.heldOutViews.first,
                                       policy: policy)
    }

    private func runSingle() async throws(TrainingError) -> TrainingOutcome {
        currentPhase = .single
        let trainer = try startTrainer(dataset: dataset)
        try restoreOptimizerState(into: trainer)
        let (end, rate) = try await train(trainer, limit: configuration.iterations)
        let (cloud, optimizer) = try complete(trainer, end: end, rate: rate, total: configuration.iterations)
        return TrainingOutcome(cloud: cloud, iterations: trainer.iteration,
                               refinedCameras: try refinedCameras(of: trainer), optimizerState: optimizer)
    }

    /// Takes the optimizer state given to the session, if any, into the fresh `trainer` as a continuation. A state that
    /// does not fit is logged with its reason, and the run keeps the fresh Adam state of the trainer.
    private func restoreOptimizerState(into trainer: GPUTrainer) throws(TrainingError) {
        guard let state = pendingOptimizerState else {
            return
        }
        pendingOptimizerState = nil
        if let reason = try trainer.restore(from: state, continuing: true) {
            logger.notice("optimizer state ignored, Adam starts fresh: \(reason, privacy: .public)")
        } else {
            logger.info("optimizer state restored at Adam step \(state.adamStep, privacy: .public)")
        }
    }

    private func runLeaveOneOut(heldOutIndex index: Int) async throws(TrainingError) -> TrainingOutcome {
        let heldOut = dataset.views[index]
        scheduler = LeaveOneOutScheduler(policy: configuration.earlyStopping, maxIterations: configuration.iterations)
        latestHeldOut = nil
        heldOutTrainingView = heldOut
        currentPhase = .selection(heldOutView: heldOut.name)
        if pendingOptimizerState != nil {
            pendingOptimizerState = nil
            logger.notice("optimizer state ignored: leave-one-out runs start with fresh Adam state")
        }

        switch try await runSelection(excluding: index) {
        case .cancelled(let outcome):
            return outcome
        case .finished:
            break
        }

        let stopIteration: Int = scheduler?.finalIterations(viewCount: dataset.views.count) ?? configuration.iterations
        heldOutTrainingView = nil
        currentPhase = .final(stopIteration: stopIteration)
        let trainer = try startTrainer(dataset: dataset)
        let (end, rate) = try await train(trainer, limit: stopIteration)
        let (cloud, optimizer) = try complete(trainer, end: end, rate: rate, total: stopIteration)
        return TrainingOutcome(cloud: cloud, heldOutView: heldOut.name, bestIteration: scheduler?.bestIteration,
                               heldOut: scheduler?.bestMetrics, iterations: trainer.iteration,
                               refinedCameras: try refinedCameras(of: trainer), optimizerState: optimizer)
    }

    /// Refined training cameras of the final run, or nil when pose refinement is off.
    private func refinedCameras(of trainer: GPUTrainer) throws(TrainingError) -> [Camera]? {
        guard configuration.poseRefinement.isEnabled else { return nil }
        return try trainer.refinedCameras()
    }

    private enum SelectionResult {
        case finished
        case cancelled(TrainingOutcome)
    }

    /// Run 1. Its trainer is released when this returns, before run 2 allocates its own.
    private func runSelection(excluding index: Int) async throws(TrainingError) -> SelectionResult {
        var reduced = dataset
        reduced.views.remove(at: index)
        let trainer = try startTrainer(dataset: reduced)
        let (end, rate) = try await train(trainer, limit: configuration.iterations)
        guard end == .cancelled else {
            currentTrainer = nil
            return .finished
        }
        let cloud = try trainer.snapshot()
        lastGoodPreview = cloud
        emit(.cancelled, trainer: trainer, rate: rate, preview: cloud, total: configuration.iterations)
        continuation.finish()
        let outcome = TrainingOutcome(cloud: cloud, heldOutView: heldOutTrainingView?.name,
                                      bestIteration: scheduler?.bestIteration, heldOut: scheduler?.bestMetrics,
                                      iterations: trainer.iteration)
        currentTrainer = nil
        return .cancelled(outcome)
    }

    // MARK: - Loop

    private enum LoopEnd: Equatable {
        case limitReached
        /// The leave-one-out scheduler decided to stop (run 1 only).
        case stopped
        case cancelled
    }

    /// Steps `trainer` until `limit`, a scheduler stop, or cancellation. Returns how it ended and the
    /// iteration rate of the last window. Emits `.running` and `.paused` events; terminal events are
    /// left to the caller (`complete` or `runSelection`).
    private func train(_ trainer: GPUTrainer, limit: Int) async throws(TrainingError) -> (end: LoopEnd, rate: Double) {
        var meter = RateMeter(now: ContinuousClock.now, iteration: trainer.iteration)
        let selectionTarget: TrainingView? = isSelectionPhase ? heldOutTrainingView : nil
        let evalInterval: Int = configuration.earlyStopping.evalInterval
        while true {
            if Task.isCancelled {
                return (.cancelled, meter.rate(now: ContinuousClock.now, iteration: trainer.iteration))
            }

            // Finished takes precedence over pause: a pause right after the last step must not park a
            // run that has nothing left to do.
            if trainer.iteration >= limit {
                return (.limitReached, meter.rate(now: ContinuousClock.now, iteration: trainer.iteration))
            }

            if isPaused {
                emit(.paused, trainer: trainer, rate: 0, preview: nil, total: limit)
                while isPaused && !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(50))
                }
                // The rate after a pause must not include the paused time.
                meter = RateMeter(now: ContinuousClock.now, iteration: trainer.iteration)
                if !Task.isCancelled {
                    emit(.running, trainer: trainer, rate: 0, preview: nil, total: limit)
                }
                continue
            }

            let report = try trainer.step()
            let iteration = report.iteration
            if let loss = report.loss {
                lastLoss = loss.loss
            }
            if let depth = report.depthLoss {
                lastDepthLoss = depth.loss
            }
            // Autosave (v1.5): not in the selection run, and not at the limit, where the run's checkpoint is written.
            if let handler = autosaveHandler, configuration.autosaveInterval > 0, !isSelectionPhase,
               iteration % configuration.autosaveInterval == 0, iteration < limit {
                try writeAutosave(trainer, handler: handler)
            }
            let isPreviewStep = configuration.previewInterval > 0 && iteration % configuration.previewInterval == 0
            let isEvalStep = configuration.evalInterval > 0 && iteration % configuration.evalInterval == 0
            var preview: GaussianCloud?
            if isPreviewStep {
                let cloud = try trainer.snapshot()
                lastGoodPreview = cloud
                preview = cloud
            }
            if isEvalStep {
                lastPSNR = try trainer.evaluatePSNR()
            }
            var isHeldOutStep = false
            var stopSelection = false
            if let target = selectionTarget, evalInterval > 0, iteration % evalInterval == 0 {
                isHeldOutStep = true
                let metrics = try trainer.evaluate(view: target)
                latestHeldOut = metrics
                if scheduler?.record(iteration: iteration, metrics: metrics) == .stop {
                    stopSelection = true
                }
            }
            if report.loss != nil || isPreviewStep || isEvalStep || isHeldOutStep {
                let rate = meter.rate(now: ContinuousClock.now, iteration: iteration)
                emit(.running, trainer: trainer, rate: rate, preview: preview, total: limit)
            }
            if stopSelection {
                return (.stopped, meter.rate(now: ContinuousClock.now, iteration: iteration))
            }
            await Task.yield()
        }
    }

    /// Hands the cloud and optimizer state at the current iteration to the autosave handler.
    private func writeAutosave(_ trainer: GPUTrainer, handler: AutosaveHandler) throws(TrainingError) {
        let cloud = try trainer.snapshot()
        let optimizer = try trainer.exportOptimizerState()
        handler(TrainingAutosave(iteration: trainer.iteration, cloud: cloud, optimizerState: optimizer))
    }

    /// Terminal event of a run that is not run 1, and its stream end. Returns the run's cloud and optimizer state.
    private func complete(_ trainer: GPUTrainer, end: LoopEnd, rate: Double,
                          total: Int) throws(TrainingError) -> (cloud: GaussianCloud, optimizer: OptimizerState) {
        let cloud = try trainer.snapshot()
        let optimizer = try trainer.exportOptimizerState()
        lastGoodPreview = cloud
        if end == .cancelled {
            emit(.cancelled, trainer: trainer, rate: rate, preview: cloud, total: total)
        } else {
            lastPSNR = try trainer.evaluatePSNR()
            emit(.finished, trainer: trainer, rate: rate, preview: cloud, total: total)
        }
        continuation.finish()
        return (cloud, optimizer)
    }

    // MARK: - Helpers

    private var isSelectionPhase: Bool {
        if case .selection = currentPhase {
            return true
        }
        return false
    }

    /// Creates the trainer of a run. The run's initial cloud comes from `initializer` with that run's dataset.
    private func startTrainer(dataset runDataset: TrainingDataset) throws(TrainingError) -> GPUTrainer {
        let initialCloud = initializer(runDataset)
        lastGoodPreview = initialCloud
        let context = try metalCall { () throws(MetalError) -> MetalContext in
            try MetalContext.makeDefault()
        }
        let trainer = try GPUTrainer(context: context, dataset: runDataset, initialCloud: initialCloud,
                                     configuration: configuration)
        currentTrainer = trainer
        return trainer
    }

    /// Held-out metrics shown with an event: the latest in run 1, the best from run 2 on.
    private var heldOutForProgress: HeldOutMetrics? {
        switch currentPhase {
        case .single:
            return nil
        case .selection:
            return latestHeldOut
        case .final:
            return scheduler?.bestMetrics
        }
    }

    private func emit(_ state: TrainingProgress.State, trainer: GPUTrainer, rate: Double,
                      preview: GaussianCloud?, total: Int) {
        let progress = TrainingProgress(state: state,
                                        iteration: trainer.iteration,
                                        totalIterations: total,
                                        loss: lastLoss,
                                        psnr: lastPSNR,
                                        gaussianCount: trainer.gaussianCount,
                                        iterationsPerSecond: rate,
                                        preview: preview,
                                        phase: currentPhase,
                                        heldOut: heldOutForProgress,
                                        bestIteration: scheduler?.bestIteration,
                                        depthLoss: lastDepthLoss)
        // The stored copy drops the preview so the session does not keep an extra model alive.
        var stored = progress
        stored.preview = nil
        lastProgress = stored
        continuation.yield(progress)
    }

    /// Yields the `.failed` event (with the last good preview) and finishes the stream.
    private func emitFailure(_ error: TrainingError) {
        var failed = lastProgress ?? TrainingProgress(state: .running, iteration: 0,
                                                      totalIterations: configuration.iterations)
        failed.state = .failed(Self.describe(error))
        failed.phase = currentPhase
        failed.preview = lastGoodPreview
        continuation.yield(failed)
        continuation.finish()
    }

    private static func describe(_ error: TrainingError) -> String {
        switch error {
        case .metal(let message):
            return message
        case .noViews:
            return "the dataset has no training views"
        case .nonFiniteLoss(let iteration):
            return "the loss became non-finite at iteration \(iteration)"
        case .initialCloudExceedsBudget(let count, let cap):
            return "the initial cloud of \(count) Gaussians exceeds the memory budget of \(cap)"
        case .alreadyRunning:
            return "training is already running"
        }
    }
}

/// Iterations per second over the window since the previous reading.
private struct RateMeter {
    private var mark: ContinuousClock.Instant
    private var markIteration: Int

    init(now: ContinuousClock.Instant, iteration: Int) {
        mark = now
        markIteration = iteration
    }

    /// Rate since the last call, then restarts the window. Zero when no time has passed.
    mutating func rate(now: ContinuousClock.Instant, iteration: Int) -> Double {
        let elapsed = mark.duration(to: now).components
        let seconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) * 1e-18
        let done = iteration - markIteration
        mark = now
        markIteration = iteration
        guard seconds > 0 else {
            return 0
        }
        return Double(done) / seconds
    }
}
