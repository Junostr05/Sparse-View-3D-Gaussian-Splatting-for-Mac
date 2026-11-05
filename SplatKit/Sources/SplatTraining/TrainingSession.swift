import SplatCore
import SplatMetal

// Headless training run (Phase 3 Task 15). The actor owns the Metal context and the GPUTrainer for the
// whole run; no Metal object crosses an isolation boundary. Progress leaves the actor only as
// `TrainingProgress` values (Sendable, previews are snapshots).
//
// Interpretations, where the plan is literal:
// - A session runs once. A second `run()`, concurrent or after the first has finished, throws
//   `.alreadyRunning`, because the progress stream is finished when the first run ends.
// - Pause is a 50 ms sleep loop checked between iterations. A step in progress is never interrupted.
// - Cancellation is checked between iterations; densification is atomic with respect to it. A
//   cancelled run returns the snapshot at the last completed iteration.
// - `.failed` carries the last good preview, so a caller can still show the last usable model.

/// Runs one `GPUTrainer` and reports its progress as an `AsyncStream`.
public actor TrainingSession {
    /// Progress events, `.bufferingNewest(4)`: a slow consumer loses old events, never the terminal one
    /// (the newest). Finished when `run()` returns or throws, and when the session is deinitialized.
    public nonisolated let progress: AsyncStream<TrainingProgress>
    private let continuation: AsyncStream<TrainingProgress>.Continuation

    private let dataset: TrainingDataset
    private let initialCloud: GaussianCloud
    private let configuration: TrainingConfiguration

    private var hasStarted = false
    private var isPaused = false
    /// The trainer of the current run, for `currentIteration`. Owned by this actor.
    private var currentTrainer: GPUTrainer?
    private var lastProgress: TrainingProgress?
    /// Last snapshot known to be finite. Starts as the initial cloud, so `.failed` always carries a model.
    private var lastGoodPreview: GaussianCloud?
    private var lastLoss: Double?
    private var lastPSNR: Double?

    public init(dataset: TrainingDataset, initialCloud: GaussianCloud, configuration: TrainingConfiguration) {
        let (stream, continuation) = AsyncStream.makeStream(of: TrainingProgress.self,
                                                            bufferingPolicy: .bufferingNewest(4))
        progress = stream
        self.continuation = continuation
        self.dataset = dataset
        self.initialCloud = initialCloud
        self.configuration = configuration
        lastGoodPreview = initialCloud
    }

    deinit {
        // Consumers of `progress` must never hang, even when `run()` was never called.
        continuation.finish()
    }

    /// Completed iterations of the current run (0 before `run()` has created the trainer).
    public var currentIteration: Int {
        currentTrainer?.iteration ?? 0
    }

    /// Trains until `configuration.iterations` are done or the task is cancelled. Returns the final
    /// cloud, or the cloud at the last completed iteration when cancelled.
    public func run() async throws(TrainingError) -> GaussianCloud {
        guard !hasStarted else {
            throw .alreadyRunning
        }
        hasStarted = true
        do {
            let trainer = try makeTrainer()
            currentTrainer = trainer
            return try await train(trainer)
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

    // MARK: - Loop

    private func train(_ trainer: GPUTrainer) async throws(TrainingError) -> GaussianCloud {
        var meter = RateMeter(now: ContinuousClock.now, iteration: trainer.iteration)
        while true {
            if Task.isCancelled {
                let cloud = try trainer.snapshot()
                lastGoodPreview = cloud
                emit(.cancelled, trainer: trainer, rate: meter.rate(now: ContinuousClock.now, iteration: trainer.iteration),
                     preview: cloud)
                continuation.finish()
                return cloud
            }

            // Finished takes precedence over pause: a pause right after the last step must not park a
            // run that has nothing left to do.
            if trainer.iteration >= configuration.iterations {
                break
            }

            if isPaused {
                emit(.paused, trainer: trainer, rate: 0, preview: nil)
                while isPaused && !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(50))
                }
                // The rate after a pause must not include the paused time.
                meter = RateMeter(now: ContinuousClock.now, iteration: trainer.iteration)
                if !Task.isCancelled {
                    emit(.running, trainer: trainer, rate: 0, preview: nil)
                }
                continue
            }

            let report = try trainer.step()
            let iteration = report.iteration
            if let loss = report.loss {
                lastLoss = loss.loss
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
            if report.loss != nil || isPreviewStep || isEvalStep {
                let rate = meter.rate(now: ContinuousClock.now, iteration: iteration)
                emit(.running, trainer: trainer, rate: rate, preview: preview)
            }
            await Task.yield()
        }

        let cloud = try trainer.snapshot()
        lastGoodPreview = cloud
        lastPSNR = try trainer.evaluatePSNR()
        let rate = meter.rate(now: ContinuousClock.now, iteration: trainer.iteration)
        emit(.finished, trainer: trainer, rate: rate, preview: cloud)
        continuation.finish()
        return cloud
    }

    // MARK: - Helpers

    private func makeTrainer() throws(TrainingError) -> GPUTrainer {
        let context = try metalCall { () throws(MetalError) -> MetalContext in
            try MetalContext.makeDefault()
        }
        return try GPUTrainer(context: context, dataset: dataset, initialCloud: initialCloud,
                              configuration: configuration)
    }

    private func emit(_ state: TrainingProgress.State, trainer: GPUTrainer, rate: Double,
                      preview: GaussianCloud?) {
        let progress = TrainingProgress(state: state,
                                        iteration: trainer.iteration,
                                        totalIterations: configuration.iterations,
                                        loss: lastLoss,
                                        psnr: lastPSNR,
                                        gaussianCount: trainer.gaussianCount,
                                        iterationsPerSecond: rate,
                                        preview: preview)
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
