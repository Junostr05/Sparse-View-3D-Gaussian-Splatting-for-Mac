import SplatCore

/// Which part of a training session an event belongs to (P11).
public enum TrainingPhase: Sendable, Equatable {
    /// A plain run: early stopping off, or fewer views than `minimumViews`.
    case single
    /// Run 1 of leave-one-out: the named view is held out and used only for evaluation.
    case selection(heldOutView: String)
    /// Run 2: a fresh retrain on all views, stopping at `stopIteration`.
    case final(stopIteration: Int)
}

/// One progress event of a `TrainingSession`. Emitted on log, preview and evaluation boundaries and
/// on the terminal states.
public struct TrainingProgress: Sendable {
    public enum State: Sendable, Equatable {
        case running
        case paused
        case finished
        case cancelled
        case failed(String)
    }

    public var state: State
    /// Completed iterations.
    public var iteration: Int
    /// Iterations of the current run: `configuration.iterations` in run 1, the stop iteration in run 2.
    public var totalIterations: Int
    /// Loss of the most recent logged step. `nil` before the first logged step.
    public var loss: Double?
    /// Mean PSNR over the training views at the most recent evaluation. `nil` before the first one.
    public var psnr: Double?
    public var gaussianCount: Int
    public var iterationsPerSecond: Double
    /// Set every `previewInterval` iterations, on `finished` and `cancelled`, and on `failed`
    /// (the last good preview, if any). Always a snapshot, never a live buffer.
    public var preview: GaussianCloud?
    /// The phase of the session this event belongs to.
    public var phase: TrainingPhase
    /// In `.selection`: the latest held-out evaluation. In `.final` and on `.finished`: the best one.
    public var heldOut: HeldOutMetrics?
    /// Best held-out iteration so far (leave-one-out only; nil for a single run).
    public var bestIteration: Int?
    /// Depth loss of the most recent logged step. `nil` when no depth term is active.
    public var depthLoss: Double?

    public init(state: State,
                iteration: Int,
                totalIterations: Int,
                loss: Double? = nil,
                psnr: Double? = nil,
                gaussianCount: Int = 0,
                iterationsPerSecond: Double = 0,
                preview: GaussianCloud? = nil,
                phase: TrainingPhase = .single,
                heldOut: HeldOutMetrics? = nil,
                bestIteration: Int? = nil,
                depthLoss: Double? = nil) {
        self.state = state
        self.iteration = iteration
        self.totalIterations = totalIterations
        self.loss = loss
        self.psnr = psnr
        self.gaussianCount = gaussianCount
        self.iterationsPerSecond = iterationsPerSecond
        self.preview = preview
        self.phase = phase
        self.heldOut = heldOut
        self.bestIteration = bestIteration
        self.depthLoss = depthLoss
    }
}
