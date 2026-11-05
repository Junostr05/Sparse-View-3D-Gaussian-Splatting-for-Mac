import SplatCore

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

    public init(state: State,
                iteration: Int,
                totalIterations: Int,
                loss: Double? = nil,
                psnr: Double? = nil,
                gaussianCount: Int = 0,
                iterationsPerSecond: Double = 0,
                preview: GaussianCloud? = nil) {
        self.state = state
        self.iteration = iteration
        self.totalIterations = totalIterations
        self.loss = loss
        self.psnr = psnr
        self.gaussianCount = gaussianCount
        self.iterationsPerSecond = iterationsPerSecond
        self.preview = preview
    }
}
