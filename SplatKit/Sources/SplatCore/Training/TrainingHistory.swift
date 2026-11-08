/// The recorded progress of one training run, for the training dashboard.
///
/// Times are wall-clock seconds from the caller's clock (for example `ProcessInfo.systemUptime`). Time spent
/// paused is excluded from `elapsedSeconds`, so the elapsed time and the rate match the iterations actually run.
/// The history keeps at most `capacity` points. When it is full, the older half is thinned to every other point,
/// so the chart keeps the whole run at a coarser resolution instead of dropping its beginning.
public struct TrainingHistory: Sendable, Equatable {
    /// Most points kept. Beyond this the older half is decimated before a new point is added.
    public static let capacity: Int = 1500
    /// Weight of the newest instantaneous rate in the moving average of iterations per second.
    public static let smoothing: Double = 0.2

    /// One progress event as the training session reports it.
    public struct Sample: Sendable, Equatable {
        /// Cumulative completed iterations of the whole project, not only of the current round.
        public var iteration: Int
        /// Loss of the most recent logged step. `nil` before the first logged step.
        public var loss: Double?
        /// Mean PSNR over the training views at the most recent evaluation. `nil` before the first one.
        public var psnr: Double?
        public var gaussianCount: Int
        /// The session's own rate, kept for reference. The history smooths its own rate from the iteration deltas.
        public var iterationsPerSecond: Double

        public init(iteration: Int,
                    loss: Double?,
                    psnr: Double?,
                    gaussianCount: Int,
                    iterationsPerSecond: Double) {
            self.iteration = iteration
            self.loss = loss
            self.psnr = psnr
            self.gaussianCount = gaussianCount
            self.iterationsPerSecond = iterationsPerSecond
        }
    }

    /// A recorded sample with the active (unpaused) time at which it was recorded.
    public struct Point: Sendable, Equatable {
        public var iteration: Int
        public var loss: Double?
        public var psnr: Double?
        public var gaussianCount: Int
        public var iterationsPerSecond: Double
        /// Active seconds since the first sample, paused time excluded.
        public var elapsedSeconds: Double
    }

    /// Recorded points, oldest first. At most `capacity`.
    public private(set) var points: [Point] = []
    /// Moving average of the iteration rate, from the deltas between points. Zero until two points with progress exist.
    public private(set) var smoothedIterationsPerSecond: Double = 0
    /// Active seconds up to the latest update, paused time excluded.
    public private(set) var elapsedSeconds: Double = 0
    public private(set) var isPaused: Bool = false
    /// The clock time up to which active time has been counted. While paused, the time of the pause.
    private var lastTime: Double?

    public init() {}

    /// True when no sample has been recorded since the last `reset()`.
    public var isEmpty: Bool {
        points.isEmpty
    }

    /// The newest point, if any.
    public var latest: Point? {
        points.last
    }

    /// Clears the history. Call it when a new run starts.
    public mutating func reset() {
        self = TrainingHistory()
    }

    /// Stops the clock. Time from now until `resume(at:)` is not counted. Does nothing when already paused.
    public mutating func pause(at time: Double) {
        guard !isPaused else {
            return
        }
        advanceClock(to: time)
        isPaused = true
        lastTime = time
    }

    /// Restarts the clock after a pause. Does nothing when not paused.
    public mutating func resume(at time: Double) {
        guard isPaused else {
            return
        }
        isPaused = false
        lastTime = time
    }

    /// Records a sample at clock time `time`. A sample with the same iteration as the newest point replaces it,
    /// so repeated events at one iteration do not add points or distort the rate.
    public mutating func append(_ sample: Sample, at time: Double) {
        advanceClock(to: time)
        let previous: Point? = points.last
        let point: Point = Point(iteration: sample.iteration,
                                 loss: sample.loss,
                                 psnr: sample.psnr,
                                 gaussianCount: sample.gaussianCount,
                                 iterationsPerSecond: sample.iterationsPerSecond,
                                 elapsedSeconds: elapsedSeconds)

        if let previous, previous.iteration == sample.iteration {
            points[points.count - 1] = point
            return
        }

        if let previous {
            updateRate(from: previous, to: point)
        }
        if points.count >= Self.capacity {
            decimateOlderHalf()
        }
        points.append(point)
    }

    /// Seconds left at the smoothed rate, for a run with `totalIterations` in total. `nil` when the rate is zero
    /// (nothing measured yet) or the run has no fixed total (an endless continuation passes `nil`).
    public func estimatedRemainingSeconds(totalIterations: Int?) -> Double? {
        guard let totalIterations, smoothedIterationsPerSecond > 0, let latest else {
            return nil
        }
        let remaining: Int = max(0, totalIterations - latest.iteration)
        return Double(remaining) / smoothedIterationsPerSecond
    }

    // MARK: - Private

    /// Adds the active time since the last update. Does nothing while paused.
    private mutating func advanceClock(to time: Double) {
        guard !isPaused else {
            return
        }
        if let lastTime {
            elapsedSeconds += max(0, time - lastTime)
        }
        lastTime = time
    }

    /// Folds the rate between two points into the moving average. Points without progress or without elapsed time
    /// (a restarted round, a repeated time) do not change it.
    private mutating func updateRate(from previous: Point, to current: Point) {
        let iterations: Int = current.iteration - previous.iteration
        let seconds: Double = current.elapsedSeconds - previous.elapsedSeconds
        guard iterations > 0, seconds > 0 else {
            return
        }
        let instantaneous: Double = Double(iterations) / seconds
        if smoothedIterationsPerSecond > 0 {
            smoothedIterationsPerSecond = Self.smoothing * instantaneous
                + (1 - Self.smoothing) * smoothedIterationsPerSecond
        } else {
            smoothedIterationsPerSecond = instantaneous
        }
    }

    /// Keeps every other point of the older half. The newer half is untouched.
    private mutating func decimateOlderHalf() {
        let half: Int = points.count / 2
        var kept: [Point] = []
        kept.reserveCapacity(half / 2 + (points.count - half))
        var index: Int = 0
        while index < half {
            kept.append(points[index])
            index += 2
        }
        kept.append(contentsOf: points[half...])
        points = kept
    }
}
