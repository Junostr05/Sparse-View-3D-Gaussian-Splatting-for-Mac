import Foundation

/// User-facing settings for continuing to train a finished model (v1.1).
/// Each round is one `TrainingSession`; `TrainingConfiguration.continued(...)` derives the
/// configuration for a round from these settings.
public struct ContinuationSettings: Codable, Sendable, Equatable {
    /// Iterations of one round (one TrainingSession). Default 5_000.
    public var roundIterations = 5_000
    /// Repeat rounds until the user stops.
    public var isEndless = false
    /// Re-enable densification so the model grows.
    public var addGaussians = false
    /// Per round the Gaussian cap is existing count × growthFactor (only when addGaussians). Default 1.5.
    public var growthFactor: Float = 1.5
    /// Hard ceiling for the Gaussian count. Default 3_000_000.
    public var maxGaussians = 3_000_000

    public init() {}
}

extension TrainingConfiguration {
    /// Configuration for a continuation round that starts from a trained cloud with `gaussianCount` Gaussians
    /// of SH degree `shDegree`. `round` is 0-based.
    public func continued(from gaussianCount: Int, shDegree: Int, round: Int,
                          settings: ContinuationSettings) -> TrainingConfiguration {
        var config = self

        // Iterations of this round.
        let roundIterations: Int = max(1, settings.roundIterations)
        config.iterations = roundIterations

        // SH: the loaded cloud keeps its full degree; the restarted counter must not truncate it.
        config.shDegree = shDegree
        config.shDegreeInterval = 0

        // Learning rates: the model is already converged, so start the position rate lower.
        config.learningRates.positionInitial = self.learningRates.positionFinal * 10

        // Early stopping is off for continuation; there is no held-out view by default.
        config.earlyStopping.isEnabled = false
        config.heldOutViews = []

        // Different sampling order per round.
        let roundIndex: UInt64 = UInt64(max(0, round))
        config.seed = self.seed &+ roundIndex

        // Densification: grow only when asked, and stop at 80% of the round.
        if settings.addGaussians {
            let stopIteration: Int = Int(Double(roundIterations) * 0.8)
            config.densification.startIteration = 0
            config.densification.stopIteration = stopIteration
            if config.densification.interval <= 0 {
                config.densification.interval = 100
            }
            config.densification.gradientThreshold = self.densification.gradientThreshold * 0.5
        } else {
            config.densification.startIteration = 0
            config.densification.stopIteration = 0
        }

        // Gaussian cap: grow by growthFactor per round up to the ceiling, never below the current count.
        if settings.addGaussians {
            let growth: Float = max(settings.growthFactor, 1)
            let grownFloat: Float = Float(gaussianCount) * growth
            let grown: Int = Int(grownFloat.rounded(.up))
            let capped: Int = min(settings.maxGaussians, grown)
            config.maxGaussians = max(gaussianCount, capped)
        } else {
            config.maxGaussians = max(self.maxGaussians, gaussianCount)
        }

        return config
    }
}
