import Foundation

/// Per-group base learning rates. Position rates are multiplied by the scene radius.
public struct LearningRates: Codable, Sendable, Equatable {
    public var positionInitial: Float = 1.6e-4   // × scene radius
    public var positionFinal: Float = 1.6e-6     // × scene radius
    public var logScale: Float = 5e-3
    public var rotation: Float = 1e-3
    public var opacity: Float = 5e-2
    public var shDC: Float = 2.5e-3
    public var shRest: Float = 1.25e-4

    public init() {}
}

/// Learning rates for one iteration, after the position schedule has been evaluated.
public struct ResolvedLearningRates: Sendable, Equatable {
    public var position: Float
    public var logScale: Float
    public var rotation: Float
    public var opacity: Float
    public var shDC: Float
    public var shRest: Float

    public init(position: Float, logScale: Float, rotation: Float,
                opacity: Float, shDC: Float, shRest: Float) {
        self.position = position
        self.logScale = logScale
        self.rotation = rotation
        self.opacity = opacity
        self.shDC = shDC
        self.shRest = shRest
    }
}

public struct AdamHyperparameters: Codable, Sendable, Equatable {
    public var beta1: Float = 0.9
    public var beta2: Float = 0.999
    public var epsilon: Float = 1e-15

    public init() {}
}

public struct DensificationPolicy: Codable, Sendable, Equatable {
    public var startIteration = 500
    public var stopIteration = 6_000
    public var interval = 100
    public var gradientThreshold: Float = 0.0002
    /// × scene radius: clone if max scale ≤ this, else split.
    public var splitScaleFraction: Float = 0.01
    public var pruneOpacity: Float = 0.005
    /// × scene radius.
    public var pruneScaleFraction: Float = 0.1
    public var pruneScaleStartIteration = 3_000
    public var splitScaleDivisor: Float = 1.6
    /// Recorded decision 15: opacity reset is off. Must stay 0 in Phase 3.
    public var opacityResetInterval = 0

    public init() {}

    public var isEnabled: Bool { startIteration < stopIteration && interval > 0 }
}

/// Extension point for Phase 4 regularizers. Unused in Phase 3.
public enum RegularizerKind: String, Codable, Sendable {
    case depthPearson
    case depthL1
    case anisotropy
    case scaleCap
    case isolationPrune
}

public struct TrainingConfiguration: Codable, Sendable, Equatable {
    public var iterations = 10_000
    public var shDegree = 1
    public var shDegreeInterval = 1_000
    public var lambdaDSSIM: Float = 0.2
    public var background: SIMD3<Float> = .zero
    public var learningRates = LearningRates()
    public var adam = AdamHyperparameters()
    public var densification = DensificationPolicy()
    public var maxGaussians = 1_500_000
    public var maxImageSide = 1_600
    public var initialOpacity: Float = 0.1
    public var randomInitCount = 10_000
    public var seed: UInt64 = 0
    public var logInterval = 10
    public var evalInterval = 500
    public var previewInterval = 200
    /// Phase 4 extension point.
    public var regularizers: Set<RegularizerKind> = []
    /// Phase 4 extension point (leave-one-out).
    public var heldOutViews: [String] = []

    public init() {}

    /// Every key is optional; missing keys fall back to the defaults above.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = TrainingConfiguration()
        iterations = try container.decodeIfPresent(Int.self, forKey: .iterations) ?? defaults.iterations
        shDegree = try container.decodeIfPresent(Int.self, forKey: .shDegree) ?? defaults.shDegree
        shDegreeInterval = try container.decodeIfPresent(Int.self, forKey: .shDegreeInterval)
            ?? defaults.shDegreeInterval
        lambdaDSSIM = try container.decodeIfPresent(Float.self, forKey: .lambdaDSSIM)
            ?? defaults.lambdaDSSIM
        background = try container.decodeIfPresent(SIMD3<Float>.self, forKey: .background)
            ?? defaults.background
        learningRates = try container.decodeIfPresent(LearningRates.self, forKey: .learningRates)
            ?? defaults.learningRates
        adam = try container.decodeIfPresent(AdamHyperparameters.self, forKey: .adam)
            ?? defaults.adam
        densification = try container.decodeIfPresent(DensificationPolicy.self, forKey: .densification)
            ?? defaults.densification
        maxGaussians = try container.decodeIfPresent(Int.self, forKey: .maxGaussians)
            ?? defaults.maxGaussians
        maxImageSide = try container.decodeIfPresent(Int.self, forKey: .maxImageSide)
            ?? defaults.maxImageSide
        initialOpacity = try container.decodeIfPresent(Float.self, forKey: .initialOpacity)
            ?? defaults.initialOpacity
        randomInitCount = try container.decodeIfPresent(Int.self, forKey: .randomInitCount)
            ?? defaults.randomInitCount
        seed = try container.decodeIfPresent(UInt64.self, forKey: .seed) ?? defaults.seed
        logInterval = try container.decodeIfPresent(Int.self, forKey: .logInterval)
            ?? defaults.logInterval
        evalInterval = try container.decodeIfPresent(Int.self, forKey: .evalInterval)
            ?? defaults.evalInterval
        previewInterval = try container.decodeIfPresent(Int.self, forKey: .previewInterval)
            ?? defaults.previewInterval
        regularizers = try container.decodeIfPresent(Set<RegularizerKind>.self, forKey: .regularizers)
            ?? defaults.regularizers
        heldOutViews = try container.decodeIfPresent([String].self, forKey: .heldOutViews)
            ?? defaults.heldOutViews
    }

    /// SH degree in use at `iteration`: grows by one every `shDegreeInterval` iterations.
    public func activeSHDegree(iteration: Int) -> Int {
        guard shDegreeInterval > 0 else { return shDegree }
        let steps = max(0, iteration) / shDegreeInterval
        return min(shDegree, steps)
    }

    /// Learning rates at `iteration`. The position rate decays exponentially (log-linear)
    /// from `positionInitial` to `positionFinal` over `iterations`, scaled by `sceneRadius`.
    public func learningRates(iteration: Int, sceneRadius: Float) -> ResolvedLearningRates {
        let progress: Double
        if iterations > 0 {
            progress = min(max(Double(iteration) / Double(iterations), 0), 1)
        } else {
            progress = 1
        }
        let logInitial = log(Double(learningRates.positionInitial))
        let logFinal = log(Double(learningRates.positionFinal))
        let logPosition = (1 - progress) * logInitial + progress * logFinal
        let position = Float(Double(sceneRadius) * exp(logPosition))

        return ResolvedLearningRates(
            position: position,
            logScale: learningRates.logScale,
            rotation: learningRates.rotation,
            opacity: learningRates.opacity,
            shDC: learningRates.shDC,
            shRest: learningRates.shRest
        )
    }
}
