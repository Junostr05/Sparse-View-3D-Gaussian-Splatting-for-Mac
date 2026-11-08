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

/// Phase 4 features that can be switched off individually. The stored set is the source of truth.
public enum RegularizerKind: String, Codable, Sendable, CaseIterable {
    case depthPearson
    case depthL1
    case anisotropy
    case scaleCap
    case isolationPrune
    case pseudoViews
}

/// The monocular depth model that fills in depth for views without a supplied map. The raw value is the
/// stable id used in storage (for example the per-estimator depth cache directory of a project).
public enum DepthEstimatorKind: String, Codable, Sendable, CaseIterable {
    /// Depth Anything V2 Small (Apache-2.0). Relative inverse depth, the default.
    case depthAnythingV2Small
    /// MoGe-2 ViT-S (MIT). Affine-invariant camera-space depth.
    case moge2Small
}

/// Strengths and gates of the depth prior (Phase 4 P4-P7).
public struct DepthPriorSettings: Codable, Sendable, Equatable {
    /// Monocular model used for views without a supplied depth map. Depth Anything V2 Small by default.
    public var estimator: DepthEstimatorKind = .depthAnythingV2Small
    public var pearsonWeight: Float = 0.05
    public var l1Weight: Float = 0.1
    /// Pixels with rendered opacity below this are excluded from the depth terms.
    public var alphaMask: Float = 0.3
    public var minimumPixels = 64
    public var startIteration = 0
    public var minimumAlignmentPoints = 8
    public var minimumInlierFraction: Float = 0.3

    public init() {}

    /// Every key is optional; missing keys fall back to the defaults above.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = DepthPriorSettings()
        estimator = try container.decodeIfPresent(DepthEstimatorKind.self, forKey: .estimator)
            ?? defaults.estimator
        pearsonWeight = try container.decodeIfPresent(Float.self, forKey: .pearsonWeight)
            ?? defaults.pearsonWeight
        l1Weight = try container.decodeIfPresent(Float.self, forKey: .l1Weight) ?? defaults.l1Weight
        alphaMask = try container.decodeIfPresent(Float.self, forKey: .alphaMask) ?? defaults.alphaMask
        minimumPixels = try container.decodeIfPresent(Int.self, forKey: .minimumPixels)
            ?? defaults.minimumPixels
        startIteration = try container.decodeIfPresent(Int.self, forKey: .startIteration)
            ?? defaults.startIteration
        minimumAlignmentPoints = try container.decodeIfPresent(Int.self, forKey: .minimumAlignmentPoints)
            ?? defaults.minimumAlignmentPoints
        minimumInlierFraction = try container.decodeIfPresent(Float.self, forKey: .minimumInlierFraction)
            ?? defaults.minimumInlierFraction
    }
}

/// Pseudo-view generation between neighbouring training cameras (Phase 4 P9).
public struct PseudoViewSettings: Codable, Sendable, Equatable {
    public var startIteration = 2_000
    public var interval = 2
    public var weight: Float = 0.05
    /// Interpolated cameras per neighbouring pair (t = k / (perPair + 1)).
    public var perPair = 3
    public var resolutionDivisor = 2
    public var maxAngleDegrees: Float = 60
    public var minimumCoverage: Float = 0.3
    public var minimumDepthFraction: Float = 0.2

    public init() {}

    /// Every key is optional; missing keys fall back to the defaults above.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = PseudoViewSettings()
        startIteration = try container.decodeIfPresent(Int.self, forKey: .startIteration)
            ?? defaults.startIteration
        interval = try container.decodeIfPresent(Int.self, forKey: .interval) ?? defaults.interval
        weight = try container.decodeIfPresent(Float.self, forKey: .weight) ?? defaults.weight
        perPair = try container.decodeIfPresent(Int.self, forKey: .perPair) ?? defaults.perPair
        resolutionDivisor = try container.decodeIfPresent(Int.self, forKey: .resolutionDivisor)
            ?? defaults.resolutionDivisor
        maxAngleDegrees = try container.decodeIfPresent(Float.self, forKey: .maxAngleDegrees)
            ?? defaults.maxAngleDegrees
        minimumCoverage = try container.decodeIfPresent(Float.self, forKey: .minimumCoverage)
            ?? defaults.minimumCoverage
        minimumDepthFraction = try container.decodeIfPresent(Float.self, forKey: .minimumDepthFraction)
            ?? defaults.minimumDepthFraction
    }
}

/// Geometry regularizers: anisotropy, scale cap and isolation pruning (Phase 4 P10).
public struct ArtifactRegularizerSettings: Codable, Sendable, Equatable {
    public var anisotropyWeight: Float = 0.1
    public var maxAxisRatio: Float = 10
    /// × max(scene radius, distance to the extent centre).
    public var scaleCapFraction: Float = 0.05
    public var isolationNeighbors = 4
    public var isolationFactor: Float = 8
    public var isolationStartIteration = 1_000
    public var isolationInterval = 500
    public var isolationMaxFraction: Float = 0.05
    public var isolationSampleCount = 2_048

    public init() {}

    /// Every key is optional; missing keys fall back to the defaults above.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = ArtifactRegularizerSettings()
        anisotropyWeight = try container.decodeIfPresent(Float.self, forKey: .anisotropyWeight)
            ?? defaults.anisotropyWeight
        maxAxisRatio = try container.decodeIfPresent(Float.self, forKey: .maxAxisRatio)
            ?? defaults.maxAxisRatio
        scaleCapFraction = try container.decodeIfPresent(Float.self, forKey: .scaleCapFraction)
            ?? defaults.scaleCapFraction
        isolationNeighbors = try container.decodeIfPresent(Int.self, forKey: .isolationNeighbors)
            ?? defaults.isolationNeighbors
        isolationFactor = try container.decodeIfPresent(Float.self, forKey: .isolationFactor)
            ?? defaults.isolationFactor
        isolationStartIteration = try container.decodeIfPresent(Int.self, forKey: .isolationStartIteration)
            ?? defaults.isolationStartIteration
        isolationInterval = try container.decodeIfPresent(Int.self, forKey: .isolationInterval)
            ?? defaults.isolationInterval
        isolationMaxFraction = try container.decodeIfPresent(Float.self, forKey: .isolationMaxFraction)
            ?? defaults.isolationMaxFraction
        isolationSampleCount = try container.decodeIfPresent(Int.self, forKey: .isolationSampleCount)
            ?? defaults.isolationSampleCount
    }
}

/// Dense depth-based point initialization (Phase 4 P8).
public struct DenseInitSettings: Codable, Sendable, Equatable {
    public var isEnabled = true
    public var maxPoints = 150_000
    public var edgeRatio: Float = 1.05
    public var consistencyTolerance: Float = 0.05
    public var includeSfMPoints = true

    public init() {}

    /// Every key is optional; missing keys fall back to the defaults above.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = DenseInitSettings()
        isEnabled = try container.decodeIfPresent(Bool.self, forKey: .isEnabled) ?? defaults.isEnabled
        maxPoints = try container.decodeIfPresent(Int.self, forKey: .maxPoints) ?? defaults.maxPoints
        edgeRatio = try container.decodeIfPresent(Float.self, forKey: .edgeRatio) ?? defaults.edgeRatio
        consistencyTolerance = try container.decodeIfPresent(Float.self, forKey: .consistencyTolerance)
            ?? defaults.consistencyTolerance
        includeSfMPoints = try container.decodeIfPresent(Bool.self, forKey: .includeSfMPoints)
            ?? defaults.includeSfMPoints
    }
}

/// Leave-one-out early stopping (Phase 4 P11).
public struct EarlyStoppingPolicy: Codable, Sendable, Equatable {
    public var isEnabled = true
    /// Below this many views the held-out view is not taken and no early stop runs.
    public var minimumViews = 4
    public var evalInterval = 250
    public var patience = 4
    public var minDelta: Double = 0.002
    public var minimumIterations = 1_500
    public var emaWeight: Double = 0.5

    public init() {}

    /// The policy a new `TrainingConfiguration` starts with: switched off, because the Mac-session
    /// measurements (2026-11-07) showed no reliable gain for twice the training time.
    public static var offByDefault: EarlyStoppingPolicy {
        var policy = EarlyStoppingPolicy()
        policy.isEnabled = false
        return policy
    }

    /// Every key is optional; missing keys fall back to the defaults above.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = EarlyStoppingPolicy()
        isEnabled = try container.decodeIfPresent(Bool.self, forKey: .isEnabled) ?? defaults.isEnabled
        minimumViews = try container.decodeIfPresent(Int.self, forKey: .minimumViews)
            ?? defaults.minimumViews
        evalInterval = try container.decodeIfPresent(Int.self, forKey: .evalInterval) ?? defaults.evalInterval
        patience = try container.decodeIfPresent(Int.self, forKey: .patience) ?? defaults.patience
        minDelta = try container.decodeIfPresent(Double.self, forKey: .minDelta) ?? defaults.minDelta
        minimumIterations = try container.decodeIfPresent(Int.self, forKey: .minimumIterations)
            ?? defaults.minimumIterations
        emaWeight = try container.decodeIfPresent(Double.self, forKey: .emaWeight) ?? defaults.emaWeight
    }
}

/// Per-image colour compensation (v1.2): a learned gain and bias per colour channel and training view, applied
/// to the render before the image loss. It absorbs exposure and white-balance changes between photos, so the
/// Gaussians do not smear them into the scene. Only the loss sees it; the stored model stays unchanged.
public struct AppearanceSettings: Codable, Sendable, Equatable {
    /// On by default (user decision 2026-11-08; +3.5 dB with exposure changes, neutral without). The `nonFiniteLoss`
    /// it exposed on real photos was a NaN in the SSIM backward pass, fixed the same day.
    public var isEnabled = true
    public var learningRate: Float = 2e-3
    /// Weight of the pull of every view's gain towards 1 and bias towards 0 (fixes the global gauge).
    public var identityWeight: Float = 0.01
    public var startIteration = 0

    public init() {}

    /// Every key is optional; missing keys fall back to the defaults above.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = AppearanceSettings()
        isEnabled = try container.decodeIfPresent(Bool.self, forKey: .isEnabled) ?? defaults.isEnabled
        learningRate = try container.decodeIfPresent(Float.self, forKey: .learningRate) ?? defaults.learningRate
        identityWeight = try container.decodeIfPresent(Float.self, forKey: .identityWeight)
            ?? defaults.identityWeight
        startIteration = try container.decodeIfPresent(Int.self, forKey: .startIteration)
            ?? defaults.startIteration
    }
}

/// MCMC densification (v1.2, after "3D Gaussian Splatting as Markov Chain Monte Carlo"): instead of clone and
/// split, dead Gaussians are relocated onto live ones, new Gaussians are added up to the budget, and the
/// positions get gradient-scaled noise. Replaces the classic densification when enabled.
public struct MCMCSettings: Codable, Sendable, Equatable {
    public var isEnabled = true
    public var startIteration = 500
    public var stopIteration = 25_000
    public var interval = 100
    /// Share of the current count added per interval, until `TrainingConfiguration.maxGaussians` is reached.
    public var growthFraction: Float = 0.05
    /// Opacity below which a Gaussian counts as dead and is relocated.
    public var deadOpacity: Float = 0.005
    public var noiseLearningRate: Float = 5e5
    public var opacityRegularizer: Float = 0.01
    public var scaleRegularizer: Float = 0.01

    public init() {}

    /// Every key is optional; missing keys fall back to the defaults above.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = MCMCSettings()
        isEnabled = try container.decodeIfPresent(Bool.self, forKey: .isEnabled) ?? defaults.isEnabled
        startIteration = try container.decodeIfPresent(Int.self, forKey: .startIteration)
            ?? defaults.startIteration
        stopIteration = try container.decodeIfPresent(Int.self, forKey: .stopIteration)
            ?? defaults.stopIteration
        interval = try container.decodeIfPresent(Int.self, forKey: .interval) ?? defaults.interval
        growthFraction = try container.decodeIfPresent(Float.self, forKey: .growthFraction)
            ?? defaults.growthFraction
        deadOpacity = try container.decodeIfPresent(Float.self, forKey: .deadOpacity) ?? defaults.deadOpacity
        noiseLearningRate = try container.decodeIfPresent(Float.self, forKey: .noiseLearningRate)
            ?? defaults.noiseLearningRate
        opacityRegularizer = try container.decodeIfPresent(Float.self, forKey: .opacityRegularizer)
            ?? defaults.opacityRegularizer
        scaleRegularizer = try container.decodeIfPresent(Float.self, forKey: .scaleRegularizer)
            ?? defaults.scaleRegularizer
    }
}

/// Anti-aliasing after Mip-Splatting (v1.2): a 3D smoothing filter limits every Gaussian to what the training
/// cameras can resolve, and a 2D box-like filter with compensation replaces the fixed 0.3 px dilation.
public struct AntiAliasingSettings: Codable, Sendable, Equatable {
    public var isEnabled = true
    /// Variance of the 3D filter, in squared pixels at the training resolution (Mip-Splatting: 0.2).
    public var filter3DVariance: Float = 0.2
    /// Variance added to the projected 2D covariance (Mip-Splatting: 0.1; the classic dilation is 0.3).
    public var filter2DVariance: Float = 0.1
    /// The per-Gaussian sampling rate is recomputed from the training cameras every this many iterations.
    public var samplingRateInterval = 100

    public init() {}

    /// Every key is optional; missing keys fall back to the defaults above.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = AntiAliasingSettings()
        isEnabled = try container.decodeIfPresent(Bool.self, forKey: .isEnabled) ?? defaults.isEnabled
        filter3DVariance = try container.decodeIfPresent(Float.self, forKey: .filter3DVariance)
            ?? defaults.filter3DVariance
        filter2DVariance = try container.decodeIfPresent(Float.self, forKey: .filter2DVariance)
            ?? defaults.filter2DVariance
        samplingRateInterval = try container.decodeIfPresent(Int.self, forKey: .samplingRateInterval)
            ?? defaults.samplingRateInterval
    }
}

/// Joint camera pose refinement (v1.2): every training camera gets a small learned rotation and translation
/// correction, optimized with the Gaussians. Helps poses that are slightly off (for example pose-free ones).
public struct PoseRefinementSettings: Codable, Sendable, Equatable {
    public var isEnabled = true
    /// Rotation learning rate in radians per Adam step.
    public var rotationLearningRate: Float = 2e-5
    /// Translation learning rate, multiplied by the scene radius.
    public var translationLearningRate: Float = 1e-4
    /// Poses stay fixed until the geometry has settled.
    public var startIteration = 1_000

    public init() {}

    /// Every key is optional; missing keys fall back to the defaults above.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = PoseRefinementSettings()
        isEnabled = try container.decodeIfPresent(Bool.self, forKey: .isEnabled) ?? defaults.isEnabled
        rotationLearningRate = try container.decodeIfPresent(Float.self, forKey: .rotationLearningRate)
            ?? defaults.rotationLearningRate
        translationLearningRate = try container.decodeIfPresent(Float.self, forKey: .translationLearningRate)
            ?? defaults.translationLearningRate
        startIteration = try container.decodeIfPresent(Int.self, forKey: .startIteration)
            ?? defaults.startIteration
    }
}

/// DropGaussian-style random dropout (experimental, v1.5 held-out investigation; off by default): in every training
/// iteration each Gaussian is left out of the forward and backward pass with probability p, and the opacity of the
/// remaining ones is scaled by 1 / (1 − p). p ramps linearly from 0 to `finalRate` over the run. Evaluation, previews
/// and exports never drop. `finalRate` 0 leaves every path unchanged. The CPU reference trainer does not implement it.
public struct GaussianDropoutSettings: Codable, Sendable, Equatable {
    public var finalRate: Float = 0

    public init() {}

    public var isEnabled: Bool { finalRate > 0 }

    /// Drop probability at `iteration` of a run of `iterations` (linear ramp, clamped to [0, 0.9]).
    public func rate(iteration: Int, iterations: Int) -> Float {
        guard finalRate > 0, iterations > 0 else { return 0 }
        let progress = Float(min(max(iteration, 0), iterations)) / Float(iterations)
        return min(max(finalRate * progress, 0), 0.9)
    }

    /// Every key is optional; missing keys fall back to the defaults above.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        finalRate = try container.decodeIfPresent(Float.self, forKey: .finalRate) ?? GaussianDropoutSettings().finalRate
    }
}

public struct TrainingConfiguration: Codable, Sendable, Equatable {
    /// Revision 1 is Phase 3 (no revision key). Revision 2 is Phase 4 (P14): a manifest
    /// without the key gets the new default regularizer set.
    public static let currentRevision = 2

    public var configurationRevision = TrainingConfiguration.currentRevision
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
    /// Phase 4 features switched on. User decision 2026-11-08: every regularizer is on by default, including scale
    /// cap and pseudo-views, although the synthetic benchmark measured both as neutral to harmful (decision 14).
    public static let defaultRegularizers: Set<RegularizerKind> = [
        .depthPearson, .depthL1, .anisotropy, .scaleCap, .isolationPrune, .pseudoViews
    ]
    public var regularizers: Set<RegularizerKind> = TrainingConfiguration.defaultRegularizers
    /// Leave-one-out: names of views excluded from training and used for evaluation.
    public var heldOutViews: [String] = []
    public var depthPrior = DepthPriorSettings()
    public var pseudoViews = PseudoViewSettings()
    public var artifacts = ArtifactRegularizerSettings()
    public var denseInit = DenseInitSettings()
    public var earlyStopping = EarlyStoppingPolicy.offByDefault
    /// v1.2 quality features: exposure compensation is on by default, the others stay off (see the decision log).
    public var appearance = AppearanceSettings()
    public var mcmc = MCMCSettings()
    public var antiAliasing = AntiAliasingSettings()
    public var poseRefinement = PoseRefinementSettings()
    /// Experimental DropGaussian dropout (see `GaussianDropoutSettings`). nil (the default) is off; optional so a
    /// configuration without it encodes exactly as before (manifest JSON and `OptimizerState.configurationHash`).
    public var dropout: GaussianDropoutSettings? = nil
    /// Iterations between autosaves of an interrupted run (v1.5). 0 turns autosave off. Not part of the
    /// configuration hash of an optimizer snapshot, so changing it keeps a snapshot usable.
    public var autosaveInterval = 2_000

    public init() {}

    /// Whether a Phase 4 feature is switched on.
    public func isEnabled(_ kind: RegularizerKind) -> Bool {
        regularizers.contains(kind)
    }

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
        // Migration (P14): a manifest without the revision key was written by Phase 3, which
        // always stored an empty regularizer set. Its stored value is ignored in favour of the
        // Phase 4 default set, and the decoded revision becomes the current one.
        if let storedRevision = try container.decodeIfPresent(Int.self, forKey: .configurationRevision) {
            configurationRevision = storedRevision
            regularizers = try container.decodeIfPresent(Set<RegularizerKind>.self, forKey: .regularizers)
                ?? defaults.regularizers
        } else {
            configurationRevision = TrainingConfiguration.currentRevision
            regularizers = defaults.regularizers
        }
        heldOutViews = try container.decodeIfPresent([String].self, forKey: .heldOutViews)
            ?? defaults.heldOutViews
        depthPrior = try container.decodeIfPresent(DepthPriorSettings.self, forKey: .depthPrior)
            ?? defaults.depthPrior
        pseudoViews = try container.decodeIfPresent(PseudoViewSettings.self, forKey: .pseudoViews)
            ?? defaults.pseudoViews
        artifacts = try container.decodeIfPresent(ArtifactRegularizerSettings.self, forKey: .artifacts)
            ?? defaults.artifacts
        denseInit = try container.decodeIfPresent(DenseInitSettings.self, forKey: .denseInit)
            ?? defaults.denseInit
        earlyStopping = try container.decodeIfPresent(EarlyStoppingPolicy.self, forKey: .earlyStopping)
            ?? defaults.earlyStopping
        appearance = try container.decodeIfPresent(AppearanceSettings.self, forKey: .appearance)
            ?? defaults.appearance
        mcmc = try container.decodeIfPresent(MCMCSettings.self, forKey: .mcmc) ?? defaults.mcmc
        antiAliasing = try container.decodeIfPresent(AntiAliasingSettings.self, forKey: .antiAliasing)
            ?? defaults.antiAliasing
        poseRefinement = try container.decodeIfPresent(PoseRefinementSettings.self, forKey: .poseRefinement)
            ?? defaults.poseRefinement
        dropout = try container.decodeIfPresent(GaussianDropoutSettings.self, forKey: .dropout)
        autosaveInterval = try container.decodeIfPresent(Int.self, forKey: .autosaveInterval)
            ?? defaults.autosaveInterval
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
