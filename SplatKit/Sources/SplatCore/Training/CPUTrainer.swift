import Foundation

// CPU reference training loop (Phase 3 Task 10, Phase 4 Task 11). Slow and deterministic on purpose:
// the Metal trainer mirrors its iteration order, schedules and densification boundaries.
//
// Interpretations of the plan, where it is literal:
// - `CPUStepResult.iteration` and `CPUTrainerError.nonFiniteLoss(iteration:)` both report the
//   trainer's `iteration` at the moment of the event: after a successful step it is the count of
//   completed steps; a failing step reports the index it would have had.
// - The scene extent also falls back to the dataset's point positions when the cameras do not
//   span a volume.
// - A failed step leaves `cloud`, `iteration`, the Adam state and the statistics untouched; only
//   the view sampler has advanced.
// - Phase 4 step order (Task 11, normative): 1 render (with depth when the view has a prior and a
//   depth term is on), 2 image and depth loss, 3 backward with statistics, 4 pseudo-view (its
//   gradient is added, no statistics), 5 anisotropy gradient, 6 Adam, 7 scale cap, 8 increment,
//   9 densification, 10 isolation pruning. With `regularizers == []` steps 1–10 reduce to the
//   Phase 3 order exactly.
// - Isolation pruning with no pruned Gaussian leaves the statistics untouched; with pruned
//   Gaussians they are reset, as after densification.

/// Errors of the CPU reference trainer.
public enum CPUTrainerError: Error, Equatable {
    /// The loss of the step at `iteration` was NaN or infinite.
    case nonFiniteLoss(iteration: Int)
    /// The dataset has no training views.
    case noViews
}

/// Phase 4 terms of one step (empty when no Phase 4 feature ran).
public struct DepthStepInfo: Sendable {
    /// Depth prior loss of the training view, or `nil` when the view was rendered without a prior.
    public var training: DepthLossResult?
    /// Depth loss of the pseudo-view of this step, or `nil` when no pseudo-view ran.
    public var pseudo: DepthLossResult?
    /// Gaussians removed by isolation pruning after this step.
    public var isolationPruned: Int

    public init(training: DepthLossResult? = nil, pseudo: DepthLossResult? = nil, isolationPruned: Int = 0) {
        self.training = training
        self.pseudo = pseudo
        self.isolationPruned = isolationPruned
    }
}

/// Outcome of one completed `CPUTrainer.step()`.
public struct CPUStepResult: Sendable {
    /// Trainer iteration after this step (the number of completed steps).
    public var iteration: Int
    /// Loss of the view rendered in this step, before the parameter update.
    public var loss: Double
    /// Densification applied after this step, or `nil` when no densification ran.
    public var densified: DensifyPlan?
    /// Phase 4 terms of this step.
    public var depth: DepthStepInfo

    public init(iteration: Int, loss: Double, densified: DensifyPlan?, depth: DepthStepInfo = DepthStepInfo()) {
        self.iteration = iteration
        self.loss = loss
        self.densified = densified
        self.depth = depth
    }
}

/// CPU reference for the training loop: render, loss, backward, Adam, and densification at
/// the policy's boundaries. Structure and order follow the Phase 3 plan (Task 10) and the
/// Phase 4 plan (Task 11) exactly.
public struct CPUTrainer: Sendable {
    /// Near plane of the depth terms, the same as `RasterSettings` default.
    static let depthNearPlane: Float = 0.01

    public private(set) var cloud: GaussianCloud
    /// Number of completed steps.
    public private(set) var iteration: Int
    public let sceneExtent: SceneExtent
    /// Pseudo-view pool, built in `init` when `.pseudoViews` is enabled.
    public let pseudoViews: [PseudoView]

    private let views: [TrainingView]
    private let configuration: TrainingConfiguration
    private var adam: AdamState
    private var statistics: DensificationStatistics
    private var sampler: ViewSampler

    public init(dataset: TrainingDataset,
                initialCloud: GaussianCloud,
                configuration: TrainingConfiguration) throws(CPUTrainerError) {
        guard !dataset.views.isEmpty else {
            throw .noViews
        }
        views = dataset.views
        self.configuration = configuration
        cloud = initialCloud
        iteration = 0
        sceneExtent = SceneExtent.estimate(cameraCenters: dataset.cameraCenters,
                                           points: dataset.points?.positions ?? [])
        if configuration.isEnabled(.pseudoViews) {
            pseudoViews = PseudoViews.make(dataset.views, settings: configuration.pseudoViews,
                                           near: CPUTrainer.depthNearPlane).views
        } else {
            pseudoViews = []
        }
        adam = AdamState(zeroLike: initialCloud)
        statistics = DensificationStatistics(count: initialCloud.count)
        sampler = ViewSampler(count: dataset.views.count, seed: configuration.seed)
    }

    /// One training iteration, in the normative order of the plan:
    /// render → loss → backward → statistics → pseudo-view → anisotropy → Adam → scale cap →
    /// increment → densify at boundaries → isolation pruning at boundaries.
    public mutating func step() throws(CPUTrainerError) -> CPUStepResult {
        let viewIndex = sampler.next()
        let view = views[viewIndex]
        let settings = RasterSettings(shDegreeLimit: configuration.activeSHDegree(iteration: iteration),
                                      background: configuration.background)
        var info = DepthStepInfo()

        // 1. Render, with depth when the view has a prior and a depth term is on.
        let depthRequested: Bool = (configuration.isEnabled(.depthPearson) || configuration.isEnabled(.depthL1))
            && iteration >= configuration.depthPrior.startIteration
        let rendered: RGBImage
        var renderedDepth: DepthRender?
        if depthRequested, view.depth != nil {
            let withDepth = CPURasterizer.renderWithDepth(cloud, camera: view.camera, settings: settings)
            rendered = withDepth.image
            renderedDepth = withDepth.depth
        } else {
            rendered = CPURasterizer.render(cloud, camera: view.camera, settings: settings)
        }

        // 2. Image loss, then the depth loss of the view.
        let loss = ImageLoss.evaluate(rendered: rendered, target: view.image,
                                      lambda: configuration.lambdaDSSIM)
        guard loss.loss.isFinite else {
            throw .nonFiniteLoss(iteration: iteration)
        }
        var depthGradient: [SIMD2<Float>]?
        if let renderedDepth, let prior = view.depth {
            let result = DepthLoss.evaluate(depth: renderedDepth, prior: prior, weights: trainingDepthWeights())
            guard result.loss.isFinite else {
                throw .nonFiniteLoss(iteration: iteration)
            }
            info.training = result
            depthGradient = result.gradient
        }

        // 3. Backward and densification statistics (Phase 3 rule).
        let backward = CPURasterizer.backwardWithScreenSpace(cloud, camera: view.camera,
                                                             settings: settings,
                                                             imageGradient: loss.gradient,
                                                             depthGradient: depthGradient)
        var gradients = backward.gradients
        let policy = configuration.densification
        if iteration < policy.stopIteration {
            statistics.accumulate(mean2D: backward.mean2D, visible: backward.visible,
                                  width: view.camera.width, height: view.camera.height)
        }

        // 4. Pseudo-view: depth-only gradient, added to this step's gradients, no statistics.
        let pseudoSettings = configuration.pseudoViews
        if configuration.isEnabled(.pseudoViews), !pseudoViews.isEmpty,
           iteration >= pseudoSettings.startIteration, pseudoSettings.interval > 0,
           (iteration - pseudoSettings.startIteration) % pseudoSettings.interval == 0 {
            let index = PseudoViews.index(iteration: iteration, count: pseudoViews.count, seed: configuration.seed)
            let pseudo = pseudoViews[index]
            let pseudoRender = CPURasterizer.renderWithDepth(cloud, camera: pseudo.camera, settings: settings)
            let weights = DepthLossWeights(pearson: pseudoSettings.weight, l1: 0,
                                           alphaMask: configuration.depthPrior.alphaMask,
                                           minimumPixels: configuration.depthPrior.minimumPixels,
                                           nearPlane: CPUTrainer.depthNearPlane)
            let result = DepthLoss.evaluate(depth: pseudoRender.depth, prior: pseudo.prior, weights: weights)
            guard result.loss.isFinite else {
                throw .nonFiniteLoss(iteration: iteration)
            }
            let zeroImageGradient = [SIMD3<Float>](repeating: .zero, count: pseudo.camera.width * pseudo.camera.height)
            let pseudoBackward = CPURasterizer.backwardWithScreenSpace(cloud, camera: pseudo.camera,
                                                                       settings: settings,
                                                                       imageGradient: zeroImageGradient,
                                                                       depthGradient: result.gradient)
            CPUTrainer.addGradients(&gradients, pseudoBackward.gradients)
            info.pseudo = result
        }

        // 5. Anisotropy gradient on the log-scales (after backward, before Adam).
        if configuration.isEnabled(.anisotropy) {
            Regularizers.addAnisotropyGradient(cloud,
                                               weight: configuration.artifacts.anisotropyWeight,
                                               maxRatio: configuration.artifacts.maxAxisRatio,
                                               into: &gradients)
        }

        // 6. Adam.
        let rates = configuration.learningRates(iteration: iteration,
                                                sceneRadius: sceneExtent.radius)
        cloud = CPUAdam.step(cloud, gradients: gradients, state: &adam,
                             rates: rates, hyper: configuration.adam)

        // 7. Scale cap projection (after Adam, no gradient).
        if configuration.isEnabled(.scaleCap) {
            Regularizers.applyScaleCap(&cloud, fraction: configuration.artifacts.scaleCapFraction,
                                       extent: sceneExtent)
        }

        // 8. Increment.
        iteration += 1

        // 9. Densification at its boundaries.
        var densified: DensifyPlan?
        if policy.isEnabled,
           iteration >= policy.startIteration,
           iteration < policy.stopIteration,
           iteration % policy.interval == 0 {
            let plan = Densification.plan(cloud,
                                          statistics: statistics,
                                          policy: policy,
                                          sceneRadius: sceneExtent.radius,
                                          iteration: iteration,
                                          maxGaussians: configuration.maxGaussians)
            cloud = Densification.apply(plan, to: cloud, state: &adam, policy: policy,
                                        seed: UInt32(truncatingIfNeeded: configuration.seed),
                                        iteration: iteration)
            statistics = DensificationStatistics(count: cloud.count)
            densified = plan
        }

        // 10. Isolation pruning at its boundaries (discrete decision, then compaction).
        let artifacts = configuration.artifacts
        if configuration.isEnabled(.isolationPrune),
           iteration >= artifacts.isolationStartIteration,
           artifacts.isolationInterval > 0,
           (iteration - artifacts.isolationStartIteration) % artifacts.isolationInterval == 0,
           iteration < configuration.iterations {
            info.isolationPruned = pruneIsolated(iteration: iteration)
        }

        return CPUStepResult(iteration: iteration, loss: loss.loss, densified: densified, depth: info)
    }

    // MARK: - Helpers

    /// Depth loss weights of the training view: each term follows its regularizer switch.
    private func trainingDepthWeights() -> DepthLossWeights {
        var weights = DepthLossWeights(settings: configuration.depthPrior, nearPlane: CPUTrainer.depthNearPlane)
        weights.pearson = configuration.isEnabled(.depthPearson) ? configuration.depthPrior.pearsonWeight : 0
        weights.l1 = configuration.isEnabled(.depthL1) ? configuration.depthPrior.l1Weight : 0
        return weights
    }

    /// Removes isolated Gaussians and compacts the parameters and Adam moments. Returns the number
    /// removed; with none removed nothing changes.
    private mutating func pruneIsolated(iteration: Int) -> Int {
        let artifacts = configuration.artifacts
        let count = cloud.count
        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(count)
        for i in 0..<count {
            positions.append(SIMD3<Float>(cloud.positions[3 * i], cloud.positions[3 * i + 1],
                                          cloud.positions[3 * i + 2]))
        }
        let mask = IsolationPruning.pruneMask(positions: positions,
                                              neighbors: artifacts.isolationNeighbors,
                                              factor: artifacts.isolationFactor,
                                              maxFraction: artifacts.isolationMaxFraction,
                                              sampleCount: artifacts.isolationSampleCount,
                                              seed: configuration.seed ^ UInt64(iteration))
        var pruned = 0
        var actions: [DensifyAction] = []
        actions.reserveCapacity(count)
        for flag in mask {
            if flag {
                pruned += 1
                actions.append(.prune)
            } else {
                actions.append(.keep)
            }
        }
        guard pruned > 0 else { return 0 }
        let plan = DensifyPlan(actions: actions, outputCount: count - pruned, pruned: pruned,
                               cloned: 0, split: 0, demoted: 0, keptAll: false)
        cloud = Densification.apply(plan, to: cloud, state: &adam, policy: configuration.densification,
                                    seed: UInt32(truncatingIfNeeded: configuration.seed), iteration: iteration)
        statistics = DensificationStatistics(count: cloud.count)
        return pruned
    }

    /// Adds `extra` into `total`, element by element.
    private static func addGradients(_ total: inout GaussianGradients, _ extra: GaussianGradients) {
        precondition(total.positions.count == extra.positions.count
                     && total.logScales.count == extra.logScales.count
                     && total.rotations.count == extra.rotations.count
                     && total.opacityLogits.count == extra.opacityLogits.count
                     && total.sh.count == extra.sh.count,
                     "gradient buffers must match")
        for i in total.positions.indices {
            total.positions[i] += extra.positions[i]
        }
        for i in total.logScales.indices {
            total.logScales[i] += extra.logScales[i]
        }
        for i in total.rotations.indices {
            total.rotations[i] += extra.rotations[i]
        }
        for i in total.opacityLogits.indices {
            total.opacityLogits[i] += extra.opacityLogits[i]
        }
        for i in total.sh.indices {
            total.sh[i] += extra.sh[i]
        }
    }
}
