import Foundation

// CPU reference training loop (Phase 3 Task 10). Slow and deterministic on purpose: the Metal
// trainer mirrors its iteration order, schedules and densification boundaries.
//
// Interpretations of the plan, where it is literal:
// - `CPUStepResult.iteration` and `CPUTrainerError.nonFiniteLoss(iteration:)` both report the
//   trainer's `iteration` at the moment of the event: after a successful step it is the count of
//   completed steps; a failing step reports the index it would have had.
// - The scene extent also falls back to the dataset's point positions when the cameras do not
//   span a volume.
// - A failed step leaves `cloud`, `iteration`, the Adam state and the statistics untouched; only
//   the view sampler has advanced.

/// Errors of the CPU reference trainer.
public enum CPUTrainerError: Error, Equatable {
    /// The loss of the step at `iteration` was NaN or infinite.
    case nonFiniteLoss(iteration: Int)
    /// The dataset has no training views.
    case noViews
}

/// Outcome of one completed `CPUTrainer.step()`.
public struct CPUStepResult: Sendable {
    /// Trainer iteration after this step (the number of completed steps).
    public var iteration: Int
    /// Loss of the view rendered in this step, before the parameter update.
    public var loss: Double
    /// Densification applied after this step, or `nil` when no densification ran.
    public var densified: DensifyPlan?

    public init(iteration: Int, loss: Double, densified: DensifyPlan?) {
        self.iteration = iteration
        self.loss = loss
        self.densified = densified
    }
}

/// CPU reference for the training loop: render, loss, backward, Adam, and densification at
/// the policy's boundaries. Structure and order follow the Phase 3 plan (Task 10) exactly.
public struct CPUTrainer: Sendable {
    public private(set) var cloud: GaussianCloud
    /// Number of completed steps.
    public private(set) var iteration: Int
    public let sceneExtent: SceneExtent

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
        adam = AdamState(zeroLike: initialCloud)
        statistics = DensificationStatistics(count: initialCloud.count)
        sampler = ViewSampler(count: dataset.views.count, seed: configuration.seed)
    }

    /// One training iteration, in the normative order of the plan:
    /// render → loss → backward → statistics → Adam → increment → densify at boundaries.
    public mutating func step() throws(CPUTrainerError) -> CPUStepResult {
        let viewIndex = sampler.next()
        let view = views[viewIndex]
        let settings = RasterSettings(shDegreeLimit: configuration.activeSHDegree(iteration: iteration),
                                      background: configuration.background)

        let rendered = CPURasterizer.render(cloud, camera: view.camera, settings: settings)
        let loss = ImageLoss.evaluate(rendered: rendered, target: view.image,
                                      lambda: configuration.lambdaDSSIM)
        guard loss.loss.isFinite else {
            throw .nonFiniteLoss(iteration: iteration)
        }

        let backward = CPURasterizer.backwardWithScreenSpace(cloud, camera: view.camera,
                                                             settings: settings,
                                                             imageGradient: loss.gradient)
        let policy = configuration.densification
        if iteration < policy.stopIteration {
            statistics.accumulate(mean2D: backward.mean2D, visible: backward.visible,
                                  width: view.camera.width, height: view.camera.height)
        }

        let rates = configuration.learningRates(iteration: iteration,
                                                sceneRadius: sceneExtent.radius)
        cloud = CPUAdam.step(cloud, gradients: backward.gradients, state: &adam,
                             rates: rates, hyper: configuration.adam)
        iteration += 1

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

        return CPUStepResult(iteration: iteration, loss: loss.loss, densified: densified)
    }
}
