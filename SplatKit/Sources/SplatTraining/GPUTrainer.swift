import Foundation
import Metal
import SplatCore
import SplatIO
import SplatMetal

// Metal training loop (Phase 3 Task 15, Phase 4 Task 17). Mirrors `CPUTrainer.step` in SplatCore in order and in
// arithmetic. Phase 4 step order (normative, Task 11 / Task 17), with the GPU encode it maps to:
//   1 sample a view; forward render (depth mode only when the view has a prior and iteration ≥ depth start)
//   2 image loss (`imageLoss.encode`); depth loss (`.pixelGradient` terms)
//   3 backward with statistics (`encodeBackward`, depth gradient passed only for depth-mode views)
//   4 pseudo-views (Task 18): on a pseudo iteration, steps 1–3 are committed first; then a pseudo forward, its depth
//     loss (slot 1), a backward that ACCUMULATES with a zero image gradient and no statistics, and steps 5–7
//   5 anisotropy gradient (`.parameterGradient` terms)
//   6 Adam (`adam.encode`)
//   7 scale cap (`.postOptimizer` terms)
//   8 increment
//   9 densification at boundaries (after a wait, synchronous, as in Phase 3)
//   10 isolation pruning at boundaries (after a wait; discrete decision on the CPU, then compaction)
// Steps 1–8 of one iteration go into ONE command buffer, the rasterize command buffer B returned by
// `encodeForward`. Steps 9 and 10 run only at a wait point, when the GPU is idle with respect to every buffer
// they touch.
//
// Command-buffer structure per iteration:
// - `encodeForward` runs projection and the tile-count scan in their own command buffer, waited on by the
//   renderer (the CPU needs the intersection total). It returns B uncommitted.
// - B receives steps 2–7. It is committed and NOT waited on. The next iteration's projection command buffer is
//   committed after B on the same queue, so GPU order is preserved, and its CPU wait also bounds the in-flight work.
// - Each step starts by synchronizing with the previous step (`synchronize`, right after its projection returns).
//   That wait is cheap: the projection was queued behind the previous step's B and depends on Adam's writes, so B has
//   effectively finished. It checks B's status and validates the previous step's image and depth losses, throwing
//   `.nonFiniteLoss(iteration: k-1)` before the current loss encode overwrites the loss state.
// - The CPU also waits at: a log step, the last iteration, a densify boundary, an isolation boundary, `snapshot` and
//   `evaluatePSNR`. The current step's losses are validated there as well and reported in `StepReport`.
//
// Wait points and buffer lifetimes (Task 17):
// - Every CPU access of a GPU buffer happens after `synchronize()`. Isolation pruning reads the positions, decides on
//   the CPU, compacts the parameters and both Adam moments with the same mask, then resets the statistics in its own
//   command buffer that is waited on before the function returns.
// - Compaction keeps the capacity, so gradient buffers and statistics need no reallocation. Densification may grow the
//   capacity, which replaces the parameters, moments and statistics: the gradient buffers are then rebuilt (as in
//   Phase 3). Depth buffers and the image gradient are sized by pixels, not by Gaussians, and are never replaced
//   by a Gaussian-count change.
// - The shared renderer projection buffers are used by one view at a time: the view's forward and backward are both
//   encoded into B, which is committed before the next view's `encodeForward` (see `SplatRenderer.encodeForward`).
// - `GPUImageLoss` and `GPUDepthLoss` keep one scratch state each. Their values are read only after the wait that
//   follows the command buffer that wrote them.
//
// Decisions and interpretations (see also the comments at each site):
// - Any error thrown by `step`, `snapshot`, `evaluatePSNR` or densification makes the trainer fatal: the same error
//   is thrown by every later call. The model may be partially updated.
// - The densify seed is `UInt32(truncatingIfNeeded: seed)`; the densifier adds the iteration itself.
// - Phase 4 terms follow `configuration.regularizers` (`RegularizerKind`). With `regularizers == []` the step is
//   exactly the Phase 3 step: no depth-mode forward, no extra encode, no isolation, no statistics change.
// - The memory budget reserves the depth buffers for every training view, even when no view has a prior. This is a
//   conservative constant, not a per-view allocation.

/// Outcome of one `GPUTrainer.step()`.
public struct StepReport: Sendable {
    /// Completed iterations after this step (the trainer's `iteration`).
    public var iteration: Int
    /// Loss of this step's view. Non-nil when the step synchronized with the GPU at a log, last, densify or
    /// isolation boundary, nil otherwise.
    public var loss: ImageLossValue?
    /// Gaussians in the view's frustum with at least one tile intersection.
    public var visibleGaussians: Int
    /// Densification applied after this step, or nil.
    public var densify: DensifySummary?
    /// Depth loss of this step's view, read at the same wait points as `loss`. Nil when the view was
    /// rendered without a depth prior, or when the step did not synchronize.
    public var depthLoss: DepthLossValue?
    /// Gaussians removed by isolation pruning after this step. Nil when no isolation boundary ran.
    public var isolationPruned: Int?
    /// Depth loss of this iteration's pseudo-view, read at the same wait points. Nil when no pseudo-view ran.
    public var pseudoDepthLoss: DepthLossValue?

    public init(iteration: Int, loss: ImageLossValue?, visibleGaussians: Int, densify: DensifySummary?,
                depthLoss: DepthLossValue? = nil, isolationPruned: Int? = nil, pseudoDepthLoss: DepthLossValue? = nil) {
        self.iteration = iteration
        self.loss = loss
        self.visibleGaussians = visibleGaussians
        self.densify = densify
        self.depthLoss = depthLoss
        self.isolationPruned = isolationPruned
        self.pseudoDepthLoss = pseudoDepthLoss
    }
}

/// Synchronous Metal trainer. Not `Sendable`: it owns Metal objects and must stay in one isolation
/// domain (`TrainingSession` is the only owner in the app). One `step()` encodes one iteration.
public final class GPUTrainer {
    /// Near plane of the depth terms, the same as the `RasterSettings` default and `CPUTrainer`.
    static let depthNearPlane: Float = 0.01

    /// Budget cap on the Gaussian count: `TrainingMemoryBudget.maxGaussians`.
    public let maxGaussians: Int
    public let sceneExtent: SceneExtent
    /// Completed iterations.
    public private(set) var iteration: Int = 0
    /// Live Gaussians.
    public var gaussianCount: Int { store.count }
    /// Extra terms, encoded after the built-in Phase 4 terms at their stages. Empty by default.
    public var auxiliaryLosses: [any AuxiliaryLoss] = [] {
        didSet { rebuildTermCache() }
    }
    /// Per-stage term lists (built-in terms first, then `auxiliaryLosses`, in array order). Rebuilt only when the
    /// term sets change, not on every step.
    private var termsByStage: [AuxiliaryStage: [any AuxiliaryLoss]] = [:]

    private let context: MetalContext
    private let configuration: TrainingConfiguration
    private let views: [TrainingView]
    private let targets: [MTLBuffer]
    private var sampler: ViewSampler
    private let store: GaussianBuffers
    private let adamState: AdamBuffers
    /// Replaced (not resized) when densification grows the capacity.
    private var gradients: GradientBuffers
    private let statistics: DensificationStatisticsBuffers
    private let renderer: SplatRenderer
    private let imageLoss: GPUImageLoss
    private let adam: GPUAdam
    private let densifier: GPUDensifier
    /// Built-in Phase 4 terms, in the order of `configuration`: depth loss, anisotropy, scale cap.
    private let phase4Terms: [any AuxiliaryLoss]
    /// The depth term, when the run has one. Its value is read at the wait points.
    private let depthTerm: DepthLossTerm?
    /// Pseudo-view pool (P9), built once in `init` from the metric views; empty unless `.pseudoViews` is on.
    private let pseudoPool: [PseudoView]
    /// Uploaded depth prior of each pseudo-view, in pool order.
    private let pseudoPriors: [GPUDepthPrior]
    /// The depth loss kernel: slot 0 for the training views (through `depthTerm`), slot 1 for pseudo-views.
    private let depthLoss: GPUDepthLoss?
    /// Per-view colour compensation (v1.2). Nil unless `configuration.appearance.isEnabled`.
    let appearance: GPUAppearance?
    /// Camera pose refinement (v1.2), nil unless `configuration.poseRefinement.isEnabled` (GPUPoseRefiner.swift).
    private let poseRefiner: GPUPoseRefiner?
    /// The training cameras in view order. Pose corrections are folded into these; without refinement they equal the
    /// dataset's cameras. Pseudo-views keep their own cameras.
    private var cameras: [Camera]
    /// Grow-only, zero-filled dL/drendered for pseudo-view backward passes. Only read by the GPU, so it stays zero.
    private var zeroImageGradient: MTLBuffer?
    /// True from a pseudo-view encode until the trainer reads its value at a wait point.
    private var pendingPseudoLoss = false
    /// One rgba32Float render target per distinct image size, created on first use. N distinct sizes
    /// allocate N textures, which `TrainingMemoryBudget.reservedImageBytes` does not fully count (it
    /// reserves one largest-view render target). Known gap for Task 16 and later: cap or share the
    /// targets once the view sizes are known to vary.
    private var renderTargets: [SIMD2<Int>: MTLTexture] = [:]
    /// Grow-only dL/drendered buffer, shared by the views.
    private var imageGradient: MTLBuffer?
    /// Grow-only `float2` per pixel: the depth forward output `(Draw, A)` and the depth gradient. Used only by views
    /// with a depth prior.
    private var renderedDepthBuffer: MTLBuffer?
    private var depthGradientBuffer: MTLBuffer?
    /// Adam steps taken before this trainer's first iteration. Zero for a fresh run; a continuation round restored
    /// from an optimizer state starts from the lineage's count (`restore(from:continuing:)`). The step of iteration
    /// `index` is `adamStepBase + index + 1`.
    private var adamStepBase = 0
    /// Anti-aliasing (Mip-Splatting). The sampling-rate kernel exists only when `antiAliasing.isEnabled`; the
    /// buffer holds ν per Gaussian and is shared with `renderer.samplingRateBuffer`.
    private let samplingRateKernel: GPUSamplingRate?
    private var samplingRateBuffer: MTLBuffer?
    /// True when ν must be recomputed before the next forward pass (start, after densification and pruning).
    private var samplingRatesStale = true
    /// Command buffers committed since the last wait point.
    private var pending: [MTLCommandBuffer] = []
    /// 0-based index of the most recent step whose losses have not been validated yet.
    private var unvalidatedLossStep: Int?
    /// Set by the first error; every later call throws it.
    private var fatal: TrainingError?

    /// Buffers of one depth-mode view. Both hold `float2` per pixel of the view.
    private struct DepthBuffers {
        let rendered: MTLBuffer
        let gradient: MTLBuffer
    }

    /// Losses read at a wait point. Both are nil when the step was already validated.
    private struct LossReadout {
        var image: ImageLossValue?
        var depth: DepthLossValue?
        var pseudoDepth: DepthLossValue?
    }

    /// - Precondition: `initialCloud.shDegree == configuration.shDegree`.
    /// - Throws: `.noViews`, `.initialCloudExceedsBudget` when the cloud is above the cap, `.metal`.
    public convenience init(context: MetalContext, dataset: TrainingDataset, initialCloud: GaussianCloud,
                            configuration: TrainingConfiguration) throws(TrainingError) {
        try self.init(context: context, dataset: dataset, initialCloud: initialCloud,
                      configuration: configuration, capacityOverride: nil)
    }

    /// Internal entry point. `capacityOverride` replaces the D2 initial capacity; it exists so tests
    /// can force a densify that grows the capacity, which the D2 default rarely reaches in small scenes.
    init(context: MetalContext, dataset: TrainingDataset, initialCloud: GaussianCloud,
         configuration: TrainingConfiguration, capacityOverride: Int?) throws(TrainingError) {
        guard !dataset.views.isEmpty else {
            throw .noViews
        }
        precondition(initialCloud.shDegree == configuration.shDegree,
                     "initial cloud SH degree \(initialCloud.shDegree) does not match the configuration")

        // Pseudo-view pool (P9), built once from the metric views; empty unless `.pseudoViews` is on.
        let pseudoPool: [PseudoView] = configuration.isEnabled(.pseudoViews)
            ? PseudoViews.make(dataset.views, settings: configuration.pseudoViews, near: GPUTrainer.depthNearPlane).views
            : []
        let pseudoSizes: [SIMD2<Int>] = pseudoPool.map { SIMD2<Int>($0.camera.width, $0.camera.height) }
        let wantsDepthLoss: Bool = configuration.isEnabled(.depthPearson) || configuration.isEnabled(.depthL1)
        let hasTrainingPrior: Bool = dataset.views.contains(where: { $0.depth != nil })
        let hasDepthTerm: Bool = wantsDepthLoss && hasTrainingPrior

        // Budget (D13). Each view keeps its own size; the reserved image bytes use the largest view.
        // Depth buffers are reserved only when a depth term or the pseudo-view pool exists, so a run with
        // `regularizers == []` reserves exactly the Phase 3 bytes.
        let viewSizes: [SIMD2<Int>] = dataset.views.map { SIMD2<Int>($0.camera.width, $0.camera.height) }
        var reservedBytes: Int = TrainingMemoryBudget.reservedImageBytes(viewSizes: viewSizes)
        if hasDepthTerm || !pseudoPool.isEmpty {
            reservedBytes += TrainingMemoryBudget.reservedDepthBytes(viewSizes: viewSizes,
                                                                     pseudoViewSizes: pseudoSizes)
        }
        let budget = TrainingMemoryBudget.maxGaussians(
            workingSetBytes: context.device.recommendedMaxWorkingSetSize,
            shDegree: configuration.shDegree,
            reservedBytes: reservedBytes,
            configuredMax: configuration.maxGaussians)
        guard initialCloud.count <= budget else {
            throw .initialCloudExceedsBudget(count: initialCloud.count, cap: budget)
        }

        // Initial capacity per decision D2: min(budget, max(2·n0, 65 536)), never below n0 (the guard above
        // makes n0 ≤ budget). Densification grows it later (GPUDensifier.densify).
        let d2Capacity: Int = min(budget, max(2 * initialCloud.count, 65_536))
        let capacity: Int = max(initialCloud.count, capacityOverride ?? d2Capacity)
        let shDegree = configuration.shDegree
        let store = try metalCall { () throws(MetalError) -> GaussianBuffers in
            try GaussianBuffers(context: context, cloud: initialCloud, capacity: capacity)
        }
        let adamState = try metalCall { () throws(MetalError) -> AdamBuffers in
            try AdamBuffers(context: context, capacity: capacity, shDegree: shDegree)
        }
        let gradients = try metalCall { () throws(MetalError) -> GradientBuffers in
            try GradientBuffers(context: context, count: capacity, shDegree: shDegree)
        }
        let statistics = try metalCall { () throws(MetalError) -> DensificationStatisticsBuffers in
            try DensificationStatisticsBuffers(context: context, capacity: capacity)
        }
        let renderer = try metalCall { () throws(MetalError) -> SplatRenderer in
            try SplatRenderer(context: context, gaussians: store)
        }
        let imageLoss = try metalCall { () throws(MetalError) -> GPUImageLoss in
            try GPUImageLoss(context: context)
        }
        let adam = try metalCall { () throws(MetalError) -> GPUAdam in
            try GPUAdam(context: context)
        }
        let densifier = try metalCall { () throws(MetalError) -> GPUDensifier in
            try GPUDensifier(context: context)
        }
        let regularizers = try metalCall { () throws(MetalError) -> GPURegularizers in
            try GPURegularizers(context: context)
        }
        var samplingRateKernel: GPUSamplingRate?
        if configuration.antiAliasing.isEnabled {
            samplingRateKernel = try metalCall { () throws(MetalError) -> GPUSamplingRate in
                try GPUSamplingRate(context: context)
            }
        }
        var targets: [MTLBuffer] = []
        targets.reserveCapacity(dataset.views.count)
        for view in dataset.views {
            let target = try metalCall { () throws(MetalError) -> MTLBuffer in
                try imageLoss.makeTarget(view.image)
            }
            targets.append(target)
        }

        // Phase 4 terms (P12), built from the switches. The depth term exists only when a depth switch is on and
        // at least one view has a prior; its priors are uploaded once, one entry per view.
        // One GPUDepthLoss serves the training views (slot 0) and the pseudo-views (slot 1).
        var depthLossObject: GPUDepthLoss?
        if hasDepthTerm || !pseudoPool.isEmpty {
            depthLossObject = try metalCall { () throws(MetalError) -> GPUDepthLoss in
                try GPUDepthLoss(context: context)
            }
        }
        var pseudoPriors: [GPUDepthPrior] = []
        if let depthLossObject, !pseudoPool.isEmpty {
            for pseudo in pseudoPool {
                let uploaded = try metalCall { () throws(MetalError) -> GPUDepthPrior in
                    try depthLossObject.makePrior(pseudo.prior)
                }
                pseudoPriors.append(uploaded)
            }
        }

        // Appearance compensation (v1.2): one gain and bias per training view, created only when enabled.
        var appearanceModel: GPUAppearance?
        if configuration.appearance.isEnabled {
            appearanceModel = try metalCall { () throws(MetalError) -> GPUAppearance in
                try GPUAppearance(context: context, viewCount: dataset.views.count)
            }
        }

        // Camera pose refinement (v1.2): one pose gradient sink per training view, created only when enabled.
        var poseRefiner: GPUPoseRefiner?
        if configuration.poseRefinement.isEnabled {
            poseRefiner = try metalCall { () throws(MetalError) -> GPUPoseRefiner in
                try GPUPoseRefiner(context: context, viewCount: dataset.views.count,
                                   settings: configuration.poseRefinement, hyper: configuration.adam)
            }
        }

        var phase4Terms: [any AuxiliaryLoss] = []
        var depthTerm: DepthLossTerm?
        if hasDepthTerm, let depthLoss = depthLossObject {
            var priors: [GPUDepthPrior?] = []
            priors.reserveCapacity(dataset.views.count)
            for view in dataset.views {
                if let prior = view.depth {
                    let uploaded = try metalCall { () throws(MetalError) -> GPUDepthPrior in
                        try depthLoss.makePrior(prior)
                    }
                    priors.append(uploaded)
                } else {
                    priors.append(nil)
                }
            }
            let depthWeights = DepthLossWeights(
                pearson: configuration.isEnabled(.depthPearson) ? configuration.depthPrior.pearsonWeight : 0,
                l1: configuration.isEnabled(.depthL1) ? configuration.depthPrior.l1Weight : 0,
                alphaMask: configuration.depthPrior.alphaMask,
                minimumPixels: configuration.depthPrior.minimumPixels,
                nearPlane: GPUTrainer.depthNearPlane)
            let term = DepthLossTerm(loss: depthLoss, priors: priors, weights: depthWeights)
            depthTerm = term
            phase4Terms.append(term)

            // v1.5 normal prior: after the depth term (it adds into the depth gradient the depth term overwrote),
            // only for views whose depth prior carries normals, and only when switched on.
            let hasNormals: Bool = dataset.views.contains(where: { $0.depth?.normals != nil })
            if configuration.normalPrior.isEnabled && hasNormals {
                let normalLoss = try metalCall { () throws(MetalError) -> GPUNormalLoss in
                    try GPUNormalLoss(context: context)
                }
                var normalPriors: [GPUNormalPrior?] = []
                normalPriors.reserveCapacity(dataset.views.count)
                for view in dataset.views {
                    if let prior = view.depth, let normals = prior.normals {
                        let uploaded = try metalCall { () throws(MetalError) -> GPUNormalPrior in
                            try normalLoss.makePrior(normals: normals, width: prior.width, height: prior.height)
                        }
                        normalPriors.append(uploaded)
                    } else {
                        normalPriors.append(nil)
                    }
                }
                let normalWeights = NormalLossWeights(weight: configuration.normalPrior.weight,
                                                      alphaMask: configuration.depthPrior.alphaMask,
                                                      minimumPixels: configuration.depthPrior.minimumPixels)
                phase4Terms.append(NormalLossTerm(loss: normalLoss, priors: normalPriors, weights: normalWeights))
            }
        }
        if configuration.isEnabled(.anisotropy) {
            phase4Terms.append(AnisotropyTerm(regularizers: regularizers,
                                              weight: configuration.artifacts.anisotropyWeight,
                                              maxRatio: configuration.artifacts.maxAxisRatio))
        }
        if configuration.isEnabled(.scaleCap) {
            phase4Terms.append(ScaleCapTerm(regularizers: regularizers,
                                            fraction: configuration.artifacts.scaleCapFraction))
        }
        // MCMC per-step terms (v1.2): empty unless `mcmc.isEnabled` (MCMCTerms.swift).
        let mcmcTerms = try MCMCTerms.make(context: context, configuration: configuration)
        phase4Terms.append(contentsOf: mcmcTerms)

        self.context = context
        self.configuration = configuration
        self.views = dataset.views
        self.targets = targets
        self.sampler = ViewSampler(count: dataset.views.count, seed: configuration.seed)
        self.store = store
        self.adamState = adamState
        self.gradients = gradients
        self.statistics = statistics
        self.renderer = renderer
        self.imageLoss = imageLoss
        self.adam = adam
        self.densifier = densifier
        self.samplingRateKernel = samplingRateKernel
        self.phase4Terms = phase4Terms
        self.depthTerm = depthTerm
        self.maxGaussians = budget
        // Same scene extent as CPUTrainer: camera centres, falling back to the dataset points.
        self.sceneExtent = SceneExtent.estimate(cameraCenters: dataset.cameraCenters,
                                                points: dataset.points?.positions ?? [])
        self.pseudoPool = pseudoPool
        self.pseudoPriors = pseudoPriors
        self.depthLoss = depthLossObject
        self.appearance = appearanceModel
        self.poseRefiner = poseRefiner
        self.cameras = dataset.views.map { $0.camera }
        rebuildTermCache()
    }

    /// One training iteration. Does not wait for the GPU unless a boundary requires it (see the file
    /// comment). Throws `.nonFiniteLoss` with the index of the step whose loss was validated.
    @discardableResult
    public func step() throws(TrainingError) -> StepReport {
        try guarded { () throws(TrainingError) -> StepReport in
            try self.advance()
        }
    }

    /// Mean PSNR over the training views, rendered with the full SH degree. Waits for the GPU.
    public func evaluatePSNR() throws(TrainingError) -> Double {
        try guarded { () throws(TrainingError) -> Double in
            try self.measurePSNR()
        }
    }

    /// The first `gaussianCount` Gaussians as a CPU cloud. Waits for the GPU first, and throws if the
    /// most recent step's loss is non-finite.
    /// With anti-aliasing on, the 3D filter is baked into the returned scales and opacities (logScale = log s',
    /// opacity = o · Π s/s'), using the current sampling rates, so exporters and the viewer see one consistent model.
    public func snapshot() throws(TrainingError) -> GaussianCloud {
        try guarded { () throws(TrainingError) -> GaussianCloud in
            _ = try self.synchronize()
            let cloud = self.store.snapshot()
            return try self.bakedSnapshot(cloud)
        }
    }

    /// Learned gain and bias of every training view, in view order. Identity when appearance is off. Waits for the GPU.
    public func appearanceParameters() throws(TrainingError) -> [AppearanceParameters] {
        try guarded { () throws(TrainingError) -> [AppearanceParameters] in
            _ = try self.synchronize()
            guard let appearance = self.appearance else {
                return [AppearanceParameters](repeating: .identity, count: self.views.count)
            }
            return appearance.readParameters()
        }
    }

    // MARK: - Iteration

    private func advance() throws(TrainingError) -> StepReport {
        let index = iteration
        let viewIndex = sampler.next()
        // Pose refinement: fold the pending pose gradients before this view's camera is read (GPUPoseRefiner).
        if let poseRefiner, poseRefiner.hasPendingGradient {
            _ = try synchronize()
        }
        let camera = cameras[viewIndex]
        let width = camera.width
        let height = camera.height
        var settings = RasterSettings(shDegreeLimit: configuration.activeSHDegree(iteration: index),
                                      background: configuration.background)
        settings.mipFilter = trainingMipFilter
        if let dropout = configuration.dropout, dropout.isEnabled {
            // DropGaussian (experimental): a fresh random subset per iteration, shared by this step's passes.
            settings.dropRate = dropout.rate(iteration: index, iterations: configuration.iterations)
            settings.dropSeed = UInt32(truncatingIfNeeded: configuration.seed &* 0x9E37_79B9 &+ UInt64(index))
        }
        try refreshSamplingRatesIfDue(iteration: index)
        let texture = try renderTarget(width: width, height: height)
        let gradientImage = try imageGradientBuffer(pixelCount: width * height)

        // Step 1: depth mode only for a view with a prior, from the depth prior's start iteration on (CPUTrainer).
        var depthPass: DepthBuffers?
        if let depthTerm, index >= configuration.depthPrior.startIteration, depthTerm.hasPrior(view: viewIndex) {
            depthPass = try ensureDepthBuffers(pixelCount: width * height)
        }

        // Projection (waited inside the renderer) and the uncommitted rasterize command buffer.
        let pass = try metalCall { () throws(MetalError) -> ForwardPass in
            try self.renderer.encodeForward(camera: camera, settings: settings, to: texture,
                                            depth: depthPass?.rendered)
        }
        var commandBuffer = pass.commandBuffer

        // Validate the previous step before this step's loss encode overwrites the loss state. The wait
        // is cheap (see the file comment); it throws `.nonFiniteLoss(iteration: index - 1)` for a NaN loss.
        if unvalidatedLossStep != nil {
            _ = try synchronize()
        }

        // Step 2: image loss, then the pixel-gradient terms (depth loss).
        // Appearance (v1.2): the loss sees A = R ⊙ gain + bias, and the gradient G = dL/dA is turned into dL/dR in
        // place before the rasterizer backward (`encodeGradient` also updates the view's gain and bias).
        let appearanceActive: Bool = appearance != nil && index >= configuration.appearance.startIteration
        var lossInput: MTLTexture = texture
        if let appearance, appearanceActive {
            lossInput = try metalCall { () throws(MetalError) -> MTLTexture in
                try appearance.encodeCompensation(rendered: texture, viewIndex: viewIndex,
                                                  commandBuffer: commandBuffer)
            }
        }
        try metalCall { () throws(MetalError) -> Void in
            try self.imageLoss.encode(rendered: lossInput, target: self.targets[viewIndex],
                                      width: width, height: height,
                                      lambda: self.configuration.lambdaDSSIM,
                                      imageGradient: gradientImage, commandBuffer: commandBuffer)
        }
        if let appearance, appearanceActive {
            try metalCall { () throws(MetalError) -> Void in
                try appearance.encodeGradient(rendered: texture, imageGradient: gradientImage, viewIndex: viewIndex,
                                              learningRate: self.configuration.appearance.learningRate,
                                              identityWeight: self.configuration.appearance.identityWeight,
                                              hyper: self.configuration.adam, commandBuffer: commandBuffer)
            }
        }
        let currentGradients = gradients
        let auxiliary = AuxiliaryContext(
            viewIndex: viewIndex,
            iteration: index,
            isPseudoView: false,
            camera: camera,
            renderedDepth: depthPass?.rendered,
            depthGradient: depthPass?.gradient,
            parameters: store,
            gradients: currentGradients,
            sceneExtent: sceneExtent,
            commandBuffer: commandBuffer)
        for term in terms(.pixelGradient) {
            try term.encode(auxiliary)
        }

        // Step 3: backward with statistics. Statistics only while the densify window is open, as in CPUTrainer
        // (index before increment). The depth gradient is passed only for a depth-mode view.
        // MCMC replaces classic densification, so it needs no statistics (v1.2).
        let collectsStatistics = index < configuration.densification.stopIteration && !configuration.mcmc.isEnabled
        let statisticsTarget: DensificationStatisticsBuffers? = collectsStatistics ? statistics : nil
        let poseSink = try poseSinkForStep(viewIndex: viewIndex, iteration: index)
        try metalCall { () throws(MetalError) -> Void in
            try self.renderer.encodeBackward(imageGradient: gradientImage, depthGradient: depthPass?.gradient,
                                             into: currentGradients, statistics: statisticsTarget,
                                             poseSink: poseSink, commandBuffer: commandBuffer)
        }
        if poseSink != nil, pass.tileIntersections > 0, store.count > 0 {
            poseRefiner?.noteAccumulation(viewIndex: viewIndex)
        }

        // Step 4 (Task 18): a pseudo iteration commits the training part (steps 1–3) and encodes the pseudo-view into
        // its own command buffer, which then carries steps 5–7 of this iteration (`encodePseudoViewStep`).
        // Otherwise steps 5–7 go into the training command buffer.
        if let poolIndex = pseudoPoolIndex(iteration: index) {
            commandBuffer.commit()
            pending.append(commandBuffer)
            commandBuffer = try encodePseudoViewStep(viewIndex: viewIndex, iteration: index, poolIndex: poolIndex,
                                                     settings: settings, gradients: currentGradients)
        } else {
            try encodeParameterUpdate(auxiliary, iteration: index)
        }

        commandBuffer.commit()
        pending.append(commandBuffer)
        unvalidatedLossStep = index
        iteration = index + 1

        // Step 8 is done. Steps 9 and 10 need the GPU idle: wait when a boundary requires it.
        let completed = iteration
        let logInterval = configuration.logInterval
        let isLogStep = logInterval > 0 && completed % logInterval == 0
        let isLastStep = completed >= configuration.iterations
        let isDensifyStep = isDensificationBoundary(completed)
        let isIsolationStep = isIsolationBoundary(completed)
        let isMCMCStep = isMCMCBoundary(completed)
        var readout = LossReadout()
        if isLogStep || isLastStep || isDensifyStep || isIsolationStep || isMCMCStep {
            readout = try synchronize()
        }

        // Step 9: densification at its boundaries. Runs only after the wait above.
        var summary: DensifySummary?
        if isDensifyStep {
            let result = try metalCall { () throws(MetalError) -> DensifySummary in
                try self.densifier.densify(parameters: self.store, m: self.adamState.m, v: self.adamState.v,
                                           statistics: self.statistics,
                                           policy: self.configuration.densification,
                                           sceneRadius: self.sceneExtent.radius,
                                           iteration: completed,
                                           seed: UInt32(truncatingIfNeeded: self.configuration.seed),
                                           maxGaussians: self.maxGaussians)
            }
            if result.grewCapacity {
                // Gradient buffers are sized to the parameter capacity; the old ones are stale.
                gradients = try makeGradients()
            }
            summary = result
            samplingRatesStale = true
        }

        // Step 9b (v1.2): MCMC relocation and growth at its boundaries, after the wait above. Replaces densification.
        if isMCMCStep {
            _ = try relocateMCMC(iteration: completed)
        }

        // Step 10: isolation pruning at its boundaries. Runs after densification, which has waited.
        var isolationPruned: Int?
        if isIsolationStep {
            isolationPruned = try pruneIsolated(iteration: completed)
            samplingRatesStale = true
        }

        return StepReport(iteration: completed, loss: readout.image, visibleGaussians: pass.visibleGaussians,
                          densify: summary, depthLoss: readout.depth, isolationPruned: isolationPruned,
                          pseudoDepthLoss: readout.pseudoDepth)
    }

    /// Terms of `stage`, built-in first, then the caller's `auxiliaryLosses`, in array order (from the cache).
    private func terms(_ stage: AuxiliaryStage) -> [any AuxiliaryLoss] {
        termsByStage[stage] ?? []
    }

    /// Rebuilds the per-stage lists. Called from `init` and whenever `auxiliaryLosses` is set.
    private func rebuildTermCache() {
        let all: [any AuxiliaryLoss] = phase4Terms + auxiliaryLosses
        var cache: [AuxiliaryStage: [any AuxiliaryLoss]] = [:]
        for stage in AuxiliaryStage.allCases {
            cache[stage] = all.filter { $0.stage == stage }
        }
        termsByStage = cache
    }

    /// Same boundary rule as CPUTrainer.step, applied to the completed iteration count.
    private func isDensificationBoundary(_ completed: Int) -> Bool {
        let policy = configuration.densification
        return policy.isEnabled
            && !configuration.mcmc.isEnabled
            && completed >= policy.startIteration
            && completed < policy.stopIteration
            && completed % policy.interval == 0
    }

    /// Same boundary rule as CPUTrainer step 10, applied to the completed iteration count.
    private func isIsolationBoundary(_ completed: Int) -> Bool {
        let artifacts = configuration.artifacts
        return configuration.isEnabled(.isolationPrune)
            && completed >= artifacts.isolationStartIteration
            && artifacts.isolationInterval > 0
            && (completed - artifacts.isolationStartIteration) % artifacts.isolationInterval == 0
            && completed < configuration.iterations
    }

    // MARK: - MCMC (v1.2)

    /// MCMC boundary rule: every `interval` completed iterations inside `startIteration..<stopIteration`.
    private func isMCMCBoundary(_ completed: Int) -> Bool {
        let settings = configuration.mcmc
        return settings.isEnabled
            && settings.interval > 0
            && completed >= settings.startIteration
            && completed < settings.stopIteration
            && completed % settings.interval == 0
    }

    /// MCMC relocation and growth on the CPU, after a wait (see MCMCRelocator). A capacity change replaces the
    /// buffers, so the statistics and gradient buffers follow it, as in densification.
    private func relocateMCMC(iteration: Int) throws(TrainingError) -> MCMCRelocator.Result {
        let result = try metalCall { () throws(MetalError) -> MCMCRelocator.Result in
            try MCMCRelocator.run(parameters: self.store, m: self.adamState.m, v: self.adamState.v,
                                  settings: self.configuration.mcmc, maxGaussians: self.maxGaussians,
                                  seed: self.configuration.seed ^ UInt64(iteration))
        }
        if result.grewCapacity {
            try metalCall { () throws(MetalError) -> Void in
                try self.statistics.grow(toCapacity: self.store.capacity)
            }
            gradients = try makeGradients()
        }
        // Relocation moves Gaussians and growth appends new ones, so the anti-aliasing sampling rates (one per
        // Gaussian) no longer match the store.
        samplingRatesStale = true
        return result
    }

    // MARK: - Synchronization

    /// Waits for every command buffer committed since the last wait point, throws if any failed, and reads and
    /// validates the image and depth losses of the most recent step if they have not been validated yet. Called
    /// at the start of each step (for the previous step) and at the boundaries (for the current step).
    private func synchronize() throws(TrainingError) -> LossReadout {
        for buffer in pending {
            buffer.waitUntilCompleted()
        }
        var gpuError: String?
        for buffer in pending where buffer.status == .error {
            gpuError = buffer.error?.localizedDescription ?? "unknown GPU error"
        }
        pending.removeAll(keepingCapacity: true)
        if let gpuError {
            throw .metal(gpuError)
        }
        // Pose refinement: every pending pose gradient is complete now; apply and fold them (GPUPoseRefiner).
        poseRefiner?.applyPending(cameras: &cameras, sceneRadius: sceneExtent.radius)

        guard let stepIndex = unvalidatedLossStep else {
            return LossReadout()
        }
        unvalidatedLossStep = nil
        // Safe: the command buffer of the most recent step has completed, so both loss states are final.
        let image = imageLoss.readValue()
        guard image.loss.isFinite else {
            throw .nonFiniteLoss(iteration: stepIndex)
        }
        let depth = depthTerm?.takePendingValue()
        if let depth, !depth.loss.isFinite {
            throw .nonFiniteLoss(iteration: stepIndex)
        }
        var pseudo: DepthLossValue?
        if pendingPseudoLoss, let depthLoss {
            pseudo = depthLoss.readValue(slot: 1)
        }
        pendingPseudoLoss = false
        if let pseudo, !pseudo.loss.isFinite {
            throw .nonFiniteLoss(iteration: stepIndex)
        }
        return LossReadout(image: image, depth: depth, pseudoDepth: pseudo)
    }

    /// Runs `body` unless an earlier error made the trainer fatal; records any error it throws.
    private func guarded<T>(_ body: () throws(TrainingError) -> T) throws(TrainingError) -> T {
        if let fatal {
            throw fatal
        }
        do {
            return try body()
        } catch {
            fatal = error
            throw error
        }
    }

    // MARK: - Pose refinement (v1.2)

    /// Pose sink of this step's training view, or nil when pose refinement is off or has not started (`startIteration`).
    private func poseSinkForStep(viewIndex: Int, iteration index: Int) throws(TrainingError) -> PoseGradientSink? {
        guard let poseRefiner, index >= configuration.poseRefinement.startIteration else {
            return nil
        }
        let capacity = store.capacity
        return try metalCall { () throws(MetalError) -> PoseGradientSink in
            try poseRefiner.sink(viewIndex: viewIndex, capacity: capacity)
        }
    }

    /// The training cameras in view order with every pose correction applied so far. Waits for the GPU. Equal to the
    /// dataset's cameras when pose refinement is off.
    public func refinedCameras() throws(TrainingError) -> [Camera] {
        try guarded { () throws(TrainingError) -> [Camera] in
            _ = try self.synchronize()
            return self.cameras
        }
    }

    private func measurePSNR() throws(TrainingError) -> Double {
        _ = try synchronize()
        // Evaluation uses the full degree of the model, not the schedule's current degree.
        var settings = RasterSettings(shDegreeLimit: configuration.shDegree,
                                      background: configuration.background)
        // Anti-aliasing: evaluate the raw parameters with the same filter the training passes use.
        settings.mipFilter = trainingMipFilter
        try refreshSamplingRatesIfDue(iteration: iteration)
        // Appearance (v1.2): training views are compared with their learned gain and bias. Identity parameters leave
        // the render unchanged, so this is exact for views before the appearance start.
        let compensation: [AppearanceParameters]? = appearance?.readParameters()
        var sum: Double = 0
        for (index, view) in views.enumerated() {
            var rendered = try metalCall { () throws(MetalError) -> RGBImage in
                try self.renderer.renderImage(camera: self.cameras[index], settings: settings)
            }
            if let compensation {
                rendered = AppearanceTransform.apply(rendered, compensation[index])
            }
            sum += ImageMetrics.psnr(rendered, view.image)
        }
        return sum / Double(views.count)
    }

    // MARK: - Isolation pruning

    /// Isolation pruning (P10, CPUTrainer step 10). Runs only at a wait point, so the GPU is idle with respect to the
    /// store, the moments and the statistics. The discrete decision is made on the CPU with the seed
    /// `seed ^ iteration`; the parameters and both moments are then compacted with the same mask, and the statistics
    /// are reset in their own command buffer that is waited on here. Nothing changes when nothing is pruned.
    /// The model is never emptied: `IsolationPruning` keeps at least one Gaussian and `compact` checks it.
    /// Returns the number of Gaussians removed.
    private func pruneIsolated(iteration: Int) throws(TrainingError) -> Int {
        let artifacts = configuration.artifacts
        let total = store.count
        precondition(adamState.m.count == total && adamState.v.count == total,
                     "Adam moments must have the parameters' count before compaction")
        let positions = store.readPositions()
        let flags = IsolationPruning.pruneMask(positions: positions,
                                               neighbors: artifacts.isolationNeighbors,
                                               factor: artifacts.isolationFactor,
                                               maxFraction: artifacts.isolationMaxFraction,
                                               sampleCount: artifacts.isolationSampleCount,
                                               seed: configuration.seed ^ UInt64(iteration))
        var keep = [Bool](repeating: true, count: total)
        var pruned = 0
        for index in 0..<total where flags[index] {
            keep[index] = false
            pruned += 1
        }
        guard pruned > 0 else {
            return 0
        }

        // Parameters and moments share one mask, so they stay aligned. Capacity is unchanged: no reallocation.
        store.compact(keeping: keep)
        adamState.m.compact(keeping: keep)
        adamState.v.compact(keeping: keep)

        // Statistics restart for the survivors, as after densification. Waited on before the CPU continues.
        guard let resetBuffer = context.queue.makeCommandBuffer() else {
            throw .metal("cannot create command buffer")
        }
        try metalCall { () throws(MetalError) -> Void in
            try self.statistics.encodeReset(commandBuffer: resetBuffer)
        }
        resetBuffer.commit()
        resetBuffer.waitUntilCompleted()
        if resetBuffer.status == .error {
            throw .metal(resetBuffer.error?.localizedDescription ?? "unknown GPU error")
        }
        return pruned
    }

    // MARK: - Anti-aliasing (Mip-Splatting)

    /// Mip filter of the training passes; classic when anti-aliasing is off. The CPU sampling-rate array is not
    /// used here: the Metal path reads `samplingRateBuffer`.
    private var trainingMipFilter: MipFilter {
        let aa = configuration.antiAliasing
        return MipFilter(isEnabled: aa.isEnabled, filter2DVariance: aa.filter2DVariance,
                         filter3DVariance: aa.filter3DVariance)
    }

    /// True when the sampling-rate buffer holds one rate for every Gaussian of the store. A safety net for any
    /// code path that changes the Gaussian count without marking the rates stale.
    private var samplingRatesCoverStore: Bool {
        guard let buffer = samplingRateBuffer else { return false }
        return buffer.length >= store.count * MemoryLayout<Float>.size
    }

    /// Recomputes ν when it is stale or when the sampling-rate interval falls on this iteration. The command buffer
    /// is committed here and runs before the forward pass that follows on the same queue, so no wait is needed.
    private func refreshSamplingRatesIfDue(iteration index: Int) throws(TrainingError) {
        guard let kernel = samplingRateKernel else { return }
        let interval = configuration.antiAliasing.samplingRateInterval
        let scheduled = interval > 0 && index % interval == 0
        guard samplingRatesStale || scheduled || !samplingRatesCoverStore else { return }
        try encodeSamplingRates(kernel)
    }

    /// Encodes ν for the current cloud into the shared buffer, commits it and returns it.
    @discardableResult
    private func encodeSamplingRates(_ kernel: GPUSamplingRate) throws(TrainingError) -> MTLCommandBuffer {
        let buffer = try growableBuffer(&samplingRateBuffer, bytes: store.capacity * MemoryLayout<Float>.size)
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw .metal("cannot create command buffer")
        }
        let cameras = self.cameras
        try metalCall { () throws(MetalError) -> Void in
            try kernel.encode(positions: self.store.positions, count: self.store.count, cameras: cameras,
                              near: GPUTrainer.depthNearPlane, into: buffer, commandBuffer: commandBuffer)
        }
        commandBuffer.commit()
        pending.append(commandBuffer)
        renderer.samplingRateBuffer = buffer
        samplingRatesStale = false
        return commandBuffer
    }

    /// `cloud` with the 3D filter baked in (`GaussianCloud.bakingSmoothing3D`) when anti-aliasing is on, using the
    /// current sampling rates. Waits for a sampling-rate pass that is still pending.
    private func bakedSnapshot(_ cloud: GaussianCloud) throws(TrainingError) -> GaussianCloud {
        guard configuration.antiAliasing.isEnabled, let kernel = samplingRateKernel else {
            return cloud
        }
        if samplingRatesStale || !samplingRatesCoverStore {
            let commandBuffer = try encodeSamplingRates(kernel)
            commandBuffer.waitUntilCompleted()
            if commandBuffer.status == .error {
                throw .metal(commandBuffer.error?.localizedDescription ?? "unknown GPU error")
            }
        }
        guard let buffer = samplingRateBuffer, cloud.count > 0,
              buffer.length >= cloud.count * MemoryLayout<Float>.size else {
            // Never read past the sampling-rate buffer (it once was smaller than the cloud and crashed the app).
            return cloud
        }
        let rates = buffer.contents().bindMemory(to: Float.self, capacity: cloud.count)
        let v3 = configuration.antiAliasing.filter3DVariance
        var filterVariances = [Float](repeating: 0, count: cloud.count)
        for i in 0..<cloud.count {
            let nu = rates[i]
            filterVariances[i] = v3 / (nu * nu)
        }
        return cloud.bakingSmoothing3D(filterVariances: filterVariances)
    }

    // MARK: - Resources

    private func renderTarget(width: Int, height: Int) throws(TrainingError) -> MTLTexture {
        let key = SIMD2<Int>(width, height)
        if let existing = renderTargets[key] {
            return existing
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderWrite, .shaderRead]
        descriptor.storageMode = .shared
        guard let texture = context.device.makeTexture(descriptor: descriptor) else {
            throw .metal("cannot allocate a \(width)x\(height) render target")
        }
        renderTargets[key] = texture
        return texture
    }

    /// Grow-only. A replaced buffer stays alive while in-flight command buffers reference it.
    private func imageGradientBuffer(pixelCount: Int) throws(TrainingError) -> MTLBuffer {
        let bytes = pixelCount * 4 * MemoryLayout<Float>.size
        if let existing = imageGradient, existing.length >= bytes {
            return existing
        }
        let buffer = try metalCall { () throws(MetalError) -> MTLBuffer in
            try self.context.makeBuffer(length: bytes)
        }
        imageGradient = buffer
        return buffer
    }

    /// Grow-only `float2` per pixel buffers for the depth forward output and the depth gradient.
    private func ensureDepthBuffers(pixelCount: Int) throws(TrainingError) -> DepthBuffers {
        let bytes = pixelCount * 2 * MemoryLayout<Float>.size
        let rendered = try growableBuffer(&renderedDepthBuffer, bytes: bytes)
        let gradient = try growableBuffer(&depthGradientBuffer, bytes: bytes)
        return DepthBuffers(rendered: rendered, gradient: gradient)
    }

    /// Returns `buffer` when it holds at least `bytes`, else replaces it with a new one of that size.
    private func growableBuffer(_ buffer: inout MTLBuffer?, bytes: Int) throws(TrainingError) -> MTLBuffer {
        if let existing = buffer, existing.length >= bytes {
            return existing
        }
        let fresh = try metalCall { () throws(MetalError) -> MTLBuffer in
            try self.context.makeBuffer(length: bytes)
        }
        buffer = fresh
        return fresh
    }

    private func makeGradients() throws(TrainingError) -> GradientBuffers {
        try metalCall { () throws(MetalError) -> GradientBuffers in
            try GradientBuffers(context: self.context, count: self.store.capacity, shDegree: self.store.shDegree)
        }
    }
}

// MARK: - Optimizer state (v1.5)

extension GPUTrainer {
    /// Copies the Adam moments, the iteration counter, and the appearance and pose-refinement state to the CPU. Waits
    /// for the GPU first, applies pending pose gradients, and throws if the most recent step's losses are non-finite.
    /// Densification statistics are not part of the state: a resumed run starts its current densification window with
    /// empty statistics (the moments themselves are exact).
    public func exportOptimizerState() throws(TrainingError) -> OptimizerState {
        try guarded { () throws(TrainingError) -> OptimizerState in
            _ = try self.synchronize()
            let count: Int = self.store.count
            // Moments count as live only once an Adam step has run. Before that they are zero, so the count is set here.
            self.adamState.m.count = count
            self.adamState.v.count = count
            let firstMoment: GaussianMoments = self.adamState.m.snapshotArrays()
            let secondMoment: GaussianMoments = self.adamState.v.snapshotArrays()
            let appearanceBlock: AppearanceOptimizerBlock? = self.appearance?.snapshotState()
            var poseBlock: PoseOptimizerBlock?
            if let refiner = self.poseRefiner {
                poseBlock = self.makePoseBlock(refiner: refiner)
            }
            return OptimizerState(
                gaussianCount: count,
                shDegree: self.store.shDegree,
                iteration: self.iteration,
                adamStep: self.adamStepBase + self.iteration,
                configurationHash: OptimizerState.configurationHash(self.configuration),
                firstMoment: firstMoment,
                secondMoment: secondMoment,
                appearance: appearanceBlock,
                pose: poseBlock)
        }
    }

    /// Loads an exported state into this trainer, which must be fresh (no step taken). Two modes:
    /// - `continuing == false`: the same run resumes at the state's iteration. The configuration hash must match, the
    ///   Adam step continues from the state, and the view sampler is replayed to the state's position, so the next
    ///   views are the ones the uninterrupted run would have visited.
    /// - `continuing == true`: a continuation round. Its own schedules start at iteration 0 as usual, but the moments,
    ///   the appearance and pose state and the Adam step counter continue from the state.
    /// Returns nil when the state was loaded. Otherwise it returns the reason and changes nothing, and the caller
    /// trains with fresh Adam state.
    public func restore(from state: OptimizerState, continuing: Bool) throws(TrainingError) -> String? {
        try guarded { () throws(TrainingError) -> String? in
            precondition(self.iteration == 0 && self.pending.isEmpty, "optimizer state is restored into a fresh trainer")
            if let reason = state.incompatibility(gaussianCount: self.store.count, shDegree: self.store.shDegree,
                                                  viewCount: self.views.count,
                                                  appearance: self.appearance != nil,
                                                  poseRefinement: self.poseRefiner != nil) {
                return reason
            }
            if !continuing {
                guard state.configurationHash == OptimizerState.configurationHash(self.configuration) else {
                    return "the optimizer state belongs to another configuration"
                }
                guard state.adamStep >= state.iteration else {
                    return "the optimizer state's Adam step is behind its iteration"
                }
            }

            self.adamState.m.upload(arrays: state.firstMoment)
            self.adamState.v.upload(arrays: state.secondMoment)
            if let block = state.appearance, let appearance = self.appearance {
                appearance.restoreState(block)
            }
            if let block = state.pose, let refiner = self.poseRefiner {
                self.restorePoses(block, refiner: refiner)
            }
            if continuing {
                self.adamStepBase = state.adamStep
            } else {
                self.iteration = state.iteration
                self.adamStepBase = state.adamStep - state.iteration
                for _ in 0..<state.iteration {
                    _ = self.sampler.next()
                }
            }
            return nil
        }
    }

    /// Pose block of the trainer's cameras and the refiner's Adam state. The caller has waited for the GPU.
    private func makePoseBlock(refiner: GPUPoseRefiner) -> PoseOptimizerBlock {
        let states: [PoseAdamState] = refiner.snapshotStates()
        var poses: [Float] = []
        poses.reserveCapacity(cameras.count * 12)
        for camera in cameras {
            let r: Matrix3 = camera.rotation
            poses += [r.c0.x, r.c0.y, r.c0.z, r.c1.x, r.c1.y, r.c1.z, r.c2.x, r.c2.y, r.c2.z]
            poses += [camera.translation.x, camera.translation.y, camera.translation.z]
        }
        var firstMoment: [Double] = []
        var secondMoment: [Double] = []
        firstMoment.reserveCapacity(states.count * 6)
        secondMoment.reserveCapacity(states.count * 6)
        for state in states {
            firstMoment += state.firstMoment
            secondMoment += state.secondMoment
        }
        return PoseOptimizerBlock(poses: poses, firstMoment: firstMoment, secondMoment: secondMoment,
                                  steps: states.map { $0.step })
    }

    /// Puts the restored refined poses into the cameras and the restored Adam state into the refiner.
    private func restorePoses(_ block: PoseOptimizerBlock, refiner: GPUPoseRefiner) {
        var restored: [PoseAdamState] = []
        restored.reserveCapacity(cameras.count)
        for view in 0..<cameras.count {
            let p: Int = view * 12
            let c0 = SIMD3<Float>(block.poses[p], block.poses[p + 1], block.poses[p + 2])
            let c1 = SIMD3<Float>(block.poses[p + 3], block.poses[p + 4], block.poses[p + 5])
            let c2 = SIMD3<Float>(block.poses[p + 6], block.poses[p + 7], block.poses[p + 8])
            cameras[view].rotation = Matrix3(c0: c0, c1: c1, c2: c2)
            cameras[view].translation = SIMD3<Float>(block.poses[p + 9], block.poses[p + 10], block.poses[p + 11])
            let m: Int = view * 6
            restored.append(PoseAdamState(firstMoment: Array(block.firstMoment[m..<(m + 6)]),
                                          secondMoment: Array(block.secondMoment[m..<(m + 6)]),
                                          step: block.steps[view]))
        }
        refiner.restoreStates(restored)
    }
}

// MARK: - Pseudo-views (Task 18, P9), held-out evaluation

// Command-buffer contract of a pseudo iteration (CPUTrainer step 4, GPU order):
// 1. The training part (steps 1–3: forward, image and depth loss, backward with statistics) is committed as B.
// 2. `encodeForward` for the pseudo camera queues its projection behind B and returns the pseudo buffer P, uncommitted.
// 3. In P: the pseudo depth loss (slot 1), `encodeBackward` with a zero image gradient, `accumulate: true` and no
//    statistics, then steps 5–7 (anisotropy, Adam, scale cap). Gradients written by B are overwritten by B's own
//    backward (accumulate off), so P adds the pseudo part on top of exactly the training gradients.
// 4. P is committed by `advance` like any step buffer and joins `pending`, so the next wait point covers B and P.
// The pseudo depth buffers and the forward's projection/render buffers are shared with the training view. This is safe
// because B is committed before P is encoded and Metal orders the command buffers of one queue for tracked resources.

extension GPUTrainer {
    /// Number of pseudo-views in the pool (P9). Zero when `.pseudoViews` is off or the dataset has no metric view.
    public var pseudoViewCount: Int { pseudoPool.count }

    /// PSNR and SSIM of `view`, rendered with the full SH degree, using the CPU metric definitions on the GPU rendering.
    /// Waits for the GPU first and throws if the most recent step's losses are non-finite. Does not change the model
    /// or the densification statistics.
    public func evaluate(view: TrainingView) throws(TrainingError) -> HeldOutMetrics {
        try guarded { () throws(TrainingError) -> HeldOutMetrics in
            try self.measureHeldOut(view)
        }
    }

    private func measureHeldOut(_ view: TrainingView) throws(TrainingError) -> HeldOutMetrics {
        _ = try synchronize()
        var settings = RasterSettings(shDegreeLimit: configuration.shDegree, background: configuration.background)
        settings.mipFilter = trainingMipFilter
        try refreshSamplingRatesIfDue(iteration: iteration)
        let rendered = try metalCall { () throws(MetalError) -> RGBImage in
            try self.renderer.renderImage(camera: view.camera, settings: settings)
        }
        return HeldOutMetrics(psnr: ImageMetrics.psnr(rendered, view.image),
                              ssim: ImageMetrics.ssim(rendered, view.image))
    }

    /// Pool index of the pseudo-view used by this iteration, or nil when the iteration has none (the CPUTrainer gate).
    func pseudoPoolIndex(iteration: Int) -> Int? {
        let schedule = configuration.pseudoViews
        guard configuration.isEnabled(.pseudoViews), !pseudoPool.isEmpty,
              iteration >= schedule.startIteration, schedule.interval > 0,
              (iteration - schedule.startIteration) % schedule.interval == 0 else {
            return nil
        }
        return PseudoViews.index(iteration: iteration, count: pseudoPool.count, seed: configuration.seed)
    }

    /// Step 4 and steps 5–7 of a pseudo iteration. Precondition: the training part has been committed.
    /// Returns the pseudo command buffer, uncommitted. Adds into `gradients` and touches no statistics.
    func encodePseudoViewStep(viewIndex: Int, iteration: Int, poolIndex: Int, settings: RasterSettings,
                              gradients: GradientBuffers) throws(TrainingError) -> MTLCommandBuffer {
        guard let depthLoss else {
            throw .metal("pseudo-view step without a depth loss kernel")
        }
        let pseudo: PseudoView = pseudoPool[poolIndex]
        let camera = pseudo.camera
        let width = camera.width
        let height = camera.height
        let texture = try renderTarget(width: width, height: height)
        let depthBuffers = try ensureDepthBuffers(pixelCount: width * height)
        let pass = try metalCall { () throws(MetalError) -> ForwardPass in
            try self.renderer.encodeForward(camera: camera, settings: settings, to: texture,
                                            depth: depthBuffers.rendered)
        }
        let commandBuffer = pass.commandBuffer

        // Depth loss into slot 1 with the pseudo weight: Pearson only (no L1), as in CPUTrainer step 4.
        let weights = DepthLossWeights(pearson: configuration.pseudoViews.weight, l1: 0,
                                       alphaMask: configuration.depthPrior.alphaMask,
                                       minimumPixels: configuration.depthPrior.minimumPixels,
                                       nearPlane: GPUTrainer.depthNearPlane)
        try metalCall { () throws(MetalError) -> Void in
            try depthLoss.encode(renderedDepth: depthBuffers.rendered, prior: self.pseudoPriors[poolIndex],
                                 weights: weights, slot: 1, depthGradient: depthBuffers.gradient,
                                 commandBuffer: commandBuffer)
        }
        pendingPseudoLoss = true

        // Backward with a zero image gradient. Accumulates into the step's gradients; statistics: nil.
        let zeros = try zeroImageGradientBuffer(pixelCount: width * height)
        try metalCall { () throws(MetalError) -> Void in
            try self.renderer.encodeBackward(imageGradient: zeros, depthGradient: depthBuffers.gradient,
                                             into: gradients, statistics: nil, accumulate: true,
                                             commandBuffer: commandBuffer)
        }

        // Context of steps 5–7 on a pseudo iteration. `isPseudoView` is true, `viewIndex` is the TRAINING view (the
        // statistics and Gaussians are not view-specific), and `camera`, `renderedDepth`, `depthGradient` and
        // `commandBuffer` are the pseudo-view's. Only `.parameterGradient` and `.postOptimizer` terms read this context.
        let auxiliary = AuxiliaryContext(
            viewIndex: viewIndex,
            iteration: iteration,
            isPseudoView: true,
            camera: camera,
            renderedDepth: depthBuffers.rendered,
            depthGradient: depthBuffers.gradient,
            parameters: store,
            gradients: gradients,
            sceneExtent: sceneExtent,
            commandBuffer: commandBuffer)
        try encodeParameterUpdate(auxiliary, iteration: iteration)
        return commandBuffer
    }

    /// Steps 5–7 of an iteration, in CPUTrainer order: parameter-gradient terms (anisotropy), Adam with the
    /// pre-increment schedule and step = index + 1, then post-optimizer terms (scale cap). Shared by the training
    /// and pseudo command buffers, so both encode the same order.
    func encodeParameterUpdate(_ auxiliary: AuxiliaryContext, iteration index: Int) throws(TrainingError) {
        for term in terms(.parameterGradient) {
            try term.encode(auxiliary)
        }
        let rates = configuration.learningRates(iteration: index, sceneRadius: sceneExtent.radius)
        try metalCall { () throws(MetalError) -> Void in
            try self.adam.encode(parameters: self.store, gradients: auxiliary.gradients, state: self.adamState,
                                 rates: rates, step: self.adamStepBase + index + 1, hyper: self.configuration.adam,
                                 commandBuffer: auxiliary.commandBuffer)
        }
        for term in terms(.postOptimizer) {
            try term.encode(auxiliary)
        }
    }

    /// Grow-only, zero-filled `float4` per pixel. Zeroed once when created; the GPU only reads it.
    /// Not part of `TrainingMemoryBudget`: it is about one quarter of the reserved image-gradient buffer, because
    /// pseudo-views render at `resolutionDivisor` (2) per side.
    func zeroImageGradientBuffer(pixelCount: Int) throws(TrainingError) -> MTLBuffer {
        let bytes = pixelCount * 4 * MemoryLayout<Float>.size
        if let existing = zeroImageGradient, existing.length >= bytes {
            return existing
        }
        let buffer = try metalCall { () throws(MetalError) -> MTLBuffer in
            try self.context.makeBuffer(length: bytes)
        }
        memset(buffer.contents(), 0, buffer.length)
        zeroImageGradient = buffer
        return buffer
    }
}

/// Test-only copy of the trainer state after the GPU is idle.
struct GPUTrainerTestReadout {
    /// Gradients left by the most recent step (Adam does not change them).
    var gradients: GaussianGradients
    /// Densification statistics: summed screen-space gradient norms and visible counts, one entry per Gaussian.
    var gradientNormSums: [Float]
    var visibleCounts: [UInt32]
}

// Test hook: not part of the training API.
extension GPUTrainer {
    /// Waits for the GPU, validates the most recent step and copies the gradients and statistics to the CPU.
    func testingReadout() throws(TrainingError) -> GPUTrainerTestReadout {
        try guarded { () throws(TrainingError) -> GPUTrainerTestReadout in
            _ = try self.synchronize()
            let count: Int = self.store.count
            let gradientCopy = self.gradients.toGaussianGradients(count: count, shDegree: self.store.shDegree)
            let norms: [Float] = count > 0
                ? Array(UnsafeBufferPointer(
                    start: self.statistics.gradientNormSums.contents().bindMemory(to: Float.self, capacity: count),
                    count: count))
                : []
            let visible: [UInt32] = count > 0
                ? Array(UnsafeBufferPointer(
                    start: self.statistics.visibleCounts.contents().bindMemory(to: UInt32.self, capacity: count),
                    count: count))
                : []
            return GPUTrainerTestReadout(gradients: gradientCopy, gradientNormSums: norms, visibleCounts: visible)
        }
    }
}
