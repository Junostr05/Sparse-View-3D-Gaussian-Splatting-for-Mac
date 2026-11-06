import Foundation
import Metal
import SplatCore
import SplatMetal

// Metal training loop (Phase 3 Task 15, Phase 4 Task 17). Mirrors `CPUTrainer.step` in SplatCore in order and in
// arithmetic. Phase 4 step order (normative, Task 11 / Task 17), with the GPU encode it maps to:
//   1 sample a view; forward render (depth mode only when the view has a prior and iteration ≥ depth start)
//   2 image loss (`imageLoss.encode`); depth loss (`.pixelGradient` terms)
//   3 backward with statistics (`encodeBackward`, depth gradient passed only for depth-mode views)
//   4 pseudo-views: not in this task (Task 18)
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

    public init(iteration: Int, loss: ImageLossValue?, visibleGaussians: Int, densify: DensifySummary?,
                depthLoss: DepthLossValue? = nil, isolationPruned: Int? = nil) {
        self.iteration = iteration
        self.loss = loss
        self.visibleGaussians = visibleGaussians
        self.densify = densify
        self.depthLoss = depthLoss
        self.isolationPruned = isolationPruned
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
    public var auxiliaryLosses: [any AuxiliaryLoss] = []

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

        // Budget (D13). Each view keeps its own size; the reserved image bytes use the largest view.
        // Depth buffers are reserved for every training view and no pseudo-view yet (Task 18 adds them).
        let viewSizes: [SIMD2<Int>] = dataset.views.map { SIMD2<Int>($0.camera.width, $0.camera.height) }
        let reservedBytes: Int = TrainingMemoryBudget.reservedImageBytes(viewSizes: viewSizes)
            + TrainingMemoryBudget.reservedDepthBytes(viewSizes: viewSizes, pseudoViewSizes: [])
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
        var phase4Terms: [any AuxiliaryLoss] = []
        var depthTerm: DepthLossTerm?
        let wantsDepthLoss: Bool = configuration.isEnabled(.depthPearson) || configuration.isEnabled(.depthL1)
        if wantsDepthLoss, dataset.views.contains(where: { $0.depth != nil }) {
            let depthLoss = try metalCall { () throws(MetalError) -> GPUDepthLoss in
                try GPUDepthLoss(context: context)
            }
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
        self.phase4Terms = phase4Terms
        self.depthTerm = depthTerm
        self.maxGaussians = budget
        // Same scene extent as CPUTrainer: camera centres, falling back to the dataset points.
        self.sceneExtent = SceneExtent.estimate(cameraCenters: dataset.cameraCenters,
                                                points: dataset.points?.positions ?? [])
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
    public func snapshot() throws(TrainingError) -> GaussianCloud {
        try guarded { () throws(TrainingError) -> GaussianCloud in
            _ = try self.synchronize()
            return self.store.snapshot()
        }
    }

    // MARK: - Iteration

    private func advance() throws(TrainingError) -> StepReport {
        let index = iteration
        let viewIndex = sampler.next()
        let view = views[viewIndex]
        let camera = view.camera
        let width = camera.width
        let height = camera.height
        let settings = RasterSettings(shDegreeLimit: configuration.activeSHDegree(iteration: index),
                                      background: configuration.background)
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
        let commandBuffer = pass.commandBuffer

        // Validate the previous step before this step's loss encode overwrites the loss state. The wait
        // is cheap (see the file comment); it throws `.nonFiniteLoss(iteration: index - 1)` for a NaN loss.
        if unvalidatedLossStep != nil {
            _ = try synchronize()
        }

        // Step 2: image loss, then the pixel-gradient terms (depth loss).
        try metalCall { () throws(MetalError) -> Void in
            try self.imageLoss.encode(rendered: texture, target: self.targets[viewIndex],
                                      width: width, height: height,
                                      lambda: self.configuration.lambdaDSSIM,
                                      imageGradient: gradientImage, commandBuffer: commandBuffer)
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
        let collectsStatistics = index < configuration.densification.stopIteration
        let statisticsTarget: DensificationStatisticsBuffers? = collectsStatistics ? statistics : nil
        try metalCall { () throws(MetalError) -> Void in
            try self.renderer.encodeBackward(imageGradient: gradientImage, depthGradient: depthPass?.gradient,
                                             into: currentGradients, statistics: statisticsTarget,
                                             commandBuffer: commandBuffer)
        }

        // Step 5: parameter-gradient terms (anisotropy), after the backward pass and before Adam.
        for term in terms(.parameterGradient) {
            try term.encode(auxiliary)
        }

        // Step 6: Adam with the pre-increment schedule and step = index + 1 (CPUAdam increments first).
        let rates = configuration.learningRates(iteration: index, sceneRadius: sceneExtent.radius)
        try metalCall { () throws(MetalError) -> Void in
            try self.adam.encode(parameters: self.store, gradients: currentGradients, state: self.adamState,
                                 rates: rates, step: index + 1, hyper: self.configuration.adam,
                                 commandBuffer: commandBuffer)
        }

        // Step 7: post-optimizer terms (scale cap), after Adam and before the increment.
        for term in terms(.postOptimizer) {
            try term.encode(auxiliary)
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
        var readout = LossReadout()
        if isLogStep || isLastStep || isDensifyStep || isIsolationStep {
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
        }

        // Step 10: isolation pruning at its boundaries. Runs after densification, which has waited.
        var isolationPruned: Int?
        if isIsolationStep {
            isolationPruned = try pruneIsolated(iteration: completed)
        }

        return StepReport(iteration: completed, loss: readout.image, visibleGaussians: pass.visibleGaussians,
                          densify: summary, depthLoss: readout.depth, isolationPruned: isolationPruned)
    }

    /// Terms of `stage`, built-in first, then the caller's `auxiliaryLosses`, in array order.
    private func terms(_ stage: AuxiliaryStage) -> [any AuxiliaryLoss] {
        (phase4Terms + auxiliaryLosses).filter { $0.stage == stage }
    }

    /// Same boundary rule as CPUTrainer.step, applied to the completed iteration count.
    private func isDensificationBoundary(_ completed: Int) -> Bool {
        let policy = configuration.densification
        return policy.isEnabled
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
        return LossReadout(image: image, depth: depth)
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

    private func measurePSNR() throws(TrainingError) -> Double {
        _ = try synchronize()
        // Evaluation uses the full degree of the model, not the schedule's current degree.
        let settings = RasterSettings(shDegreeLimit: configuration.shDegree,
                                      background: configuration.background)
        var sum: Double = 0
        for view in views {
            let rendered = try metalCall { () throws(MetalError) -> RGBImage in
                try self.renderer.renderImage(camera: view.camera, settings: settings)
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
