import Foundation
import Metal
import SplatCore
import SplatMetal

// Metal training loop (Phase 3 Task 15). Mirrors `CPUTrainer.step` in SplatCore in order and in
// arithmetic: view sampler → forward → loss → backward (+statistics) → Adam → densify at boundaries.
//
// Command-buffer structure per iteration:
// - `encodeForward` runs projection and the tile-count scan in their own command buffer, waited on
//   by the renderer (the CPU needs the intersection total). It returns the rasterize command buffer
//   B uncommitted.
// - B receives the loss, the auxiliary losses, the backward pass (with statistics), and Adam. It is
//   committed and NOT waited on. The next iteration's projection command buffer is committed after
//   B on the same queue, so GPU order is preserved, and its CPU wait also bounds the in-flight work.
// - Each step starts by synchronizing with the previous step (`synchronize`, right after its
//   projection returns). That wait is cheap: the projection was queued behind the previous step's
//   command buffer B and depends on Adam's writes, so B has effectively finished. It checks B's
//   status and validates the previous step's loss, throwing `.nonFiniteLoss(iteration: k-1)` before
//   the current loss encode overwrites the loss state. A NaN is therefore reported at the second
//   `step()` call at the latest, and never later than the next wait point.
// - The CPU also waits at: a log step, the last iteration, a densify boundary, `snapshot` and
//   `evaluatePSNR`, where the current step's loss is validated as well.
// - The `GPUImageLoss` contract ("do not encode a new loss while a previous command buffer that uses
//   it is still running") still holds, although the previous step's buffer B is queued before the
//   projection and is not waited on before the forward pass: Metal orders tracked resources
//   submitted to one queue, and the CPU writes no scratch memory of the loss object between
//   encodes. The synchronization at the start of each step makes the previous loss read happen
//   after B completes, which is the only CPU access to that state.
//
// Decisions and interpretations (see also the comments at each site):
// - Any error thrown by `step`, `snapshot`, `evaluatePSNR` or densification makes the trainer
//   fatal: the same error is thrown by every later call. The model may be partially updated.
// - The densify seed is `UInt32(truncatingIfNeeded: seed)`; the densifier adds the iteration itself.

/// Phase 4 extension point: an extra loss term encoded after the photometric loss, into the same
/// image gradient and parameter gradients. Empty in Phase 3.
public protocol AuxiliaryLoss: AnyObject {
    func encode(view: Int, iteration: Int, commandBuffer: MTLCommandBuffer) throws(TrainingError)
}

/// Outcome of one `GPUTrainer.step()`.
public struct StepReport: Sendable {
    /// Completed iterations after this step (the trainer's `iteration`).
    public var iteration: Int
    /// Loss of this step's view. Non-nil when the step synchronized with the GPU at a log, last or
    /// densify boundary, nil otherwise.
    public var loss: ImageLossValue?
    /// Gaussians in the view's frustum with at least one tile intersection.
    public var visibleGaussians: Int
    /// Densification applied after this step, or nil.
    public var densify: DensifySummary?

    public init(iteration: Int, loss: ImageLossValue?, visibleGaussians: Int, densify: DensifySummary?) {
        self.iteration = iteration
        self.loss = loss
        self.visibleGaussians = visibleGaussians
        self.densify = densify
    }
}

/// Synchronous Metal trainer. Not `Sendable`: it owns Metal objects and must stay in one isolation
/// domain (`TrainingSession` is the only owner in the app). One `step()` encodes one iteration.
public final class GPUTrainer {
    /// Budget cap on the Gaussian count: `TrainingMemoryBudget.maxGaussians`.
    public let maxGaussians: Int
    public let sceneExtent: SceneExtent
    /// Completed iterations.
    public private(set) var iteration: Int = 0
    /// Live Gaussians.
    public var gaussianCount: Int { store.count }
    /// Empty in Phase 3.
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
    /// One rgba32Float render target per distinct image size, created on first use. N distinct sizes
    /// allocate N textures, which `TrainingMemoryBudget.reservedImageBytes` does not fully count (it
    /// reserves one largest-view render target). Known gap for Task 16 and later: cap or share the
    /// targets once the view sizes are known to vary.
    private var renderTargets: [SIMD2<Int>: MTLTexture] = [:]
    /// Grow-only dL/drendered buffer, shared by the views.
    private var imageGradient: MTLBuffer?
    /// Command buffers committed since the last wait point.
    private var pending: [MTLCommandBuffer] = []
    /// 0-based index of the most recent step whose loss has not been validated yet.
    private var unvalidatedLossStep: Int?
    /// Set by the first error; every later call throws it.
    private var fatal: TrainingError?

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
        let viewSizes: [SIMD2<Int>] = dataset.views.map { SIMD2<Int>($0.camera.width, $0.camera.height) }
        let budget = TrainingMemoryBudget.maxGaussians(
            workingSetBytes: context.device.recommendedMaxWorkingSetSize,
            shDegree: configuration.shDegree,
            reservedBytes: TrainingMemoryBudget.reservedImageBytes(viewSizes: viewSizes),
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
        var targets: [MTLBuffer] = []
        targets.reserveCapacity(dataset.views.count)
        for view in dataset.views {
            let target = try metalCall { () throws(MetalError) -> MTLBuffer in
                try imageLoss.makeTarget(view.image)
            }
            targets.append(target)
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

        // Projection (waited inside the renderer) and the uncommitted rasterize command buffer.
        let pass = try metalCall { () throws(MetalError) -> ForwardPass in
            try self.renderer.encodeForward(camera: camera, settings: settings, to: texture)
        }
        let commandBuffer = pass.commandBuffer

        // Validate the previous step before this step's loss encode overwrites the loss state. The wait
        // is cheap (see the file comment); it throws `.nonFiniteLoss(iteration: index - 1)` for a NaN loss.
        if unvalidatedLossStep != nil {
            _ = try synchronize()
        }

        try metalCall { () throws(MetalError) -> Void in
            try self.imageLoss.encode(rendered: texture, target: self.targets[viewIndex],
                                      width: width, height: height,
                                      lambda: self.configuration.lambdaDSSIM,
                                      imageGradient: gradientImage, commandBuffer: commandBuffer)
        }
        for auxiliary in auxiliaryLosses {
            try auxiliary.encode(view: viewIndex, iteration: index, commandBuffer: commandBuffer)
        }

        // Statistics only while the densify window is open, as in CPUTrainer (index before increment).
        let collectsStatistics = index < configuration.densification.stopIteration
        let statisticsTarget: DensificationStatisticsBuffers? = collectsStatistics ? statistics : nil
        let currentGradients = gradients
        try metalCall { () throws(MetalError) -> Void in
            try self.renderer.encodeBackward(imageGradient: gradientImage, into: currentGradients,
                                             statistics: statisticsTarget, commandBuffer: commandBuffer)
        }

        // Adam with the pre-increment schedule and step = index + 1 (CPUAdam increments first).
        let rates = configuration.learningRates(iteration: index, sceneRadius: sceneExtent.radius)
        try metalCall { () throws(MetalError) -> Void in
            try self.adam.encode(parameters: self.store, gradients: currentGradients, state: self.adamState,
                                 rates: rates, step: index + 1, hyper: self.configuration.adam,
                                 commandBuffer: commandBuffer)
        }
        commandBuffer.commit()
        pending.append(commandBuffer)
        unvalidatedLossStep = index
        iteration = index + 1

        let completed = iteration
        let logInterval = configuration.logInterval
        let isLogStep = logInterval > 0 && completed % logInterval == 0
        let isLastStep = completed >= configuration.iterations
        let isDensifyStep = isDensificationBoundary(completed)
        var lossValue: ImageLossValue?
        if isLogStep || isLastStep || isDensifyStep {
            lossValue = try synchronize()
        }

        // Densification runs only after the wait above: the GPU is idle with respect to every buffer.
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
        return StepReport(iteration: completed, loss: lossValue,
                          visibleGaussians: pass.visibleGaussians, densify: summary)
    }

    /// Same boundary rule as CPUTrainer.step, applied to the completed iteration count.
    private func isDensificationBoundary(_ completed: Int) -> Bool {
        let policy = configuration.densification
        return policy.isEnabled
            && completed >= policy.startIteration
            && completed < policy.stopIteration
            && completed % policy.interval == 0
    }

    // MARK: - Synchronization

    /// Waits for every command buffer committed since the last wait point, throws if any failed, and
    /// reads and validates the loss of the most recent step if it has not been validated yet. Called at
    /// the start of each step (for the previous step) and at the boundaries (for the current step).
    private func synchronize() throws(TrainingError) -> ImageLossValue? {
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
            return nil
        }
        unvalidatedLossStep = nil
        // Safe: the command buffer of the most recent `encode` has completed.
        let value = imageLoss.readValue()
        guard value.loss.isFinite else {
            throw .nonFiniteLoss(iteration: stepIndex)
        }
        return value
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

    private func makeGradients() throws(TrainingError) -> GradientBuffers {
        try metalCall { () throws(MetalError) -> GradientBuffers in
            try GradientBuffers(context: self.context, count: self.store.capacity, shDegree: self.store.shDegree)
        }
    }
}
