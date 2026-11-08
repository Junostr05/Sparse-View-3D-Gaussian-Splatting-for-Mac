import Metal
import SplatCore

/// Value of one GPU depth loss evaluation (Phase 4 P7), read back after its command buffer completed.
public struct DepthLossValue: Sendable, Equatable {
    /// `λ_P·(1 − ρ)` (0 when degenerate) plus `λ_L1·l1`.
    public var loss: Double
    /// ρ, 0 when degenerate.
    public var pearson: Double
    /// Normalised L1 value before weighting, 0 when not applicable.
    public var l1: Double
    /// Number of pixels in the mask.
    public var validPixels: Int
    /// True when the Pearson term was skipped (too few pixels or a near-constant signal).
    public var degenerate: Bool

    public init(loss: Double, pearson: Double, l1: Double, validPixels: Int, degenerate: Bool) {
        self.loss = loss
        self.pearson = pearson
        self.l1 = l1
        self.validPixels = validPixels
        self.degenerate = degenerate
    }
}

/// Mirrors `DepthLossUniforms` in `Shaders/Depth.metal` (48 bytes).
struct DepthLossUniforms {
    /// λ_P, λ_L1, alphaMask, nearPlane.
    var weights: SIMD4<Float>
    /// l1Normalizer, domain (0 depth, 1 inverse depth), L1 applies (1/0), unused.
    var extra: SIMD4<Float>
    /// width, height, minimumPixels, slot.
    var size: SIMD4<UInt32>
}

/// Mirrors `DepthPartial` in `Shaders/Depth.metal` (48 bytes), one per threadgroup.
struct DepthPartial {
    var sums: SIMD4<Float>
    var counts: SIMD4<UInt32>
    /// (Σw over the mask, Σw over measured pixels, 0, 0).
    var weights: SIMD4<Float>
}

/// Mirrors `DepthStats` in `Shaders/Depth.metal` (64 bytes), one per slot.
struct DepthStats {
    var counts: SIMD4<UInt32>
    var means: SIMD4<Float>
    var spread: SIMD4<Float>
    /// (W, W1, 0, 0): the weight sums the means and the L1 value are divided by.
    var weightSums: SIMD4<Float>
}

/// A depth prior uploaded to the GPU: target, metric depth and per-pixel weight, one Float per pixel.
/// Created by `GPUDepthLoss.makePrior`.
public final class GPUDepthPrior {
    public let width: Int
    public let height: Int
    public let domain: DepthDomain
    public let isMeasuredMetric: Bool
    public let l1Normalizer: Float

    /// Prior in `domain`; ≤ 0 or non-finite means invalid.
    let targetBuffer: MTLBuffer
    /// Camera-z metric depth; 0 means invalid. All zeros when the prior has no metric depth.
    let metricBuffer: MTLBuffer
    /// True when the prior carries metric depth, which the L1 term needs.
    let hasMetric: Bool
    /// Per-pixel loss weight (`DepthPrior.weights`), all 1 when the prior has none.
    let weightBuffer: MTLBuffer

    init(width: Int, height: Int, domain: DepthDomain, isMeasuredMetric: Bool, l1Normalizer: Float,
         targetBuffer: MTLBuffer, metricBuffer: MTLBuffer, hasMetric: Bool, weightBuffer: MTLBuffer) {
        self.width = width
        self.height = height
        self.domain = domain
        self.isMeasuredMetric = isMeasuredMetric
        self.l1Normalizer = l1Normalizer
        self.targetBuffer = targetBuffer
        self.metricBuffer = metricBuffer
        self.hasMetric = hasMetric
        self.weightBuffer = weightBuffer
    }
}

/// GPU counterpart of `DepthLoss` (Pearson in the prior's domain plus an optional normalised L1 against
/// measured metric depth). Gradients are written per pixel with respect to `(Draw, A)`.
///
/// Two slots (0 = training view, 1 = pseudo-view) can be encoded into one command buffer and read back
/// independently. Call `readValue(slot:)` only after the command buffer of the most recent `encode` into
/// that slot has completed.
public final class GPUDepthLoss {
    /// Number of independent result slots.
    public static let slotCount = 2
    /// Threadgroup edge length of the reduction kernels (16×16 threads).
    private static let tileSize = 16

    private let context: MetalContext
    private let momentsAPipeline: MTLComputePipelineState
    private let finalizeAPipeline: MTLComputePipelineState
    private let momentsBPipeline: MTLComputePipelineState
    private let finalizeBPipeline: MTLComputePipelineState
    private let gradientPipeline: MTLComputePipelineState

    /// `slotCount` consecutive `DepthStats` entries, `storageModeShared`.
    private let statsBuffer: MTLBuffer
    /// Per-slot grow-only `DepthPartial` scratch, one entry per threadgroup.
    private var partialBuffers: [MTLBuffer?] = [nil, nil]
    /// Grow-only buffer of ones shared by every prior without weights, so the unweighted path needs no
    /// extra memory per view. A prior keeps the buffer it was given; growing replaces it for later priors.
    private var onesBuffer: MTLBuffer?

    /// A buffer of at least `count` ones (`Float`), shared between unweighted priors.
    private func sharedOnes(count: Int) throws(MetalError) -> MTLBuffer {
        let length: Int = max(count, 1) * MemoryLayout<Float>.stride
        if let existing = onesBuffer, existing.length >= length {
            return existing
        }
        let buffer = try context.makeBuffer(length: length)
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: max(count, 1))
        for i in 0..<max(count, 1) {
            pointer[i] = 1
        }
        onesBuffer = buffer
        return buffer
    }

    public init(context: MetalContext) throws(MetalError) {
        self.context = context
        momentsAPipeline = try context.makePipeline("depthMomentsA", minThreads: Self.tileSize * Self.tileSize)
        finalizeAPipeline = try context.makePipeline("depthFinalizeA")
        momentsBPipeline = try context.makePipeline("depthMomentsB", minThreads: Self.tileSize * Self.tileSize)
        finalizeBPipeline = try context.makePipeline("depthFinalizeB")
        gradientPipeline = try context.makePipeline("depthGradient")
        statsBuffer = try context.makeBuffer(length: Self.slotCount * MemoryLayout<DepthStats>.stride)
    }

    /// Uploads the prior's target, metric depth and weights. Metric depth is zero-filled when absent,
    /// weights are one-filled (the unweighted loss).
    public func makePrior(_ prior: DepthPrior) throws(MetalError) -> GPUDepthPrior {
        let count: Int = prior.width * prior.height
        precondition(prior.correlationTarget.count == count, "prior target must match its size")
        let floatStride: Int = MemoryLayout<Float>.stride

        let targetBuffer = try context.makeBuffer(length: count * floatStride)
        let targetPointer = targetBuffer.contents().bindMemory(to: Float.self, capacity: max(count, 1))
        for i in 0..<count {
            targetPointer[i] = prior.correlationTarget[i]
        }

        let metricBuffer = try context.makeBuffer(length: count * floatStride)
        let metricPointer = metricBuffer.contents().bindMemory(to: Float.self, capacity: max(count, 1))
        if let metric = prior.metricDepth {
            precondition(metric.count == count, "metric depth must match the pixel count")
            for i in 0..<count {
                metricPointer[i] = metric[i]
            }
        } else {
            for i in 0..<count {
                metricPointer[i] = 0
            }
        }

        let weightBuffer: MTLBuffer
        if let weights = prior.weights {
            precondition(weights.count == count, "depth weights must match the pixel count")
            weightBuffer = try context.makeBuffer(length: count * floatStride)
            let weightPointer = weightBuffer.contents().bindMemory(to: Float.self, capacity: max(count, 1))
            for i in 0..<count {
                weightPointer[i] = weights[i]
            }
        } else {
            weightBuffer = try sharedOnes(count: count)
        }

        return GPUDepthPrior(width: prior.width, height: prior.height, domain: prior.domain,
                             isMeasuredMetric: prior.isMeasuredMetric,
                             l1Normalizer: prior.l1Normalizer,
                             targetBuffer: targetBuffer, metricBuffer: metricBuffer,
                             hasMetric: prior.metricDepth != nil, weightBuffer: weightBuffer)
    }

    /// Encodes the loss of `renderedDepth` (float2 `(Draw, A)` per pixel) against `prior` into `slot`.
    /// Writes `depthGradient` (float2 `(∂L/∂Draw, ∂L/∂A)` per pixel, overwritten everywhere).
    public func encode(renderedDepth: MTLBuffer, prior: GPUDepthPrior, weights: DepthLossWeights, slot: Int,
                       depthGradient: MTLBuffer, commandBuffer: MTLCommandBuffer) throws(MetalError) {
        guard slot >= 0, slot < Self.slotCount else {
            throw .commandFailed("depth loss slot \(slot) is out of range")
        }
        let width: Int = prior.width
        let height: Int = prior.height
        guard width > 0, height > 0 else {
            throw .commandFailed("depth prior has no pixels")
        }
        let count: Int = width * height
        let pixelStride: Int = MemoryLayout<SIMD2<Float>>.stride
        guard renderedDepth.length >= count * pixelStride else {
            throw .commandFailed("renderedDepth must hold width·height (Draw, A) pairs")
        }
        guard depthGradient.length >= count * pixelStride else {
            throw .commandFailed("depthGradient must hold width·height float2 values")
        }

        let tile: Int = Self.tileSize
        let tilesX: Int = (width + tile - 1) / tile
        let tilesY: Int = (height + tile - 1) / tile
        let groups: Int = tilesX * tilesY
        let partials = try context.ensureBuffer(&partialBuffers[slot],
                                                length: groups * MemoryLayout<DepthPartial>.stride)

        let l1Applies: Bool = prior.isMeasuredMetric && prior.hasMetric && prior.l1Normalizer > 0 && weights.l1 > 0
        let domainValue: Float = prior.domain == .inverseDepth ? 1 : 0
        var uniforms = DepthLossUniforms(
            weights: SIMD4<Float>(weights.pearson, weights.l1, weights.alphaMask, weights.nearPlane),
            extra: SIMD4<Float>(prior.l1Normalizer, domainValue, l1Applies ? 1 : 0, 0),
            size: SIMD4<UInt32>(UInt32(width), UInt32(height),
                                UInt32(clamping: max(0, weights.minimumPixels)), UInt32(slot)))
        let uniformLength: Int = MemoryLayout<DepthLossUniforms>.stride
        let statsOffset: Int = slot * MemoryLayout<DepthStats>.stride

        let gridSize = MTLSize(width: tilesX, height: tilesY, depth: 1)
        let groupSize = MTLSize(width: tile, height: tile, depth: 1)
        let singleThread = MTLSize(width: 1, height: 1, depth: 1)

        // Stage 1: per-threadgroup sums over the mask.
        let momentsAEncoder = try context.makeComputeEncoder(commandBuffer)
        momentsAEncoder.setComputePipelineState(momentsAPipeline)
        momentsAEncoder.setBuffer(renderedDepth, offset: 0, index: 0)
        momentsAEncoder.setBuffer(prior.targetBuffer, offset: 0, index: 1)
        momentsAEncoder.setBuffer(prior.metricBuffer, offset: 0, index: 2)
        momentsAEncoder.setBuffer(partials, offset: 0, index: 3)
        momentsAEncoder.setBytes(&uniforms, length: uniformLength, index: 4)
        momentsAEncoder.setBuffer(prior.weightBuffer, offset: 0, index: 5)
        momentsAEncoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
        momentsAEncoder.endEncoding()

        // Stage 2: means and the normalised L1 value.
        let finalizeAEncoder = try context.makeComputeEncoder(commandBuffer)
        finalizeAEncoder.setComputePipelineState(finalizeAPipeline)
        finalizeAEncoder.setBuffer(partials, offset: 0, index: 0)
        finalizeAEncoder.setBuffer(statsBuffer, offset: statsOffset, index: 1)
        finalizeAEncoder.setBytes(&uniforms, length: uniformLength, index: 2)
        finalizeAEncoder.dispatchThreadgroups(singleThread, threadsPerThreadgroup: singleThread)
        finalizeAEncoder.endEncoding()

        // Stage 3: centred second moments about the means.
        let momentsBEncoder = try context.makeComputeEncoder(commandBuffer)
        momentsBEncoder.setComputePipelineState(momentsBPipeline)
        momentsBEncoder.setBuffer(renderedDepth, offset: 0, index: 0)
        momentsBEncoder.setBuffer(prior.targetBuffer, offset: 0, index: 1)
        momentsBEncoder.setBuffer(statsBuffer, offset: statsOffset, index: 2)
        momentsBEncoder.setBuffer(partials, offset: 0, index: 3)
        momentsBEncoder.setBytes(&uniforms, length: uniformLength, index: 4)
        momentsBEncoder.setBuffer(prior.weightBuffer, offset: 0, index: 5)
        momentsBEncoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
        momentsBEncoder.endEncoding()

        // Stage 4: correlation, degenerate flag and loss value.
        let finalizeBEncoder = try context.makeComputeEncoder(commandBuffer)
        finalizeBEncoder.setComputePipelineState(finalizeBPipeline)
        finalizeBEncoder.setBuffer(partials, offset: 0, index: 0)
        finalizeBEncoder.setBuffer(statsBuffer, offset: statsOffset, index: 1)
        finalizeBEncoder.setBytes(&uniforms, length: uniformLength, index: 2)
        finalizeBEncoder.dispatchThreadgroups(singleThread, threadsPerThreadgroup: singleThread)
        finalizeBEncoder.endEncoding()

        // Stage 5: per-pixel gradient.
        let gradientEncoder = try context.makeComputeEncoder(commandBuffer)
        gradientEncoder.setComputePipelineState(gradientPipeline)
        gradientEncoder.setBuffer(renderedDepth, offset: 0, index: 0)
        gradientEncoder.setBuffer(prior.targetBuffer, offset: 0, index: 1)
        gradientEncoder.setBuffer(prior.metricBuffer, offset: 0, index: 2)
        gradientEncoder.setBuffer(statsBuffer, offset: statsOffset, index: 3)
        gradientEncoder.setBuffer(depthGradient, offset: 0, index: 4)
        gradientEncoder.setBytes(&uniforms, length: uniformLength, index: 5)
        gradientEncoder.setBuffer(prior.weightBuffer, offset: 0, index: 6)
        gradientEncoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                                        threadsPerThreadgroup: groupSize)
        gradientEncoder.endEncoding()
    }

    /// Value of the most recent `encode` into `slot`. Precondition: that command buffer has completed.
    public func readValue(slot: Int) -> DepthLossValue {
        precondition(slot >= 0 && slot < Self.slotCount, "depth loss slot out of range")
        let pointer = statsBuffer.contents().bindMemory(to: DepthStats.self, capacity: Self.slotCount)
        let stats: DepthStats = pointer[slot]
        return DepthLossValue(loss: Double(stats.spread.w),
                              pearson: Double(stats.spread.z),
                              l1: Double(stats.means.z),
                              validPixels: Int(stats.counts.x),
                              degenerate: stats.counts.z != 0)
    }
}
