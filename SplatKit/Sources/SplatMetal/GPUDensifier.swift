import Foundation
import Metal
import SplatCore

// GPU densification (Phase 3, Task 14). Mirrors `Densification` in SplatCore line by line; the
// kernels live in Shaders/Densify.metal. Layouts of the uniform structs below must match there.

// MARK: - Uniform mirrors

/// MSL `DensifyStatsUniforms`, 32 bytes.
struct DensifyStatsUniforms {
    /// x: width / 2, y: height / 2, zw unused.
    var halfSize: SIMD4<Float>
    /// x: Gaussian count, yzw unused.
    var info: SIMD4<UInt32>
}

/// MSL `DensifyDecideUniforms`, 48 bytes.
struct DensifyDecideUniforms {
    /// x: scene radius, y: prune opacity, z: prune scale fraction, w: gradient threshold.
    var params: SIMD4<Float>
    /// x: split scale fraction, yzw unused.
    var params2: SIMD4<Float>
    /// x: Gaussian count, y: iteration, z: prune scale start iteration, w unused.
    var info: SIMD4<Int32>
}

/// MSL `DensifyFinalizeUniforms`, 16 bytes.
struct DensifyFinalizeUniforms {
    /// x: Gaussian count, y: growths allowed, z: 1 when every Gaussian is kept, w unused.
    var info: SIMD4<UInt32>
}

/// MSL `DensifyScatterUniforms`, 32 bytes.
struct DensifyScatterUniforms {
    /// x: Gaussian count, y: floats per Gaussian in the group, z: stream seed, w unused.
    var info: SIMD4<UInt32>
    /// x: log of the split scale divisor, yzw unused.
    var extra: SIMD4<Float>
}

// MARK: - Statistics

/// Per-Gaussian densification statistics on the GPU: Σ screen-space gradient norms and visible counts.
///
/// Owned by the training loop. `encodeBackward(statistics:)` adds one view per call; `GPUDensifier.densify`
/// resets the statistics after it runs. Call `encodeReset` when a densify window starts.
/// `grow` and `read` follow the usual CPU/GPU rule: the GPU must be idle when they run.
public final class DensificationStatisticsBuffers {
    /// Float per Gaussian. Replaced by `grow`.
    public private(set) var gradientNormSums: MTLBuffer
    /// UInt32 per Gaussian. Replaced by `grow`.
    public private(set) var visibleCounts: MTLBuffer
    public private(set) var capacity: Int

    private let context: MetalContext
    private let accumulatePipeline: MTLComputePipelineState

    public init(context: MetalContext, capacity: Int) throws(MetalError) {
        precondition(capacity >= 0, "capacity must be non-negative")
        self.context = context
        accumulatePipeline = try context.makePipeline("accumulateDensifyStats")
        self.capacity = capacity
        gradientNormSums = try DensificationStatisticsBuffers.makeZeroedBuffer(context, capacity: capacity, elementSize: 4)
        visibleCounts = try DensificationStatisticsBuffers.makeZeroedBuffer(context, capacity: capacity, elementSize: 4)
    }

    /// Zeroes every entry on the GPU.
    public func encodeReset(commandBuffer: MTLCommandBuffer) throws(MetalError) {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw .commandFailed("cannot create blit encoder")
        }
        blit.fill(buffer: gradientNormSums, range: 0..<gradientNormSums.length, value: 0)
        blit.fill(buffer: visibleCounts, range: 0..<visibleCounts.length, value: 0)
        blit.endEncoding()
    }

    /// Reallocates both buffers to `newCapacity` entries. Contents are discarded (zeroed), so call this
    /// only before a reset or after reading the statistics. A `newCapacity` not above the current capacity
    /// is a no-op.
    public func grow(toCapacity newCapacity: Int) throws(MetalError) {
        guard newCapacity > capacity else { return }
        let sums = try DensificationStatisticsBuffers.makeZeroedBuffer(context, capacity: newCapacity, elementSize: 4)
        let counts = try DensificationStatisticsBuffers.makeZeroedBuffer(context, capacity: newCapacity, elementSize: 4)
        gradientNormSums = sums
        visibleCounts = counts
        capacity = newCapacity
    }

    /// The first `count` entries as CPU statistics. GPU work using these buffers must have completed.
    public func read(count: Int) -> DensificationStatistics {
        precondition(count >= 0 && count <= capacity, "count \(count) outside 0...\(capacity)")
        var statistics = DensificationStatistics(count: count)
        guard count > 0 else { return statistics }
        let sums = gradientNormSums.contents().bindMemory(to: Float.self, capacity: count)
        let counts = visibleCounts.contents().bindMemory(to: UInt32.self, capacity: count)
        for i in 0..<count {
            statistics.gradientNormSums[i] = sums[i]
            statistics.visibleCounts[i] = counts[i]
        }
        return statistics
    }

    /// Adds one view: `accumulator` is the Phase 2 accumulator (12 floats per Gaussian), `tilesTouched`
    /// the per-Gaussian tile counts of the same forward pass. Called by `SplatRenderer.encodeBackward`.
    func encodeAccumulate(accumulator: MTLBuffer, tilesTouched: MTLBuffer, count: Int,
                          width: Int, height: Int, commandBuffer: MTLCommandBuffer) throws(MetalError) {
        precondition(count <= capacity, "statistics hold \(capacity) Gaussians, need \(count)")
        guard count > 0 else { return }
        var uniforms = DensifyStatsUniforms(
            halfSize: SIMD4<Float>(Float(width) / 2, Float(height) / 2, 0, 0),
            info: SIMD4<UInt32>(UInt32(count), 0, 0, 0))
        let encoder = try context.makeComputeEncoder(commandBuffer)
        encoder.setComputePipelineState(accumulatePipeline)
        encoder.setBuffer(tilesTouched, offset: 0, index: 0)
        encoder.setBuffer(accumulator, offset: 0, index: 1)
        encoder.setBuffer(gradientNormSums, offset: 0, index: 2)
        encoder.setBuffer(visibleCounts, offset: 0, index: 3)
        encoder.setBytes(&uniforms, length: MemoryLayout<DensifyStatsUniforms>.size, index: 4)
        MetalContext.dispatch1D(encoder, pipeline: accumulatePipeline, count: count)
        encoder.endEncoding()
    }

    /// Zero-filled buffer of `capacity` entries. The CPU zeroes it because `makeBuffer` does not.
    private static func makeZeroedBuffer(_ context: MetalContext, capacity: Int,
                                         elementSize: Int) throws(MetalError) -> MTLBuffer {
        let buffer = try context.makeBuffer(length: capacity * elementSize)
        _ = buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: buffer.length)
        return buffer
    }
}

// MARK: - Densifier

/// Result of one `GPUDensifier.densify` call.
public struct DensifySummary: Sendable, Equatable {
    public var before: Int
    public var after: Int
    public var pruned: Int
    public var cloned: Int
    public var split: Int
    /// Growth candidates that became keeps because the budget was reached.
    public var demoted: Int
    /// True when every Gaussian was marked for pruning and the model was kept intact.
    public var keptAll: Bool
    /// True when the parameter, moment and statistics buffers were reallocated. Gradient buffers
    /// sized to the old capacity must be recreated by the caller.
    public var grewCapacity: Bool

    public init(before: Int, after: Int, pruned: Int, cloned: Int, split: Int, demoted: Int,
                keptAll: Bool, grewCapacity: Bool) {
        self.before = before
        self.after = after
        self.pruned = pruned
        self.cloned = cloned
        self.split = split
        self.demoted = demoted
        self.keptAll = keptAll
        self.grewCapacity = grewCapacity
    }
}

/// Densification on the GPU, mirroring `Densification.plan` and `Densification.apply`.
///
/// Pipeline (every step is a single command buffer, waited on before the CPU reads results):
/// 1. decide per Gaussian, then scan the grow and keep flags. The CPU derives the total growth,
///    the kept count and the granted growth, and predicts the output count.
/// 2. grow every buffer if the predicted count exceeds the capacity (CPU `memcpy` via `GaussianBuffers.grow`).
/// 3. finalize (never-empty rule, budget demotion) and scan the output counts into offsets. The GPU
///    count must equal the prediction; a mismatch throws before any parameter is written.
/// 4. reset the statistics, then per group scatter into one scratch buffer and blit the first
///    `after` entries back. Positions are scattered before logScales and rotations are replaced.
///
/// Precondition: the GPU is idle with respect to all passed buffers.
public final class GPUDensifier {
    private let context: MetalContext
    private let prefixSum: GPUPrefixSum
    private let decidePipeline: MTLComputePipelineState
    private let finalizePipeline: MTLComputePipelineState
    private let scatterCopyPipeline: MTLComputePipelineState
    private let scatterMomentsPipeline: MTLComputePipelineState
    private let scatterLogScalesPipeline: MTLComputePipelineState
    private let scatterPositionsPipeline: MTLComputePipelineState

    // UInt32 per input Gaussian. They are indexed by input index only, so they follow the input capacity
    // and never need to be resized during a densify call.
    private var actions: MTLBuffer?
    private var growFlags: MTLBuffer?
    private var growRanks: MTLBuffer?
    private var keptFlags: MTLBuffer?
    private var keptScans: MTLBuffer?
    private var outCounts: MTLBuffer?
    private var offsets: MTLBuffer?
    /// One group's compacted output: `floats` per output Gaussian, sized for the largest group.
    private var scratch: MTLBuffer?

    public init(context: MetalContext) throws(MetalError) {
        self.context = context
        prefixSum = try GPUPrefixSum(context: context)
        decidePipeline = try context.makePipeline("densifyDecide")
        finalizePipeline = try context.makePipeline("densifyFinalize")
        scatterCopyPipeline = try context.makePipeline("densifyScatterCopy")
        scatterMomentsPipeline = try context.makePipeline("densifyScatterMoments")
        scatterLogScalesPipeline = try context.makePipeline("densifyScatterLogScales")
        scatterPositionsPipeline = try context.makePipeline("densifyScatterPositions")
    }

    /// Which scatter kernel handles a parameter group.
    private enum ScatterKind {
        /// Rotations, opacity, SH: every output is a copy.
        case copy
        /// Adam moments: zero for split children and for the second clone child.
        case moments
        /// Log scales: split children are shifted by −log(divisor).
        case logScales
        /// Positions: split children are sampled around the parent.
        case positions
    }

    /// Values shared by every scatter in one densify call.
    private struct ScatterContext {
        let actions: MTLBuffer
        let offsets: MTLBuffer
        let scratch: MTLBuffer
        let count: Int
        let after: Int
        let logDivisor: Float
        let streamSeed: UInt32
        let commandBuffer: MTLCommandBuffer
    }

    /// Densifies the first `parameters.count` Gaussians in place. `m` and `v` (Adam moments) must have the
    /// same count, capacity and SH degree. `iteration` is the training iteration; `seed` is the training seed.
    /// Synchronous: waits for the GPU. On return `parameters`, `m` and `v` hold the new count and
    /// `statistics` is reset and covers the new count.
    ///
    /// Sizing and ownership for callers:
    /// - `m`, `v` and `statistics` are grown together with `parameters`, so they must have the same capacity.
    ///   Gradient buffers (`GradientBuffers`) must be sized to `parameters.capacity`, not to the count.
    /// - When `summary.grewCapacity` is true, every buffer object that was allocated at the old capacity is
    ///   stale. Callers must not keep references to `parameters`' groups, `m`, `v`, `statistics` or gradient
    ///   buffers across this call; re-read them from the objects after it returns.
    ///
    /// Failure: the decision and count check happen before any parameter write. If the scatter command
    /// buffer fails, some groups may already be overwritten while `count` is unchanged, so the model is
    /// inconsistent. The caller (training) must treat a thrown error from this call as fatal for this model.
    ///
    /// Throws `MetalError` on GPU failure. A count mismatch between the CPU prediction and the GPU scan
    /// is reported as `.commandFailed` before any parameter buffer is written.
    public func densify(parameters: GaussianBuffers, m: GaussianBuffers, v: GaussianBuffers,
                        statistics: DensificationStatisticsBuffers, policy: DensificationPolicy,
                        sceneRadius: Float, iteration: Int, seed: UInt32, maxGaussians: Int) throws(MetalError) -> DensifySummary {
        let n = parameters.count
        precondition(m.count == n && v.count == n, "moments must have the parameters' count")
        precondition(m.capacity == parameters.capacity && v.capacity == parameters.capacity,
                     "moments must have the parameters' capacity")
        precondition(m.shDegree == parameters.shDegree && v.shDegree == parameters.shDegree,
                     "moments must have the parameters' SH degree")
        precondition(statistics.capacity >= n, "statistics hold \(statistics.capacity) Gaussians, need \(n)")
        guard n > 0 else {
            return DensifySummary(before: 0, after: 0, pruned: 0, cloned: 0, split: 0, demoted: 0,
                                  keptAll: false, grewCapacity: false)
        }
        // The scratch buffer (up to maxFloats × capacity × 4 bytes, e.g. ~290 MB at 1.5 M Gaussians with
        // SH degree 3) is only needed inside this call. Releasing it keeps that memory from staying
        // allocated between densify windows, which run every few hundred iterations.
        defer { scratch = nil }

        let inputCapacity = parameters.capacity
        let indexBytes = 4 * inputCapacity
        let actionBuffer = try context.ensureBuffer(&actions, length: indexBytes)
        let growFlagBuffer = try context.ensureBuffer(&growFlags, length: indexBytes)
        let growRankBuffer = try context.ensureBuffer(&growRanks, length: indexBytes)
        let keptFlagBuffer = try context.ensureBuffer(&keptFlags, length: indexBytes)
        let keptScanBuffer = try context.ensureBuffer(&keptScans, length: indexBytes)
        let outCountBuffer = try context.ensureBuffer(&outCounts, length: indexBytes)
        let offsetBuffer = try context.ensureBuffer(&offsets, length: indexBytes)

        // Step 1: decide, then scan the flags. Commands are serial within a command buffer, so the two
        // scans can share the prefix-sum scratch buffers.
        guard let decideBuffer = context.queue.makeCommandBuffer() else {
            throw .commandFailed("cannot create command buffer")
        }
        var decideUniforms = DensifyDecideUniforms(
            params: SIMD4<Float>(sceneRadius, policy.pruneOpacity, policy.pruneScaleFraction, policy.gradientThreshold),
            params2: SIMD4<Float>(policy.splitScaleFraction, 0, 0, 0),
            info: SIMD4<Int32>(Int32(clamping: n), Int32(clamping: iteration),
                               Int32(clamping: policy.pruneScaleStartIteration), 0))
        let decideEncoder = try context.makeComputeEncoder(decideBuffer)
        decideEncoder.setComputePipelineState(decidePipeline)
        decideEncoder.setBuffer(parameters.logScales, offset: 0, index: 0)
        decideEncoder.setBuffer(parameters.opacityLogits, offset: 0, index: 1)
        decideEncoder.setBuffer(statistics.gradientNormSums, offset: 0, index: 2)
        decideEncoder.setBuffer(statistics.visibleCounts, offset: 0, index: 3)
        decideEncoder.setBuffer(actionBuffer, offset: 0, index: 4)
        decideEncoder.setBuffer(growFlagBuffer, offset: 0, index: 5)
        decideEncoder.setBuffer(keptFlagBuffer, offset: 0, index: 6)
        decideEncoder.setBytes(&decideUniforms, length: MemoryLayout<DensifyDecideUniforms>.size, index: 7)
        MetalContext.dispatch1D(decideEncoder, pipeline: decidePipeline, count: n)
        decideEncoder.endEncoding()
        try prefixSum.encodeExclusiveScan(input: growFlagBuffer, output: growRankBuffer, count: n,
                                          commandBuffer: decideBuffer)
        try prefixSum.encodeExclusiveScan(input: keptFlagBuffer, output: keptScanBuffer, count: n,
                                          commandBuffer: decideBuffer)
        decideBuffer.commit()
        decideBuffer.waitUntilCompleted()
        try Self.check(decideBuffer)

        // Totals of an exclusive scan: last exclusive value plus last input.
        let growFlagValues = growFlagBuffer.contents().bindMemory(to: UInt32.self, capacity: n)
        let growRankValues = growRankBuffer.contents().bindMemory(to: UInt32.self, capacity: n)
        let keptFlagValues = keptFlagBuffer.contents().bindMemory(to: UInt32.self, capacity: n)
        let keptScanValues = keptScanBuffer.contents().bindMemory(to: UInt32.self, capacity: n)
        let growthTotal = Int(growRankValues[n - 1]) + Int(growFlagValues[n - 1])
        let keptRaw = Int(keptScanValues[n - 1]) + Int(keptFlagValues[n - 1])

        // Mirrors `Densification.plan`. Never empty the model: when nothing survives, keep everything
        // and grant no growth. Otherwise the budget covers all non-prune slots, and granted growth is
        // the first `allowed` candidates in index order.
        let keptAll = keptRaw == 0
        let kept = keptAll ? n : keptRaw
        let growth = keptAll ? 0 : growthTotal
        let allowed = max(0, maxGaussians - kept)
        let granted = min(growth, allowed)
        let demoted = growth - granted
        let predictedAfter = kept + granted

        // Step 2: grow when the output does not fit. A 1.5x growth, capped at the budget but never below
        // the predicted count (this only exceeds the budget when the input already did).
        var grewCapacity = false
        if predictedAfter > parameters.capacity {
            let grown = (3 * parameters.capacity + 1) / 2
            let target = max(predictedAfter, min(maxGaussians, grown))
            try parameters.grow(toCapacity: target)
            try m.grow(toCapacity: target)
            try v.grow(toCapacity: target)
            grewCapacity = true
        }
        // Statistics must cover the output even when no parameter buffer grew, because statistics.capacity can
        // lag parameters.capacity. Grow is a no-op when already large enough; the contents are reset below.
        try statistics.grow(toCapacity: max(predictedAfter, parameters.capacity))

        let maxFloats = GaussianBuffers.floatsPerGaussian(shDegree: parameters.shDegree).max() ?? 1
        let scratchBuffer = try context.ensureBuffer(&scratch, length: 4 * maxFloats * parameters.capacity)

        // Step 3: finalize and scan the output counts. Checked before any scatter writes.
        guard let finalizeBuffer = context.queue.makeCommandBuffer() else {
            throw .commandFailed("cannot create command buffer")
        }
        var finalizeUniforms = DensifyFinalizeUniforms(
            info: SIMD4<UInt32>(UInt32(n), UInt32(clamping: allowed), keptAll ? 1 : 0, 0))
        let finalizeEncoder = try context.makeComputeEncoder(finalizeBuffer)
        finalizeEncoder.setComputePipelineState(finalizePipeline)
        finalizeEncoder.setBuffer(actionBuffer, offset: 0, index: 0)
        finalizeEncoder.setBuffer(growRankBuffer, offset: 0, index: 1)
        finalizeEncoder.setBuffer(outCountBuffer, offset: 0, index: 2)
        finalizeEncoder.setBytes(&finalizeUniforms, length: MemoryLayout<DensifyFinalizeUniforms>.size, index: 3)
        MetalContext.dispatch1D(finalizeEncoder, pipeline: finalizePipeline, count: n)
        finalizeEncoder.endEncoding()
        try prefixSum.encodeExclusiveScan(input: outCountBuffer, output: offsetBuffer, count: n,
                                          commandBuffer: finalizeBuffer)
        finalizeBuffer.commit()
        finalizeBuffer.waitUntilCompleted()
        try Self.check(finalizeBuffer)

        let outCountValues = outCountBuffer.contents().bindMemory(to: UInt32.self, capacity: n)
        let offsetValues = offsetBuffer.contents().bindMemory(to: UInt32.self, capacity: n)
        let gpuAfter = Int(offsetValues[n - 1]) + Int(outCountValues[n - 1])
        guard gpuAfter == predictedAfter else {
            throw .commandFailed("densification count mismatch: GPU \(gpuAfter), CPU \(predictedAfter)")
        }

        // Step 4: reset the statistics, scatter every group into scratch and blit the first `after`
        // entries back. Order: positions (reads old logScales and rotations), rotations, opacity, SH, and
        // logScales last. Moments are independent of each other.
        guard let scatterBuffer = context.queue.makeCommandBuffer() else {
            throw .commandFailed("cannot create command buffer")
        }
        try statistics.encodeReset(commandBuffer: scatterBuffer)
        let scatterContext = ScatterContext(
            actions: actionBuffer,
            offsets: offsetBuffer,
            scratch: scratchBuffer,
            count: n,
            after: predictedAfter,
            logDivisor: log(policy.splitScaleDivisor),
            streamSeed: seed &+ UInt32(truncatingIfNeeded: iteration),
            commandBuffer: scatterBuffer)
        for index in [0, 2, 3, 4, 1] {
            let kind: ScatterKind = index == 0 ? .positions : (index == 1 ? .logScales : .copy)
            try encodeScatter(kind, store: parameters, index: index, context: scatterContext)
        }
        for index in 0..<5 {
            try encodeScatter(.moments, store: m, index: index, context: scatterContext)
        }
        for index in 0..<5 {
            try encodeScatter(.moments, store: v, index: index, context: scatterContext)
        }
        scatterBuffer.commit()
        scatterBuffer.waitUntilCompleted()
        try Self.check(scatterBuffer)

        // Read the final decisions for the summary.
        var pruned = 0
        var cloned = 0
        var split = 0
        let finalActions = actionBuffer.contents().bindMemory(to: UInt32.self, capacity: n)
        for i in 0..<n {
            let action = finalActions[i]
            if action == DensifyAction.prune.rawValue {
                pruned += 1
            } else if action == DensifyAction.clone.rawValue {
                cloned += 1
            } else if action == DensifyAction.split.rawValue {
                split += 1
            }
        }

        parameters.count = gpuAfter
        m.count = gpuAfter
        v.count = gpuAfter

        return DensifySummary(before: n, after: gpuAfter, pruned: pruned, cloned: cloned, split: split,
                              demoted: demoted, keptAll: keptAll, grewCapacity: grewCapacity)
    }

    /// Encodes one group's scatter into scratch, then the blit of the first `after` entries back into the
    /// group. Destination = source group, because the output replaces the input.
    private func encodeScatter(_ kind: ScatterKind, store: GaussianBuffers, index: Int,
                               context scatter: ScatterContext) throws(MetalError) {
        let floats = GaussianBuffers.floatsPerGaussian(shDegree: store.shDegree)[index]
        var uniforms = DensifyScatterUniforms(
            info: SIMD4<UInt32>(UInt32(scatter.count), UInt32(floats), scatter.streamSeed, 0),
            extra: SIMD4<Float>(scatter.logDivisor, 0, 0, 0))
        let encoder = try context.makeComputeEncoder(scatter.commandBuffer)
        switch kind {
        case .positions:
            encoder.setComputePipelineState(scatterPositionsPipeline)
            encoder.setBuffer(store.positions, offset: 0, index: 0)
            encoder.setBuffer(store.logScales, offset: 0, index: 1)
            encoder.setBuffer(store.rotations, offset: 0, index: 2)
            encoder.setBuffer(scatter.actions, offset: 0, index: 3)
            encoder.setBuffer(scatter.offsets, offset: 0, index: 4)
            encoder.setBuffer(scatter.scratch, offset: 0, index: 5)
            encoder.setBytes(&uniforms, length: MemoryLayout<DensifyScatterUniforms>.size, index: 6)
        case .copy, .moments, .logScales:
            let pipeline: MTLComputePipelineState
            switch kind {
            case .copy: pipeline = scatterCopyPipeline
            case .moments: pipeline = scatterMomentsPipeline
            default: pipeline = scatterLogScalesPipeline
            }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(store.groups[index], offset: 0, index: 0)
            encoder.setBuffer(scatter.actions, offset: 0, index: 1)
            encoder.setBuffer(scatter.offsets, offset: 0, index: 2)
            encoder.setBuffer(scatter.scratch, offset: 0, index: 3)
            encoder.setBytes(&uniforms, length: MemoryLayout<DensifyScatterUniforms>.size, index: 4)
        }
        MetalContext.dispatch1D(encoder, pipeline: pipelineFor(kind), count: scatter.count)
        encoder.endEncoding()

        let destination = store.groups[index]
        let bytes = scatter.after * floats * MemoryLayout<Float>.size
        guard bytes > 0 else { return }
        guard let blit = scatter.commandBuffer.makeBlitCommandEncoder() else {
            throw .commandFailed("cannot create blit encoder")
        }
        blit.copy(from: scatter.scratch, sourceOffset: 0, to: destination, destinationOffset: 0, size: bytes)
        blit.endEncoding()
    }

    private func pipelineFor(_ kind: ScatterKind) -> MTLComputePipelineState {
        switch kind {
        case .positions: return scatterPositionsPipeline
        case .copy: return scatterCopyPipeline
        case .moments: return scatterMomentsPipeline
        case .logScales: return scatterLogScalesPipeline
        }
    }

    private static func check(_ commandBuffer: MTLCommandBuffer) throws(MetalError) {
        if commandBuffer.status == .error {
            throw .commandFailed(commandBuffer.error?.localizedDescription ?? "unknown GPU error")
        }
    }
}
