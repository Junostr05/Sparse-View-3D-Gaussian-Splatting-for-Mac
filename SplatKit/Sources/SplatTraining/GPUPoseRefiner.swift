import Foundation
import Metal
import SplatCore
import SplatMetal

// Joint camera pose refinement on the GPU (v1.2). Each training view's pose gradient is accumulated on the GPU by its
// backward pass (six sums per view, float atomics). The CPU applies one Adam step per view and folds the correction
// into the view's camera at the trainer's wait points, then zeroes the sums. The Gaussian gradients are not touched.
//
// Ordering: `GPUTrainer` waits for the pending gradients before it reads a camera whose view has one pending, so the
// correction lands before that view is rendered again. Every fold therefore uses only gradients computed with the
// camera that is folded, and the mean over the accumulations of a view is exact (one per view in the trainer's loop).
//
// Pseudo-views are not refined (their cameras are interpolated, not measured), and the held-out view of
// leave-one-out is not in the training set.

final class GPUPoseRefiner {
    private let settings: PoseRefinementSettings
    private let hyper: AdamHyperparameters
    private let context: MetalContext
    private let reducer: GPUPoseGradient
    /// Six float sums per training view in the order (dL/dω, dL/dτ). Accumulated on the GPU, read and zeroed on the CPU.
    private let sums: MTLBuffer
    /// Six floats per Gaussian (grow-only), written by `projectBackward` and reduced by the view's backward.
    private var scratch: MTLBuffer?
    /// Pose-gradient accumulations per view since the last apply.
    private var counts: [Int]
    /// Sum of `counts`.
    private var pendingCount = 0
    /// Adam state of each view's pose.
    private var states: [PoseAdamState]

    init(context: MetalContext, viewCount: Int, settings: PoseRefinementSettings,
         hyper: AdamHyperparameters) throws(MetalError) {
        self.context = context
        self.settings = settings
        self.hyper = hyper
        reducer = try GPUPoseGradient(context: context)
        sums = try context.makeBuffer(length: max(1, viewCount) * 6 * MemoryLayout<Float>.size)
        memset(sums.contents(), 0, sums.length)
        counts = [Int](repeating: 0, count: viewCount)
        states = [PoseAdamState](repeating: PoseAdamState(), count: viewCount)
    }

    /// Adam state of every view's pose, in view order. Call only when no gradient is pending (after a wait).
    func snapshotStates() -> [PoseAdamState] {
        states
    }

    /// Replaces the Adam state of every view's pose. Call on a fresh refiner, before any gradient is accumulated.
    func restoreStates(_ restored: [PoseAdamState]) {
        precondition(restored.count == states.count, "one pose state per view")
        precondition(pendingCount == 0, "pose state is restored only before any gradient is pending")
        states = restored
    }

    /// True when some view has a gradient that `applyPending` has not consumed yet.
    var hasPendingGradient: Bool {
        pendingCount > 0
    }

    /// The pose sink of training view `viewIndex`. The scratch buffer is grown to hold `capacity` Gaussians.
    func sink(viewIndex: Int, capacity: Int) throws(MetalError) -> PoseGradientSink {
        let bytes = max(1, capacity) * 6 * MemoryLayout<Float>.size
        let buffer: MTLBuffer
        if let existing = scratch, existing.length >= bytes {
            buffer = existing
        } else {
            buffer = try context.makeBuffer(length: bytes)
            scratch = buffer
        }
        return PoseGradientSink(reducer: reducer, scratch: buffer, sums: sums, base: 6 * viewIndex)
    }

    /// Records that a backward pass of `viewIndex` was encoded with a sink. Call only after `encodeBackward` dispatched it.
    func noteAccumulation(viewIndex: Int) {
        counts[viewIndex] += 1
        pendingCount += 1
    }

    /// Applies one Adam step to every view with pending gradient and folds the step into `cameras` (the views' cameras,
    /// in view order). The GPU must be idle with respect to `sums`: call after waiting on the pending command buffers.
    /// The learning rates are `rotationLearningRate` and `translationLearningRate × sceneRadius`.
    func applyPending(cameras: inout [Camera], sceneRadius: Float) {
        guard pendingCount > 0 else { return }
        let pointer = sums.contents().bindMemory(to: Float.self, capacity: counts.count * 6)
        let rotationRate = Double(settings.rotationLearningRate)
        let translationRate = Double(settings.translationLearningRate * sceneRadius)
        for view in counts.indices where counts[view] > 0 {
            let average: Double = Double(counts[view])
            let base: Int = 6 * view
            // GPU order is (dL/dω, dL/dτ); the sums are divided by the accumulation count.
            var rotation = SIMD3<Double>(0, 0, 0)
            var translation = SIMD3<Double>(0, 0, 0)
            for axis in 0..<3 {
                let omegaSum: Double = Double(pointer[base + axis])
                let tauSum: Double = Double(pointer[base + 3 + axis])
                rotation[axis] = omegaSum / average
                translation[axis] = tauSum / average
            }
            let gradient = PoseGradient(rotation: rotation, translation: translation)
            let delta = PoseRefinement.adamStep(&states[view], gradient: gradient,
                                                rotationRate: rotationRate, translationRate: translationRate,
                                                hyper: hyper)
            cameras[view] = PoseRefinement.fold(cameras[view], rotation: delta.rotation,
                                                translation: delta.translation)
            counts[view] = 0
        }
        pendingCount = 0
        memset(sums.contents(), 0, sums.length)
    }
}
