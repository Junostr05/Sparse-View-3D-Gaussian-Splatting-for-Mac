import Foundation
import Metal
import SplatCore

/// GPU first and second Adam moments. Each is a `GaussianBuffers` store with the same SoA layout as
/// the parameters. `count` is kept equal to the parameter count by `GPUAdam.encode`.
public final class AdamBuffers {
    public let m: GaussianBuffers
    public let v: GaussianBuffers

    /// Zero-filled moments with room for `capacity` Gaussians.
    public init(context: MetalContext, capacity: Int, shDegree: Int) throws(MetalError) {
        m = try GaussianBuffers(context: context, capacity: capacity, shDegree: shDegree)
        v = try GaussianBuffers(context: context, capacity: capacity, shDegree: shDegree)
    }

    /// Grows both stores, copying the first `count` moments. GPU work using them must have completed.
    public func grow(toCapacity newCapacity: Int) throws(MetalError) {
        try m.grow(toCapacity: newCapacity)
        try v.grow(toCapacity: newCapacity)
    }
}

/// Metal Adam optimizer (PyTorch formulation with bias correction). Mirrors `CPUAdam` in SplatCore.
public final class GPUAdam {
    private let context: MetalContext
    private let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws(MetalError) {
        self.context = context
        pipeline = try context.makePipeline("adamStep")
    }

    /// Encodes one Adam step for all five parameter groups over `parameters.count` Gaussians.
    /// `step` is the already-incremented counter (>= 1). Gradients and moments must have capacity for
    /// `parameters.count` Gaussians and the same SH degree. Only the first `parameters.count`
    /// Gaussians of each group are touched; the spare capacity keeps its values.
    public func encode(parameters: GaussianBuffers, gradients: GradientBuffers, state: AdamBuffers,
                       rates: ResolvedLearningRates, step: Int, hyper: AdamHyperparameters,
                       commandBuffer: MTLCommandBuffer) throws(MetalError) {
        precondition(step >= 1, "Adam step counter starts at 1")
        let n = parameters.count
        precondition(gradients.capacity >= n, "gradient capacity \(gradients.capacity) is below \(n) Gaussians")
        precondition(state.m.capacity >= n && state.v.capacity >= n,
                     "moment capacity is below \(n) Gaussians")
        precondition(gradients.shDegree == parameters.shDegree
                         && state.m.shDegree == parameters.shDegree
                         && state.v.shDegree == parameters.shDegree,
                     "SH degree mismatch between parameters, gradients and moments")
        state.m.count = n
        state.v.count = n
        guard n > 0 else { return }

        // Same Float arithmetic as CPUAdam.step: bias corrections in Double, then rounded to Float.
        let t = Double(step)
        let bc1: Float = Float(1 - pow(Double(hyper.beta1), t))
        let bc2: Float = Float(1 - pow(Double(hyper.beta2), t))
        let invSqrtBC2: Float = 1 / bc2.squareRoot()

        let floats = GaussianBuffers.floatsPerGaussian(shDegree: parameters.shDegree)
        let parameterBuffers = parameters.groups
        let gradientBuffers = [gradients.positions, gradients.logScales, gradients.rotations,
                               gradients.opacityLogits, gradients.sh]
        let firstMoments = state.m.groups
        let secondMoments = state.v.groups

        // Per group: head rate (first headCount floats of each Gaussian), tail rate, and headCount.
        // Only SH splits DC (3 floats) from the rest; every other group uses its rate for all floats.
        let headRates: [Float] = [rates.position, rates.logScale, rates.rotation, rates.opacity, rates.shDC]
        let tailRates: [Float] = [rates.position, rates.logScale, rates.rotation, rates.opacity, rates.shRest]
        let headCounts: [Int] = [floats[0], floats[1], floats[2], floats[3], 3]

        let encoder = try context.makeComputeEncoder(commandBuffer)
        encoder.setComputePipelineState(pipeline)
        for group in 0..<5 {
            let stride = floats[group]
            let total = n * stride
            precondition(total <= Int(UInt32.max), "parameter group too large for the Adam kernel")
            var uniforms = AdamUniforms(
                rates: SIMD4<Float>(headRates[group] / bc1, tailRates[group] / bc1, invSqrtBC2, hyper.epsilon),
                betas: SIMD4<Float>(hyper.beta1, hyper.beta2, 0, 0),
                layout: SIMD4<UInt32>(UInt32(total), UInt32(stride), UInt32(headCounts[group]), 0))
            encoder.setBuffer(parameterBuffers[group], offset: 0, index: 0)
            encoder.setBuffer(gradientBuffers[group], offset: 0, index: 1)
            encoder.setBuffer(firstMoments[group], offset: 0, index: 2)
            encoder.setBuffer(secondMoments[group], offset: 0, index: 3)
            encoder.setBytes(&uniforms, length: MemoryLayout<AdamUniforms>.stride, index: 4)
            MetalContext.dispatch1D(encoder, pipeline: pipeline, count: total)
        }
        encoder.endEncoding()
    }
}

/// Mirror of `AdamUniforms` in Shaders/Adam.metal.
private struct AdamUniforms {
    var rates: SIMD4<Float>
    var betas: SIMD4<Float>
    var layout: SIMD4<UInt32>
}
