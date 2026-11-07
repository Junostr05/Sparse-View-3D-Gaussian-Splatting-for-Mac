import Metal
import SplatCore

// GPU MCMC kernels (v1.2). Mirror `MCMCRegularizers` and `MCMCNoise` in SplatCore; the kernels live in
// Shaders/MCMC.metal. Layouts of the uniform structs below must match there.

/// MSL `MCMCRegularizerUniforms`, 32 bytes.
private struct MCMCRegularizerUniforms {
    /// x: opacityWeight / n, y: scaleWeight / n, zw unused.
    var params: SIMD4<Float>
    /// x: Gaussian count, yzw unused.
    var info: SIMD4<UInt32>
}

/// MSL `MCMCNoiseUniforms`, 32 bytes.
private struct MCMCNoiseUniforms {
    /// x: step, y: gate steepness, z: gate opacity x0, w unused.
    var params: SIMD4<Float>
    /// x: Gaussian count, y: seed, zw unused.
    var info: SIMD4<UInt32>
}

/// Encodes the MCMC per-step terms on the GPU. Each call only encodes work: the caller commits and waits.
public final class GPUMCMC {
    private let context: MetalContext
    private let regularizerPipeline: MTLComputePipelineState
    private let noisePipeline: MTLComputePipelineState

    public init(context: MetalContext) throws(MetalError) {
        self.context = context
        regularizerPipeline = try context.makePipeline("mcmcRegularizerGradient")
        noisePipeline = try context.makePipeline("mcmcNoise")
    }

    /// Adds the regularizer gradients of `MCMCRegularizers.addGradients` into `gradients` for the first
    /// `parameters.count` Gaussians. The kernel accumulates, as the anisotropy kernel does.
    public func encodeRegularizers(parameters: GaussianBuffers, gradients: GradientBuffers,
                                   opacityWeight: Float, scaleWeight: Float,
                                   commandBuffer: MTLCommandBuffer) throws(MetalError) {
        let n = parameters.count
        precondition(gradients.capacity >= n, "gradient capacity \(gradients.capacity) is below \(n) Gaussians")
        precondition(gradients.shDegree == parameters.shDegree, "SH degree mismatch between parameters and gradients")
        guard n > 0 else { return }

        let invCount: Float = Float(1.0 / Double(n))
        var uniforms = MCMCRegularizerUniforms(
            params: SIMD4<Float>(opacityWeight * invCount, scaleWeight * invCount, 0, 0),
            info: SIMD4<UInt32>(UInt32(n), 0, 0, 0))
        let encoder = try context.makeComputeEncoder(commandBuffer)
        encoder.setComputePipelineState(regularizerPipeline)
        encoder.setBuffer(parameters.opacityLogits, offset: 0, index: 0)
        encoder.setBuffer(parameters.logScales, offset: 0, index: 1)
        encoder.setBuffer(gradients.opacityLogits, offset: 0, index: 2)
        encoder.setBuffer(gradients.logScales, offset: 0, index: 3)
        encoder.setBytes(&uniforms, length: MemoryLayout<MCMCRegularizerUniforms>.stride, index: 4)
        MetalContext.dispatch1D(encoder, pipeline: regularizerPipeline, count: n)
        encoder.endEncoding()
    }

    /// SGLD position noise, in place, for the first `parameters.count` Gaussians:
    /// `p += step · M(o) · Σ ε` with `ε[a] = SplatRandom.normal(seed, gid, a)`. `step` is the noise learning rate
    /// times the position learning rate of the iteration. Runs after Adam, so it reads the updated parameters.
    public func encodeNoise(parameters: GaussianBuffers, step: Float, seed: UInt32,
                            commandBuffer: MTLCommandBuffer) throws(MetalError) {
        let n = parameters.count
        guard n > 0 else { return }

        var uniforms = MCMCNoiseUniforms(
            params: SIMD4<Float>(step, MCMCNoise.gateSteepness, MCMCNoise.gateOpacity, 0),
            info: SIMD4<UInt32>(UInt32(n), seed, 0, 0))
        let encoder = try context.makeComputeEncoder(commandBuffer)
        encoder.setComputePipelineState(noisePipeline)
        encoder.setBuffer(parameters.positions, offset: 0, index: 0)
        encoder.setBuffer(parameters.logScales, offset: 0, index: 1)
        encoder.setBuffer(parameters.rotations, offset: 0, index: 2)
        encoder.setBuffer(parameters.opacityLogits, offset: 0, index: 3)
        encoder.setBytes(&uniforms, length: MemoryLayout<MCMCNoiseUniforms>.stride, index: 4)
        MetalContext.dispatch1D(encoder, pipeline: noisePipeline, count: n)
        encoder.endEncoding()
    }
}
