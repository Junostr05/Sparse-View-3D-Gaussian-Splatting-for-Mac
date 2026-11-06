import Metal
import SplatCore

// GPU artifact regularizers (Phase 4, Task 16). Mirrors `Regularizers` in SplatCore; the kernels live in
// Shaders/Regularizers.metal. Layouts of the uniform structs below must match there.

/// MSL `AnisotropyUniforms`, 32 bytes.
private struct AnisotropyUniforms {
    /// x: weight, y: maxRatio, z: 1 / count, w unused.
    var params: SIMD4<Float>
    /// x: Gaussian count, yzw unused.
    var info: SIMD4<UInt32>
}

/// MSL `ScaleCapUniforms`, 48 bytes.
private struct ScaleCapUniforms {
    /// x: fraction, y: extent radius, z: centre x, w: centre y.
    var a: SIMD4<Float>
    /// x: centre z, yzw unused.
    var b: SIMD4<Float>
    /// x: Gaussian count, yzw unused.
    var info: SIMD4<UInt32>
}

/// Encodes the artifact regularizers on the GPU. Both calls only encode work: the caller commits the
/// command buffer and waits for it before reading the buffers on the CPU.
public final class GPURegularizers {
    private let context: MetalContext
    private let anisotropyPipeline: MTLComputePipelineState
    private let scaleCapPipeline: MTLComputePipelineState

    public init(context: MetalContext) throws(MetalError) {
        self.context = context
        anisotropyPipeline = try context.makePipeline("anisotropyGradient")
        scaleCapPipeline = try context.makePipeline("scaleCap")
    }

    /// Adds the anisotropy gradient of `λ/n · Σ max(0, s_max/s_min − maxRatio)` into `gradients.logScales`
    /// for the first `parameters.count` Gaussians. The gradient buffers must be zeroed by the caller where
    /// a fresh gradient is wanted, because the kernel accumulates.
    public func encodeAnisotropy(parameters: GaussianBuffers, gradients: GradientBuffers, weight: Float,
                                 maxRatio: Float, commandBuffer: MTLCommandBuffer) throws(MetalError) {
        let n = parameters.count
        precondition(gradients.capacity >= n, "gradient capacity \(gradients.capacity) is below \(n) Gaussians")
        precondition(gradients.shDegree == parameters.shDegree, "SH degree mismatch between parameters and gradients")
        guard n > 0 else { return }

        let invCount: Float = Float(1.0 / Double(n))
        var uniforms = AnisotropyUniforms(params: SIMD4<Float>(weight, maxRatio, invCount, 0),
                                          info: SIMD4<UInt32>(UInt32(n), 0, 0, 0))
        let encoder = try context.makeComputeEncoder(commandBuffer)
        encoder.setComputePipelineState(anisotropyPipeline)
        encoder.setBuffer(parameters.logScales, offset: 0, index: 0)
        encoder.setBuffer(gradients.logScales, offset: 0, index: 1)
        encoder.setBytes(&uniforms, length: MemoryLayout<AnisotropyUniforms>.stride, index: 2)
        MetalContext.dispatch1D(encoder, pipeline: anisotropyPipeline, count: n)
        encoder.endEncoding()
    }

    /// Scale cap projection, in place on `parameters.logScales` for the first `parameters.count` Gaussians:
    /// `logScale_a = min(logScale_a, log(fraction · max(R, ‖μ − c‖)))` with `R` and `c` from `extent`.
    public func encodeScaleCap(parameters: GaussianBuffers, fraction: Float, extent: SceneExtent,
                               commandBuffer: MTLCommandBuffer) throws(MetalError) {
        let n = parameters.count
        guard n > 0 else { return }

        var uniforms = ScaleCapUniforms(
            a: SIMD4<Float>(fraction, extent.radius, extent.center.x, extent.center.y),
            b: SIMD4<Float>(extent.center.z, 0, 0, 0),
            info: SIMD4<UInt32>(UInt32(n), 0, 0, 0))
        let encoder = try context.makeComputeEncoder(commandBuffer)
        encoder.setComputePipelineState(scaleCapPipeline)
        encoder.setBuffer(parameters.logScales, offset: 0, index: 0)
        encoder.setBuffer(parameters.positions, offset: 0, index: 1)
        encoder.setBytes(&uniforms, length: MemoryLayout<ScaleCapUniforms>.stride, index: 2)
        MetalContext.dispatch1D(encoder, pipeline: scaleCapPipeline, count: n)
        encoder.endEncoding()
    }
}
