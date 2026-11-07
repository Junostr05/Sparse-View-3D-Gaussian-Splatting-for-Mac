import Metal
import SplatCore

/// Where one training view's camera-pose gradient goes on the GPU (pose refinement). `encodeBackward` writes six
/// floats per Gaussian into `scratch` (dL/dω, then dL/dτ, from `projectBackward`), and `reducer` adds their six sums
/// into `sums[base ..< base + 6]` with float atomics. The owner zero-fills `sums` before the first use and must hold
/// `scratch` for at least the Gaussian count.
public struct PoseGradientSink {
    public let reducer: GPUPoseGradient
    public let scratch: MTLBuffer
    public let sums: MTLBuffer
    /// Float index of the first of the six sums in `sums`.
    public let base: Int

    public init(reducer: GPUPoseGradient, scratch: MTLBuffer, sums: MTLBuffer, base: Int) {
        self.reducer = reducer
        self.scratch = scratch
        self.sums = sums
        self.base = base
    }
}

/// The pose-gradient reduction kernel (`reducePoseGradient`, Shaders/PoseGradient.metal).
public final class GPUPoseGradient {
    /// Threads per threadgroup, fixed by the shader's shared array.
    public static let threadsPerGroup = 256

    private let context: MetalContext
    private let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws(MetalError) {
        self.context = context
        pipeline = try context.makePipeline("reducePoseGradient", minThreads: Self.threadsPerGroup)
    }

    /// Adds the six sums of the first `count` scratch entries to `sums[base ..< base + 6]`. Encodes only; the caller
    /// commits. Does nothing for `count == 0`.
    public func encode(scratch: MTLBuffer, count: Int, sums: MTLBuffer, base: Int,
                       commandBuffer: MTLCommandBuffer) throws(MetalError) {
        guard count > 0 else { return }
        var countValue = UInt32(count)
        var baseValue = UInt32(base)
        let encoder = try context.makeComputeEncoder(commandBuffer)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(scratch, offset: 0, index: 0)
        encoder.setBuffer(sums, offset: 0, index: 1)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&baseValue, length: MemoryLayout<UInt32>.size, index: 3)
        let groups = (count + Self.threadsPerGroup - 1) / Self.threadsPerGroup
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threadsPerGroup, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
