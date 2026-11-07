import Foundation
import Metal
import SplatCore

/// Per-Gaussian sampling rate ν of the Mip-Splatting 3D filter, computed on the GPU over all Gaussians and
/// training cameras. The CPU reference is `SamplingRate.compute` (SplatCore).
public final class GPUSamplingRate {
    private let context: MetalContext
    private let computePipeline: MTLComputePipelineState
    private let fillPipeline: MTLComputePipelineState
    private var cameraBuffer: MTLBuffer?
    private var visibleSumBuffer: MTLBuffer?
    private var visibleCountBuffer: MTLBuffer?

    public init(context: MetalContext) throws(MetalError) {
        self.context = context
        computePipeline = try context.makePipeline("samplingRateCompute")
        fillPipeline = try context.makePipeline("samplingRateFill")
    }

    /// Encodes ν for the first `count` Gaussians of `positions` (3 floats each) into `rates` (≥ 4·count bytes).
    /// The reduction counters are zeroed by a blit in the same command buffer. Nothing is waited for: the caller
    /// commits `commandBuffer`, and later work on the same queue sees the result.
    public func encode(positions: MTLBuffer, count: Int, cameras: [Camera], near: Float,
                       into rates: MTLBuffer, commandBuffer: MTLCommandBuffer) throws(MetalError) {
        guard count > 0 else { return }
        guard rates.length >= count * MemoryLayout<Float>.size else {
            throw .commandFailed("sampling rate buffer holds fewer than \(count) values")
        }
        let cameraCount = cameras.count
        let stride = MemoryLayout<SamplingCameraUniforms>.stride
        let cameraStore = try context.ensureBuffer(&cameraBuffer, length: max(1, cameraCount) * stride)
        let visibleSum = try context.ensureBuffer(&visibleSumBuffer, length: MemoryLayout<Float>.size)
        let visibleCount = try context.ensureBuffer(&visibleCountBuffer, length: MemoryLayout<UInt32>.size)

        if cameraCount > 0 {
            let pointer = cameraStore.contents().bindMemory(to: SamplingCameraUniforms.self, capacity: cameraCount)
            for (i, camera) in cameras.enumerated() {
                let rot = camera.rotation
                let t = camera.translation
                pointer[i] = SamplingCameraUniforms(
                    rotRow0: SIMD4(rot.c0.x, rot.c1.x, rot.c2.x, t.x),
                    rotRow1: SIMD4(rot.c0.y, rot.c1.y, rot.c2.y, t.y),
                    rotRow2: SIMD4(rot.c0.z, rot.c1.z, rot.c2.z, t.z),
                    intrinsics: SIMD4(camera.fx, camera.fy, camera.cx, camera.cy),
                    size: SIMD4(Float(camera.width), Float(camera.height), near, 0))
            }
        }
        var uniforms = SamplingUniforms(counts: SIMD4(UInt32(count), UInt32(cameraCount), 0, 0))

        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw .commandFailed("cannot create blit encoder")
        }
        blit.fill(buffer: visibleSum, range: 0..<MemoryLayout<Float>.size, value: 0)
        blit.fill(buffer: visibleCount, range: 0..<MemoryLayout<UInt32>.size, value: 0)
        blit.endEncoding()

        let computeEncoder = try context.makeComputeEncoder(commandBuffer)
        computeEncoder.setComputePipelineState(computePipeline)
        computeEncoder.setBuffer(positions, offset: 0, index: 0)
        computeEncoder.setBuffer(cameraStore, offset: 0, index: 1)
        computeEncoder.setBytes(&uniforms, length: MemoryLayout<SamplingUniforms>.size, index: 2)
        computeEncoder.setBuffer(rates, offset: 0, index: 3)
        computeEncoder.setBuffer(visibleSum, offset: 0, index: 4)
        computeEncoder.setBuffer(visibleCount, offset: 0, index: 5)
        MetalContext.dispatch1D(computeEncoder, pipeline: computePipeline, count: count)
        computeEncoder.endEncoding()

        let fillEncoder = try context.makeComputeEncoder(commandBuffer)
        fillEncoder.setComputePipelineState(fillPipeline)
        fillEncoder.setBuffer(rates, offset: 0, index: 0)
        fillEncoder.setBuffer(visibleSum, offset: 0, index: 1)
        fillEncoder.setBuffer(visibleCount, offset: 0, index: 2)
        fillEncoder.setBytes(&uniforms, length: MemoryLayout<SamplingUniforms>.size, index: 3)
        MetalContext.dispatch1D(fillEncoder, pipeline: fillPipeline, count: count)
        fillEncoder.endEncoding()
    }
}
