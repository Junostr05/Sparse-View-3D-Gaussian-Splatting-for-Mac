import Metal
import SplatCore

/// Value of one GPU normal loss evaluation (v1.5), read back after its command buffer completed.
public struct NormalLossValue: Sendable, Equatable {
    /// `λ_n · mean(1 − n·m)`, 0 when degenerate.
    public var loss: Double
    /// Mean cosine between the rendered and the prior normals over the valid pixels.
    public var meanCosine: Double
    public var validPixels: Int
    public var degenerate: Bool

    public init(loss: Double, meanCosine: Double, validPixels: Int, degenerate: Bool) {
        self.loss = loss
        self.meanCosine = meanCosine
        self.validPixels = validPixels
        self.degenerate = degenerate
    }
}

/// Mirrors `NormalLossUniforms` in `Shaders/Normal.metal` (48 bytes).
struct NormalLossUniforms {
    var intrinsics: SIMD4<Float>
    var params: SIMD4<Float>
    var size: SIMD4<UInt32>
}

/// Mirrors `NormalPartial` in `Shaders/Normal.metal` (32 bytes).
struct NormalPartial {
    var sums: SIMD4<Float>
    var counts: SIMD4<UInt32>
}

/// Mirrors `NormalStats` in `Shaders/Normal.metal` (32 bytes).
struct NormalStats {
    var counts: SIMD4<UInt32>
    var values: SIMD4<Float>
}

/// A normal prior uploaded to the GPU: one `float4 (nx, ny, nz, 0)` per pixel, zero where there is no prior.
public final class GPUNormalPrior {
    public let width: Int
    public let height: Int
    let buffer: MTLBuffer

    init(width: Int, height: Int, buffer: MTLBuffer) {
        self.width = width
        self.height = height
        self.buffer = buffer
    }
}

/// GPU counterpart of `NormalLoss`. Unlike the depth loss it ADDS its gradient into `depthGradient`, so it is
/// encoded after the depth loss of the same view, which overwrites that buffer first.
public final class GPUNormalLoss {
    private static let tileSize = 16

    private let context: MetalContext
    private let momentsPipeline: MTLComputePipelineState
    private let finalizePipeline: MTLComputePipelineState
    private let gradientPipeline: MTLComputePipelineState
    private let statsBuffer: MTLBuffer
    private var partialBuffer: MTLBuffer?

    public init(context: MetalContext) throws(MetalError) {
        self.context = context
        momentsPipeline = try context.makePipeline("normalMoments", minThreads: Self.tileSize * Self.tileSize)
        finalizePipeline = try context.makePipeline("normalFinalize")
        gradientPipeline = try context.makePipeline("normalGradient")
        statsBuffer = try context.makeBuffer(length: MemoryLayout<NormalStats>.stride)
    }

    /// Uploads `normals` (row-major, `width · height` entries, zero means no prior).
    public func makePrior(normals: [SIMD3<Float>], width: Int, height: Int) throws(MetalError) -> GPUNormalPrior {
        let count: Int = width * height
        precondition(normals.count == count, "normals must match the size")
        let buffer = try context.makeBuffer(length: max(count, 1) * MemoryLayout<SIMD4<Float>>.stride)
        let pointer = buffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: max(count, 1))
        for i in 0..<count {
            let n: SIMD3<Float> = normals[i]
            pointer[i] = SIMD4<Float>(n.x, n.y, n.z, 0)
        }
        return GPUNormalPrior(width: width, height: height, buffer: buffer)
    }

    /// Encodes the loss of `renderedDepth` (float2 `(Draw, A)` per pixel) against `prior` and adds its gradient
    /// into `depthGradient` (float2 `(∂L/∂Draw, ∂L/∂A)` per pixel).
    public func encode(renderedDepth: MTLBuffer, prior: GPUNormalPrior, intrinsics: NormalLossIntrinsics,
                       weights: NormalLossWeights, depthGradient: MTLBuffer,
                       commandBuffer: MTLCommandBuffer) throws(MetalError) {
        let width: Int = prior.width
        let height: Int = prior.height
        guard width > 0, height > 0 else {
            throw .commandFailed("normal prior has no pixels")
        }
        let count: Int = width * height
        let pixelStride: Int = MemoryLayout<SIMD2<Float>>.stride
        guard renderedDepth.length >= count * pixelStride, depthGradient.length >= count * pixelStride else {
            throw .commandFailed("renderedDepth and depthGradient must hold width·height float2 values")
        }
        let tile: Int = Self.tileSize
        let tilesX: Int = (width + tile - 1) / tile
        let tilesY: Int = (height + tile - 1) / tile
        let partials = try context.ensureBuffer(&partialBuffer, length: tilesX * tilesY * MemoryLayout<NormalPartial>.stride)
        var uniforms = NormalLossUniforms(
            intrinsics: SIMD4<Float>(intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy),
            params: SIMD4<Float>(weights.weight, weights.alphaMask, 0, 0),
            size: SIMD4<UInt32>(UInt32(width), UInt32(height), UInt32(clamping: max(0, weights.minimumPixels)), 0))
        let uniformLength: Int = MemoryLayout<NormalLossUniforms>.stride
        let groupSize = MTLSize(width: tile, height: tile, depth: 1)
        let singleThread = MTLSize(width: 1, height: 1, depth: 1)

        let moments = try context.makeComputeEncoder(commandBuffer)
        moments.setComputePipelineState(momentsPipeline)
        moments.setBuffer(renderedDepth, offset: 0, index: 0)
        moments.setBuffer(prior.buffer, offset: 0, index: 1)
        moments.setBuffer(partials, offset: 0, index: 2)
        moments.setBytes(&uniforms, length: uniformLength, index: 3)
        moments.dispatchThreadgroups(MTLSize(width: tilesX, height: tilesY, depth: 1), threadsPerThreadgroup: groupSize)
        moments.endEncoding()

        let finalize = try context.makeComputeEncoder(commandBuffer)
        finalize.setComputePipelineState(finalizePipeline)
        finalize.setBuffer(partials, offset: 0, index: 0)
        finalize.setBuffer(statsBuffer, offset: 0, index: 1)
        finalize.setBytes(&uniforms, length: uniformLength, index: 2)
        finalize.dispatchThreadgroups(singleThread, threadsPerThreadgroup: singleThread)
        finalize.endEncoding()

        let gradient = try context.makeComputeEncoder(commandBuffer)
        gradient.setComputePipelineState(gradientPipeline)
        gradient.setBuffer(renderedDepth, offset: 0, index: 0)
        gradient.setBuffer(prior.buffer, offset: 0, index: 1)
        gradient.setBuffer(statsBuffer, offset: 0, index: 2)
        gradient.setBuffer(depthGradient, offset: 0, index: 3)
        gradient.setBytes(&uniforms, length: uniformLength, index: 4)
        gradient.dispatchThreads(MTLSize(width: width, height: height, depth: 1), threadsPerThreadgroup: groupSize)
        gradient.endEncoding()
    }

    /// Value of the most recent `encode`. Precondition: its command buffer has completed.
    public func readValue() -> NormalLossValue {
        let stats: NormalStats = statsBuffer.contents().bindMemory(to: NormalStats.self, capacity: 1)[0]
        return NormalLossValue(loss: Double(stats.values.x), meanCosine: Double(stats.values.y),
                               validPixels: Int(stats.counts.x), degenerate: stats.counts.y != 0)
    }
}
