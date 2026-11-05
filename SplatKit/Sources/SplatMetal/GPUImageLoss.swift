import Metal
import SplatCore

/// Loss value of one view, reduced on the CPU in `Double`.
public struct ImageLossValue: Sendable, Equatable {
    /// `(1 − λ)·l1 + λ·(1 − ssim)`.
    public var loss: Double
    /// Mean absolute error over W·H·3 values.
    public var l1: Double
    /// Mean SSIM over W·H·3 values.
    public var ssim: Double

    public init(loss: Double, l1: Double, ssim: Double) {
        self.loss = loss
        self.l1 = l1
        self.ssim = ssim
    }
}

/// Mirrors `LossUniforms` in `Shaders/Loss.metal` (48 bytes).
struct LossUniforms {
    /// width, height, tilesX, unused.
    var size: SIMD4<UInt32>
    /// l1 gradient scale (1−λ)/N, SSIM gradient scale −λ/N, λ, unused.
    var scales: SIMD4<Float>
    /// C1, C2, unused, unused.
    var stability: SIMD4<Float>
}

/// GPU counterpart of `ImageLoss` (L1 + D-SSIM) with a closed-form backward pass.
///
/// `encode` writes dL/drendered into a caller buffer (RGBA float per pixel, A = 0) that
/// `SplatRenderer.encodeBackward` consumes. Scratch buffers are grow-only and owned by this object.
/// Do not encode a new loss while a previous command buffer that uses this object is still running,
/// and call `readValue` only after the command buffer of the most recent `encode` has completed.
public final class GPUImageLoss {
    /// Threadgroup edge length of every kernel (16×16 threads).
    private static let tileSize = 16

    private let context: MetalContext
    private let momentsPipeline: MTLComputePipelineState
    private let mapsPipeline: MTLComputePipelineState
    private let blurPipeline: MTLComputePipelineState
    private let gradientPipeline: MTLComputePipelineState

    /// 15 planes: horizontal moments, later the first 9 planes hold the horizontally blurred maps.
    private var momentBuffer: MTLBuffer?
    /// 9 planes: per-pixel SSIM backward terms.
    private var mapBuffer: MTLBuffer?
    /// float2 (ΣS, Σ|x−y|) per threadgroup.
    private var partialBuffer: MTLBuffer?

    /// State of the most recent `encode`, consumed by `readValue`.
    private struct LastEncode {
        var groups: Int
        var values: Int      // W·H·3
        var lambda: Double
        var partials: MTLBuffer
    }
    private var lastEncode: LastEncode?

    public init(context: MetalContext) throws(MetalError) {
        self.context = context
        momentsPipeline = try context.makePipeline("ssimMomentsH", minThreads: Self.tileSize * Self.tileSize)
        mapsPipeline = try context.makePipeline("ssimMapsV", minThreads: Self.tileSize * Self.tileSize)
        blurPipeline = try context.makePipeline("blurH9")
        gradientPipeline = try context.makePipeline("lossGradient")
    }

    /// Target image as RGBA float per pixel (A = 0), `storageModeShared`.
    public func makeTarget(_ image: RGBImage) throws(MetalError) -> MTLBuffer {
        let count = image.width * image.height
        precondition(image.pixels.count == count, "image pixel count does not match its size")
        let buffer = try context.makeBuffer(length: count * MemoryLayout<SIMD4<Float>>.stride)
        if count > 0 {
            let pointer = buffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: count)
            for i in 0..<count {
                let p = image.pixels[i]
                pointer[i] = SIMD4<Float>(p.x, p.y, p.z, 0)
            }
        }
        return buffer
    }

    /// Encodes the loss and its gradient for `rendered` against `target` into `commandBuffer`.
    /// `rendered` must be an `rgba32Float` texture of `width` × `height` with shader-read usage
    /// (for example the texture given to `SplatRenderer.encodeForward`). `target` comes from `makeTarget`.
    /// Writes dL/drendered into `imageGradient` (at least W·H·4 floats) and the per-threadgroup partial sums.
    public func encode(rendered: MTLTexture, target: MTLBuffer, width: Int, height: Int, lambda: Float,
                       imageGradient: MTLBuffer, commandBuffer: MTLCommandBuffer) throws(MetalError) {
        guard width > 0, height > 0 else {
            throw .commandFailed("image has no pixels")
        }
        guard rendered.width == width, rendered.height == height, rendered.pixelFormat == .rgba32Float else {
            throw .commandFailed("rendered must be an rgba32Float texture of \(width)×\(height)")
        }
        let count = width * height
        let floatSize = MemoryLayout<Float>.size
        guard target.length >= count * 4 * floatSize else {
            throw .commandFailed("target must hold width·height RGBA floats")
        }
        guard imageGradient.length >= count * 4 * floatSize else {
            throw .commandFailed("imageGradient must hold width·height RGBA floats")
        }

        let tile = Self.tileSize
        let tilesX = (width + tile - 1) / tile
        let tilesY = (height + tile - 1) / tile
        let groups = tilesX * tilesY

        let moments = try context.ensureBuffer(&momentBuffer, length: 15 * count * floatSize)
        let maps = try context.ensureBuffer(&mapBuffer, length: 9 * count * floatSize)
        let partials = try context.ensureBuffer(&partialBuffer, length: groups * 2 * floatSize)

        // Scales are computed in Double and rounded once, as the CPU reference does.
        let lambdaValue: Double = Double(lambda)
        let valueCount: Double = Double(count * 3)
        var uniforms = LossUniforms(
            size: SIMD4<UInt32>(UInt32(width), UInt32(height), UInt32(tilesX), 0),
            scales: SIMD4<Float>(Float((1 - lambdaValue) / valueCount),
                                 Float(-lambdaValue / valueCount),
                                 lambda,
                                 0),
            stability: SIMD4<Float>(ImageLoss.c1, ImageLoss.c2, 0, 0))
        let taps: [Float] = ImageLoss.window
        let groupSize = MTLSize(width: tile, height: tile, depth: 1)
        let gridSize = MTLSize(width: tilesX, height: tilesY, depth: 1)

        // Stage 1: horizontal moments.
        let momentsEncoder = try context.makeComputeEncoder(commandBuffer)
        momentsEncoder.setComputePipelineState(momentsPipeline)
        momentsEncoder.setTexture(rendered, index: 0)
        momentsEncoder.setBuffer(target, offset: 0, index: 0)
        momentsEncoder.setBuffer(moments, offset: 0, index: 1)
        momentsEncoder.setBytes(&uniforms, length: MemoryLayout<LossUniforms>.stride, index: 2)
        taps.withUnsafeBytes { bytes in
            momentsEncoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 3)
        }
        momentsEncoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
        momentsEncoder.endEncoding()

        // Stage 2: vertical blur, SSIM map, SSIM backward terms, per-threadgroup partial sums.
        let mapsEncoder = try context.makeComputeEncoder(commandBuffer)
        mapsEncoder.setComputePipelineState(mapsPipeline)
        mapsEncoder.setTexture(rendered, index: 0)
        mapsEncoder.setBuffer(target, offset: 0, index: 0)
        mapsEncoder.setBuffer(moments, offset: 0, index: 1)
        mapsEncoder.setBuffer(maps, offset: 0, index: 2)
        mapsEncoder.setBuffer(partials, offset: 0, index: 3)
        mapsEncoder.setBytes(&uniforms, length: MemoryLayout<LossUniforms>.stride, index: 4)
        taps.withUnsafeBytes { bytes in
            mapsEncoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 5)
        }
        mapsEncoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
        mapsEncoder.endEncoding()

        // Stage 3: horizontal blur of the nine backward planes, into the first nine moment planes.
        let blurEncoder = try context.makeComputeEncoder(commandBuffer)
        blurEncoder.setComputePipelineState(blurPipeline)
        blurEncoder.setBuffer(maps, offset: 0, index: 0)
        blurEncoder.setBuffer(moments, offset: 0, index: 1)
        blurEncoder.setBytes(&uniforms, length: MemoryLayout<LossUniforms>.stride, index: 2)
        taps.withUnsafeBytes { bytes in
            blurEncoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 3)
        }
        blurEncoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
        blurEncoder.endEncoding()

        // Stage 4: vertical blur fused with the image gradient.
        let gradientEncoder = try context.makeComputeEncoder(commandBuffer)
        gradientEncoder.setComputePipelineState(gradientPipeline)
        gradientEncoder.setTexture(rendered, index: 0)
        gradientEncoder.setBuffer(target, offset: 0, index: 0)
        gradientEncoder.setBuffer(moments, offset: 0, index: 1)
        gradientEncoder.setBuffer(imageGradient, offset: 0, index: 2)
        gradientEncoder.setBytes(&uniforms, length: MemoryLayout<LossUniforms>.stride, index: 3)
        taps.withUnsafeBytes { bytes in
            gradientEncoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 4)
        }
        gradientEncoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
        gradientEncoder.endEncoding()

        lastEncode = LastEncode(groups: groups, values: count * 3, lambda: lambdaValue, partials: partials)
    }

    /// Loss of the most recent `encode`, summing the partials on the CPU in `Double`.
    /// Precondition: `encode` was called and its command buffer has completed.
    public func readValue() -> ImageLossValue {
        guard let last = lastEncode else {
            preconditionFailure("readValue called before encode")
        }
        let pointer = last.partials.contents().bindMemory(to: Float.self, capacity: last.groups * 2)
        var sumS: Double = 0
        var sumL: Double = 0
        for g in 0..<last.groups {
            sumS += Double(pointer[2 * g])
            sumL += Double(pointer[2 * g + 1])
        }
        let n: Double = Double(last.values)
        let l1: Double = sumL / n
        let ssim: Double = sumS / n
        let loss: Double = (1 - last.lambda) * l1 + last.lambda * (1 - ssim)
        return ImageLossValue(loss: loss, l1: l1, ssim: ssim)
    }
}
