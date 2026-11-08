import Foundation
import Metal
import SplatCore

/// Mirrors `AppearancePixelUniforms` in `Shaders/Appearance.metal` (16 bytes).
struct AppearancePixelUniforms {
    var pixelCount: UInt32
    var width: UInt32
    var viewIndex: UInt32
    var groupCount: UInt32
}

/// Mirrors `AppearanceUpdateUniforms` in `Shaders/Appearance.metal` (48 bytes).
struct AppearanceUpdateUniforms {
    var viewIndex: UInt32
    var groupCount: UInt32
    /// learningRate / bias correction 1, 1 / sqrt(bias correction 2), identity scale 2λ/3, epsilon.
    var rates: SIMD4<Float>
    /// beta1, beta2.
    var betas: SIMD4<Float>
}

/// Per-image appearance compensation on the GPU (v1.2): a gain and bias per training view, with Adam moments.
/// Mirrors `AppearanceState` in SplatCore. Nothing is read back per step: the parameters stay in GPU buffers and
/// are copied to the CPU only by `readParameters`, which the caller runs after the GPU is idle.
///
/// Encoding order for one training view (all in the view's rasterize command buffer):
/// `encodeCompensation` (forward), the image loss on its output, then `encodeGradient` (backward and update).
/// Calls for the same object must be encoded in that order, and `readParameters` only after the GPU has completed.
public final class GPUAppearance {
    /// Threadgroups of the backward reduction. Each threadgroup reduces its share of the pixels.
    static let groupCount: Int = 64
    /// Threads per backward threadgroup. The kernel's shared memory assumes this value.
    static let threadsPerGroup: Int = 256
    /// Gain r, g, b and bias r, g, b.
    static let valuesPerView: Int = 6

    public let viewCount: Int
    private let context: MetalContext
    private let compensationPipeline: MTLComputePipelineState
    private let backwardPipeline: MTLComputePipelineState
    private let updatePipeline: MTLComputePipelineState
    /// 6 floats per view: gain r, g, b, then bias r, g, b. Initialised to the identity.
    private let parameters: MTLBuffer
    private let firstMoments: MTLBuffer
    private let secondMoments: MTLBuffer
    /// 6 floats per backward threadgroup, written by `appearanceBackward` and summed by `appearanceUpdate`.
    private let partials: MTLBuffer
    /// Compensated render per image size; grow-only, like the trainer's render targets.
    private var compensatedTextures: [SIMD2<Int>: MTLTexture] = [:]
    /// Adam steps taken per view, counted when the update is encoded (host side).
    private var stepCounts: [Int]

    /// Identity parameters and zero moments for `viewCount` views.
    public init(context: MetalContext, viewCount: Int) throws(MetalError) {
        precondition(viewCount > 0, "appearance compensation needs at least one view")
        let values: Int = viewCount * Self.valuesPerView
        let floatSize: Int = MemoryLayout<Float>.stride
        let compensation = try context.makePipeline("appearanceForward")
        let backward = try context.makePipeline("appearanceBackward", minThreads: Self.threadsPerGroup)
        let update = try context.makePipeline("appearanceUpdate", minThreads: Self.valuesPerView)
        let parameterBuffer = try context.makeBuffer(length: values * floatSize)
        let firstBuffer = try context.makeBuffer(length: values * floatSize)
        let secondBuffer = try context.makeBuffer(length: values * floatSize)
        let partialBuffer = try context.makeBuffer(length: Self.groupCount * Self.valuesPerView * floatSize)

        let pointer = parameterBuffer.contents().bindMemory(to: Float.self, capacity: values)
        for view in 0..<viewCount {
            for k in 0..<Self.valuesPerView {
                pointer[view * Self.valuesPerView + k] = k < 3 ? 1 : 0
            }
        }
        memset(firstBuffer.contents(), 0, firstBuffer.length)
        memset(secondBuffer.contents(), 0, secondBuffer.length)

        self.viewCount = viewCount
        self.context = context
        compensationPipeline = compensation
        backwardPipeline = backward
        updatePipeline = update
        parameters = parameterBuffer
        firstMoments = firstBuffer
        secondMoments = secondBuffer
        partials = partialBuffer
        stepCounts = [Int](repeating: 0, count: viewCount)
    }

    /// Encodes `A = R ⊙ gain + bias` of `viewIndex` for the render `rendered` and returns the texture holding A, of
    /// the same size. The texture is reused by the next call with the same size.
    public func encodeCompensation(rendered: MTLTexture, viewIndex: Int,
                                   commandBuffer: MTLCommandBuffer) throws(MetalError) -> MTLTexture {
        precondition(viewIndex >= 0 && viewIndex < viewCount, "appearance view index out of range")
        guard rendered.pixelFormat == .rgba32Float else {
            throw .commandFailed("appearance needs an rgba32Float render")
        }
        let width: Int = rendered.width
        let height: Int = rendered.height
        let count: Int = width * height
        let output = try compensatedTexture(width: width, height: height)
        var uniforms = AppearancePixelUniforms(pixelCount: UInt32(count), width: UInt32(width),
                                               viewIndex: UInt32(viewIndex), groupCount: UInt32(Self.groupCount))

        let encoder = try context.makeComputeEncoder(commandBuffer)
        encoder.setComputePipelineState(compensationPipeline)
        encoder.setTexture(rendered, index: 0)
        encoder.setTexture(output, index: 1)
        encoder.setBuffer(parameters, offset: 0, index: 0)
        encoder.setBytes(&uniforms, length: MemoryLayout<AppearancePixelUniforms>.stride, index: 1)
        MetalContext.dispatch1D(encoder, pipeline: compensationPipeline, count: count)
        encoder.endEncoding()
        return output
    }

    /// Encodes the backward and the Adam update of `viewIndex`.
    ///
    /// `imageGradient` must hold G = dL/dA (RGBA floats per pixel, as written by `GPUImageLoss`). It is replaced in
    /// place by dL/dR = G ⊙ gain with alpha 0, which is the image gradient of the rasterizer backward. `rendered` is
    /// the view's render R. The update uses the photometric gradient plus the identity pull, `learningRate`,
    /// `identityWeight` and `hyper`, and advances the view's own step count.
    public func encodeGradient(rendered: MTLTexture, imageGradient: MTLBuffer, viewIndex: Int,
                               learningRate: Float, identityWeight: Float, hyper: AdamHyperparameters,
                               commandBuffer: MTLCommandBuffer) throws(MetalError) {
        precondition(viewIndex >= 0 && viewIndex < viewCount, "appearance view index out of range")
        guard rendered.pixelFormat == .rgba32Float else {
            throw .commandFailed("appearance needs an rgba32Float render")
        }
        let width: Int = rendered.width
        let height: Int = rendered.height
        let count: Int = width * height
        guard imageGradient.length >= count * 4 * MemoryLayout<Float>.stride else {
            throw .commandFailed("image gradient must hold width·height RGBA floats")
        }
        var pixelUniforms = AppearancePixelUniforms(pixelCount: UInt32(count), width: UInt32(width),
                                                    viewIndex: UInt32(viewIndex), groupCount: UInt32(Self.groupCount))

        let backwardEncoder = try context.makeComputeEncoder(commandBuffer)
        backwardEncoder.setComputePipelineState(backwardPipeline)
        backwardEncoder.setTexture(rendered, index: 0)
        backwardEncoder.setBuffer(imageGradient, offset: 0, index: 0)
        backwardEncoder.setBuffer(parameters, offset: 0, index: 1)
        backwardEncoder.setBuffer(partials, offset: 0, index: 2)
        backwardEncoder.setBytes(&pixelUniforms, length: MemoryLayout<AppearancePixelUniforms>.stride, index: 3)
        backwardEncoder.dispatchThreadgroups(MTLSize(width: Self.groupCount, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: Self.threadsPerGroup, height: 1,
                                                                            depth: 1))
        backwardEncoder.endEncoding()

        // Same Float arithmetic as CPUAdam.update: bias corrections in Double, then rounded to Float.
        stepCounts[viewIndex] += 1
        let step: Double = Double(stepCounts[viewIndex])
        let bc1: Float = Float(1 - pow(Double(hyper.beta1), step))
        let bc2: Float = Float(1 - pow(Double(hyper.beta2), step))
        let invSqrtBC2: Float = 1 / bc2.squareRoot()
        let stepSize: Float = learningRate / bc1
        let identityScale: Float = 2 * identityWeight / 3
        var updateUniforms = AppearanceUpdateUniforms(
            viewIndex: UInt32(viewIndex), groupCount: UInt32(Self.groupCount),
            rates: SIMD4<Float>(stepSize, invSqrtBC2, identityScale, hyper.epsilon),
            betas: SIMD4<Float>(hyper.beta1, hyper.beta2, 0, 0))

        let updateEncoder = try context.makeComputeEncoder(commandBuffer)
        updateEncoder.setComputePipelineState(updatePipeline)
        updateEncoder.setBuffer(parameters, offset: 0, index: 0)
        updateEncoder.setBuffer(firstMoments, offset: 0, index: 1)
        updateEncoder.setBuffer(secondMoments, offset: 0, index: 2)
        updateEncoder.setBuffer(partials, offset: 0, index: 3)
        updateEncoder.setBytes(&updateUniforms, length: MemoryLayout<AppearanceUpdateUniforms>.stride, index: 4)
        updateEncoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                           threadsPerThreadgroup: MTLSize(width: Self.valuesPerView, height: 1,
                                                                          depth: 1))
        updateEncoder.endEncoding()
    }

    /// Gains, biases, Adam moments and step counts of every view, copied from the GPU. The GPU must be idle.
    public func snapshotState() -> AppearanceOptimizerBlock {
        let values: Int = viewCount * Self.valuesPerView
        return AppearanceOptimizerBlock(parameters: Self.read(parameters, count: values),
                                        firstMoment: Self.read(firstMoments, count: values),
                                        secondMoment: Self.read(secondMoments, count: values),
                                        stepCounts: stepCounts)
    }

    /// Replaces the gains, biases, moments and step counts with `block`, which must describe `viewCount` views. The
    /// GPU must be idle with respect to these buffers.
    public func restoreState(_ block: AppearanceOptimizerBlock) {
        let values: Int = viewCount * Self.valuesPerView
        precondition(block.viewCount == viewCount && block.parameters.count == values
                         && block.firstMoment.count == values && block.secondMoment.count == values,
                     "appearance state does not match \(viewCount) views")
        Self.write(block.parameters, to: parameters)
        Self.write(block.firstMoment, to: firstMoments)
        Self.write(block.secondMoment, to: secondMoments)
        stepCounts = block.stepCounts
    }

    private static func read(_ buffer: MTLBuffer, count: Int) -> [Float] {
        guard count > 0 else { return [] }
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private static func write(_ values: [Float], to buffer: MTLBuffer) {
        guard !values.isEmpty else { return }
        values.withUnsafeBytes { raw in
            buffer.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
        }
    }

    /// Gains and biases of every view, copied from the GPU. The GPU must be idle with respect to these buffers.
    public func readParameters() -> [AppearanceParameters] {
        let pointer = parameters.contents().bindMemory(to: Float.self, capacity: viewCount * Self.valuesPerView)
        var result: [AppearanceParameters] = []
        result.reserveCapacity(viewCount)
        for view in 0..<viewCount {
            let base: Int = view * Self.valuesPerView
            let gain = SIMD3<Float>(pointer[base], pointer[base + 1], pointer[base + 2])
            let bias = SIMD3<Float>(pointer[base + 3], pointer[base + 4], pointer[base + 5])
            result.append(AppearanceParameters(gain: gain, bias: bias))
        }
        return result
    }

    /// Test hook: the sum over all backward threadgroups of the partial reduction (gain r, g, b, bias r, g, b) of the
    /// most recent `encodeGradient`. The GPU must be idle.
    func partialSums() -> [Float] {
        let pointer = partials.contents().bindMemory(to: Float.self, capacity: Self.groupCount * Self.valuesPerView)
        var sums = [Float](repeating: 0, count: Self.valuesPerView)
        for group in 0..<Self.groupCount {
            for k in 0..<Self.valuesPerView {
                sums[k] += pointer[group * Self.valuesPerView + k]
            }
        }
        return sums
    }

    /// Test hook: overwrites the parameters of every view, one entry per view. The GPU must be idle.
    func testingSetParameters(_ values: [AppearanceParameters]) {
        precondition(values.count == viewCount, "one entry per view")
        let pointer = parameters.contents().bindMemory(to: Float.self, capacity: viewCount * Self.valuesPerView)
        for (view, value) in values.enumerated() {
            let base: Int = view * Self.valuesPerView
            pointer[base] = value.gain.x
            pointer[base + 1] = value.gain.y
            pointer[base + 2] = value.gain.z
            pointer[base + 3] = value.bias.x
            pointer[base + 4] = value.bias.y
            pointer[base + 5] = value.bias.z
        }
    }

    private func compensatedTexture(width: Int, height: Int) throws(MetalError) -> MTLTexture {
        let key = SIMD2<Int>(width, height)
        if let existing = compensatedTextures[key] {
            return existing
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderWrite, .shaderRead]
        descriptor.storageMode = .shared
        guard let texture = context.device.makeTexture(descriptor: descriptor) else {
            throw .commandFailed("cannot allocate a \(width)x\(height) compensation target")
        }
        compensatedTextures[key] = texture
        return texture
    }
}
