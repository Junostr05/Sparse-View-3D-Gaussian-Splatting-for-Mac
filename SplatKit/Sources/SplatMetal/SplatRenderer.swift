import Foundation
import Metal
import SplatCore

public struct FrameStats: Sendable, Equatable {
    public var gpuTimeMilliseconds: Double
    public var visibleGaussians: Int
    public var tileIntersections: Int

    public init(gpuTimeMilliseconds: Double, visibleGaussians: Int, tileIntersections: Int) {
        self.gpuTimeMilliseconds = gpuTimeMilliseconds
        self.visibleGaussians = visibleGaussians
        self.tileIntersections = tileIntersections
    }
}

/// A forward pass whose rasterize work is encoded into `commandBuffer` but not committed.
public struct ForwardPass {
    /// Uncommitted. Encode backward work into it (`SplatRenderer.encodeBackward`) or commit it as is.
    public let commandBuffer: MTLCommandBuffer
    public let visibleGaussians: Int
    public let tileIntersections: Int
    /// GPU time of the projection command buffer, which has already completed.
    let projectionSeconds: Double

    init(commandBuffer: MTLCommandBuffer, visibleGaussians: Int, tileIntersections: Int, projectionSeconds: Double) {
        self.commandBuffer = commandBuffer
        self.visibleGaussians = visibleGaussians
        self.tileIntersections = tileIntersections
        self.projectionSeconds = projectionSeconds
    }
}

/// Forward tile rasterizer over a `GaussianBuffers` store. The store is read at encode time, so
/// changes to its `count` or contents apply to the next frame.
///
/// Frame flow: command buffer A projects all Gaussians and scans the per-Gaussian tile counts; the
/// CPU reads the total number of tile intersections; command buffer B duplicates keys, sorts them,
/// builds per-tile ranges and rasterizes. `encodeForward` returns B uncommitted.
public final class SplatRenderer {
    static let tileSize = 16

    private let context: MetalContext
    private let prefixSum: GPUPrefixSum
    private let radixSort: GPURadixSort
    private let projectPipeline: MTLComputePipelineState
    private let duplicatePipeline: MTLComputePipelineState
    private let rangesPipeline: MTLComputePipelineState
    private let rasterPipeline: MTLComputePipelineState
    private let rasterBackwardPipeline: MTLComputePipelineState
    private let projectBackwardPipeline: MTLComputePipelineState

    /// The Gaussian store this renderer reads. `setCloud` may replace it.
    public private(set) var gaussians: GaussianBuffers

    // Per-frame buffers, grow-only. Internal read access lets tests inspect the last frame.
    private(set) var meanDepth: MTLBuffer?
    private(set) var conicOpacity: MTLBuffer?
    private(set) var colors: MTLBuffer?
    private(set) var tileRects: MTLBuffer?
    private(set) var tilesTouched: MTLBuffer?
    private(set) var offsets: MTLBuffer?
    private(set) var sortKeys: MTLBuffer?
    private(set) var sortValues: MTLBuffer?
    private(set) var tileRangeBuffer: MTLBuffer?
    /// Per pixel: transmittance after the last blended sample (written by `rasterizeTiles`).
    private(set) var finalTransmittance: MTLBuffer?
    /// Per pixel: sorted index one past the last blended sample, 0 if none.
    private(set) var lastContributor: MTLBuffer?
    /// Per Gaussian: 12 float accumulator slots used by `rasterizeBackward`.
    private(set) var gradientAccumulator: MTLBuffer?

    /// What the backward pass needs from the most recent successful forward pass.
    private struct LastFrame {
        var camera: Camera
        var settings: RasterSettings
        var tilesX: Int
        var tilesY: Int
        var total: Int
        /// Gaussian count at forward time; `encodeBackward` rejects a store whose count changed.
        var count: Int
    }
    private var lastFrame: LastFrame?

    /// Renderer over a fresh store that holds `cloud`.
    public convenience init(context: MetalContext, cloud: GaussianCloud) throws(MetalError) {
        let store = try GaussianBuffers(context: context, cloud: cloud)
        try self.init(context: context, gaussians: store)
    }

    /// Renderer over a shared store. The store's `count` and buffers are read at every frame.
    public init(context: MetalContext, gaussians: GaussianBuffers) throws(MetalError) {
        self.context = context
        prefixSum = try GPUPrefixSum(context: context)
        radixSort = try GPURadixSort(context: context)
        projectPipeline = try context.makePipeline("projectGaussians")
        duplicatePipeline = try context.makePipeline("duplicateKeys")
        rangesPipeline = try context.makePipeline("tileRanges")
        rasterPipeline = try context.makePipeline("rasterizeTiles", minThreads: Self.tileSize * Self.tileSize)
        rasterBackwardPipeline = try context.makePipeline("rasterizeBackward", minThreads: Self.tileSize * Self.tileSize)
        projectBackwardPipeline = try context.makePipeline("projectBackward")
        self.gaussians = gaussians
    }

    /// Replaces the scene. The store is updated in place when the SH degree matches and the cloud
    /// fits its capacity; otherwise a new store is created. Clears the state kept for `backward`.
    public func setCloud(_ cloud: GaussianCloud) throws(MetalError) {
        lastFrame = nil
        if cloud.shDegree == gaussians.shDegree && cloud.count <= gaussians.capacity {
            gaussians.upload(cloud)
        } else {
            gaussians = try GaussianBuffers(context: context, cloud: cloud)
        }
    }

    @discardableResult
    public func render(camera: Camera, settings: RasterSettings, to texture: MTLTexture) throws(MetalError) -> FrameStats {
        // Cleared first so a failed or empty render never leaves a stale frame for `backward`.
        lastFrame = nil
        guard camera.width > 0, camera.height > 0 else {
            return FrameStats(gpuTimeMilliseconds: 0, visibleGaussians: 0, tileIntersections: 0)
        }

        let pass = try encodeForward(camera: camera, settings: settings, to: texture)
        let commandBuffer = pass.commandBuffer
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if commandBuffer.status == .error {
            lastFrame = nil
            throw .commandFailed(commandBuffer.error?.localizedDescription ?? "unknown GPU error")
        }
        let rasterSeconds = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        return FrameStats(gpuTimeMilliseconds: (pass.projectionSeconds + rasterSeconds) * 1000.0,
                          visibleGaussians: pass.visibleGaussians,
                          tileIntersections: pass.tileIntersections)
    }

    /// Encodes one forward pass into `texture`. Projection and the tile-count scan run and complete
    /// first (the CPU needs the intersection total). Duplicate, sort, ranges and rasterize are encoded
    /// into a new command buffer that is returned uncommitted, so the caller can append backward work
    /// and commit once. Sets the state `encodeBackward` uses. Throws for an empty camera.
    public func encodeForward(camera: Camera, settings: RasterSettings, to texture: MTLTexture) throws(MetalError) -> ForwardPass {
        lastFrame = nil
        let width = camera.width
        let height = camera.height
        guard width > 0, height > 0 else {
            throw .commandFailed("camera has no pixels")
        }
        let tile = Self.tileSize
        let tilesX = (width + tile - 1) / tile
        let tilesY = (height + tile - 1) / tile
        let tileCount = tilesX * tilesY
        let store = gaussians
        let n = store.count

        // Per-Gaussian buffers (minimum size handled by ensureBuffer).
        let meanDepthBuffer = try context.ensureBuffer(&meanDepth, length: n * 16)
        let conicBuffer = try context.ensureBuffer(&conicOpacity, length: n * 16)
        let colorBuffer = try context.ensureBuffer(&colors, length: n * 16)
        let rectBuffer = try context.ensureBuffer(&tileRects, length: n * 16)
        let touchedBuffer = try context.ensureBuffer(&tilesTouched, length: n * 4)
        let offsetBuffer = try context.ensureBuffer(&offsets, length: n * 4)
        let rangeBuffer = try context.ensureBuffer(&tileRangeBuffer, length: tileCount * 8)
        let finalTBuffer = try context.ensureBuffer(&finalTransmittance, length: width * height * 4)
        let lastContributorBuffer = try context.ensureBuffer(&lastContributor, length: width * height * 4)

        var projectionSeconds = 0.0
        var total = 0
        var visible = 0

        // Command buffer A: projection + scan of tiles touched.
        if n > 0 {
            guard let commandBufferA = context.queue.makeCommandBuffer() else {
                throw .commandFailed("cannot create command buffer")
            }
            var uniforms = makeProjectUniforms(camera: camera, settings: settings, tilesX: tilesX, tilesY: tilesY)

            let encoder = try context.makeComputeEncoder(commandBufferA)
            encoder.setComputePipelineState(projectPipeline)
            encoder.setBuffer(store.positions, offset: 0, index: 0)
            encoder.setBuffer(store.logScales, offset: 0, index: 1)
            encoder.setBuffer(store.rotations, offset: 0, index: 2)
            encoder.setBuffer(store.opacityLogits, offset: 0, index: 3)
            encoder.setBuffer(store.sh, offset: 0, index: 4)
            encoder.setBuffer(meanDepthBuffer, offset: 0, index: 5)
            encoder.setBuffer(conicBuffer, offset: 0, index: 6)
            encoder.setBuffer(colorBuffer, offset: 0, index: 7)
            encoder.setBuffer(rectBuffer, offset: 0, index: 8)
            encoder.setBuffer(touchedBuffer, offset: 0, index: 9)
            encoder.setBytes(&uniforms, length: MemoryLayout<ProjectUniforms>.size, index: 10)
            MetalContext.dispatch1D(encoder, pipeline: projectPipeline, count: n)
            encoder.endEncoding()

            try prefixSum.encodeExclusiveScan(input: touchedBuffer, output: offsetBuffer, count: n,
                                              commandBuffer: commandBufferA)
            commandBufferA.commit()
            commandBufferA.waitUntilCompleted()
            try Self.check(commandBufferA)
            projectionSeconds = commandBufferA.gpuEndTime - commandBufferA.gpuStartTime

            let touched = touchedBuffer.contents().bindMemory(to: UInt32.self, capacity: n)
            let scanned = offsetBuffer.contents().bindMemory(to: UInt32.self, capacity: n)
            total = Int(scanned[n - 1]) + Int(touched[n - 1])
            for i in 0..<n where touched[i] > 0 {
                visible += 1
            }
        }

        let keyBuffer = try context.ensureBuffer(&sortKeys, length: total * 8)
        let valueBuffer = try context.ensureBuffer(&sortValues, length: total * 4)

        // Command buffer B: duplicate, sort, ranges, rasterize. Returned uncommitted.
        guard let commandBufferB = context.queue.makeCommandBuffer() else {
            throw .commandFailed("cannot create command buffer")
        }

        if total > 0 {
            var dupUniforms = DuplicateUniforms(info: SIMD4(UInt32(n), UInt32(tilesX), UInt32(tilesY), 0))
            let encoder = try context.makeComputeEncoder(commandBufferB)
            encoder.setComputePipelineState(duplicatePipeline)
            encoder.setBuffer(touchedBuffer, offset: 0, index: 0)
            encoder.setBuffer(offsetBuffer, offset: 0, index: 1)
            encoder.setBuffer(rectBuffer, offset: 0, index: 2)
            encoder.setBuffer(meanDepthBuffer, offset: 0, index: 3)
            encoder.setBuffer(keyBuffer, offset: 0, index: 4)
            encoder.setBuffer(valueBuffer, offset: 0, index: 5)
            encoder.setBytes(&dupUniforms, length: MemoryLayout<DuplicateUniforms>.size, index: 6)
            MetalContext.dispatch1D(encoder, pipeline: duplicatePipeline, count: n)
            encoder.endEncoding()

            let tileBits = max(1, UInt32.bitWidth - UInt32(tileCount - 1).leadingZeroBitCount)
            try radixSort.encodeSort(keys: keyBuffer, values: valueBuffer, count: total,
                                     keyBits: 32 + tileBits, commandBuffer: commandBufferB)
        }

        guard let blit = commandBufferB.makeBlitCommandEncoder() else {
            throw .commandFailed("cannot create blit encoder")
        }
        blit.fill(buffer: rangeBuffer, range: 0..<rangeBuffer.length, value: 0)
        blit.endEncoding()

        if total > 0 {
            var totalValue = UInt32(total)
            let encoder = try context.makeComputeEncoder(commandBufferB)
            encoder.setComputePipelineState(rangesPipeline)
            encoder.setBuffer(keyBuffer, offset: 0, index: 0)
            encoder.setBuffer(rangeBuffer, offset: 0, index: 1)
            encoder.setBytes(&totalValue, length: MemoryLayout<UInt32>.size, index: 2)
            MetalContext.dispatch1D(encoder, pipeline: rangesPipeline, count: total)
            encoder.endEncoding()
        }

        do {
            var rasterUniforms = makeRasterUniforms(camera: camera, settings: settings, tilesX: tilesX, tilesY: tilesY)
            let encoder = try context.makeComputeEncoder(commandBufferB)
            encoder.setComputePipelineState(rasterPipeline)
            encoder.setBuffer(meanDepthBuffer, offset: 0, index: 0)
            encoder.setBuffer(conicBuffer, offset: 0, index: 1)
            encoder.setBuffer(colorBuffer, offset: 0, index: 2)
            encoder.setBuffer(valueBuffer, offset: 0, index: 3)
            encoder.setBuffer(rangeBuffer, offset: 0, index: 4)
            encoder.setBytes(&rasterUniforms, length: MemoryLayout<RasterUniforms>.size, index: 5)
            encoder.setTexture(texture, index: 0)
            encoder.setBuffer(finalTBuffer, offset: 0, index: 6)
            encoder.setBuffer(lastContributorBuffer, offset: 0, index: 7)
            encoder.dispatchThreadgroups(MTLSize(width: tilesX, height: tilesY, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: Self.tileSize, height: Self.tileSize, depth: 1))
            encoder.endEncoding()
        }

        lastFrame = LastFrame(camera: camera, settings: settings, tilesX: tilesX, tilesY: tilesY,
                              total: total, count: n)
        return ForwardPass(commandBuffer: commandBufferB, visibleGaussians: visible,
                           tileIntersections: total, projectionSeconds: projectionSeconds)
    }

    public func renderImage(camera: Camera, settings: RasterSettings) throws(MetalError) -> RGBImage {
        let width = camera.width
        let height = camera.height
        guard width > 0, height > 0 else {
            return RGBImage(width: max(0, width), height: max(0, height), pixels: [])
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderWrite, .shaderRead]
        descriptor.storageMode = .shared
        guard let texture = context.device.makeTexture(descriptor: descriptor) else {
            throw .bufferAllocation
        }
        try render(camera: camera, settings: settings, to: texture)

        var raw = [Float](repeating: 0, count: width * height * 4)
        raw.withUnsafeMutableBytes { bytes in
            texture.getBytes(bytes.baseAddress!,
                             bytesPerRow: width * 4 * MemoryLayout<Float>.size,
                             from: MTLRegionMake2D(0, 0, width, height),
                             mipmapLevel: 0)
        }
        var pixels = [SIMD3<Float>]()
        pixels.reserveCapacity(width * height)
        for i in 0..<(width * height) {
            pixels.append(SIMD3(raw[4 * i], raw[4 * i + 1], raw[4 * i + 2]))
        }
        return RGBImage(width: width, height: height, pixels: pixels)
    }

    private static func check(_ commandBuffer: MTLCommandBuffer) throws(MetalError) {
        if commandBuffer.status == .error {
            throw .commandFailed(commandBuffer.error?.localizedDescription ?? "unknown GPU error")
        }
    }

    private func makeProjectUniforms(camera: Camera, settings: RasterSettings,
                                     tilesX: Int, tilesY: Int) -> ProjectUniforms {
        let store = gaussians
        let evalDegree = max(0, min(store.shDegree, settings.shDegreeLimit))
        let rot = camera.rotation
        let t = camera.translation
        let c = camera.center
        return ProjectUniforms(
            rotRow0: SIMD4(rot.c0.x, rot.c1.x, rot.c2.x, t.x),
            rotRow1: SIMD4(rot.c0.y, rot.c1.y, rot.c2.y, t.y),
            rotRow2: SIMD4(rot.c0.z, rot.c1.z, rot.c2.z, t.z),
            camCenter: SIMD4(c.x, c.y, c.z, 0),
            intrinsics: SIMD4(camera.fx, camera.fy, camera.cx, camera.cy),
            limits: SIMD4(settings.nearPlane, settings.farPlane, settings.eps2D, 0),
            counts: SIMD4(UInt32(store.count), UInt32(camera.width), UInt32(camera.height), UInt32(tilesX)),
            shInfo: SIMD4(UInt32(tilesY), UInt32(evalDegree),
                          UInt32(GaussianCloud.coefficientCount(degree: store.shDegree)), 0))
    }

    private func makeRasterUniforms(camera: Camera, settings: RasterSettings,
                                    tilesX: Int, tilesY: Int) -> RasterUniforms {
        let bg = settings.background
        return RasterUniforms(
            background: SIMD4(bg.x, bg.y, bg.z, 0),
            info: SIMD4(UInt32(camera.width), UInt32(camera.height), UInt32(tilesX), UInt32(tilesY)))
    }
}

extension SplatRenderer {
    /// Encodes the backward pass for the most recent forward pass (same camera and settings).
    ///
    /// - Parameter imageGradient: `dL/dpixel` as `width·height` RGBA floats (A ignored).
    /// - Parameter gradients: destination buffers; their capacity must be at least the store's count.
    /// - Parameter statistics: when given, this view's screen-space gradient norms and visible counts are
    ///   added to it (`accumulateDensifyStats`, after `rasterizeBackward`). The gradients do not change.
    ///   Its capacity must be at least the store's count.
    ///
    /// Throws if the Gaussian count changed since the forward pass. The accumulator and the gradient
    /// buffers are zeroed on the GPU first. When no tile is intersected, the gradients stay zero and
    /// no kernel is dispatched (statistics are not touched either, since no Gaussian is visible).
    public func encodeBackward(imageGradient: MTLBuffer, into gradients: GradientBuffers,
                               statistics: DensificationStatisticsBuffers? = nil,
                               commandBuffer: MTLCommandBuffer) throws(MetalError) {
        guard let frame = lastFrame else { throw .commandFailed("no frame rendered") }
        let store = gaussians
        guard frame.count == store.count else {
            throw .commandFailed("gaussian count changed since render")
        }
        guard gradients.capacity >= store.count, gradients.shDegree == store.shDegree else {
            throw .commandFailed("gradient buffers do not match the cloud")
        }
        let n = store.count
        if let statistics, statistics.capacity < n {
            throw .commandFailed("statistics capacity \(statistics.capacity) is below the cloud count \(n)")
        }
        let width = frame.camera.width
        let height = frame.camera.height
        guard imageGradient.length >= width * height * 4 * MemoryLayout<Float>.size else {
            throw .commandFailed("imageGradient must hold width·height RGBA floats")
        }
        let accumulator = try context.ensureBuffer(
            &gradientAccumulator, length: n * 12 * MemoryLayout<Float>.size)

        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw .commandFailed("cannot create blit encoder")
        }
        blit.fill(buffer: accumulator, range: 0..<accumulator.length, value: 0)
        for buffer in [gradients.positions, gradients.logScales, gradients.rotations,
                       gradients.opacityLogits, gradients.sh] {
            blit.fill(buffer: buffer, range: 0..<buffer.length, value: 0)
        }
        blit.endEncoding()

        guard frame.total > 0, n > 0 else { return }
        guard let meanDepthBuffer = meanDepth, let conicBuffer = conicOpacity, let colorBuffer = colors,
              let touchedBuffer = tilesTouched, let valueBuffer = sortValues,
              let rangeBuffer = tileRangeBuffer, let finalTBuffer = finalTransmittance,
              let lastBuffer = lastContributor else {
            throw .commandFailed("forward buffers are missing")
        }

        // Pixel stage: one threadgroup per tile, same layout as the forward rasterizer.
        var rasterUniforms = makeRasterUniforms(camera: frame.camera, settings: frame.settings,
                                                tilesX: frame.tilesX, tilesY: frame.tilesY)
        let rasterEncoder = try context.makeComputeEncoder(commandBuffer)
        rasterEncoder.setComputePipelineState(rasterBackwardPipeline)
        rasterEncoder.setBuffer(meanDepthBuffer, offset: 0, index: 0)
        rasterEncoder.setBuffer(conicBuffer, offset: 0, index: 1)
        rasterEncoder.setBuffer(colorBuffer, offset: 0, index: 2)
        rasterEncoder.setBuffer(valueBuffer, offset: 0, index: 3)
        rasterEncoder.setBuffer(rangeBuffer, offset: 0, index: 4)
        rasterEncoder.setBytes(&rasterUniforms, length: MemoryLayout<RasterUniforms>.size, index: 5)
        rasterEncoder.setBuffer(imageGradient, offset: 0, index: 6)
        rasterEncoder.setBuffer(finalTBuffer, offset: 0, index: 7)
        rasterEncoder.setBuffer(lastBuffer, offset: 0, index: 8)
        rasterEncoder.setBuffer(accumulator, offset: 0, index: 9)
        rasterEncoder.dispatchThreadgroups(
            MTLSize(width: frame.tilesX, height: frame.tilesY, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.tileSize, height: Self.tileSize, depth: 1))
        rasterEncoder.endEncoding()

        // Statistics hook: reads the accumulator slots 7-8 that rasterizeBackward just wrote (serial encoders).
        if let statistics {
            try statistics.encodeAccumulate(accumulator: accumulator, tilesTouched: touchedBuffer, count: n,
                                            width: width, height: height, commandBuffer: commandBuffer)
        }

        // Gaussian stage: one thread per Gaussian, chain rule to the stored parameters.
        var projectUniforms = makeProjectUniforms(camera: frame.camera, settings: frame.settings,
                                                  tilesX: frame.tilesX, tilesY: frame.tilesY)
        let projectEncoder = try context.makeComputeEncoder(commandBuffer)
        projectEncoder.setComputePipelineState(projectBackwardPipeline)
        projectEncoder.setBuffer(store.positions, offset: 0, index: 0)
        projectEncoder.setBuffer(store.logScales, offset: 0, index: 1)
        projectEncoder.setBuffer(store.rotations, offset: 0, index: 2)
        projectEncoder.setBuffer(store.opacityLogits, offset: 0, index: 3)
        projectEncoder.setBuffer(store.sh, offset: 0, index: 4)
        projectEncoder.setBuffer(accumulator, offset: 0, index: 5)
        projectEncoder.setBuffer(touchedBuffer, offset: 0, index: 6)
        projectEncoder.setBuffer(gradients.positions, offset: 0, index: 7)
        projectEncoder.setBuffer(gradients.logScales, offset: 0, index: 8)
        projectEncoder.setBuffer(gradients.rotations, offset: 0, index: 9)
        projectEncoder.setBuffer(gradients.opacityLogits, offset: 0, index: 10)
        projectEncoder.setBuffer(gradients.sh, offset: 0, index: 11)
        projectEncoder.setBytes(&projectUniforms, length: MemoryLayout<ProjectUniforms>.size, index: 12)
        MetalContext.dispatch1D(projectEncoder, pipeline: projectBackwardPipeline, count: n)
        projectEncoder.endEncoding()
    }

    /// Gradients of `L = Σ imageGradient · image` with respect to every Gaussian parameter, for the
    /// most recent forward pass. Synchronous: waits for the GPU.
    public func backward(imageGradient: [SIMD3<Float>]) throws(MetalError) -> GaussianGradients {
        guard let frame = lastFrame else { throw .commandFailed("no frame rendered") }
        let pixelCount = frame.camera.width * frame.camera.height
        guard imageGradient.count == pixelCount else {
            throw .commandFailed("imageGradient must hold \(pixelCount) pixels")
        }

        let gradientImage = try context.makeBuffer(length: pixelCount * 4 * MemoryLayout<Float>.size)
        if pixelCount > 0 {
            let pointer = gradientImage.contents().bindMemory(to: Float.self, capacity: pixelCount * 4)
            for i in 0..<pixelCount {
                pointer[4 * i] = imageGradient[i].x
                pointer[4 * i + 1] = imageGradient[i].y
                pointer[4 * i + 2] = imageGradient[i].z
                pointer[4 * i + 3] = 0
            }
        }

        let store = gaussians
        let gradients = try GradientBuffers(context: context, count: store.count, shDegree: store.shDegree)
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw .commandFailed("cannot create command buffer")
        }
        try encodeBackward(imageGradient: gradientImage, into: gradients, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try Self.check(commandBuffer)
        return gradients.toGaussianGradients(count: store.count, shDegree: store.shDegree)
    }
}
