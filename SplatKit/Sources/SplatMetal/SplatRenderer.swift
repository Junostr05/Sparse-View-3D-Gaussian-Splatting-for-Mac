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
    private let depthColormapPipeline: MTLComputePipelineState

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
    /// Per Gaussian: 12 float accumulator slots used by `rasterizeBackward` (slot 9 = depth, Phase 4).
    private(set) var gradientAccumulator: MTLBuffer?
    /// Bound to the depth slots of the forward and backward kernels when no depth buffer is given.
    private var dummyDepth: MTLBuffer?

    /// The viewer display of the forward pass (`setDisplay`). `.splat` is the normal rendering, which training uses.
    private(set) var display: ViewerDisplay = .splat
    /// The colour array the override buffer was filled from. `nil` when the buffer is stale or unused.
    private var uploadedColorOverride: [SIMD4<Float>]?
    /// Per Gaussian: the override colour of the points, shapes and heatmap modes (`setDisplay`). Grow-only.
    private var colorOverrideBuffer: MTLBuffer?
    /// Float2 per pixel for the depth view (`renderDepthColorized`). Grow-only.
    private var depthViewBuffer: MTLBuffer?
    /// Depth view: pixels with less accumulated opacity than this (A = 1 − T_final) show the background.
    static let minimumDepthAlpha: Float = 0.5

    /// Per Gaussian: sampling rate ν of the Mip-Splatting 3D filter (`GPUSamplingRate`). Read only when the
    /// settings' `mipFilter` is on; the trainer keeps it current. Must hold at least `gaussians.count` floats.
    public var samplingRateBuffer: MTLBuffer?

    /// What the backward pass needs from the most recent successful forward pass.
    private struct LastFrame {
        var camera: Camera
        var settings: RasterSettings
        var tilesX: Int
        var tilesY: Int
        var total: Int
        /// Gaussian count at forward time; `encodeBackward` rejects a store whose count changed.
        var count: Int
        /// Display mode of the forward pass. `encodeBackward` needs `.splat`: the other modes change the geometry.
        var viewMode: ViewerDisplayMode
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
        depthColormapPipeline = try context.makePipeline("depthColormap")
        self.gaussians = gaussians
    }

    /// Replaces the scene. The store is updated in place when the SH degree matches and the cloud
    /// fits its capacity; otherwise a new store is created. Clears the state kept for `backward`.
    /// The colour override is dropped (the SH colours are drawn) until `setDisplay` is called with the colours of the new cloud.
    public func setCloud(_ cloud: GaussianCloud) throws(MetalError) {
        lastFrame = nil
        uploadedColorOverride = nil
        display.colorOverride = nil
        if cloud.shDegree == gaussians.shDegree && cloud.count <= gaussians.capacity {
            gaussians.upload(cloud)
        } else {
            gaussians = try GaussianBuffers(context: context, cloud: cloud)
        }
    }

    /// Sets how the following frames are drawn. The default `.splat` is the normal rendering. The colour override
    /// of the points, shapes and heatmap modes is uploaded when its array changes; it must hold at least one colour
    /// per Gaussian of the current cloud. Without an override those modes draw the SH colours.
    public func setDisplay(_ newDisplay: ViewerDisplay) throws(MetalError) {
        if let colors = newDisplay.colorOverride, uploadedColorOverride != colors {
            let n: Int = gaussians.count
            guard colors.count >= n else {
                throw .commandFailed("colour override has \(colors.count) colours for \(n) Gaussians")
            }
            let buffer = try context.ensureBuffer(&colorOverrideBuffer, length: n * MemoryLayout<SIMD4<Float>>.stride)
            let pointer = buffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: max(n, 1))
            for i in 0..<n {
                pointer[i] = colors[i]
            }
            uploadedColorOverride = colors
        }
        display = newDisplay
    }

    /// Projection mode of the display (`ViewModeUniforms.modeInfo.x`). The depth view projects like the splats.
    private var projectionCode: UInt32 {
        switch display.mode {
        case .splat, .depth: return 0
        case .points: return 1
        case .shapes: return 2
        case .heatmap: return 3
        }
    }

    /// True when the projection reads the colour override: a non-splat mode with an uploaded override.
    private var colorOverrideActive: Bool {
        projectionCode != 0 && display.colorOverride != nil && uploadedColorOverride != nil
    }

    /// Viewer crop box in the rendered scene (`setCrop`). `nil` is off, which is what training and checkpoints use.
    private(set) var crop: CropBox?

    /// Sets or clears the viewer crop. Gaussians whose centre lies outside the box are culled in the projection of
    /// every forward pass until it is cleared. Training never calls this, so its passes are unchanged.
    public func setCrop(_ crop: CropBox?) {
        self.crop = crop
    }

    private func makeViewModeUniforms() -> ViewModeUniforms {
        let mode: UInt32 = projectionCode
        let overrideFlag: UInt32 = colorOverrideActive ? 1 : 0
        var cropCentre: SIMD4<Float> = .zero
        var cropHalfExtents: SIMD4<Float> = .zero
        var cropRow0: SIMD4<Float> = .zero
        var cropRow1: SIMD4<Float> = .zero
        var cropRow2: SIMD4<Float> = .zero
        if let crop {
            let rows: [SIMD3<Float>] = crop.localRows
            cropCentre = SIMD4<Float>(crop.center.x, crop.center.y, crop.center.z, 1)
            cropHalfExtents = SIMD4<Float>(crop.halfExtents.x, crop.halfExtents.y, crop.halfExtents.z, 0)
            cropRow0 = SIMD4<Float>(rows[0].x, rows[0].y, rows[0].z, 0)
            cropRow1 = SIMD4<Float>(rows[1].x, rows[1].y, rows[1].z, 0)
            cropRow2 = SIMD4<Float>(rows[2].x, rows[2].y, rows[2].z, 0)
        }
        return ViewModeUniforms(modeInfo: SIMD4<UInt32>(mode, overrideFlag, 0, 0),
                                styleInfo: SIMD4<Float>(display.pointRadiusPixels, display.opacityScale,
                                                        display.scaleScale, 0),
                                cropCentre: cropCentre,
                                cropHalfExtents: cropHalfExtents,
                                cropRow0: cropRow0,
                                cropRow1: cropRow1,
                                cropRow2: cropRow2)
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
    ///
    /// Shared state and ordering: the projection buffers (`meanDepth`, `conicOpacity`, `colors`, tile rects,
    /// sort keys and values, tile ranges, the per-pixel transmittance and last-contributor buffers and the
    /// screen-space gradient accumulator) belong to this renderer and are shared by all views. One view's
    /// forward AND backward command buffer must therefore be COMMITTED before the next view's `encodeForward`
    /// is called. Encoding view B's forward before committing view A's silently makes A train on B's geometry.
    /// Command buffers on the queue execute in commit order, so committing A first is enough.
    ///
    /// - Parameter depth: float2 per pixel, ≥ width·height·8 bytes, receives `(Draw, A)` = `(Σ αᵢTᵢzᵢ, 1 − T_final)`.
    ///   `nil` is the Phase 3 forward pass (the image is identical either way).
    public func encodeForward(camera: Camera, settings: RasterSettings, to texture: MTLTexture,
                              depth: MTLBuffer? = nil) throws(MetalError) -> ForwardPass {
        lastFrame = nil
        let width = camera.width
        let height = camera.height
        guard width > 0, height > 0 else {
            throw .commandFailed("camera has no pixels")
        }
        if let depth, depth.length < width * height * 2 * MemoryLayout<Float>.size {
            throw .commandFailed("depth must hold width·height float2 values")
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
            var mipUniforms = makeMipUniforms(settings: settings)
            let ratesBinding = try samplingRatesBinding(settings: settings)
            var viewUniforms = makeViewModeUniforms()
            // The colour override is bound only when it is active; otherwise a 16-byte dummy that is never read.
            let colorBinding: MTLBuffer
            if colorOverrideActive, let overrideBuffer = colorOverrideBuffer {
                colorBinding = overrideBuffer
            } else {
                colorBinding = try dummyDepthBuffer()
            }

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
            encoder.setBytes(&mipUniforms, length: MemoryLayout<MipFilterUniforms>.size, index: 11)
            encoder.setBuffer(ratesBinding, offset: 0, index: 12)
            encoder.setBytes(&viewUniforms, length: MemoryLayout<ViewModeUniforms>.size, index: 13)
            encoder.setBuffer(colorBinding, offset: 0, index: 14)
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

        // The depth binding is always set (a 16-byte dummy when depth is off): the kernel writes it only in depth mode.
        let depthBinding: MTLBuffer
        if let depth {
            depthBinding = depth
        } else {
            depthBinding = try dummyDepthBuffer()
        }
        do {
            var rasterUniforms = makeRasterUniforms(camera: camera, settings: settings, tilesX: tilesX, tilesY: tilesY,
                                                    depthMode: depth != nil)
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
            encoder.setBuffer(depthBinding, offset: 0, index: 8)
            encoder.dispatchThreadgroups(MTLSize(width: tilesX, height: tilesY, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: Self.tileSize, height: Self.tileSize, depth: 1))
            encoder.endEncoding()
        }

        lastFrame = LastFrame(camera: camera, settings: settings, tilesX: tilesX, tilesY: tilesY,
                              total: total, count: n, viewMode: display.mode)
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
                                     tilesX: Int, tilesY: Int, accumulate: Bool = false,
                                     poseOn: Bool = false) -> ProjectUniforms {
        let store = gaussians
        let evalDegree = max(0, min(store.shDegree, settings.shDegreeLimit))
        let rot = camera.rotation
        let t = camera.translation
        let c = camera.center
        // In mip mode the kernels take the 2D filter variance v2 in the eps2D slot.
        let eps2D = settings.mipFilter.isEnabled ? settings.mipFilter.filter2DVariance : settings.eps2D
        return ProjectUniforms(
            rotRow0: SIMD4(rot.c0.x, rot.c1.x, rot.c2.x, t.x),
            rotRow1: SIMD4(rot.c0.y, rot.c1.y, rot.c2.y, t.y),
            rotRow2: SIMD4(rot.c0.z, rot.c1.z, rot.c2.z, t.z),
            camCenter: SIMD4(c.x, c.y, c.z, 0),
            intrinsics: SIMD4(camera.fx, camera.fy, camera.cx, camera.cy),
            limits: SIMD4(settings.nearPlane, settings.farPlane, eps2D, poseOn ? 1 : 0),
            counts: SIMD4(UInt32(store.count), UInt32(camera.width), UInt32(camera.height), UInt32(tilesX)),
            shInfo: SIMD4(UInt32(tilesY), UInt32(evalDegree),
                          UInt32(GaussianCloud.coefficientCount(degree: store.shDegree)),
                          accumulate ? 1 : 0))
    }

    private func makeMipUniforms(settings: RasterSettings) -> MipFilterUniforms {
        let mip = settings.mipFilter
        return MipFilterUniforms(info: SIMD4(mip.isEnabled ? 1 : 0, mip.filter3DVariance, 0, 0))
    }

    /// The sampling-rate buffer when the mip filters are on (an error if it is missing), else any valid buffer,
    /// which the kernels do not read.
    private func samplingRatesBinding(settings: RasterSettings) throws(MetalError) -> MTLBuffer {
        if settings.mipFilter.isEnabled {
            guard let buffer = samplingRateBuffer else {
                throw .commandFailed("mip filter needs samplingRateBuffer")
            }
            return buffer
        }
        return gaussians.positions
    }

    private func makeRasterUniforms(camera: Camera, settings: RasterSettings,
                                    tilesX: Int, tilesY: Int, depthMode: Bool = false) -> RasterUniforms {
        let bg = settings.background
        return RasterUniforms(
            background: SIMD4(bg.x, bg.y, bg.z, depthMode ? 1 : 0),
            info: SIMD4(UInt32(camera.width), UInt32(camera.height), UInt32(tilesX), UInt32(tilesY)))
    }

    /// 16-byte buffer bound to the depth slots when depth is off. The kernels never touch it in that mode.
    private func dummyDepthBuffer() throws(MetalError) -> MTLBuffer {
        try context.ensureBuffer(&dummyDepth, length: 16)
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
    /// - Parameter depthGradient: float2 per pixel `(∂L/∂Draw, ∂L/∂A)`, ≥ width·height·8 bytes. `nil` is the
    ///   Phase 3 image-only backward. The forward pass must have been rendered for the same camera; its depth
    ///   channel is not needed (the backward uses the saved transmittance and colour/depth of the samples).
    /// - Parameter accumulate: add the parameter gradients into `gradients` instead of overwriting them. The
    ///   gradient buffers are not zero-filled, and culled Gaussians are not written. The accumulator is always zeroed.
    ///   The screen-space accumulator is shared by all views: a view's backward must be committed before the next
    ///   view's `encodeForward` (see `encodeForward`), and per-view contributions are summed by accumulating into
    ///   the same `gradients`.
    ///
    /// Throws if the Gaussian count changed since the forward pass. The accumulator and, unless `accumulate`,
    /// the gradient buffers are zeroed on the GPU first. When no tile is intersected, the gradients stay as
    /// they are (zero when not accumulating) and no kernel is dispatched (statistics are not touched either,
    /// since no Gaussian is visible).
    ///
    /// - Parameter poseSink: when given, the camera-pose gradient of this view is written to the sink's scratch and
    ///   reduced into its six sums (`PoseGradientSink`), in the same command buffer. Parameter gradients are unchanged.
    ///   Nothing is written when no tile is intersected.
    public func encodeBackward(imageGradient: MTLBuffer, depthGradient: MTLBuffer? = nil,
                               into gradients: GradientBuffers,
                               statistics: DensificationStatisticsBuffers? = nil,
                               accumulate: Bool = false,
                               poseSink: PoseGradientSink? = nil,
                               commandBuffer: MTLCommandBuffer) throws(MetalError) {
        guard let frame = lastFrame else { throw .commandFailed("no frame rendered") }
        guard frame.viewMode == .splat else {
            throw .commandFailed("backward needs the splat display; the last frame used \(frame.viewMode.rawValue)")
        }
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
        if let depthGradient, depthGradient.length < width * height * 2 * MemoryLayout<Float>.size {
            throw .commandFailed("depthGradient must hold width·height float2 values")
        }
        let accumulator = try context.ensureBuffer(
            &gradientAccumulator, length: n * 12 * MemoryLayout<Float>.size)

        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw .commandFailed("cannot create blit encoder")
        }
        blit.fill(buffer: accumulator, range: 0..<accumulator.length, value: 0)
        if !accumulate {
            for buffer in [gradients.positions, gradients.logScales, gradients.rotations,
                           gradients.opacityLogits, gradients.sh] {
                blit.fill(buffer: buffer, range: 0..<buffer.length, value: 0)
            }
        }
        blit.endEncoding()

        guard frame.total > 0, n > 0 else { return }
        guard let meanDepthBuffer = meanDepth, let conicBuffer = conicOpacity, let colorBuffer = colors,
              let touchedBuffer = tilesTouched, let valueBuffer = sortValues,
              let rangeBuffer = tileRangeBuffer, let finalTBuffer = finalTransmittance,
              let lastBuffer = lastContributor else {
            throw .commandFailed("forward buffers are missing")
        }
        let depthBinding: MTLBuffer
        if let depthGradient {
            depthBinding = depthGradient
        } else {
            depthBinding = try dummyDepthBuffer()
        }

        // Pixel stage: one threadgroup per tile, same layout as the forward rasterizer.
        var rasterUniforms = makeRasterUniforms(camera: frame.camera, settings: frame.settings,
                                                tilesX: frame.tilesX, tilesY: frame.tilesY,
                                                depthMode: depthGradient != nil)
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
        rasterEncoder.setBuffer(depthBinding, offset: 0, index: 10)
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
        if let poseSink {
            guard poseSink.scratch.length >= n * 6 * MemoryLayout<Float>.size,
                  poseSink.sums.length >= (poseSink.base + 6) * MemoryLayout<Float>.size else {
                throw .commandFailed("pose gradient buffers are too small for the cloud")
            }
        }
        var projectUniforms = makeProjectUniforms(camera: frame.camera, settings: frame.settings,
                                                  tilesX: frame.tilesX, tilesY: frame.tilesY,
                                                  accumulate: accumulate, poseOn: poseSink != nil)
        var mipUniforms = makeMipUniforms(settings: frame.settings)
        let ratesBinding = try samplingRatesBinding(settings: frame.settings)
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
        projectEncoder.setBytes(&mipUniforms, length: MemoryLayout<MipFilterUniforms>.size, index: 13)
        projectEncoder.setBuffer(ratesBinding, offset: 0, index: 14)
        projectEncoder.setBuffer(poseSink?.scratch ?? gradients.positions, offset: 0, index: 15)
        MetalContext.dispatch1D(projectEncoder, pipeline: projectBackwardPipeline, count: n)
        projectEncoder.endEncoding()

        // Pose gradient (pose refinement): reduce the per-Gaussian contributions into the view's sums.
        if let poseSink {
            try poseSink.reducer.encode(scratch: poseSink.scratch, count: n, sums: poseSink.sums,
                                        base: poseSink.base, commandBuffer: commandBuffer)
        }
    }

    /// Rendered depth of the view, for tests and evaluation. Synchronous: waits for the GPU.
    /// Returns `Draw = Σ αᵢTᵢzᵢ` (`raw`) and `A = 1 − T_final` (`alpha`), row-major width·height.
    /// Like `render`, it leaves the frame state that `backward` needs.
    /// Throws for an empty camera, like `encodeForward`.
    public func renderDepth(camera: Camera, settings: RasterSettings) throws(MetalError) -> DepthRender {
        lastFrame = nil
        let width = camera.width
        let height = camera.height
        guard width > 0, height > 0 else {
            throw .commandFailed("camera has no pixels")
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderWrite, .shaderRead]
        descriptor.storageMode = .shared
        guard let texture = context.device.makeTexture(descriptor: descriptor) else {
            throw .bufferAllocation
        }
        let depthBuffer = try context.makeBuffer(length: width * height * 2 * MemoryLayout<Float>.size)
        let pass = try encodeForward(camera: camera, settings: settings, to: texture, depth: depthBuffer)
        let commandBuffer = pass.commandBuffer
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try Self.check(commandBuffer)

        let pixelCount = width * height
        let values = depthBuffer.contents().bindMemory(to: Float.self, capacity: 2 * pixelCount)
        var raw = [Float](repeating: 0, count: pixelCount)
        var alpha = [Float](repeating: 0, count: pixelCount)
        for i in 0..<pixelCount {
            raw[i] = values[2 * i]
            alpha[i] = values[2 * i + 1]
        }
        return DepthRender(raw: raw, alpha: alpha)
    }

    /// Depth view: the rendered depth of the splats, mapped to colour with a turbo-like ramp, near = warm.
    /// Pixels with less than `minimumDepthAlpha` opacity show `settings.background`. The range of the ramp is the
    /// 2nd to 98th percentile of the valid depths of this frame. The splats are always projected normally here,
    /// whatever display is set. Synchronous: waits for the GPU. Like `renderDepth`, it keeps no state for `backward`.
    @discardableResult
    public func renderDepthColorized(camera: Camera, settings: RasterSettings,
                                     to texture: MTLTexture) throws(MetalError) -> FrameStats {
        lastFrame = nil
        let width = camera.width
        let height = camera.height
        guard width > 0, height > 0 else {
            return FrameStats(gpuTimeMilliseconds: 0, visibleGaussians: 0, tileIntersections: 0)
        }
        let pixelCount = width * height
        let depthBuffer = try context.ensureBuffer(&depthViewBuffer, length: pixelCount * 2 * MemoryLayout<Float>.size)

        let savedDisplay = display
        display = .splat
        defer { display = savedDisplay }
        let pass = try encodeForward(camera: camera, settings: settings, to: texture, depth: depthBuffer)
        let forwardBuffer = pass.commandBuffer
        forwardBuffer.commit()
        forwardBuffer.waitUntilCompleted()
        try Self.check(forwardBuffer)
        lastFrame = nil

        let range = Self.depthRange(of: depthBuffer, pixelCount: pixelCount)
        var uniforms = DepthColormapUniforms(
            range: SIMD4<Float>(range.low, range.high, Self.minimumDepthAlpha, 0),
            background: SIMD4<Float>(settings.background.x, settings.background.y, settings.background.z, 0),
            size: SIMD4<UInt32>(UInt32(width), UInt32(height), 0, 0))
        guard let colormapBuffer = context.queue.makeCommandBuffer() else {
            throw .commandFailed("cannot create command buffer")
        }
        let encoder = try context.makeComputeEncoder(colormapBuffer)
        encoder.setComputePipelineState(depthColormapPipeline)
        encoder.setBuffer(depthBuffer, offset: 0, index: 0)
        encoder.setBytes(&uniforms, length: MemoryLayout<DepthColormapUniforms>.size, index: 1)
        encoder.setTexture(texture, index: 0)
        encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: Self.tileSize, height: Self.tileSize, depth: 1))
        encoder.endEncoding()
        colormapBuffer.commit()
        colormapBuffer.waitUntilCompleted()
        try Self.check(colormapBuffer)

        let forwardSeconds = pass.projectionSeconds + (forwardBuffer.gpuEndTime - forwardBuffer.gpuStartTime)
        let colormapSeconds = colormapBuffer.gpuEndTime - colormapBuffer.gpuStartTime
        return FrameStats(gpuTimeMilliseconds: (forwardSeconds + colormapSeconds) * 1000.0,
                          visibleGaussians: pass.visibleGaussians,
                          tileIntersections: pass.tileIntersections)
    }

    /// The 2nd and 98th percentile of the depth (Draw / A) over the valid pixels, from a 4096-bin histogram over
    /// the valid range: two passes over the buffer. Bin centres are returned. `(0, 0)` without valid pixels.
    static func depthRange(of buffer: MTLBuffer, pixelCount: Int) -> (low: Float, high: Float) {
        let binCount = 4096
        let values = buffer.contents().bindMemory(to: Float.self, capacity: 2 * max(pixelCount, 1))
        var low = Float.infinity
        var high = -Float.infinity
        var valid = 0
        for i in 0..<pixelCount {
            let alpha = values[2 * i + 1]
            guard alpha >= minimumDepthAlpha, alpha > 0 else { continue }
            let depth = values[2 * i] / alpha
            guard depth.isFinite else { continue }
            valid += 1
            low = min(low, depth)
            high = max(high, depth)
        }
        guard valid > 0 else { return (0, 0) }
        let width = (high - low) / Float(binCount)
        guard width > 0 else { return (low, high) }

        var histogram = [Int](repeating: 0, count: binCount)
        for i in 0..<pixelCount {
            let alpha = values[2 * i + 1]
            guard alpha >= minimumDepthAlpha, alpha > 0 else { continue }
            let depth = values[2 * i] / alpha
            guard depth.isFinite else { continue }
            let bin = min(Int((depth - low) / width), binCount - 1)
            histogram[bin] += 1
        }
        func quantile(_ fraction: Float) -> Float {
            let target = max(1, Int((Float(valid) * fraction).rounded(.up)))
            var cumulative = 0
            for bin in 0..<binCount {
                cumulative += histogram[bin]
                if cumulative >= target {
                    return low + (Float(bin) + 0.5) * width
                }
            }
            return high
        }
        return (quantile(0.02), quantile(0.98))
    }

    /// Gradients of `L = Σ imageGradient · image + Σ (gD·Draw + gA·A)` with respect to every Gaussian
    /// parameter, for the most recent forward pass. `depthGradient` holds `(∂L/∂Draw, ∂L/∂A)` per pixel;
    /// `nil` is the Phase 3 image-only backward. Synchronous: waits for the GPU.
    public func backward(imageGradient: [SIMD3<Float>],
                         depthGradient: [SIMD2<Float>]? = nil) throws(MetalError) -> GaussianGradients {
        guard let frame = lastFrame else { throw .commandFailed("no frame rendered") }
        let pixelCount = frame.camera.width * frame.camera.height
        guard imageGradient.count == pixelCount else {
            throw .commandFailed("imageGradient must hold \(pixelCount) pixels")
        }
        if let depthGradient, depthGradient.count != pixelCount {
            throw .commandFailed("depthGradient must hold \(pixelCount) pixels")
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

        var depthBuffer: MTLBuffer? = nil
        if let depthGradient {
            let buffer = try context.makeBuffer(length: max(1, pixelCount) * 2 * MemoryLayout<Float>.size)
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: 2 * pixelCount)
            for i in 0..<pixelCount {
                pointer[2 * i] = depthGradient[i].x
                pointer[2 * i + 1] = depthGradient[i].y
            }
            depthBuffer = buffer
        }

        let store = gaussians
        let gradients = try GradientBuffers(context: context, count: store.count, shDegree: store.shDegree)
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw .commandFailed("cannot create command buffer")
        }
        try encodeBackward(imageGradient: gradientImage, depthGradient: depthBuffer, into: gradients,
                           commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try Self.check(commandBuffer)
        return gradients.toGaussianGradients(count: store.count, shDegree: store.shDegree)
    }
}
