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

/// Forward tile rasterizer for a fixed `GaussianCloud`. The Gaussian data is uploaded once.
///
/// Frame flow: command buffer A projects all Gaussians and scans the per-Gaussian tile counts; the
/// CPU reads the total number of tile intersections; command buffer B duplicates keys, sorts them,
/// builds per-tile ranges and rasterizes.
public final class SplatRenderer {
    static let tileSize = 16

    private let context: MetalContext
    private let prefixSum: GPUPrefixSum
    private let radixSort: GPURadixSort
    private let projectPipeline: MTLComputePipelineState
    private let duplicatePipeline: MTLComputePipelineState
    private let rangesPipeline: MTLComputePipelineState
    private let rasterPipeline: MTLComputePipelineState

    private let count: Int
    private let shDegree: Int
    private let positions: MTLBuffer
    private let logScales: MTLBuffer
    private let rotations: MTLBuffer
    private let opacityLogits: MTLBuffer
    private let shCoefficients: MTLBuffer

    // Per-frame buffers, grow-only.
    private var meanDepth: MTLBuffer?
    private var conicOpacity: MTLBuffer?
    private var colors: MTLBuffer?
    private var tileRects: MTLBuffer?
    private var tilesTouched: MTLBuffer?
    private var offsets: MTLBuffer?
    private var sortKeys: MTLBuffer?
    private var sortValues: MTLBuffer?
    private var tileRangeBuffer: MTLBuffer?

    public init(context: MetalContext, cloud: GaussianCloud) throws(MetalError) {
        self.context = context
        prefixSum = try GPUPrefixSum(context: context)
        radixSort = try GPURadixSort(context: context)
        projectPipeline = try context.makePipeline("projectGaussians")
        duplicatePipeline = try context.makePipeline("duplicateKeys")
        rangesPipeline = try context.makePipeline("tileRanges")
        rasterPipeline = try context.makePipeline("rasterizeTiles", minThreads: Self.tileSize * Self.tileSize)

        count = cloud.count
        shDegree = cloud.shDegree
        positions = try Self.upload(cloud.positions, context: context)
        logScales = try Self.upload(cloud.logScales, context: context)
        rotations = try Self.upload(cloud.rotations, context: context)
        opacityLogits = try Self.upload(cloud.opacityLogits, context: context)
        shCoefficients = try Self.upload(cloud.sh, context: context)
    }

    private static func upload(_ array: [Float], context: MetalContext) throws(MetalError) -> MTLBuffer {
        let buffer = try context.makeBuffer(length: array.count * MemoryLayout<Float>.size)
        if !array.isEmpty {
            array.withUnsafeBytes { raw in
                buffer.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
            }
        }
        return buffer
    }

    @discardableResult
    public func render(camera: Camera, settings: RasterSettings, to texture: MTLTexture) throws(MetalError) -> FrameStats {
        let width = camera.width
        let height = camera.height
        guard width > 0, height > 0 else {
            return FrameStats(gpuTimeMilliseconds: 0, visibleGaussians: 0, tileIntersections: 0)
        }
        let tile = Self.tileSize
        let tilesX = (width + tile - 1) / tile
        let tilesY = (height + tile - 1) / tile
        let tileCount = tilesX * tilesY
        let n = count

        // Per-Gaussian buffers (minimum size handled by ensureBuffer).
        let meanDepthBuffer = try context.ensureBuffer(&meanDepth, length: n * 16)
        let conicBuffer = try context.ensureBuffer(&conicOpacity, length: n * 16)
        let colorBuffer = try context.ensureBuffer(&colors, length: n * 16)
        let rectBuffer = try context.ensureBuffer(&tileRects, length: n * 16)
        let touchedBuffer = try context.ensureBuffer(&tilesTouched, length: n * 4)
        let offsetBuffer = try context.ensureBuffer(&offsets, length: n * 4)
        let rangeBuffer = try context.ensureBuffer(&tileRangeBuffer, length: tileCount * 8)

        var gpuSeconds = 0.0
        var total = 0
        var visible = 0

        // Command buffer A: projection + scan of tiles touched.
        if n > 0 {
            guard let commandBufferA = context.queue.makeCommandBuffer() else {
                throw .commandFailed("cannot create command buffer")
            }
            let evalDegree = max(0, min(shDegree, settings.shDegreeLimit))
            let rot = camera.rotation
            let t = camera.translation
            let c = camera.center
            var uniforms = ProjectUniforms(
                rotRow0: SIMD4(rot.c0.x, rot.c1.x, rot.c2.x, t.x),
                rotRow1: SIMD4(rot.c0.y, rot.c1.y, rot.c2.y, t.y),
                rotRow2: SIMD4(rot.c0.z, rot.c1.z, rot.c2.z, t.z),
                camCenter: SIMD4(c.x, c.y, c.z, 0),
                intrinsics: SIMD4(camera.fx, camera.fy, camera.cx, camera.cy),
                limits: SIMD4(settings.nearPlane, settings.farPlane, settings.eps2D, 0),
                counts: SIMD4(UInt32(n), UInt32(width), UInt32(height), UInt32(tilesX)),
                shInfo: SIMD4(UInt32(tilesY), UInt32(evalDegree),
                              UInt32(GaussianCloud.coefficientCount(degree: shDegree)), 0))

            let encoder = try context.makeComputeEncoder(commandBufferA)
            encoder.setComputePipelineState(projectPipeline)
            encoder.setBuffer(positions, offset: 0, index: 0)
            encoder.setBuffer(logScales, offset: 0, index: 1)
            encoder.setBuffer(rotations, offset: 0, index: 2)
            encoder.setBuffer(opacityLogits, offset: 0, index: 3)
            encoder.setBuffer(shCoefficients, offset: 0, index: 4)
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
            gpuSeconds += commandBufferA.gpuEndTime - commandBufferA.gpuStartTime

            let touched = touchedBuffer.contents().bindMemory(to: UInt32.self, capacity: n)
            let scanned = offsetBuffer.contents().bindMemory(to: UInt32.self, capacity: n)
            total = Int(scanned[n - 1]) + Int(touched[n - 1])
            for i in 0..<n where touched[i] > 0 {
                visible += 1
            }
        }

        let keyBuffer = try context.ensureBuffer(&sortKeys, length: total * 8)
        let valueBuffer = try context.ensureBuffer(&sortValues, length: total * 4)

        // Command buffer B: duplicate, sort, ranges, rasterize.
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
            let bg = settings.background
            var rasterUniforms = RasterUniforms(
                background: SIMD4(bg.x, bg.y, bg.z, 0),
                info: SIMD4(UInt32(width), UInt32(height), UInt32(tilesX), UInt32(tilesY)))
            let encoder = try context.makeComputeEncoder(commandBufferB)
            encoder.setComputePipelineState(rasterPipeline)
            encoder.setBuffer(meanDepthBuffer, offset: 0, index: 0)
            encoder.setBuffer(conicBuffer, offset: 0, index: 1)
            encoder.setBuffer(colorBuffer, offset: 0, index: 2)
            encoder.setBuffer(valueBuffer, offset: 0, index: 3)
            encoder.setBuffer(rangeBuffer, offset: 0, index: 4)
            encoder.setBytes(&rasterUniforms, length: MemoryLayout<RasterUniforms>.size, index: 5)
            encoder.setTexture(texture, index: 0)
            encoder.dispatchThreadgroups(MTLSize(width: tilesX, height: tilesY, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: Self.tileSize, height: Self.tileSize, depth: 1))
            encoder.endEncoding()
        }

        commandBufferB.commit()
        commandBufferB.waitUntilCompleted()
        try Self.check(commandBufferB)
        gpuSeconds += commandBufferB.gpuEndTime - commandBufferB.gpuStartTime

        return FrameStats(gpuTimeMilliseconds: gpuSeconds * 1000.0,
                          visibleGaussians: visible,
                          tileIntersections: total)
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
}
