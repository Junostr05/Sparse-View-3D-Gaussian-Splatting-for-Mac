import Metal

/// Multi-level exclusive scan of `UInt32` on the GPU.
///
/// Level 0 scans blocks of 1024 elements (256 threads x 4 elements, `simd_prefix_exclusive_sum`
/// plus simdgroup totals in threadgroup memory) and emits one sum per block. The block sums are
/// scanned recursively and added back to every element of their block.
public final class GPUPrefixSum {
    static let blockSize = 1024
    static let threadsPerBlock = 256

    private let context: MetalContext
    private let scanPipeline: MTLComputePipelineState
    private let addPipeline: MTLComputePipelineState
    /// Per level: buffer receiving that level's block sums, and buffer receiving their scan.
    private var sumsBuffers: [MTLBuffer?] = []
    private var scannedSumsBuffers: [MTLBuffer?] = []

    public init(context: MetalContext) throws(MetalError) {
        self.context = context
        scanPipeline = try context.makePipeline("scanBlocks", minThreads: Self.threadsPerBlock)
        addPipeline = try context.makePipeline("addBlockOffsets")
    }

    /// Encodes an exclusive scan of `count` UInt32 values from `input` into `output`.
    /// `input` and `output` must be different buffers.
    public func encodeExclusiveScan(input: MTLBuffer, output: MTLBuffer, count: Int,
                                    commandBuffer: MTLCommandBuffer) throws(MetalError) {
        guard count > 0 else { return }
        try encodeLevel(input: input, output: output, count: count, level: 0, commandBuffer: commandBuffer)
    }

    private func encodeLevel(input: MTLBuffer, output: MTLBuffer, count: Int, level: Int,
                             commandBuffer: MTLCommandBuffer) throws(MetalError) {
        let numBlocks = (count + Self.blockSize - 1) / Self.blockSize
        while sumsBuffers.count <= level {
            sumsBuffers.append(nil)
            scannedSumsBuffers.append(nil)
        }
        let sums = try context.ensureBuffer(&sumsBuffers[level], length: numBlocks * 4)

        var countValue = UInt32(count)
        let encoder = try context.makeComputeEncoder(commandBuffer)
        encoder.setComputePipelineState(scanPipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBuffer(sums, offset: 0, index: 2)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: numBlocks, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Self.threadsPerBlock, height: 1, depth: 1))
        encoder.endEncoding()

        guard numBlocks > 1 else { return }

        let scannedSums = try context.ensureBuffer(&scannedSumsBuffers[level], length: numBlocks * 4)
        try encodeLevel(input: sums, output: scannedSums, count: numBlocks, level: level + 1,
                        commandBuffer: commandBuffer)

        let addEncoder = try context.makeComputeEncoder(commandBuffer)
        addEncoder.setComputePipelineState(addPipeline)
        addEncoder.setBuffer(output, offset: 0, index: 0)
        addEncoder.setBuffer(scannedSums, offset: 0, index: 1)
        addEncoder.setBytes(&countValue, length: MemoryLayout<UInt32>.size, index: 2)
        MetalContext.dispatch1D(addEncoder, pipeline: addPipeline, count: count)
        addEncoder.endEncoding()
    }
}
