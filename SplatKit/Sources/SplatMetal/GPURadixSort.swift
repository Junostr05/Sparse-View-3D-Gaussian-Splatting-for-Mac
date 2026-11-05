import Metal

/// Stable LSD radix sort of `UInt64` keys with `UInt32` values, 8 bits per pass.
///
/// Per pass: (1) `radixHistogram` builds one 256-bin histogram per block of 2048 elements and
/// stores it digit-major (`hist[digit * numBlocks + block]`); (2) `GPUPrefixSum` scans that array
/// so every (digit, block) pair gets its global start offset; (3) `radixScatter` walks its block in
/// 8 chunks of 256 elements, ranks each element among equal digits in thread order (simdgroup
/// ballots + per-simdgroup digit counts), and writes to `offset + rank`. Because blocks are ordered
/// within each digit and ranks follow input order, the sort is stable.
public final class GPURadixSort {
    static let threads = 256
    static let blockSize = 2048

    private let context: MetalContext
    private let prefixSum: GPUPrefixSum
    private let histogramPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState
    private var scratchKeys: MTLBuffer?
    private var scratchValues: MTLBuffer?
    private var histogram: MTLBuffer?
    private var histogramOffsets: MTLBuffer?

    public init(context: MetalContext) throws(MetalError) {
        self.context = context
        prefixSum = try GPUPrefixSum(context: context)
        histogramPipeline = try context.makePipeline("radixHistogram", minThreads: Self.threads)
        scatterPipeline = try context.makePipeline("radixScatter", minThreads: Self.threads)
        guard scatterPipeline.threadExecutionWidth == 32 else {
            throw .shaderCompilation("radix sort requires a simdgroup width of 32")
        }
    }

    /// Sorts `keys`/`values` (first `count` elements) in place by key; the result ends up in the
    /// passed buffers. Both buffers must hold at least `count` elements.
    public func encodeSort(keys: MTLBuffer, values: MTLBuffer, count: Int, keyBits: Int,
                           commandBuffer: MTLCommandBuffer) throws(MetalError) {
        guard count > 0, keyBits > 0 else { return }
        let passes = (min(keyBits, 64) + 7) / 8
        let numBlocks = (count + Self.blockSize - 1) / Self.blockSize

        let sKeys = try context.ensureBuffer(&scratchKeys, length: count * 8)
        let sValues = try context.ensureBuffer(&scratchValues, length: count * 4)
        let hist = try context.ensureBuffer(&histogram, length: 256 * numBlocks * 4)
        let histOffsets = try context.ensureBuffer(&histogramOffsets, length: 256 * numBlocks * 4)

        var srcKeys = keys, srcValues = values
        var dstKeys = sKeys, dstValues = sValues

        for pass in 0..<passes {
            var params = SIMD4<UInt32>(UInt32(count), UInt32(pass * 8), UInt32(numBlocks), 0)

            let histEncoder = try context.makeComputeEncoder(commandBuffer)
            histEncoder.setComputePipelineState(histogramPipeline)
            histEncoder.setBuffer(srcKeys, offset: 0, index: 0)
            histEncoder.setBuffer(hist, offset: 0, index: 1)
            histEncoder.setBytes(&params, length: MemoryLayout<SIMD4<UInt32>>.size, index: 2)
            histEncoder.dispatchThreadgroups(MTLSize(width: numBlocks, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: Self.threads, height: 1, depth: 1))
            histEncoder.endEncoding()

            try prefixSum.encodeExclusiveScan(input: hist, output: histOffsets, count: 256 * numBlocks,
                                              commandBuffer: commandBuffer)

            let scatterEncoder = try context.makeComputeEncoder(commandBuffer)
            scatterEncoder.setComputePipelineState(scatterPipeline)
            scatterEncoder.setBuffer(srcKeys, offset: 0, index: 0)
            scatterEncoder.setBuffer(srcValues, offset: 0, index: 1)
            scatterEncoder.setBuffer(dstKeys, offset: 0, index: 2)
            scatterEncoder.setBuffer(dstValues, offset: 0, index: 3)
            scatterEncoder.setBuffer(histOffsets, offset: 0, index: 4)
            scatterEncoder.setBytes(&params, length: MemoryLayout<SIMD4<UInt32>>.size, index: 5)
            scatterEncoder.dispatchThreadgroups(MTLSize(width: numBlocks, height: 1, depth: 1),
                                                threadsPerThreadgroup: MTLSize(width: Self.threads, height: 1, depth: 1))
            scatterEncoder.endEncoding()

            swap(&srcKeys, &dstKeys)
            swap(&srcValues, &dstValues)
        }

        // After an odd number of passes the sorted data sits in the scratch buffers.
        if passes % 2 == 1 {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw .commandFailed("cannot create blit encoder")
            }
            blit.copy(from: srcKeys, sourceOffset: 0, to: keys, destinationOffset: 0, size: count * 8)
            blit.copy(from: srcValues, sourceOffset: 0, to: values, destinationOffset: 0, size: count * 4)
            blit.endEncoding()
        }
    }
}
