import Metal
import Testing
@testable import SplatMetal

@Suite(.enabled(if: TestEnvironment.hasMetal))
struct SortTests {
    private func makeFilledBuffer<T>(_ context: MetalContext, _ array: [T]) throws -> MTLBuffer {
        let buffer = try context.makeBuffer(length: array.count * MemoryLayout<T>.stride)
        if !array.isEmpty {
            array.withUnsafeBytes { raw in
                buffer.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
            }
        }
        return buffer
    }

    private func readBack<T>(_ buffer: MTLBuffer, as type: T.Type, count: Int) -> [T] {
        let pointer = buffer.contents().bindMemory(to: T.self, capacity: max(count, 1))
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    @Test(arguments: [0, 1, 1000, 1_000_003])
    func exclusiveScanMatchesCPU(count: Int) throws {
        let context = try makeTestContext()
        let scan = try GPUPrefixSum(context: context)

        var rng = SplitMix64(seed: UInt64(count) &+ 7)
        let input: [UInt32] = (0..<count).map { _ in UInt32.random(in: 0..<8, using: &rng) }
        var expected = [UInt32](repeating: 0, count: count)
        var running: UInt32 = 0
        for i in 0..<count {
            expected[i] = running
            running &+= input[i]
        }

        let inBuffer = try makeFilledBuffer(context, input)
        let outBuffer = try context.makeBuffer(length: count * 4)
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try scan.encodeExclusiveScan(input: inBuffer, output: outBuffer, count: count, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)

        let result = readBack(outBuffer, as: UInt32.self, count: count)
        #expect(result == expected)
    }

    @Test(arguments: [0, 1, 1000, 1_000_003])
    func radixSortMatchesStableCPUSort(count: Int) throws {
        let context = try makeTestContext()
        let sorter = try GPURadixSort(context: context)

        var rng = SplitMix64(seed: UInt64(count) &+ 99)
        // 48-bit keys drawn from a small pool so that many keys are equal.
        let mask: UInt64 = (1 << 48) - 1
        let pool: [UInt64] = (0..<64).map { _ in rng.next() & mask }
        let keys: [UInt64] = (0..<count).map { _ in
            let r = rng.next()
            return r % 4 == 0 ? (r >> 8) & mask : pool[Int((r >> 8) % 64)]
        }
        let values: [UInt32] = (0..<count).map { UInt32($0) }

        let expected = zip(keys, values).sorted { ($0.0, $0.1) < ($1.0, $1.1) }

        let keyBuffer = try makeFilledBuffer(context, keys)
        let valueBuffer = try makeFilledBuffer(context, values)
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try sorter.encodeSort(keys: keyBuffer, values: valueBuffer, count: count, keyBits: 48,
                              commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)

        let sortedKeys = readBack(keyBuffer, as: UInt64.self, count: count)
        let sortedValues = readBack(valueBuffer, as: UInt32.self, count: count)
        #expect(sortedKeys == expected.map { $0.0 })
        #expect(sortedValues == expected.map { $0.1 })
    }

    @Test func shaderTypeLayoutsMatchMSL() {
        #expect(MemoryLayout<ProjectUniforms>.size == 144)
        #expect(MemoryLayout<ProjectUniforms>.size % 16 == 0)
        #expect(MemoryLayout<DuplicateUniforms>.size == 16)
        #expect(MemoryLayout<RasterUniforms>.size == 32)
        #expect(MemoryLayout<RasterUniforms>.size % 16 == 0)
    }
}
