import Foundation
import Metal

/// Opt-in GPU stage profiler (v1.5 performance work). Off unless `GPUStageProfiler.shared` is set: then every
/// compute encoder made through `MetalContext.makeComputeEncoder` samples GPU timestamps at its start and end
/// (`MTLCounterSamplingPoint.atStageBoundary`), labelled by its call site, and every command buffer that has a
/// compute encoder reports its GPU time on completion. Measurement only: it never changes what is encoded.
public final class GPUStageProfiler: @unchecked Sendable {
    nonisolated(unsafe) public static var shared: GPUStageProfiler?

    public struct Entry: Sendable {
        public var seconds: Double = 0
        public var count: Int = 0
    }

    private let sampleBuffer: MTLCounterSampleBuffer
    private let sampleCount: Int
    private let lock = NSLock()
    private var nextSample = 0
    private var stages: [String: Entry] = [:]
    private var commandBufferSeconds: Double = 0
    private var commandBufferCount = 0
    private var registered: Set<ObjectIdentifier> = []
    private var pendingRanges: [ObjectIdentifier: [(label: String, index: Int)]] = [:]
    /// GPU timestamp ticks → seconds.
    private let secondsPerTick: Double

    public init?(device: MTLDevice, sampleCount: Int = 4_096) {
        guard device.supportsCounterSampling(.atStageBoundary),
              let set = device.counterSets?.first(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue }) else {
            return nil
        }
        let descriptor = MTLCounterSampleBufferDescriptor()
        descriptor.counterSet = set
        descriptor.sampleCount = sampleCount
        descriptor.storageMode = .shared
        let made: MTLCounterSampleBuffer
        do { made = try device.makeCounterSampleBuffer(descriptor: descriptor) } catch { print("GPUStageProfiler: \(error)"); return nil }
        let buffer = made
        sampleBuffer = buffer
        self.sampleCount = sampleCount
        // Calibrate GPU ticks against the CPU clock (nanoseconds) over a short interval.
        let (cpu0, gpu0) = device.sampleTimestamps()
        Thread.sleep(forTimeInterval: 0.05)
        let (cpu1, gpu1) = device.sampleTimestamps()
        let gpuTicks = Double(gpu1 &- gpu0)
        secondsPerTick = gpuTicks > 0 ? Double(cpu1 &- cpu0) * 1e-9 / gpuTicks : 1e-9
    }

    /// Clears the collected totals.
    public func reset() {
        lock.lock()
        stages.removeAll()
        commandBufferSeconds = 0
        commandBufferCount = 0
        lock.unlock()
    }

    /// Per call-site totals, and the summed GPU time and count of the profiled command buffers.
    public func results() -> (stages: [String: Entry], commandBufferSeconds: Double, commandBuffers: Int) {
        lock.lock()
        defer { lock.unlock() }
        return (stages, commandBufferSeconds, commandBufferCount)
    }

    func makeEncoder(_ commandBuffer: MTLCommandBuffer, label: String) -> MTLComputeCommandEncoder? {
        lock.lock()
        let index = nextSample
        nextSample = (nextSample + 2) % sampleCount
        let key = ObjectIdentifier(commandBuffer)
        pendingRanges[key, default: []].append((label, index))
        let needsHandler = registered.insert(key).inserted
        lock.unlock()
        if needsHandler {
            commandBuffer.addCompletedHandler { [weak self] buffer in
                self?.complete(buffer, key: key)
            }
        }
        let descriptor = MTLComputePassDescriptor()
        let attachment = descriptor.sampleBufferAttachments[0]!
        attachment.sampleBuffer = sampleBuffer
        attachment.startOfEncoderSampleIndex = index
        attachment.endOfEncoderSampleIndex = index + 1
        return commandBuffer.makeComputeCommandEncoder(descriptor: descriptor)
    }

    private func complete(_ buffer: MTLCommandBuffer, key: ObjectIdentifier) {
        lock.lock()
        let ranges = pendingRanges.removeValue(forKey: key) ?? []
        registered.remove(key)
        lock.unlock()
        var measured: [(String, Double)] = []
        for range in ranges {
            guard let data = try? sampleBuffer.resolveCounterRange(range.index..<(range.index + 2)) else { continue }
            let pair: [UInt64] = data.withUnsafeBytes { raw in
                Array(raw.bindMemory(to: UInt64.self))
            }
            guard pair.count >= 2, pair[1] >= pair[0], pair[0] != 0,
                  pair[1] != MTLCounterErrorValue, pair[0] != MTLCounterErrorValue else { continue }
            measured.append((range.label, Double(pair[1] - pair[0]) * secondsPerTick))
        }
        let total = buffer.gpuEndTime - buffer.gpuStartTime
        lock.lock()
        for (label, seconds) in measured {
            stages[label, default: Entry()].seconds += seconds
            stages[label, default: Entry()].count += 1
        }
        commandBufferSeconds += total
        commandBufferCount += 1
        lock.unlock()
    }
}
