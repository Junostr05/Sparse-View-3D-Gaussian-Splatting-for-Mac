import Foundation
import Metal

public enum MetalError: Error {
    case noDevice
    case shaderCompilation(String)
    case bufferAllocation
    /// A command buffer or encoder could not be created, or the GPU reported an error.
    case commandFailed(String)
}

/// Owns the Metal device, command queue and the runtime-compiled shader library.
public final class MetalContext {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let library: MTLLibrary

    static let shaderFiles = ["Common", "PrefixSum", "RadixSort", "Rasterizer", "Backward", "Adam", "Loss", "Densify", "Depth", "Regularizers", "Appearance", "MCMC", "SamplingRate"]

    init(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) {
        self.device = device
        self.queue = queue
        self.library = library
    }

    public static func makeDefault() throws(MetalError) -> MetalContext {
        guard let device = MTLCreateSystemDefaultDevice() else { throw .noDevice }
        guard let queue = device.makeCommandQueue() else { throw .noDevice }

        var source = ""
        for name in shaderFiles {
            guard let url = Bundle.module.url(forResource: name, withExtension: "metal", subdirectory: "Shaders") else {
                throw .shaderCompilation("missing shader resource \(name).metal")
            }
            do {
                source += try String(contentsOf: url, encoding: .utf8)
                source += "\n"
            } catch {
                throw .shaderCompilation("cannot read \(name).metal: \(error.localizedDescription)")
            }
        }

        let options = MTLCompileOptions()
        options.mathMode = .safe
        options.languageVersion = .version3_0   // float atomics (atomic_fetch_add on atomic<float>) need MSL 3.0
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: options)
        } catch {
            throw .shaderCompilation(error.localizedDescription)
        }
        return MetalContext(device: device, queue: queue, library: library)
    }

    /// Creates a compute pipeline for the kernel `name`. Requires the pipeline to support
    /// at least `minThreads` threads per threadgroup.
    func makePipeline(_ name: String, minThreads: Int = 1) throws(MetalError) -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw .shaderCompilation("missing kernel \(name)")
        }
        let pipeline: MTLComputePipelineState
        do {
            pipeline = try device.makeComputePipelineState(function: function)
        } catch {
            throw .shaderCompilation("pipeline \(name): \(error.localizedDescription)")
        }
        guard pipeline.maxTotalThreadsPerThreadgroup >= minThreads else {
            throw .shaderCompilation("pipeline \(name) supports only \(pipeline.maxTotalThreadsPerThreadgroup) threads per threadgroup")
        }
        return pipeline
    }

    /// Shared-storage buffer, never smaller than 16 bytes.
    public func makeBuffer(length: Int) throws(MetalError) -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: max(16, length), options: .storageModeShared) else {
            throw .bufferAllocation
        }
        return buffer
    }

    /// Grow-only: replaces `buffer` only if it is smaller than `length` bytes (or nil).
    func ensureBuffer(_ buffer: inout MTLBuffer?, length: Int) throws(MetalError) -> MTLBuffer {
        if let existing = buffer, existing.length >= max(16, length) {
            return existing
        }
        let created = try makeBuffer(length: length)
        buffer = created
        return created
    }

    func makeComputeEncoder(_ commandBuffer: MTLCommandBuffer) throws(MetalError) -> MTLComputeCommandEncoder {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandFailed("cannot create compute encoder")
        }
        return encoder
    }

    /// One thread per element (`count` threads in a 1D grid).
    static func dispatch1D(_ encoder: MTLComputeCommandEncoder, pipeline: MTLComputePipelineState, count: Int) {
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }
}
