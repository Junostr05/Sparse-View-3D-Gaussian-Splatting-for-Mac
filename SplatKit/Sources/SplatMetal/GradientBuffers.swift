import Metal
import SplatCore

/// Device buffers that receive the parameter gradients of one backward pass. Layouts match
/// `GaussianGradients`: `sh` is [gaussian][coefficient][channel] with K = (shDegree + 1)².
public final class GradientBuffers {
    public let positions: MTLBuffer
    public let logScales: MTLBuffer
    public let rotations: MTLBuffer
    public let opacityLogits: MTLBuffer
    public let sh: MTLBuffer

    let count: Int
    let shDegree: Int

    public init(context: MetalContext, count: Int, shDegree: Int) throws(MetalError) {
        precondition(count >= 0, "count must be non-negative")
        precondition((0...3).contains(shDegree), "shDegree must be 0...3")
        let coefficients = GaussianCloud.coefficientCount(degree: shDegree)
        let floatSize = MemoryLayout<Float>.size
        positions = try context.makeBuffer(length: 3 * count * floatSize)
        logScales = try context.makeBuffer(length: 3 * count * floatSize)
        rotations = try context.makeBuffer(length: 4 * count * floatSize)
        opacityLogits = try context.makeBuffer(length: count * floatSize)
        sh = try context.makeBuffer(length: count * coefficients * 3 * floatSize)
        self.count = count
        self.shDegree = shDegree
    }

    /// Copies the gradients to the CPU. `count` and `shDegree` must match the allocation.
    public func toGaussianGradients(count: Int, shDegree: Int) -> GaussianGradients {
        let coefficients = GaussianCloud.coefficientCount(degree: shDegree)
        return GaussianGradients(
            positions: Self.read(positions, count: 3 * count),
            logScales: Self.read(logScales, count: 3 * count),
            rotations: Self.read(rotations, count: 4 * count),
            opacityLogits: Self.read(opacityLogits, count: count),
            sh: Self.read(sh, count: count * coefficients * 3))
    }

    private static func read(_ buffer: MTLBuffer, count: Int) -> [Float] {
        guard count > 0 else { return [] }
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}
