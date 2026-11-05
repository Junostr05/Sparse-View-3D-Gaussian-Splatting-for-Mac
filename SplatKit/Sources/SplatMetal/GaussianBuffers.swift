import Metal
import SplatCore

/// Capacity-based structure-of-arrays store of Gaussian parameters in shared Metal buffers.
///
/// The layout matches `GaussianCloud`: one buffer per parameter group, `floatsPerGaussian` floats per
/// Gaussian. Only the first `count` Gaussians are live; the rest of each allocation is spare capacity
/// and is zero-filled. The renderer re-reads `count` and the buffer references at every encode, so a
/// change here is visible to the next frame.
///
/// CPU access (`upload`, `snapshot`, `grow`, and changes to `count`) must not overlap GPU work that
/// uses these buffers: wait for the command buffers that touch them first.
public final class GaussianBuffers {
    public let shDegree: Int
    public private(set) var capacity: Int
    /// Live Gaussians. Precondition: `0 <= count <= capacity`.
    public var count: Int {
        didSet {
            precondition(count >= 0 && count <= capacity, "count \(count) outside 0...\(capacity)")
        }
    }
    public private(set) var positions: MTLBuffer
    public private(set) var logScales: MTLBuffer
    public private(set) var rotations: MTLBuffer
    public private(set) var opacityLogits: MTLBuffer
    public private(set) var sh: MTLBuffer

    private let context: MetalContext

    /// Zero-filled store with room for `capacity` Gaussians and no live Gaussians (`count == 0`).
    public init(context: MetalContext, capacity: Int, shDegree: Int) throws(MetalError) {
        precondition(capacity >= 0, "capacity must be non-negative")
        precondition((0...3).contains(shDegree), "shDegree must be 0...3")
        let buffers = try GaussianBuffers.allocate(context: context, capacity: capacity, shDegree: shDegree)
        self.context = context
        self.shDegree = shDegree
        self.capacity = capacity
        self.count = 0
        positions = buffers[0]
        logScales = buffers[1]
        rotations = buffers[2]
        opacityLogits = buffers[3]
        sh = buffers[4]
    }

    /// Store that holds `cloud`. `capacity` defaults to `cloud.count` and must not be smaller.
    public convenience init(context: MetalContext, cloud: GaussianCloud, capacity: Int? = nil) throws(MetalError) {
        let resolved = capacity ?? cloud.count
        precondition(resolved >= cloud.count, "capacity \(resolved) is below the cloud count \(cloud.count)")
        try self.init(context: context, capacity: resolved, shDegree: cloud.shDegree)
        upload(cloud)
    }

    /// Floats per Gaussian for each parameter group, in `groups` order: `[3, 3, 4, 1, 3K]`.
    public static func floatsPerGaussian(shDegree: Int) -> [Int] {
        let k = GaussianCloud.coefficientCount(degree: shDegree)
        return [3, 3, 4, 1, 3 * k]
    }

    /// The five parameter buffers in the order of `floatsPerGaussian`.
    public var groups: [MTLBuffer] {
        [positions, logScales, rotations, opacityLogits, sh]
    }

    /// Copies `cloud` into the first `cloud.count` slots and sets `count` to `cloud.count`.
    /// Precondition: same SH degree and `cloud.count <= capacity`.
    public func upload(_ cloud: GaussianCloud) {
        precondition(cloud.shDegree == shDegree, "SH degree \(cloud.shDegree) does not match \(shDegree)")
        precondition(cloud.count <= capacity, "cloud has \(cloud.count) Gaussians, capacity is \(capacity)")
        GaussianBuffers.write(cloud.positions, to: positions)
        GaussianBuffers.write(cloud.logScales, to: logScales)
        GaussianBuffers.write(cloud.rotations, to: rotations)
        GaussianBuffers.write(cloud.opacityLogits, to: opacityLogits)
        GaussianBuffers.write(cloud.sh, to: sh)
        count = cloud.count
    }

    /// Copies per-Gaussian arrays (for example Adam moments `m` and `v`, which are typed
    /// `GaussianMoments`, a typealias of `GaussianGradients`) and sets `count` from their lengths.
    /// Precondition: same SH degree, consistent lengths, and `count <= capacity`.
    public func upload(arrays: GaussianGradients) {
        let n = arrays.opacityLogits.count
        let k = GaussianCloud.coefficientCount(degree: shDegree)
        precondition(arrays.positions.count == 3 * n && arrays.logScales.count == 3 * n
                         && arrays.rotations.count == 4 * n && arrays.sh.count == n * k * 3,
                     "array lengths are inconsistent with \(n) Gaussians of SH degree \(shDegree)")
        precondition(n <= capacity, "arrays hold \(n) Gaussians, capacity is \(capacity)")
        GaussianBuffers.write(arrays.positions, to: positions)
        GaussianBuffers.write(arrays.logScales, to: logScales)
        GaussianBuffers.write(arrays.rotations, to: rotations)
        GaussianBuffers.write(arrays.opacityLogits, to: opacityLogits)
        GaussianBuffers.write(arrays.sh, to: sh)
        count = n
    }

    /// The first `count` Gaussians as a `GaussianCloud`. GPU work must have completed.
    public func snapshot() -> GaussianCloud {
        let arrays = snapshotArrays()
        do {
            return try GaussianCloud(shDegree: shDegree,
                                     positions: arrays.positions,
                                     logScales: arrays.logScales,
                                     rotations: arrays.rotations,
                                     opacityLogits: arrays.opacityLogits,
                                     sh: arrays.sh)
        } catch {
            preconditionFailure("live Gaussian buffers are inconsistent: \(error)")
        }
    }

    /// The first `count` Gaussians as per-group arrays. GPU work must have completed.
    public func snapshotArrays() -> GaussianGradients {
        let n = count
        let k = GaussianCloud.coefficientCount(degree: shDegree)
        return GaussianGradients(
            positions: GaussianBuffers.read(positions, count: 3 * n),
            logScales: GaussianBuffers.read(logScales, count: 3 * n),
            rotations: GaussianBuffers.read(rotations, count: 4 * n),
            opacityLogits: GaussianBuffers.read(opacityLogits, count: n),
            sh: GaussianBuffers.read(sh, count: n * k * 3))
    }

    /// Reallocates every group to `newCapacity` Gaussians and copies the first `count` Gaussians.
    /// The rest of the new allocation is zero-filled. A `newCapacity` not above the current capacity
    /// is a no-op. GPU work using these buffers must have completed, because the buffers are replaced.
    public func grow(toCapacity newCapacity: Int) throws(MetalError) {
        guard newCapacity > capacity else { return }
        let fresh = try GaussianBuffers.allocate(context: context, capacity: newCapacity, shDegree: shDegree)
        let floats = GaussianBuffers.floatsPerGaussian(shDegree: shDegree)
        let old = groups
        for index in 0..<old.count {
            let bytes = count * floats[index] * MemoryLayout<Float>.size
            if bytes > 0 {
                fresh[index].contents().copyMemory(from: old[index].contents(), byteCount: bytes)
            }
        }
        positions = fresh[0]
        logScales = fresh[1]
        rotations = fresh[2]
        opacityLogits = fresh[3]
        sh = fresh[4]
        capacity = newCapacity
    }

    // MARK: - Helpers

    private static func allocate(context: MetalContext, capacity: Int, shDegree: Int) throws(MetalError) -> [MTLBuffer] {
        var buffers: [MTLBuffer] = []
        buffers.reserveCapacity(5)
        for floats in floatsPerGaussian(shDegree: shDegree) {
            let buffer = try context.makeBuffer(length: capacity * floats * MemoryLayout<Float>.size)
            _ = buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: buffer.length)
            buffers.append(buffer)
        }
        return buffers
    }

    private static func write(_ values: [Float], to buffer: MTLBuffer) {
        guard !values.isEmpty else { return }
        values.withUnsafeBytes { raw in
            buffer.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
        }
    }

    private static func read(_ buffer: MTLBuffer, count: Int) -> [Float] {
        guard count > 0 else { return [] }
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}
