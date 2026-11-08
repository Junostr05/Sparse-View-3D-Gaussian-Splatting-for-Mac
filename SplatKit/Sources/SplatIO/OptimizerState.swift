import Foundation
import SplatCore

/// Errors of `OptimizerState` decoding. Each case names the first problem found; nothing is partially decoded.
public enum OptimizerStateError: Error, Equatable, Sendable {
    /// The data does not start with the optimizer-state magic.
    case badMagic
    /// The format version is not one this build reads.
    case unsupportedVersion(UInt32)
    /// The data ends before the field at byte `offset` could be read (a truncated file).
    case truncated(offset: Int)
    /// The trailing CRC-32 does not match the payload (a corrupted file).
    case checksumMismatch
    /// A header value or array size contradicts the rest of the file.
    case inconsistent(String)
}

/// The optimizer state of one training run, stored as `checkpoints/optimizer.bin` in a project and as the
/// autosave sidecar of an interrupted run. It holds the Adam first and second moments of every parameter group,
/// in the order of the Gaussians of the matching PLY checkpoint, plus the counters needed to continue exactly:
/// the iteration, the Adam step, the configuration hash, and the appearance and pose-refinement state when those
/// are on.
///
/// Binary layout, little-endian (every supported Mac), in this order:
/// - magic `GSOPTIM1` (8 bytes), format version (UInt32, 1)
/// - Gaussian count N (UInt64), SH degree (UInt32), group count (UInt32, 5) and floats per Gaussian of each group
/// - iteration (UInt64), Adam step (UInt64), configuration hash (UInt64), flags (UInt32: bit 0 appearance, bit 1 pose)
/// - first moments, then second moments: positions (3N), log scales (3N), rotations (4N), opacity logits (N),
///   SH (N·K·3 with K = (degree + 1)²) as Float32
/// - if appearance: view count V (UInt32), parameters (6V Float32), first moments (6V), second moments (6V),
///   step counts (V UInt64)
/// - if pose: view count V (UInt32), poses (12V Float32), first moments (6V Float64), second moments (6V Float64),
///   steps (V UInt64)
/// - CRC-32 (IEEE, UInt32) of every byte before it
///
/// Pure value type: no Metal. The GPU trainer builds and reads it (`GPUTrainer.exportOptimizerState`,
/// `GPUTrainer.restore`).
public struct OptimizerState: Sendable, Equatable {
    public static let formatVersion: UInt32 = 1
    static let magic: [UInt8] = Array("GSOPTIM1".utf8)
    static let flagAppearance: UInt32 = 1 << 0
    static let flagPose: UInt32 = 1 << 1
    static let groupCount = 5

    /// Gaussians the moments describe. Must equal the Gaussian count of the checkpoint they belong to.
    public var gaussianCount: Int
    /// SH degree of the moments (0...3).
    public var shDegree: Int
    /// Iterations completed by the trainer that exported the state.
    public var iteration: Int
    /// Adam steps taken over the whole lineage, which sets the bias correction of the next step.
    public var adamStep: Int
    /// `OptimizerState.configurationHash` of the configuration the run used.
    public var configurationHash: UInt64
    /// First moments, one array per parameter group (`GaussianMoments`, the layout of `GaussianCloud`).
    public var firstMoment: GaussianMoments
    /// Second moments, same layout.
    public var secondMoment: GaussianMoments
    /// Colour compensation state, nil when appearance compensation is off.
    public var appearance: AppearanceOptimizerBlock?
    /// Pose refinement state, nil when pose refinement is off.
    public var pose: PoseOptimizerBlock?

    public init(gaussianCount: Int, shDegree: Int, iteration: Int, adamStep: Int, configurationHash: UInt64,
                firstMoment: GaussianMoments, secondMoment: GaussianMoments,
                appearance: AppearanceOptimizerBlock? = nil, pose: PoseOptimizerBlock? = nil) {
        precondition((0...3).contains(shDegree), "shDegree must be 0...3")
        precondition(gaussianCount >= 0 && iteration >= 0 && adamStep >= 0, "counters are non-negative")
        precondition(Self.moments(firstMoment, fit: gaussianCount, shDegree: shDegree)
                     && Self.moments(secondMoment, fit: gaussianCount, shDegree: shDegree),
                     "moment arrays do not match \(gaussianCount) Gaussians of SH degree \(shDegree)")
        self.gaussianCount = gaussianCount
        self.shDegree = shDegree
        self.iteration = iteration
        self.adamStep = adamStep
        self.configurationHash = configurationHash
        self.firstMoment = firstMoment
        self.secondMoment = secondMoment
        self.appearance = appearance
        self.pose = pose
    }

    /// Floats per Gaussian of each parameter group, in `GaussianCloud` order: `[3, 3, 4, 1, 3K]`.
    public static func groupFloats(shDegree: Int) -> [Int] {
        [3, 3, 4, 1, 3 * GaussianCloud.coefficientCount(degree: shDegree)]
    }

    /// Hash of a training configuration, used to recognise the run a snapshot belongs to. FNV-1a over the JSON of
    /// the configuration with two fields normalised: `autosaveInterval` (which does not change the optimization)
    /// and the regularizer set, which is encoded in sorted order. Stable across launches, unlike `Hasher`.
    public static func configurationHash(_ configuration: TrainingConfiguration) -> UInt64 {
        var normalized = configuration
        normalized.autosaveInterval = 0
        normalized.regularizers = []
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var bytes: [UInt8] = Array((try? encoder.encode(normalized)) ?? Data())
        for kind in configuration.regularizers.map(\.rawValue).sorted() {
            bytes.append(contentsOf: Array(kind.utf8))
            bytes.append(0)
        }
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in bytes {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return hash
    }

    /// Why this state cannot seed a trainer with the given shape, or nil when it can. A state that does not fit is
    /// ignored by the caller (fresh Adam), so the reason is for the log only. `appearance` and `poseRefinement` are
    /// the switches of the trainer's configuration: a block must be present exactly when its feature is on.
    public func incompatibility(gaussianCount count: Int, shDegree degree: Int, viewCount: Int,
                                appearance appearanceOn: Bool, poseRefinement poseOn: Bool) -> String? {
        if gaussianCount != count {
            return "the optimizer state has \(gaussianCount) Gaussians, the model has \(count)"
        }
        if shDegree != degree {
            return "the optimizer state has SH degree \(shDegree), the model has \(degree)"
        }
        if (appearance != nil) != appearanceOn {
            return appearanceOn ? "the optimizer state has no appearance state" : "the optimizer state has appearance state but appearance is off"
        }
        if let appearance, appearance.viewCount != viewCount {
            return "the appearance state covers \(appearance.viewCount) views, the dataset has \(viewCount)"
        }
        if (pose != nil) != poseOn {
            return poseOn ? "the optimizer state has no pose refinement state" : "the optimizer state has pose state but pose refinement is off"
        }
        if let pose, pose.viewCount != viewCount {
            return "the pose state covers \(pose.viewCount) views, the dataset has \(viewCount)"
        }
        return nil
    }

    // MARK: - Encoding

    /// The binary form described in the type comment.
    public func encode() -> Data {
        var writer = ByteWriter()
        writer.append(Self.magic)
        writer.u32(Self.formatVersion)
        writer.u64(UInt64(gaussianCount))
        writer.u32(UInt32(shDegree))
        let layout = Self.groupFloats(shDegree: shDegree)
        writer.u32(UInt32(layout.count))
        for floats in layout {
            writer.u32(UInt32(floats))
        }
        writer.u64(UInt64(iteration))
        writer.u64(UInt64(adamStep))
        writer.u64(configurationHash)
        var flags: UInt32 = 0
        if appearance != nil {
            flags |= Self.flagAppearance
        }
        if pose != nil {
            flags |= Self.flagPose
        }
        writer.u32(flags)
        for moments in [firstMoment, secondMoment] {
            writer.floats(moments.positions)
            writer.floats(moments.logScales)
            writer.floats(moments.rotations)
            writer.floats(moments.opacityLogits)
            writer.floats(moments.sh)
        }
        if let appearance {
            writer.u32(UInt32(appearance.viewCount))
            writer.floats(appearance.parameters)
            writer.floats(appearance.firstMoment)
            writer.floats(appearance.secondMoment)
            for steps in appearance.stepCounts {
                writer.u64(UInt64(steps))
            }
        }
        if let pose {
            writer.u32(UInt32(pose.viewCount))
            writer.floats(pose.poses)
            writer.doubles(pose.firstMoment)
            writer.doubles(pose.secondMoment)
            for steps in pose.steps {
                writer.u64(UInt64(steps))
            }
        }
        writer.u32(writer.checksum())
        return writer.data
    }

    // MARK: - Decoding

    /// Decodes and validates a state. Throws on the first problem; a truncated, corrupted or inconsistent file never
    /// yields a partial state.
    public init(decoding data: Data) throws(OptimizerStateError) {
        let outcome: Result<OptimizerState, OptimizerStateError> = data.withUnsafeBytes {
            (raw: UnsafeRawBufferPointer) -> Result<OptimizerState, OptimizerStateError> in
            do throws(OptimizerStateError) {
                return .success(try Self.parse(raw))
            } catch {
                return .failure(error)
            }
        }
        switch outcome {
        case .success(let state):
            self = state
        case .failure(let error):
            throw error
        }
    }

    private static func parse(_ raw: UnsafeRawBufferPointer) throws(OptimizerStateError) -> OptimizerState {
        var reader = try ByteReader(raw: raw)
        guard try reader.bytes(magic.count).elementsEqual(magic) else {
            throw .badMagic
        }
        let version = try reader.u32()
        guard version == formatVersion else {
            throw .unsupportedVersion(version)
        }
        let countValue = try reader.u64()
        // Each Gaussian needs at least one opacity float, so a count above the payload size is truncation.
        guard countValue <= UInt64(reader.payloadBytes / 4) else {
            throw .truncated(offset: reader.offset)
        }
        let count = Int(countValue)
        let degree = Int(try reader.u32())
        guard (0...3).contains(degree) else {
            throw .inconsistent("SH degree \(degree)")
        }
        let groups = Int(try reader.u32())
        guard groups == groupCount else {
            throw .inconsistent("\(groups) parameter groups, expected \(groupCount)")
        }
        let expectedLayout = groupFloats(shDegree: degree)
        for expected in expectedLayout {
            let floats = Int(try reader.u32())
            guard floats == expected else {
                throw .inconsistent("parameter group layout does not match SH degree \(degree)")
            }
        }
        let iteration = try reader.u64()
        let adamStep = try reader.u64()
        let hash = try reader.u64()
        let flags = try reader.u32()
        guard flags & ~(flagAppearance | flagPose) == 0 else {
            throw .inconsistent("unknown flags \(flags)")
        }

        var moments: [GaussianMoments] = []
        for _ in 0..<2 {
            moments.append(GaussianMoments(
                positions: try reader.floats(count * expectedLayout[0]),
                logScales: try reader.floats(count * expectedLayout[1]),
                rotations: try reader.floats(count * expectedLayout[2]),
                opacityLogits: try reader.floats(count * expectedLayout[3]),
                sh: try reader.floats(count * expectedLayout[4])))
        }

        var appearance: AppearanceOptimizerBlock?
        if flags & flagAppearance != 0 {
            let views = try reader.count()
            appearance = AppearanceOptimizerBlock(
                parameters: try reader.floats(views * 6),
                firstMoment: try reader.floats(views * 6),
                secondMoment: try reader.floats(views * 6),
                stepCounts: try reader.counts(views))
        }
        var pose: PoseOptimizerBlock?
        if flags & flagPose != 0 {
            let views = try reader.count()
            pose = PoseOptimizerBlock(
                poses: try reader.floats(views * 12),
                firstMoment: try reader.doubles(views * 6),
                secondMoment: try reader.doubles(views * 6),
                steps: try reader.counts(views))
        }

        guard reader.offset == reader.payloadBytes else {
            throw .inconsistent("\(reader.payloadBytes - reader.offset) bytes after the last field")
        }
        // The trailer lies after the payload, so it is read directly and not through the payload bounds.
        let stored = UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: reader.payloadBytes, as: UInt32.self))
        let payload = UnsafeRawBufferPointer(rebasing: raw[0..<reader.payloadBytes])
        guard stored == ByteReader.checksum(payload) else {
            throw .checksumMismatch
        }
        guard Int(iteration) <= Int.max / 2, Int(adamStep) <= Int.max / 2 else {
            throw .inconsistent("counters out of range")
        }
        return OptimizerState(gaussianCount: count, shDegree: degree, iteration: Int(iteration),
                              adamStep: Int(adamStep), configurationHash: hash,
                              firstMoment: moments[0], secondMoment: moments[1],
                              appearance: appearance, pose: pose)
    }

    /// True when every array of `moments` has the size `count` Gaussians of `shDegree` need.
    private static func moments(_ moments: GaussianMoments, fit count: Int, shDegree degree: Int) -> Bool {
        let layout = groupFloats(shDegree: degree)
        return moments.positions.count == count * layout[0]
            && moments.logScales.count == count * layout[1]
            && moments.rotations.count == count * layout[2]
            && moments.opacityLogits.count == count * layout[3]
            && moments.sh.count == count * layout[4]
    }
}

// MARK: - Byte helpers

/// Little-endian writer. Float arrays are copied as their bit patterns.
private struct ByteWriter {
    var data = Data()

    mutating func append(_ bytes: [UInt8]) {
        data.append(contentsOf: bytes)
    }

    mutating func u32(_ value: UInt32) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    mutating func u64(_ value: UInt64) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    mutating func floats(_ values: [Float]) {
        values.withUnsafeBytes { data.append(contentsOf: $0) }
    }

    mutating func doubles(_ values: [Double]) {
        values.withUnsafeBytes { data.append(contentsOf: $0) }
    }

    /// CRC-32 of everything written so far.
    func checksum() -> UInt32 {
        data.withUnsafeBytes { ByteReader.checksum($0) }
    }
}

/// Bounds-checked little-endian reader over the payload of a state. The last four bytes are the trailer and are
/// never read as data.
private struct ByteReader {
    let raw: UnsafeRawBufferPointer
    /// Bytes before the CRC trailer.
    let payloadBytes: Int
    var offset = 0

    init(raw: UnsafeRawBufferPointer) throws(OptimizerStateError) {
        // The magic and the version must fit before the trailer.
        guard raw.count >= 8 + 4 + 4 else {
            throw .truncated(offset: raw.count)
        }
        self.raw = raw
        payloadBytes = raw.count - 4
    }

    mutating func take(_ count: Int) throws(OptimizerStateError) -> UnsafeRawBufferPointer {
        guard count >= 0, count <= payloadBytes - offset else {
            throw .truncated(offset: offset)
        }
        let slice = UnsafeRawBufferPointer(rebasing: raw[offset..<(offset + count)])
        offset += count
        return slice
    }

    mutating func bytes(_ count: Int) throws(OptimizerStateError) -> [UInt8] {
        Array(try take(count))
    }

    mutating func u32() throws(OptimizerStateError) -> UInt32 {
        UInt32(littleEndian: try take(4).loadUnaligned(as: UInt32.self))
    }

    mutating func u64() throws(OptimizerStateError) -> UInt64 {
        UInt64(littleEndian: try take(8).loadUnaligned(as: UInt64.self))
    }

    /// A view or Gaussian count: UInt32, bounded by the payload so the arrays it sizes cannot be huge.
    mutating func count() throws(OptimizerStateError) -> Int {
        let value = Int(try u32())
        guard value <= payloadBytes / 4 else {
            throw .truncated(offset: offset)
        }
        return value
    }

    /// `count` UInt64 values, each converted to Int and checked to be non-negative.
    mutating func counts(_ count: Int) throws(OptimizerStateError) -> [Int] {
        var values: [Int] = []
        values.reserveCapacity(count)
        for _ in 0..<count {
            let value = try u64()
            guard value <= UInt64(Int.max / 2) else {
                throw .inconsistent("step counter out of range")
            }
            values.append(Int(value))
        }
        return values
    }

    mutating func floats(_ count: Int) throws(OptimizerStateError) -> [Float] {
        let source = try take(count * 4)
        guard count > 0, let base = source.baseAddress else {
            return []
        }
        return [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
            UnsafeMutableRawPointer(buffer.baseAddress!).copyMemory(from: base, byteCount: count * 4)
            initialized = count
        }
    }

    mutating func doubles(_ count: Int) throws(OptimizerStateError) -> [Double] {
        let source = try take(count * 8)
        guard count > 0, let base = source.baseAddress else {
            return []
        }
        return [Double](unsafeUninitializedCapacity: count) { buffer, initialized in
            UnsafeMutableRawPointer(buffer.baseAddress!).copyMemory(from: base, byteCount: count * 8)
            initialized = count
        }
    }

    /// CRC-32 (IEEE 802.3, reflected, table driven).
    static func checksum(_ bytes: UnsafeRawBufferPointer) -> UInt32 {
        var crc: UInt32 = 0xFFFF_FFFF
        for byte in bytes {
            crc = crcTable[Int((crc ^ UInt32(byte)) & 0xFF)] ^ (crc >> 8)
        }
        return crc ^ 0xFFFF_FFFF
    }

    private static let crcTable: [UInt32] = (0..<256).map { (index: Int) -> UInt32 in
        var value = UInt32(index)
        for _ in 0..<8 {
            value = (value & 1) != 0 ? (0xEDB8_8320 ^ (value >> 1)) : (value >> 1)
        }
        return value
    }
}
