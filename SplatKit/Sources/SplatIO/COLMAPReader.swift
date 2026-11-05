import Foundation
import SplatCore

/// One camera from `cameras.bin`. `params` holds the model's parameters in COLMAP order.
public struct COLMAPCamera: Sendable, Equatable {
    public var id: UInt32
    public var modelID: Int32
    public var width: Int
    public var height: Int
    public var params: [Double]
}

/// One registered image from `images.bin`. The 2D observations are read past and not kept.
public struct COLMAPImage: Sendable, Equatable {
    public var id: UInt32
    /// Unit quaternion `(qw, qx, qy, qz)` of the world→camera rotation.
    public var rotation: SIMD4<Double>
    /// World→camera translation.
    public var translation: SIMD3<Double>
    public var cameraID: UInt32
    public var name: String
}

/// One sparse point from `points3D.bin`. Its track is read past and not kept.
public struct COLMAPPoint: Sendable, Equatable {
    public var position: SIMD3<Double>
    public var color: SIMD3<UInt8>
}

/// Reads COLMAP binary reconstructions (`cameras.bin`, `images.bin`, `points3D.bin`).
///
/// All values are little-endian. Every read is bounds-checked; truncated or corrupt input throws
/// `DatasetError.truncated(file:)` and never reads outside the buffer.
public enum COLMAPReader {
    /// Number of `params` values for a COLMAP camera model id, or `nil` for an unknown id.
    public static func parameterCount(modelID: Int32) -> Int? {
        guard let model = colmapModelInfo(modelID) else { return nil }
        return model.paramCount
    }

    public static func readCameras(_ data: Data) throws(DatasetError) -> [COLMAPCamera] {
        try parse(data, file: "cameras.bin") { (cursor) throws(DatasetError) -> [COLMAPCamera] in
            let count = try cursor.readCount(minimumRecordSize: 24)
            var cameras: [COLMAPCamera] = []
            cameras.reserveCapacity(count)
            for _ in 0..<count {
                let id = try cursor.readUInt32()
                let modelID = try cursor.readInt32()
                let width = try cursor.readSize()
                let height = try cursor.readSize()
                guard let paramCount = COLMAPReader.parameterCount(modelID: modelID) else {
                    throw .unknownCameraModel(modelID)
                }
                var params: [Double] = []
                params.reserveCapacity(paramCount)
                for _ in 0..<paramCount {
                    params.append(try cursor.readDouble())
                }
                cameras.append(COLMAPCamera(id: id, modelID: modelID,
                                            width: width, height: height, params: params))
            }
            return cameras
        }
    }

    /// Images sorted by name.
    public static func readImages(_ data: Data) throws(DatasetError) -> [COLMAPImage] {
        try parse(data, file: "images.bin") { (cursor) throws(DatasetError) -> [COLMAPImage] in
            // Minimum record: id, quaternion, translation, camera id, one terminator, 2D count.
            let count = try cursor.readCount(minimumRecordSize: 73)
            var images: [COLMAPImage] = []
            images.reserveCapacity(count)
            for _ in 0..<count {
                let id = try cursor.readUInt32()
                let rotation = SIMD4(try cursor.readDouble(),
                                     try cursor.readDouble(),
                                     try cursor.readDouble(),
                                     try cursor.readDouble())
                let translation = SIMD3(try cursor.readDouble(),
                                        try cursor.readDouble(),
                                        try cursor.readDouble())
                let cameraID = try cursor.readUInt32()
                let name = try cursor.readName()
                let points2D = try cursor.readCount(minimumRecordSize: 24)
                try cursor.skip(points2D * 24)
                images.append(COLMAPImage(id: id,
                                          rotation: rotation,
                                          translation: translation,
                                          cameraID: cameraID,
                                          name: name))
            }
            return images.sorted { $0.name < $1.name }
        }
    }

    public static func readPoints3D(_ data: Data) throws(DatasetError) -> [COLMAPPoint] {
        try parse(data, file: "points3D.bin") { (cursor) throws(DatasetError) -> [COLMAPPoint] in
            // Minimum record: id, position, colour, error, track length.
            let count = try cursor.readCount(minimumRecordSize: 51)
            var points: [COLMAPPoint] = []
            points.reserveCapacity(count)
            for _ in 0..<count {
                _ = try cursor.readUInt64()
                let position = SIMD3(try cursor.readDouble(),
                                     try cursor.readDouble(),
                                     try cursor.readDouble())
                let color = SIMD3(try cursor.readUInt8(),
                                  try cursor.readUInt8(),
                                  try cursor.readUInt8())
                _ = try cursor.readDouble()
                let trackLength = try cursor.readCount(minimumRecordSize: 8)
                try cursor.skip(trackLength * 8)
                points.append(COLMAPPoint(position: position, color: color))
            }
            return points
        }
    }

    // MARK: - Parsing

    /// Runs `body` over the bytes of `data`, converting its typed errors through `Result`
    /// (`withUnsafeBytes` is untyped-throwing).
    private static func parse<T>(
        _ data: Data,
        file: String,
        _ body: (inout ByteCursor) throws(DatasetError) -> T
    ) throws(DatasetError) -> T {
        let result: Result<T, DatasetError> = data.withUnsafeBytes { raw in
            var cursor = ByteCursor(raw, file: file)
            do throws(DatasetError) {
                return .success(try body(&cursor))
            } catch {
                return .failure(error)
            }
        }
        return try result.get()
    }
}

extension CameraModel {
    /// Maps a COLMAP camera. Ids 0–4 are supported; a known but unsupported model throws
    /// `.unsupportedCameraModel(name)`, and an unknown id throws `.unknownCameraModel(id)`.
    public init(colmap camera: COLMAPCamera) throws(DatasetError) {
        let modelID = camera.modelID
        guard let model = colmapModelInfo(modelID) else {
            throw .unknownCameraModel(modelID)
        }
        let p = camera.params
        // Hand-built cameras may carry the wrong number of parameters; treat that as malformed.
        guard p.count == model.paramCount else {
            throw .truncated(file: "cameras.bin")
        }
        switch modelID {
        case 0:
            self = .simplePinhole(f: p[0], cx: p[1], cy: p[2])
        case 1:
            self = .pinhole(fx: p[0], fy: p[1], cx: p[2], cy: p[3])
        case 2:
            self = .simpleRadial(f: p[0], cx: p[1], cy: p[2], k: p[3])
        case 3:
            self = .radial(f: p[0], cx: p[1], cy: p[2], k1: p[3], k2: p[4])
        case 4:
            self = .opencv(fx: p[0], fy: p[1], cx: p[2], cy: p[3],
                           k1: p[4], k2: p[5], p1: p[6], p2: p[7])
        default:
            throw .unsupportedCameraModel(model.name)
        }
    }
}

// MARK: - Model table

/// COLMAP camera model names and parameter counts, indexed by model id.
private let colmapModels: [(name: String, paramCount: Int)] = [
    ("SIMPLE_PINHOLE", 3),
    ("PINHOLE", 4),
    ("SIMPLE_RADIAL", 4),
    ("RADIAL", 5),
    ("OPENCV", 8),
    ("OPENCV_FISHEYE", 8),
    ("FULL_OPENCV", 12),
    ("FOV", 5),
    ("SIMPLE_RADIAL_FISHEYE", 4),
    ("RADIAL_FISHEYE", 5),
    ("THIN_PRISM_FISHEYE", 12),
    ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
]

private func colmapModelInfo(_ id: Int32) -> (name: String, paramCount: Int)? {
    guard id >= 0, Int(id) < colmapModels.count else { return nil }
    return colmapModels[Int(id)]
}

// MARK: - Bounds-checked reader

/// Sequential little-endian reader over a raw buffer. Every access is checked against the
/// remaining bytes and throws `.truncated(file:)` instead of reading past the end.
private struct ByteCursor {
    let raw: UnsafeRawBufferPointer
    let file: String
    private(set) var offset = 0

    init(_ raw: UnsafeRawBufferPointer, file: String) {
        self.raw = raw
        self.file = file
    }

    var remaining: Int { raw.count - offset }

    /// Reserves `count` bytes and returns the offset of the first one.
    private mutating func reserve(_ count: Int) throws(DatasetError) -> Int {
        guard count >= 0, count <= remaining else {
            throw .truncated(file: file)
        }
        let start = offset
        offset += count
        return start
    }

    mutating func skip(_ count: Int) throws(DatasetError) {
        _ = try reserve(count)
    }

    /// Reads a leading `u64` record count, rejecting it before any allocation when the remaining
    /// bytes cannot hold that many records of `minimumRecordSize` bytes.
    mutating func readCount(minimumRecordSize: Int) throws(DatasetError) -> Int {
        let value = try readUInt64()
        guard value <= UInt64(remaining / minimumRecordSize) else {
            throw .truncated(file: file)
        }
        return Int(value)
    }

    /// Reads a `u64` dimension that must fit in `Int`.
    mutating func readSize() throws(DatasetError) -> Int {
        let value = try readUInt64()
        guard let size = Int(exactly: value) else {
            throw .truncated(file: file)
        }
        return size
    }

    /// Reads a null-terminated UTF-8 string (invalid sequences are replaced).
    mutating func readName() throws(DatasetError) -> String {
        var end = offset
        while end < raw.count, raw[end] != 0 {
            end += 1
        }
        guard end < raw.count else {
            throw .truncated(file: file)
        }
        let name = String(decoding: raw[offset..<end], as: UTF8.self)
        offset = end + 1
        return name
    }

    mutating func readUInt8() throws(DatasetError) -> UInt8 {
        let start = try reserve(1)
        return raw.loadUnaligned(fromByteOffset: start, as: UInt8.self)
    }

    mutating func readUInt32() throws(DatasetError) -> UInt32 {
        let start = try reserve(4)
        return UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: start, as: UInt32.self))
    }

    mutating func readInt32() throws(DatasetError) -> Int32 {
        Int32(bitPattern: try readUInt32())
    }

    mutating func readUInt64() throws(DatasetError) -> UInt64 {
        let start = try reserve(8)
        return UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: start, as: UInt64.self))
    }

    mutating func readDouble() throws(DatasetError) -> Double {
        Double(bitPattern: try readUInt64())
    }
}
