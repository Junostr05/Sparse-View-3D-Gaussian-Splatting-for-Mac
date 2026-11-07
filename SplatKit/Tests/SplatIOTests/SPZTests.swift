import Foundation
import Testing
import SplatCore
@testable import SplatIO

/// Unit quaternion (w, x, y, z) with w ≥ 0.3.
private func unitQuaternion(_ rng: inout SeededGenerator) -> [Float] {
    while true {
        let v: [Float] = (0..<4).map { _ in Float.random(in: -1...1, using: &rng) }
        let squared: Float = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
        let norm: Float = squared.squareRoot()
        guard norm > 0.1 else { continue }
        var q: [Float] = v.map { $0 / norm }
        if abs(q[0]) < 0.3 { continue }
        if q[0] < 0 {
            q = q.map { -$0 }
        }
        return q
    }
}

/// Deterministic cloud inside every SPZ quantisation range: positions ±40, log-scales −6…0,
/// logits ±4, DC ±1.7 (colour scale 0.15 covers ±3.33), higher SH ±0.99.
private func randomCloud(degree: Int, count: Int, seed: UInt64) throws -> GaussianCloud {
    var rng = SeededGenerator(seed: seed)
    let k = GaussianCloud.coefficientCount(degree: degree)
    var positions: [Float] = []
    var logScales: [Float] = []
    var rotations: [Float] = []
    var opacity: [Float] = []
    var sh: [Float] = []
    for _ in 0..<count {
        for _ in 0..<3 {
            positions.append(Float.random(in: -40...40, using: &rng))
        }
        for _ in 0..<3 {
            logScales.append(Float.random(in: -6...0, using: &rng))
        }
        rotations.append(contentsOf: unitQuaternion(&rng))
        opacity.append(Float.random(in: -4...4, using: &rng))
        for coefficient in 0..<k {
            let bound: Float = coefficient == 0 ? 1.7 : 0.99
            for _ in 0..<3 {
                sh.append(Float.random(in: -bound...bound, using: &rng))
            }
        }
    }
    return try GaussianCloud(shDegree: degree,
                             positions: positions,
                             logScales: logScales,
                             rotations: rotations,
                             opacityLogits: opacity,
                             sh: sh)
}

/// Bound on the logit error after one-byte alpha quantisation (see `SplatWriterTests`).
private func alphaLogitTolerance(_ logit: Double) -> Double {
    let p = 1 / (1 + exp(-logit))
    let h = 0.5 / 255 + 1e-9
    let low = min(max(p - h, 1e-6), 1 - 1e-6)
    let high = min(max(p + h, 1e-6), 1 - 1e-6)
    let slowest = min(low * (1 - low), high * (1 - high))
    return h / slowest + 1e-4
}

/// Checks a decoded SPZ cloud against the original. The order is preserved, so records are compared
/// index by index. Each bound is the quantisation step of its field.
private func expectQuantisedMatch(_ original: GaussianCloud, _ read: GaussianCloud) {
    #expect(read.count == original.count)
    #expect(read.shDegree == original.shDegree)
    guard read.count == original.count else { return }

    let k = GaussianCloud.coefficientCount(degree: original.shDegree)
    var failures: [String] = []
    for i in 0..<original.count {
        for c in 0..<3 {
            // Position: 24-bit fixed point, step 1/4096, so the rounding error is ≤ 1/8192.
            let positionError = abs(Double(read.positions[3 * i + c]) - Double(original.positions[3 * i + c]))
            if positionError > 0.5 / 4096 + 1e-6 {
                failures.append("position \(i).\(c): \(positionError)")
            }
            // Log-scale: one byte, step 1/16, so the rounding error is ≤ 1/32.
            let scaleError = abs(Double(read.logScales[3 * i + c]) - Double(original.logScales[3 * i + c]))
            if scaleError > 0.5 / 16 + 1e-6 {
                failures.append("scale \(i).\(c): \(scaleError)")
            }
            // DC colour: one byte in the 0.15 scale, so the error is ≤ 0.5/(255·0.15).
            let dcError = abs(Double(read.sh[(i * k) * 3 + c]) - Double(original.sh[(i * k) * 3 + c]))
            if dcError > 0.5 / 255 / 0.15 + 1e-5 {
                failures.append("dc \(i).\(c): \(dcError)")
            }
        }
        // Opacity: one byte of probability (error ≤ 0.5/255), converted to a logit bound.
        let alphaTolerance = alphaLogitTolerance(Double(original.opacityLogits[i]))
        let alphaError = abs(Double(read.opacityLogits[i]) - Double(original.opacityLogits[i]))
        if alphaError > alphaTolerance {
            failures.append("alpha \(i): \(alphaError)")
        }
        // Rotation: the stored sign is w ≥ 0. Compare against the canonical original. xyz error
        // ≤ 0.5/127.5 per component, and w = sqrt(1 − |xyz|²) with w ≥ 0.3 adds ≤ 0.023. Bound 0.03.
        var canonical: [Double] = (0..<4).map { Double(original.rotations[4 * i + $0]) }
        if canonical[0] < 0 {
            canonical = canonical.map { -$0 }
        }
        for q in 0..<4 {
            let error = abs(Double(read.rotations[4 * i + q]) - canonical[q])
            if error > 0.03 {
                failures.append("rotation \(i).\(q): \(error)")
            }
        }
        // SH rest: one byte, step 1/128, error ≤ 0.5/128.
        for coefficient in 1..<k {
            for c in 0..<3 {
                let index = (i * k + coefficient) * 3 + c
                let error = abs(Double(read.sh[index]) - Double(original.sh[index]))
                if error > 0.5 / 128 + 1e-6 {
                    failures.append("sh \(i).\(coefficient).\(c): \(error)")
                }
            }
        }
        if failures.count > 10 { break }
    }
    #expect(failures.isEmpty, "\(failures.prefix(10))")
}

/// Splits a file into its gzip payload, the uncompressed SPZ bytes.
private func payload(of file: Data) throws -> [UInt8] {
    try Gzip.decompress(file)
}

/// Builds a gzip file around `bytes` with stored blocks, for malformed-input tests.
private func gzipFile(_ bytes: [UInt8]) -> Data {
    Data(Gzip.compressStored(bytes))
}

private func header(magic: UInt32 = SPZWriter.magic,
                    version: UInt32 = 2,
                    numPoints: UInt32,
                    shDegree: UInt8 = 0,
                    fractionalBits: UInt8 = 12,
                    flags: UInt8 = 0) -> [UInt8] {
    var bytes: [UInt8] = []
    for value in [magic, version, numPoints] {
        bytes.append(UInt8(value & 0xff))
        bytes.append(UInt8((value >> 8) & 0xff))
        bytes.append(UInt8((value >> 16) & 0xff))
        bytes.append(UInt8(value >> 24))
    }
    bytes += [shDegree, fractionalBits, flags, 0]
    return bytes
}

/// Unit quaternion (w, x, y, z) with no sign or w restriction, for the frame-conversion tests.
private func anyUnitQuaternion(_ rng: inout SeededGenerator) -> [Float] {
    while true {
        let v: [Float] = (0..<4).map { _ in Float.random(in: -1...1, using: &rng) }
        let squared: Float = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
        let norm: Float = squared.squareRoot()
        if norm > 0.1 {
            return v.map { $0 / norm }
        }
    }
}

/// Uniformly distributed unit direction.
private func randomUnitDirection(_ rng: inout SeededGenerator) -> SIMD3<Float> {
    while true {
        let v = SIMD3<Float>(Float.random(in: -1...1, using: &rng),
                             Float.random(in: -1...1, using: &rng),
                             Float.random(in: -1...1, using: &rng))
        let squared: Float = (v * v).sum()
        let length: Float = squared.squareRoot()
        if length > 0.1 && length <= 1 {
            return v / length
        }
    }
}

/// The frame map F = diag(1, −1, −1) as a matrix, for the rotation consistency test.
private let frameMatrix = Matrix3(c0: SIMD3<Float>(1, 0, 0),
                                  c1: SIMD3<Float>(0, -1, 0),
                                  c2: SIMD3<Float>(0, 0, -1))

/// Largest absolute component of a vector.
private func maxAbsComponent(_ v: SIMD3<Float>) -> Float {
    let x: Float = abs(v.x)
    let y: Float = abs(v.y)
    let z: Float = abs(v.z)
    return max(x, max(y, z))
}

private func quaternion(_ values: [Float], _ i: Int) -> SIMD4<Float> {
    SIMD4<Float>(values[4 * i], values[4 * i + 1], values[4 * i + 2], values[4 * i + 3])
}

@Suite struct SPZTests {
    @Test(arguments: [0, 1, 2, 3])
    func roundTripsWithQuantisationBounds(degree: Int) throws {
        let cloud = try randomCloud(degree: degree, count: 200, seed: UInt64(40 + degree))
        let file = try SPZWriter.write(cloud)
        let read = try SPZReader.read(file)
        expectQuantisedMatch(cloud, read)
    }

    @Test func tenThousandRandomGaussians() throws {
        let cloud = try randomCloud(degree: 3, count: 10_000, seed: 99)
        let file = try SPZWriter.write(cloud)
        let read = try SPZReader.read(file)
        expectQuantisedMatch(cloud, read)
    }

    @Test func headerFieldsAreWrittenAsSpecified() throws {
        let cloud = try randomCloud(degree: 2, count: 7, seed: 3)
        let bytes = try payload(of: try SPZWriter.write(cloud))

        #expect(Array(bytes[0..<4]) == [0x4e, 0x47, 0x53, 0x50]) // "NGSP", little-endian 0x5053474e
        #expect(bytes[4..<8].elementsEqual([2, 0, 0, 0]))
        #expect(bytes[8..<12].elementsEqual([7, 0, 0, 0]))
        #expect(bytes[12] == 2) // SH degree
        #expect(bytes[13] == 12) // fractional bits
        #expect(bytes[14] == 0) // flags
        #expect(bytes[15] == 0) // reserved
        // 16 header bytes, then 9 + 1 + 3 + 3 + 3 + 3·8 bytes per point.
        #expect(bytes.count == 16 + 7 * (9 + 1 + 3 + 3 + 3 + 24))
    }

    @Test func emptyCloudIsRejectedOnWriteAndRead() throws {
        let empty = try GaussianCloud(shDegree: 0, positions: [], logScales: [], rotations: [],
                                      opacityLogits: [], sh: [])
        #expect(throws: SplatIOError.empty) {
            try SPZWriter.write(empty)
        }
        #expect(throws: SplatIOError.empty) {
            try SPZReader.read(gzipFile(header(numPoints: 0)))
        }
    }

    @Test func nonFiniteValuesThrowOnWrite() throws {
        let cloud = try GaussianCloud(shDegree: 0,
                                      positions: [Float.nan, 0, 0],
                                      logScales: [0, 0, 0],
                                      rotations: [1, 0, 0, 0],
                                      opacityLogits: [0],
                                      sh: [0, 0, 0])
        #expect(throws: SplatIOError.nonFinite) {
            try SPZWriter.write(cloud)
        }
    }

    @Test func positionOutsideFixedPointRangeThrows() throws {
        let cloud = try GaussianCloud(shDegree: 0,
                                      positions: [3000, 0, 0],
                                      logScales: [0, 0, 0],
                                      rotations: [1, 0, 0, 0],
                                      opacityLogits: [0],
                                      sh: [0, 0, 0])
        #expect(throws: SplatIOError.self) {
            try SPZWriter.write(cloud)
        }
    }

    @Test func storedGzipStreamIsReadable() throws {
        // The same SPZ payload, wrapped in a gzip stream of stored blocks, must read back identically.
        let cloud = try randomCloud(degree: 1, count: 50, seed: 8)
        let original = try SPZWriter.write(cloud)
        let restored = gzipFile(try payload(of: original))

        let fromOriginal = try SPZReader.read(original)
        let fromStored = try SPZReader.read(restored)
        #expect(fromStored.positions == fromOriginal.positions)
        #expect(fromStored.logScales == fromOriginal.logScales)
        #expect(fromStored.rotations == fromOriginal.rotations)
        #expect(fromStored.opacityLogits == fromOriginal.opacityLogits)
        #expect(fromStored.sh == fromOriginal.sh)
    }

    #if canImport(Compression)
    @Test func deflateStreamFromCompressionIsSmallerAndReadable() throws {
        // Apple platforms: the compressed stream must be readable and actually compressed.
        let zeros = [UInt8](repeating: 0, count: 10_000)
        let compressed = Gzip.compress(zeros)
        #expect(compressed.count < zeros.count)
        #expect(try Gzip.decompress(Data(compressed)) == zeros)
    }
    #endif

    @Test func compressedStreamIsRejectedWithoutCompressionOnLinux() {
        #if !canImport(Compression)
        // Gzip header, one final block with BTYPE = 01 (fixed Huffman), then the trailer.
        let bytes: [UInt8] = [0x1f, 0x8b, 0x08, 0, 0, 0, 0, 0, 0, 0xff, 0x03, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #expect(throws: SplatIOError.unsupportedCompression) {
            try SPZReader.read(Data(bytes))
        }
        #endif
    }

    // MARK: - Malformed input

    @Test func rejectsNonGzipInput() {
        #expect(throws: SplatIOError.self) {
            try SPZReader.read(Data([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]))
        }
        #expect(throws: SplatIOError.self) {
            try SPZReader.read(Data())
        }
    }

    @Test func rejectsBadMagic() {
        let bytes = header(magic: 0x1234_5678, numPoints: 1) + [UInt8](repeating: 0, count: 19 + 9)
        #expect(throws: SplatIOError.self) {
            try SPZReader.read(gzipFile(bytes))
        }
    }

    @Test func rejectsNumPointsBeyondPayload() {
        let bytes = header(numPoints: 1000) + [UInt8](repeating: 0, count: 40)
        #expect(throws: SplatIOError.self) {
            try SPZReader.read(gzipFile(bytes))
        }
        let huge = header(numPoints: UInt32.max) + [UInt8](repeating: 0, count: 40)
        #expect(throws: SplatIOError.self) {
            try SPZReader.read(gzipFile(huge))
        }
    }

    @Test func rejectsTruncatedPayload() throws {
        let cloud = try randomCloud(degree: 0, count: 20, seed: 2)
        var bytes = try payload(of: try SPZWriter.write(cloud))
        bytes.removeLast(10)
        #expect(throws: SplatIOError.self) {
            try SPZReader.read(gzipFile(bytes))
        }
    }

    @Test func rejectsTrailingBytesWithoutExtensionFlag() throws {
        let cloud = try randomCloud(degree: 0, count: 5, seed: 2)
        var bytes = try payload(of: try SPZWriter.write(cloud))
        bytes.append(0)
        #expect(throws: SplatIOError.self) {
            try SPZReader.read(gzipFile(bytes))
        }
    }

    @Test func rejectsShDegreeAboveThree() {
        let bytes = header(numPoints: 1, shDegree: 4) + [UInt8](repeating: 0, count: 200)
        #expect(throws: SplatIOError.unsupportedFormat("SPZ SH degree 4")) {
            try SPZReader.read(gzipFile(bytes))
        }
    }

    @Test func rejectsVersionThreeAndVersionFour() {
        let version3 = header(version: 3, numPoints: 1) + [UInt8](repeating: 0, count: 100)
        #expect(throws: SplatIOError.unsupportedFormat("SPZ version 3")) {
            try SPZReader.read(gzipFile(version3))
        }
        let version4 = Array("NGSP".utf8) + [UInt8](repeating: 0, count: 60)
        #expect(throws: SplatIOError.unsupportedFormat("SPZ version 4 (ZSTD container)")) {
            try SPZReader.read(Data(version4))
        }
    }

    @Test func rejectsCorruptChecksum() throws {
        let cloud = try randomCloud(degree: 0, count: 4, seed: 6)
        var file = try SPZWriter.write(cloud)
        // The CRC-32 sits in the 8-byte trailer; flip one of its bytes.
        file[file.count - 8] ^= 0xff
        #expect(throws: SplatIOError.self) {
            try SPZReader.read(file)
        }
    }

    @Test func rejectsZeroPoints() {
        #expect(throws: SplatIOError.empty) {
            try SPZReader.read(gzipFile(header(numPoints: 0)))
        }
    }

    @Test func loadsThroughSceneLoader() throws {
        let cloud = try randomCloud(degree: 0, count: 12, seed: 21)
        let file = try SPZWriter.write(cloud)
        let scene = try SceneLoader.load(file, fileExtension: "SPZ")
        #expect(scene.cloud.count == 12)
        #expect(scene.removedNonFinite == 0)
    }

    // MARK: - Frame conversion (RDF in memory, RUB in the file)

    @Test func shSignTableMatchesBasisParity() {
        // Y_i(F·d) = s_i · Y_i(d) for all 16 basis functions of degree 3, checked with the project's evaluation.
        #expect(SPZFrame.shSigns.count == 16)
        var rng = SeededGenerator(seed: 17)
        var maxError: Float = 0
        for i in 0..<16 {
            var coefficients = [Float](repeating: 0, count: 16 * 3)
            coefficients[3 * i] = 1 // unit weight on basis i, red channel only
            for _ in 0..<64 {
                let d = randomUnitDirection(&rng)
                let mirrored = SIMD3<Float>(d.x, -d.y, -d.z)
                let y: Float = SphericalHarmonics.evaluate(degree: 3, coefficients: coefficients[...], direction: d).x
                let yMirrored: Float = SphericalHarmonics.evaluate(degree: 3, coefficients: coefficients[...], direction: mirrored).x
                let expected: Float = SPZFrame.shSigns[i] * y
                maxError = max(maxError, abs(yMirrored - expected))
            }
        }
        #expect(maxError < 1e-5)
    }

    @Test func shRenderEquivalenceUnderFrameConversion() throws {
        // A Gaussian seen from d in RDF must colour the same as the converted Gaussian seen from F·d in RUB.
        let cloud = try randomCloud(degree: 3, count: 40, seed: 77)
        let converted = try SPZFrame.convert(cloud)
        let k = GaussianCloud.coefficientCount(degree: 3)
        var rng = SeededGenerator(seed: 5)
        var maxError: Float = 0
        for i in 0..<cloud.count {
            let range: Range<Int> = (i * k * 3)..<((i + 1) * k * 3)
            for _ in 0..<16 {
                let d = randomUnitDirection(&rng)
                let mirrored = SIMD3<Float>(d.x, -d.y, -d.z)
                let original = SphericalHarmonics.evaluate(degree: 3, coefficients: cloud.sh[range], direction: d)
                let rendered = SphericalHarmonics.evaluate(degree: 3, coefficients: converted.sh[range], direction: mirrored)
                let difference: SIMD3<Float> = rendered - original
                maxError = max(maxError, maxAbsComponent(difference))
            }
        }
        #expect(maxError < 1e-5)
    }

    @Test func rotationConversionIsConjugationByFrame() throws {
        // R(q') = F · R(q) · F for random unit quaternions, with R from the project's Matrix3.rotation.
        var rng = SeededGenerator(seed: 31)
        let count = 500
        var rotations: [Float] = []
        for _ in 0..<count {
            rotations.append(contentsOf: anyUnitQuaternion(&rng))
        }
        let cloud = try GaussianCloud(shDegree: 0,
                                      positions: [Float](repeating: 0, count: 3 * count),
                                      logScales: [Float](repeating: 0, count: 3 * count),
                                      rotations: rotations,
                                      opacityLogits: [Float](repeating: 0, count: count),
                                      sh: [Float](repeating: 0, count: 3 * count))
        let converted = try SPZFrame.convert(cloud)
        var maxError: Float = 0
        for i in 0..<count {
            let expected: Matrix3 = frameMatrix * Matrix3.rotation(quaternion: quaternion(cloud.rotations, i)) * frameMatrix
            let actual: Matrix3 = Matrix3.rotation(quaternion: quaternion(converted.rotations, i))
            let d0: SIMD3<Float> = expected.c0 - actual.c0
            let d1: SIMD3<Float> = expected.c1 - actual.c1
            let d2: SIMD3<Float> = expected.c2 - actual.c2
            maxError = max(maxError, maxAbsComponent(d0), maxAbsComponent(d1), maxAbsComponent(d2))
        }
        #expect(maxError < 1e-5)
    }

    @Test func fileStoresTheRUBFrame() throws {
        // One Gaussian: position (1, 2, 3); quaternion (w, x, y, z) = (0.8, 0, 0.6, 0); SH degree 1 with the
        // band-1 y coefficient at 0.5 on the red channel. The stored bytes must be the RUB values.
        let cloud = try GaussianCloud(shDegree: 1,
                                      positions: [1, 2, 3],
                                      logScales: [0, 0, 0],
                                      rotations: [0.8, 0, 0.6, 0],
                                      opacityLogits: [0],
                                      sh: [0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0])
        let bytes = try payload(of: try SPZWriter.write(cloud))

        // Positions follow the 16-byte header: 24-bit little-endian fixed point, value · 4096.
        func fixed(_ offset: Int) -> Int {
            let raw = Int(bytes[offset]) | Int(bytes[offset + 1]) << 8 | Int(bytes[offset + 2]) << 16
            return raw >= 0x80_0000 ? raw - 0x100_0000 : raw
        }
        #expect(fixed(16) == 4096) // x unchanged
        #expect(fixed(19) == -8192) // y stored as -2
        #expect(fixed(22) == -12288) // z stored as -3

        // Rotations start at 16 + 9 + 1 + 3 + 3 = 32. The stored xyz is (0, −0.6, 0): y byte round(−0.6 · 127.5 + 127.5) = 51.
        #expect(abs(Int(bytes[33]) - 51) <= 1)

        // The first rest coefficient starts at 35. Its sign for band 1, y is −1, so 0.5 is stored as −0.5, byte 64.
        #expect(bytes[35] == 64)
    }
}
