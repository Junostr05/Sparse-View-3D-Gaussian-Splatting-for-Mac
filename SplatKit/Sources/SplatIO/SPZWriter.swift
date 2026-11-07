import Foundation
import SplatCore

/// Writes SPZ version 2 files, the legacy gzip layout of Niantic's SPZ format (MIT), implemented from
/// the published specification.
///
/// The whole file is one gzip stream. Its uncompressed payload is:
/// - header, 16 bytes: magic `NGSP` (0x5053474e, little-endian), version 2, numPoints, shDegree,
///   fractionalBits (12), flags (0), reserved (0). Integers are little-endian UInt32 or UInt8.
/// - positions: per Gaussian x, y, z as signed 24-bit fixed point, value · 2^12, round to nearest
/// - alphas: per Gaussian UInt8, round(sigmoid(opacityLogit) · 255)
/// - colours: per Gaussian R, G, B as UInt8, round(((dc · 0.15) + 0.5) · 255). `dc` is the 3DGS DC coefficient.
/// - log-scales: per Gaussian x, y, z as UInt8, round((logScale + 10) · 16)
/// - rotations: per Gaussian x, y, z as UInt8, round(q · 127.5 + 127.5) of the unit quaternion with w ≥ 0.
///   w is recovered by the reader as sqrt(1 − x² − y² − z²).
/// - spherical harmonics: the (K − 1) higher-order coefficients per Gaussian, each R, G, B as UInt8,
///   round(x · 128 + 128). Coefficients are stored in order (coefficient-major, channel-minor).
///
/// Behaviour specific to this platform:
/// - On Apple platforms the payload is deflated with the `Compression` framework (`COMPRESSION_ZLIB`).
/// - Elsewhere it is written as stored (uncompressed) deflate blocks. The result is still a valid gzip
///   stream and reads back with the same reader, but it is larger.
///
/// Quantisation: positions round to 1/4096, alphas and colours to 1/255, scales to 1/16 in log space,
/// rotations to 1/127.5 per component, SH to 1/128. Scales outside [−10, 5.9375] and SH outside [−1, 1)
/// are clamped. Positions outside ±2048 cannot be stored and throw.
///
/// Coordinate frame: `GaussianCloud` is in the 3DGS / COLMAP PLY frame, RDF (x right, y down, z forward).
/// The SPZ data is stored in RUB (x right, y up, z back), the frame of Niantic's official tools
/// (`load-spz.cc`, `splat-types.h`). The writer therefore maps the cloud from RDF to RUB before quantising:
/// positions and the xyz part of each quaternion get the signs (1, −1, −1), and each higher-order SH
/// coefficient c_i becomes s_i · c_i (see `SPZFrame`). The map F = diag(1, −1, −1) is its own inverse,
/// so `SPZReader` applies it again.
///
/// Throws `SplatIOError.empty` for an empty cloud (the format requires numPoints ≥ 1),
/// `.nonFinite` for NaN or infinite values, and `.outOfRange` for positions that do not fit.
public enum SPZWriter {
    static let magic: UInt32 = 0x5053_474e
    static let version: UInt32 = 2
    static let fractionalBits: UInt8 = 12
    private static let headerSize = 16

    public static func write(_ cloud: GaussianCloud) throws(SplatIOError) -> Data {
        let n = cloud.count
        guard n > 0 else { throw .empty }
        let k = GaussianCloud.coefficientCount(degree: cloud.shDegree)
        let restCount = k - 1
        let perPoint = 9 + 1 + 3 + 3 + 3 + 3 * restCount
        var payload: [UInt8] = []
        payload.reserveCapacity(headerSize + n * perPoint)

        // Header.
        appendLittleEndian32(magic, to: &payload)
        appendLittleEndian32(version, to: &payload)
        appendLittleEndian32(UInt32(n), to: &payload)
        payload.append(UInt8(cloud.shDegree))
        payload.append(fractionalBits)
        payload.append(0) // flags: no antialiasing, no extensions
        payload.append(0) // reserved

        let positionScale: Double = Double(1 << Int(fractionalBits))

        // Positions: 24-bit signed fixed point, three per Gaussian, in the RUB storage frame.
        for i in 0..<n {
            for c in 0..<3 {
                let value: Float = cloud.positions[3 * i + c] * SPZFrame.axisSigns[c]
                guard value.isFinite else { throw .nonFinite }
                let scaled: Double = (Double(value) * positionScale).rounded()
                guard scaled >= -8_388_608, scaled <= 8_388_607 else {
                    throw .outOfRange("position \(value) does not fit 24-bit fixed point with \(fractionalBits) fractional bits")
                }
                let fixed = Int(scaled)
                payload.append(UInt8(fixed & 0xff))
                payload.append(UInt8((fixed >> 8) & 0xff))
                payload.append(UInt8((fixed >> 16) & 0xff))
            }
        }

        // Alphas.
        for i in 0..<n {
            let logit = cloud.opacityLogits[i]
            guard logit.isFinite else { throw .nonFinite }
            let probability: Double = 1 / (1 + exp(-Double(logit)))
            payload.append(clampedByte(probability * 255))
        }

        // Colours (DC term).
        for i in 0..<n {
            for c in 0..<3 {
                let dc = cloud.sh[(i * k) * 3 + c]
                guard dc.isFinite else { throw .nonFinite }
                let encoded: Double = Double(dc) * (0.15 * 255) + 0.5 * 255
                payload.append(clampedByte(encoded))
            }
        }

        // Log-scales.
        for i in 0..<n {
            for c in 0..<3 {
                let logScale = cloud.logScales[3 * i + c]
                guard logScale.isFinite else { throw .nonFinite }
                payload.append(clampedByte((Double(logScale) + 10) * 16))
            }
        }

        // Rotations: unit quaternion with w >= 0, xyz stored, w recovered by the reader.
        for i in 0..<n {
            var quaternion: [Double] = []
            quaternion.reserveCapacity(4)
            for q in 0..<4 {
                let value = cloud.rotations[4 * i + q]
                guard value.isFinite else { throw .nonFinite }
                quaternion.append(Double(value))
            }
            var squaredNorm: Double = 0
            for value in quaternion {
                squaredNorm += value * value
            }
            let norm: Double = squaredNorm.squareRoot()
            if norm > 0 {
                quaternion = quaternion.map { $0 / norm }
            } else {
                quaternion = [1, 0, 0, 0]
            }
            // quaternion is (w, x, y, z) in RDF. Conjugation by F = diag(1, −1, −1) (a 180° turn about x)
            // keeps w and maps the xyz part by the axis signs, so the stored xyz is in the RUB frame.
            for component in 1..<4 {
                quaternion[component] *= Double(SPZFrame.axisSigns[component - 1])
            }
            // The reader expects w >= 0.
            let sign: Double = quaternion[0] < 0 ? -1 : 1
            for component in 1..<4 {
                payload.append(clampedByte(sign * quaternion[component] * 127.5 + 127.5))
            }
        }

        // Higher-order SH: coefficient-major, channel-minor, DC excluded. Each coefficient gets its basis sign.
        for i in 0..<n {
            for coefficient in 1..<k {
                for c in 0..<3 {
                    let value: Float = cloud.sh[(i * k + coefficient) * 3 + c] * SPZFrame.shSigns[coefficient]
                    guard value.isFinite else { throw .nonFinite }
                    payload.append(clampedByte(Double(value) * 128 + 128))
                }
            }
        }

        return Data(Gzip.compress(payload))
    }

    // MARK: - Helpers

    /// Rounds to the nearest integer and clamps to 0...255.
    private static func clampedByte(_ value: Double) -> UInt8 {
        guard value.isFinite else { return 0 }
        let rounded: Double = value.rounded()
        return UInt8(min(max(rounded, 0), 255))
    }

    private static func appendLittleEndian32(_ value: UInt32, to bytes: inout [UInt8]) {
        bytes.append(UInt8(value & 0xff))
        bytes.append(UInt8((value >> 8) & 0xff))
        bytes.append(UInt8((value >> 16) & 0xff))
        bytes.append(UInt8(value >> 24))
    }
}

/// The change of frame between `GaussianCloud` (RDF, the PLY frame) and the SPZ storage frame (RUB).
///
/// The map is F = diag(1, −1, −1), a 180° rotation about x, and it is its own inverse. It is the table that
/// `coordinateConverter(RDF, RUB)` builds in the official `splat-types.h`: flipP = (1, −1, −1), flipQ = (1, −1, −1)
/// on the xyz components (w is never flipped), and the per-coefficient SH flips of the rest coefficients.
enum SPZFrame {
    /// Sign of x, y, z for positions, and for the xyz part of a quaternion.
    static let axisSigns: [Float] = [1, -1, -1]

    /// Sign s_i per SH basis index i, in `SphericalHarmonics` order (index 0 is DC, 1…3 band 1, 4…8 band 2,
    /// 9…15 band 3), such that Y_i(F·d) = s_i · Y_i(d) for every unit direction d.
    ///
    /// Derivation: F sends (x, y, z) to (x, −y, −z), so a monomial x^a y^b z^c changes sign by (−1)^(b+c).
    /// Every basis function has one parity in (y, z), which gives its sign:
    /// - band 0: constant, +1;
    /// - band 1: y (−1), z (−1), x (+1);
    /// - band 2: xy (−1), yz (+1), 2z² − x² − y² (+1), xz (−1), x² − y² (+1);
    /// - band 3: y(3x² − y²) (−1), xyz (+1), y(4z² − x² − y²) (−1), z(2z² − 3x² − 3y²) (−1),
    ///   x(4z² − x² − y²) (+1), z(x² − y²) (−1), x(x² − 3y²) (+1).
    ///
    /// `SPZTests` checks every entry numerically against `SphericalHarmonics.evaluate`.
    static let shSigns: [Float] = [
        1,
        -1, -1, 1,
        -1, 1, 1, -1, 1,
        -1, 1, -1, -1, 1, -1, 1,
    ]

    /// Applies the map to a whole cloud. `SPZWriter` and `SPZReader` apply the same signs element by element;
    /// this form states the map in one place and is the reference for the tests.
    static func convert(_ cloud: GaussianCloud) throws(GaussianCloudError) -> GaussianCloud {
        let n = cloud.count
        let k = GaussianCloud.coefficientCount(degree: cloud.shDegree)
        var positions: [Float] = cloud.positions
        var rotations: [Float] = cloud.rotations
        var sh: [Float] = cloud.sh
        for i in 0..<n {
            for c in 0..<3 {
                positions[3 * i + c] *= axisSigns[c]
            }
            // Quaternion stored (w, x, y, z): w is kept, the xyz part takes the axis signs.
            for q in 1..<4 {
                rotations[4 * i + q] *= axisSigns[q - 1]
            }
            for coefficient in 0..<k {
                for c in 0..<3 {
                    sh[(i * k + coefficient) * 3 + c] *= shSigns[coefficient]
                }
            }
        }
        return try GaussianCloud(shDegree: cloud.shDegree,
                                 positions: positions,
                                 logScales: cloud.logScales,
                                 rotations: rotations,
                                 opacityLogits: cloud.opacityLogits,
                                 sh: sh)
    }
}
