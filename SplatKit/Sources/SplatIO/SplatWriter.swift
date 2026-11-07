import Foundation
import SplatCore

/// Writes the headerless antimatter15 `.splat` format, the inverse of `AntimatterSplatReader`.
///
/// Record layout (32 bytes, little-endian), matching the reader:
/// - bytes 0–11: position, 3 × Float32, exact
/// - bytes 12–23: linear scale, 3 × Float32, `exp(logScale)`
/// - bytes 24–27: RGBA, 4 × UInt8. RGB = round(clamp((0.5 + C0·dc)·255)) and A = round(clamp(sigmoid(opacity)·255))
/// - bytes 28–31: rotation (w, x, y, z), 4 × UInt8, each round(clamp(q·128 + 128)) after normalising q
///
/// Only the DC colour is stored. The format has no higher-order spherical harmonics, so view-dependent
/// colour is dropped.
///
/// Order: Gaussians are sorted by descending `exp(sx + sy + sz) · sigmoid(opacity)` (their screen-space
/// size times opacity), ties by original index. The reader does not depend on the order, and viewers
/// sort for drawing themselves, so this ordering only puts the most visible splats first.
///
/// Non-finite values cannot be represented. They are written as zero bytes; the position keeps its
/// bit pattern. Call `removingNonFinite()` first if that matters.
public enum SplatWriter {
    private static let recordSize = 32
    /// Real SH constant for degree 0, as in `AntimatterSplatReader`.
    private static let C0: Double = 0.28209479177387814

    public static func write(_ cloud: GaussianCloud) -> Data {
        let n = cloud.count
        let k = GaussianCloud.coefficientCount(degree: cloud.shDegree)

        var sortKeys = [Double](repeating: 0, count: n)
        for i in 0..<n {
            let scaleSum: Double = Double(cloud.logScales[3 * i])
                + Double(cloud.logScales[3 * i + 1])
                + Double(cloud.logScales[3 * i + 2])
            let size: Double = exp(scaleSum)
            let opacity: Double = sigmoid(Double(cloud.opacityLogits[i]))
            let key: Double = size * opacity
            // NaN keys sort last, so they cannot break the strict weak ordering.
            sortKeys[i] = key.isNaN ? -Double.infinity : key
        }
        let order = (0..<n).sorted { a, b in
            if sortKeys[a] != sortKeys[b] {
                return sortKeys[a] > sortKeys[b]
            }
            return a < b
        }

        var bytes: [UInt8] = []
        bytes.reserveCapacity(n * recordSize)
        for i in order {
            // Position: exact Float32 bit patterns.
            for c in 0..<3 {
                appendLittleEndian32(cloud.positions[3 * i + c].bitPattern, to: &bytes)
            }
            // Linear scale.
            for c in 0..<3 {
                let linear: Float = Float(exp(Double(cloud.logScales[3 * i + c])))
                appendLittleEndian32(linear.bitPattern, to: &bytes)
            }
            // RGBA from the DC term (SH coefficient 0, channels interleaved).
            for c in 0..<3 {
                let dc: Double = Double(cloud.sh[(i * k) * 3 + c])
                let colour: Double = (0.5 + C0 * dc) * 255
                bytes.append(clampedByte(colour))
            }
            let alpha: Double = sigmoid(Double(cloud.opacityLogits[i])) * 255
            bytes.append(clampedByte(alpha))

            // Rotation (w, x, y, z), normalised first. A zero quaternion becomes the identity.
            var quaternion: [Double] = (0..<4).map { Double(cloud.rotations[4 * i + $0]) }
            var squaredNorm: Double = 0
            for value in quaternion {
                squaredNorm += value * value
            }
            let norm: Double = squaredNorm.squareRoot()
            if norm.isFinite && norm > 0 {
                quaternion = quaternion.map { $0 / norm }
            } else {
                quaternion = [1, 0, 0, 0]
            }
            for value in quaternion {
                bytes.append(clampedByte(value * 128 + 128))
            }
        }
        return Data(bytes)
    }

    // MARK: - Helpers

    private static func sigmoid(_ x: Double) -> Double {
        1 / (1 + exp(-x))
    }

    /// Rounds to the nearest integer and clamps to 0...255. NaN maps to 0.
    static func clampedByte(_ value: Double) -> UInt8 {
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
