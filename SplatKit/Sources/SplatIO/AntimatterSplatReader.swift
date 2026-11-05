import Foundation
import SplatCore

/// Reads the headerless `.splat` format: a sequence of 32-byte little-endian records,
/// each holding one degree-0 Gaussian.
///
/// Record layout:
/// - bytes 0–11: position, 3 × Float32
/// - bytes 12–23: linear scale, 3 × Float32
/// - bytes 24–27: RGBA, 4 × UInt8
/// - bytes 28–31: rotation (w, x, y, z), 4 × UInt8, decoded as `(b − 128) / 128`
public enum AntimatterSplatReader {
    private static let recordSize = 32
    /// Real SH constant for degree 0; kept private so this reader does not depend on SplatCore.
    private static let C0: Float = 0.28209479177387814

    public static func read(_ data: Data) throws(SplatIOError) -> GaussianCloud {
        guard !data.isEmpty else { throw .empty }
        guard data.count % recordSize == 0 else { throw .invalidSize(data.count) }
        let n = data.count / recordSize

        var positions = [Float](repeating: 0, count: 3 * n)
        var logScales = [Float](repeating: 0, count: 3 * n)
        var rotations = [Float](repeating: 0, count: 4 * n)
        var opacityLogits = [Float](repeating: 0, count: n)
        var sh = [Float](repeating: 0, count: n * 3)

        data.withUnsafeBytes { raw in
            for i in 0..<n {
                let base = i * recordSize
                for c in 0..<3 {
                    positions[3 * i + c] = loadFloat(raw, at: base + 4 * c)
                    let linearScale = loadFloat(raw, at: base + 12 + 4 * c)
                    logScales[3 * i + c] = log(max(linearScale, 1e-8))
                }

                let rgba = base + 24
                for c in 0..<3 {
                    let byte = raw[rgba + c]
                    sh[3 * i + c] = (Float(byte) / 255 - 0.5) / C0
                }
                // Computed in Double: in Float, `1 − alpha` near 1 loses about 4 digits.
                let alpha = min(max(Double(raw[rgba + 3]) / 255, 1e-4), 1 - 1e-4)
                opacityLogits[i] = Float(log(alpha / (1 - alpha)))

                // Decode, then normalize; a zero quaternion becomes the identity.
                var quaternion = SIMD4<Float>(0, 0, 0, 0)
                for q in 0..<4 {
                    quaternion[q] = (Float(raw[base + 28 + q]) - 128) / 128
                }
                let length = (quaternion * quaternion).sum().squareRoot()
                let normalized = length > 0 ? quaternion / length : SIMD4<Float>(1, 0, 0, 0)
                for q in 0..<4 {
                    rotations[4 * i + q] = normalized[q]
                }
            }
        }

        do {
            return try GaussianCloud(
                shDegree: 0,
                positions: positions,
                logScales: logScales,
                rotations: rotations,
                opacityLogits: opacityLogits,
                sh: sh
            )
        } catch {
            throw .unsupportedFormat("inconsistent data")
        }
    }

    @inline(__always)
    private static func loadFloat(_ raw: UnsafeRawBufferPointer, at offset: Int) -> Float {
        Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self)))
    }
}
