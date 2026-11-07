import Foundation
import SplatCore

/// Reads SPZ files written in the legacy version 2 layout described in `SPZWriter`.
///
/// Supported: version 2, SH degree 0–3, fractionalBits up to 30, any flags (the extension bit allows
/// trailing data after the payload). The whole file must be a gzip stream.
///
/// Not supported, and reported as `SplatIOError.unsupportedFormat`:
/// - the current `NGSP` version 4 container (ZSTD streams, table of contents),
/// - version 3 (smallest-three quaternion encoding; the bit layout is not implemented here),
/// - version 1 (float16 positions),
/// - SH degree 4.
///
/// Deflate streams: on Apple platforms every stream is inflated with the `Compression` framework. Elsewhere
/// only stored blocks are accepted; a compressed stream throws `SplatIOError.unsupportedCompression`.
///
/// Decoded values: positions are multiplied by 2^−fractionalBits. Alphas are `invSigmoid(b / 255)` and are
/// clamped to [1e-4, 1 − 1e-4] first, as `AntimatterSplatReader` does, so the decoded logit stays finite.
/// Colours are `((b / 255) − 0.5) / 0.15`, log-scales `b / 16 − 10`, rotations `w = sqrt(max(0, 1 − |xyz|²))`
/// with `xyz = (b − 127.5) / 127.5`, and SH `(b − 128) / 128`.
///
/// Malformed input throws and never traps: bad magic, truncated or oversized data, a zero or
/// too-large point count, bad gzip framing or checksums.
public enum SPZReader {
    private static let headerSize = 16
    private static let colorScale: Float = 0.15

    public static func read(_ data: Data) throws(SplatIOError) -> GaussianCloud {
        if data.count >= 4, data.prefix(4) == Data("NGSP".utf8) {
            // Current container: the magic is the first four bytes, not inside a gzip stream.
            throw .unsupportedFormat("SPZ version 4 (ZSTD container)")
        }
        let payload = try Gzip.decompress(data)
        guard payload.count >= headerSize else {
            throw .corrupt("SPZ payload shorter than its header")
        }

        let magic = readLittleEndian32(payload, at: 0)
        guard magic == SPZWriter.magic else {
            throw .corrupt("bad SPZ magic")
        }
        let version = readLittleEndian32(payload, at: 4)
        guard version == SPZWriter.version else {
            throw .unsupportedFormat("SPZ version \(version)")
        }
        let numPoints = readLittleEndian32(payload, at: 8)
        let shDegree = Int(payload[12])
        let fractionalBits = Int(payload[13])
        let flags = payload[14]

        guard numPoints > 0 else { throw .empty }
        guard numPoints <= UInt32(Int32.max) else {
            throw .corrupt("SPZ numPoints \(numPoints) is out of range")
        }
        guard shDegree <= 3 else {
            throw .unsupportedFormat("SPZ SH degree \(shDegree)")
        }
        guard fractionalBits <= 30 else {
            throw .corrupt("SPZ fractionalBits \(fractionalBits) is out of range")
        }

        let n = Int(numPoints)
        let k = GaussianCloud.coefficientCount(degree: shDegree)
        let restCount = k - 1
        let perPoint = 9 + 1 + 3 + 3 + 3 + 3 * restCount

        // Bound the point count by the bytes present before computing the exact expected size.
        let available = payload.count - headerSize
        guard n <= available / perPoint else {
            throw .corrupt("SPZ payload truncated: \(numPoints) points need \(n * perPoint) bytes, \(available) present")
        }
        let expected = n * perPoint
        if available > expected && flags & 0x02 == 0 {
            throw .corrupt("SPZ payload has \(available - expected) unexpected trailing bytes")
        }

        let positionBase = headerSize
        let alphaBase = positionBase + 9 * n
        let colorBase = alphaBase + n
        let scaleBase = colorBase + 3 * n
        let rotationBase = scaleBase + 3 * n
        let shBase = rotationBase + 3 * n

        var positions = [Float](repeating: 0, count: 3 * n)
        var logScales = [Float](repeating: 0, count: 3 * n)
        var rotations = [Float](repeating: 0, count: 4 * n)
        var opacityLogits = [Float](repeating: 0, count: n)
        var sh = [Float](repeating: 0, count: n * k * 3)

        let positionDivisor: Float = Float(1 << fractionalBits)
        let minAlpha: Double = 1e-4

        for i in 0..<n {
            for c in 0..<3 {
                let offset = positionBase + 9 * i + 3 * c
                var fixed = Int(payload[offset])
                    | Int(payload[offset + 1]) << 8
                    | Int(payload[offset + 2]) << 16
                if fixed & 0x80_0000 != 0 {
                    fixed -= 0x100_0000
                }
                positions[3 * i + c] = Float(fixed) / positionDivisor
            }

            let alphaByte = payload[alphaBase + i]
            let alpha: Double = min(max(Double(alphaByte) / 255, minAlpha), 1 - minAlpha)
            opacityLogits[i] = Float(log(alpha / (1 - alpha)))

            for c in 0..<3 {
                let colorByte = payload[colorBase + 3 * i + c]
                sh[(i * k) * 3 + c] = (Float(colorByte) / 255 - 0.5) / colorScale
                logScales[3 * i + c] = Float(payload[scaleBase + 3 * i + c]) / 16 - 10
            }

            var xyz: [Float] = []
            xyz.reserveCapacity(3)
            for c in 0..<3 {
                xyz.append((Float(payload[rotationBase + 3 * i + c]) - 127.5) / 127.5)
            }
            var squaredNorm: Float = 0
            for value in xyz {
                squaredNorm += value * value
            }
            let w: Float = max(0, 1 - squaredNorm).squareRoot()
            rotations[4 * i] = w
            rotations[4 * i + 1] = xyz[0]
            rotations[4 * i + 2] = xyz[1]
            rotations[4 * i + 3] = xyz[2]

            for coefficient in 0..<restCount {
                for c in 0..<3 {
                    let byte = payload[shBase + i * restCount * 3 + coefficient * 3 + c]
                    sh[(i * k + coefficient + 1) * 3 + c] = (Float(byte) - 128) / 128
                }
            }
        }

        do {
            return try GaussianCloud(
                shDegree: shDegree,
                positions: positions,
                logScales: logScales,
                rotations: rotations,
                opacityLogits: opacityLogits,
                sh: sh
            )
        } catch {
            throw .corrupt("inconsistent SPZ data")
        }
    }

    private static func readLittleEndian32(_ bytes: [UInt8], at offset: Int) -> UInt32 {
        UInt32(bytes[offset])
            | UInt32(bytes[offset + 1]) << 8
            | UInt32(bytes[offset + 2]) << 16
            | UInt32(bytes[offset + 3]) << 24
    }
}
