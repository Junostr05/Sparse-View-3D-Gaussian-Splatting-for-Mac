import Foundation
import Testing
import SplatCore
import SplatIO

private let splatC0: Double = 0.28209479177387814

/// Unit quaternion (w, x, y, z) with w ≥ 0.3, so a w-error bound derived from xyz quantisation holds.
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

/// Deterministic cloud. DC is in ±1.7 (inside the `.splat` colour range), higher SH in ±0.99.
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

/// Bound on the logit error after one-byte alpha quantisation. The decoded probability is within
/// h = 0.5/255 of the true one. The logit slope 1/(t(1 − t)) is largest at the interval end nearest
/// 0 or 1, so the bound uses the smaller of t(1 − t) at the two ends.
private func alphaLogitTolerance(_ logit: Double) -> Double {
    let p = 1 / (1 + exp(-logit))
    let h = 0.5 / 255 + 1e-9
    let low = min(max(p - h, 1e-6), 1 - 1e-6)
    let high = min(max(p + h, 1e-6), 1 - 1e-6)
    let slowest = min(low * (1 - low), high * (1 - high))
    return h / slowest + 1e-4
}

/// Maps each original index to the index of the record with the same exact position.
/// Positions are unique random floats, so the match is one-to-one.
private func matchByPosition(_ original: GaussianCloud, _ read: GaussianCloud) -> [Int]? {
    var index: [[UInt32]: Int] = [:]
    for j in 0..<read.count {
        let key: [UInt32] = (0..<3).map { read.positions[3 * j + $0].bitPattern }
        index[key] = j
    }
    var mapping: [Int] = []
    for i in 0..<original.count {
        let key: [UInt32] = (0..<3).map { original.positions[3 * i + $0].bitPattern }
        guard let j = index[key] else { return nil }
        mapping.append(j)
    }
    return mapping
}

@Suite struct SplatWriterTests {
    @Test func recordLayoutMatchesReaderConventions() throws {
        // One Gaussian: DC 0 gives RGB 0.5·255 = 127.5, which rounds to 128. Opacity logit 0 gives
        // alpha 127.5, also 128. Identity rotation gives w = 1·128 + 128 → 255 (clamped) and x, y, z = 128.
        let cloud = try GaussianCloud(shDegree: 0,
                                      positions: [1, 2, 3],
                                      logScales: [0, 0, 0],
                                      rotations: [1, 0, 0, 0],
                                      opacityLogits: [0],
                                      sh: [0, 0, 0])
        let data = SplatWriter.write(cloud)

        #expect(data.count == 32)
        #expect([UInt8](data[24..<28]) == [128, 128, 128, 128])
        #expect([UInt8](data[28..<32]) == [255, 128, 128, 128])
    }

    @Test func roundTripsThroughAntimatterReader() throws {
        // Degree 3 input: only the DC term is stored, so the reader must return degree 0.
        let cloud = try randomCloud(degree: 3, count: 500, seed: 11)
        let read = try AntimatterSplatReader.read(SplatWriter.write(cloud))

        #expect(read.count == cloud.count)
        #expect(read.shDegree == 0)
        guard let mapping = matchByPosition(cloud, read) else {
            Issue.record("some positions did not survive the round trip exactly")
            return
        }

        var worstColour: Double = 0
        var worstAlpha: Double = 0
        var worstRotation: Double = 0
        var worstScale: Double = 0
        var failures: [String] = []
        for i in 0..<cloud.count {
            let j = mapping[i]

            // Position: exact.
            for c in 0..<3 where read.positions[3 * j + c] != cloud.positions[3 * i + c] {
                failures.append("position \(i)")
            }

            // Scale: linear scale is stored as Float32, so the log comes back to float precision.
            for c in 0..<3 {
                let error = abs(Double(read.logScales[3 * j + c]) - Double(cloud.logScales[3 * i + c]))
                worstScale = max(worstScale, error)
            }

            // DC colour: one byte step is 1/255 in RGB, which is 1/(255·C0) in DC.
            let colourTolerance: Double = 0.5 / 255 / splatC0 + 1e-5
            for c in 0..<3 {
                let original = Double(cloud.sh[(i * 16) * 3 + c])
                let decoded = Double(read.sh[(j * 1) * 3 + c])
                let error = abs(decoded - original)
                worstColour = max(worstColour, error)
                if error > colourTolerance {
                    failures.append("dc \(i) channel \(c): \(error)")
                }
            }

            // Opacity: alpha is one byte, so the probability has error 0.5/255. Convert that to a
            // logit bound: d logit / d p = 1 / (p (1 − p)).
            let alphaTolerance = alphaLogitTolerance(Double(cloud.opacityLogits[i]))
            let alphaError = abs(Double(read.opacityLogits[j]) - Double(cloud.opacityLogits[i]))
            worstAlpha = max(worstAlpha, alphaError)
            if alphaError > alphaTolerance {
                failures.append("alpha \(i): \(alphaError) > \(alphaTolerance)")
            }

            // Rotation: each component is one byte, step 1/128, and the reader renormalises.
            for q in 0..<4 {
                let error = abs(Double(read.rotations[4 * j + q]) - Double(cloud.rotations[4 * i + q]))
                worstRotation = max(worstRotation, error)
                if error > 1.0 / 128 {
                    failures.append("rotation \(i) component \(q): \(error)")
                }
            }
        }
        #expect(failures.isEmpty, "\(failures.prefix(5))")
        #expect(worstScale < 1e-5)
        #expect(worstColour <= 0.5 / 255 / splatC0 + 1e-5)
        #expect(worstAlpha < 0.1)
        #expect(worstRotation <= 1.0 / 128)
    }

    @Test func storesGaussiansInDescendingSizeTimesOpacityOrder() throws {
        let cloud = try randomCloud(degree: 0, count: 300, seed: 5)
        let read = try AntimatterSplatReader.read(SplatWriter.write(cloud))
        guard let mapping = matchByPosition(cloud, read) else {
            Issue.record("positions did not survive the round trip exactly")
            return
        }
        // Original index of each record, in file order.
        var originalAt = [Int](repeating: 0, count: cloud.count)
        for i in 0..<cloud.count {
            originalAt[mapping[i]] = i
        }

        // Keys from the unquantised cloud, the same quantities the writer sorts by.
        var previous = Double.infinity
        for j in 0..<read.count {
            let i = originalAt[j]
            let logSum: Double = Double(cloud.logScales[3 * i]) + Double(cloud.logScales[3 * i + 1]) + Double(cloud.logScales[3 * i + 2])
            let opacity: Double = 1 / (1 + exp(-Double(cloud.opacityLogits[i])))
            let key: Double = exp(logSum) * opacity
            #expect(key <= previous)
            previous = key
        }
    }

    @Test func nonFiniteValuesDoNotTrap() throws {
        let cloud = try GaussianCloud(shDegree: 0,
                                      positions: [Float.nan, 0, 0],
                                      logScales: [0, Float.infinity, 0],
                                      rotations: [Float.nan, 0, 0, 0],
                                      opacityLogits: [Float.nan],
                                      sh: [Float.infinity, 0, 0])
        let data = SplatWriter.write(cloud)
        #expect(data.count == 32)
    }

    @Test func emptyCloudWritesNoBytes() throws {
        let cloud = try GaussianCloud(shDegree: 0, positions: [], logScales: [], rotations: [],
                                      opacityLogits: [], sh: [])
        #expect(SplatWriter.write(cloud).isEmpty)
    }
}
