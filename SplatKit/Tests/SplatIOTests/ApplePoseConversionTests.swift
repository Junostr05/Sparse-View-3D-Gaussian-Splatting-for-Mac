import Foundation
import Testing
@testable import SplatIO

// MARK: - Test helpers (independent of the implementation)

/// Deterministic generator (SplitMix64), so the property tests are repeatable.
private struct ConversionSeededGenerator {
    var state: UInt64

    mutating func nextUnit() -> Double {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z: UInt64 = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z = z ^ (z >> 31)
        return Double(z >> 11) / Double(UInt64(1) << 53)
    }

    /// Uniform in [-1, 1).
    mutating func nextSigned() -> Double {
        return 2 * nextUnit() - 1
    }
}

private func crossProduct(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> SIMD3<Double> {
    return SIMD3<Double>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
}

/// Rotates `v` by the unit quaternion `(x, y, z, w)` with `v + w t + u x t`, where `t = 2 (u x v)`.
/// This is a different formula from the matrix in the implementation.
private func rotate(_ v: SIMD3<Double>, byXYZW q: SIMD4<Double>) -> SIMD3<Double> {
    let u = SIMD3<Double>(q.x, q.y, q.z)
    let t: SIMD3<Double> = 2 * crossProduct(u, v)
    let result: SIMD3<Double> = v + q.w * t + crossProduct(u, t)
    return result
}

private func dot(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double {
    return a.x * b.x + a.y * b.y + a.z * b.z
}

private func close(_ a: SIMD3<Double>, _ b: SIMD3<Double>, tolerance: Double = 1e-9) -> Bool {
    return abs(a.x - b.x) < tolerance && abs(a.y - b.y) < tolerance && abs(a.z - b.z) < tolerance
}

private func unitQuaternion(from raw: SIMD4<Double>) -> SIMD4<Double> {
    let length: Double = (raw.x * raw.x + raw.y * raw.y + raw.z * raw.z + raw.w * raw.w).squareRoot()
    return raw / length
}

// MARK: - Known values

@Test func identityQuaternionGivesHalfTurnAboutXAndHandComputedTranslation() {
    let result = ApplePoseConversion.colmapPose(centre: SIMD3<Double>(1, 2, 3),
                                                quaternionXYZW: SIMD4<Double>(0, 0, 0, 1))

    // Apple's camera is the world frame, so the COLMAP world-to-camera rotation is diag(1, -1, -1):
    // a 180 degree turn about X. As (w, x, y, z) that is (0, 1, 0, 0), up to sign.
    let expectedRotation = SIMD4<Double>(0, 1, 0, 0)
    let dotValue: Double = result.rotation.x * expectedRotation.x
        + result.rotation.y * expectedRotation.y
        + result.rotation.z * expectedRotation.z
        + result.rotation.w * expectedRotation.w
    #expect(abs(dotValue) > 1 - 1e-12)

    // t = -R * C = -(diag(1, -1, -1) * (1, 2, 3)) = -(1, -2, -3) = (-1, 2, 3).
    #expect(close(result.translation, SIMD3<Double>(-1, 2, 3)))
}

@Test func focalLengthUsesDiagonalRule() {
    let focal = ApplePoseConversion.focalLengthPixels(focalLength35mm: 35, width: 900, height: 1200)
    // The diagonal of 900 x 1200 is 1500 pixels.
    #expect(abs(focal - 35.0 * 1500.0 / 43.267) < 1e-9)
}

@Test func focalLengthScalesWithPixelDiagonal() {
    let small = ApplePoseConversion.focalLengthPixels(focalLength35mm: 26, width: 90, height: 120)
    let large = ApplePoseConversion.focalLengthPixels(focalLength35mm: 26, width: 180, height: 240)
    #expect(abs(large - 2 * small) < 1e-9)
}

// MARK: - Property tests against an independent computation

@Test func colmapPoseMatchesIndependentComputationForRandomPoses() {
    var generator = ConversionSeededGenerator(state: 0xC0FFEE)
    let flip = SIMD3<Double>(1, -1, -1)
    let basis: [SIMD3<Double>] = [SIMD3<Double>(1, 0, 0), SIMD3<Double>(0, 1, 0), SIMD3<Double>(0, 0, 1)]
    let probes: [SIMD3<Double>] = [SIMD3<Double>(0.3, -1.2, 2.5), SIMD3<Double>(-4, 0.5, 1)]

    for _ in 0..<500 {
        // Random Apple camera-to-world rotation, unit length by construction.
        var raw = SIMD4<Double>(0, 0, 0, 0)
        repeat {
            raw = SIMD4<Double>(generator.nextSigned(), generator.nextSigned(),
                                generator.nextSigned(), generator.nextSigned())
        } while (raw.x * raw.x + raw.y * raw.y + raw.z * raw.z + raw.w * raw.w) < 0.05
        let appleQuaternion = unitQuaternion(from: raw)
        let centre = SIMD3<Double>(generator.nextSigned() * 5,
                                   generator.nextSigned() * 5,
                                   generator.nextSigned() * 5)

        // Expected COLMAP pose, built from the columns of the Apple rotation.
        // Apple camera-to-world columns come from rotating the basis vectors.
        let appleColumns: [SIMD3<Double>] = basis.map { rotate($0, byXYZW: appleQuaternion) }
        // Flip Y and Z (the second and third camera axes) for the COLMAP / OpenCV camera.
        let cvColumns: [SIMD3<Double>] = [
            appleColumns[0],
            flip.y * appleColumns[1],
            flip.z * appleColumns[2],
        ]
        // World-to-camera is the transpose: row i of it is cvColumns[i].
        let expectedTranslation = SIMD3<Double>(-dot(cvColumns[0], centre),
                                                -dot(cvColumns[1], centre),
                                                -dot(cvColumns[2], centre))

        let result = ApplePoseConversion.colmapPose(centre: centre, quaternionXYZW: appleQuaternion)

        // Compare the rotation by applying both to probe vectors, which is sign-independent.
        for probe in probes + basis {
            let expectedImage = SIMD3<Double>(dot(cvColumns[0], probe),
                                              dot(cvColumns[1], probe),
                                              dot(cvColumns[2], probe))
            let actualImage = rotate(probe, byXYZW: SIMD4<Double>(result.rotation.y, result.rotation.z,
                                                                  result.rotation.w, result.rotation.x))
            #expect(close(actualImage, expectedImage, tolerance: 1e-9))
        }
        #expect(close(result.translation, expectedTranslation, tolerance: 1e-9))
    }
}

@Test func colmapPoseOutputIsUnitWithNonNegativeW() {
    var generator = ConversionSeededGenerator(state: 42)
    for _ in 0..<500 {
        var raw = SIMD4<Double>(0, 0, 0, 0)
        repeat {
            raw = SIMD4<Double>(generator.nextSigned(), generator.nextSigned(),
                                generator.nextSigned(), generator.nextSigned())
        } while (raw.x * raw.x + raw.y * raw.y + raw.z * raw.z + raw.w * raw.w) < 0.05
        let result = ApplePoseConversion.colmapPose(centre: SIMD3<Double>(0, 0, 0),
                                                    quaternionXYZW: raw)
        // The rotation is (w, x, y, z), so the scalar part is `.x`.
        let q = result.rotation
        let length: Double = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).squareRoot()
        #expect(abs(length - 1) < 1e-12)
        #expect(q.x >= 0)
    }
}

@Test func colmapPoseNormalizesTheInputQuaternion() {
    let unit = ApplePoseConversion.colmapPose(centre: SIMD3<Double>(1, -2, 0.5),
                                              quaternionXYZW: SIMD4<Double>(0.1, 0.4, -0.2, 0.9))
    let scaled = ApplePoseConversion.colmapPose(centre: SIMD3<Double>(1, -2, 0.5),
                                                quaternionXYZW: SIMD4<Double>(0.2, 0.8, -0.4, 1.8))
    let dotValue: Double = unit.rotation.x * scaled.rotation.x
        + unit.rotation.y * scaled.rotation.y
        + unit.rotation.z * scaled.rotation.z
        + unit.rotation.w * scaled.rotation.w
    #expect(abs(dotValue) > 1 - 1e-12)
    #expect(close(unit.translation, scaled.translation))
}

@Test func colmapPoseHandlesRotationsOnEveryShepperdBranch() {
    // Half turns about each axis and the identity exercise the trace and diagonal branches.
    let cases: [SIMD4<Double>] = [
        SIMD4<Double>(0, 0, 0, 1),
        SIMD4<Double>(1, 0, 0, 0),
        SIMD4<Double>(0, 1, 0, 0),
        SIMD4<Double>(0, 0, 1, 0),
    ]
    for quaternion in cases {
        let result = ApplePoseConversion.colmapPose(centre: SIMD3<Double>(0, 0, 0),
                                                    quaternionXYZW: quaternion)
        // The rotation is (w, x, y, z), so the scalar part is `.x`.
        let q = result.rotation
        let length: Double = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).squareRoot()
        #expect(abs(length - 1) < 1e-12)
        #expect(q.x >= 0)
    }
}
