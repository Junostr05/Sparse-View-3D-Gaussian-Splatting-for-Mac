import Foundation
import Testing
import SplatCore

/// 3 Gaussians, degree 0: #0 valid, #1 has NaN position, #2 has +Inf in sh.
func threeGaussiansWithBadOnes() throws(GaussianCloudError) -> GaussianCloud {
    try GaussianCloud(
        shDegree: 0,
        positions: [0, 0, 0, Float.nan, 0, 0, 0, 0, 0],
        logScales: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        rotations: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        opacityLogits: [0, 0, 0],
        sh: [0, 0, 0, 0, 0, 0, Float.infinity, 0, 0]
    )
}

@Test func quaternionIdentityAndNormalization() {
    #expect(Matrix3.rotation(quaternion: SIMD4(2, 0, 0, 0)) == .identity)
    #expect(Matrix3.rotation(quaternion: .zero) == .identity)
    let r = Matrix3.rotation(quaternion: SIMD4(cos(.pi / 4), 0, 0, sin(.pi / 4))) // 90° about z
    #expect(approxEqual(r * SIMD3(1, 0, 0), SIMD3(0, 1, 0), tol: 1e-6))
}

@Test func cameraCenter() {
    let cam = Camera(width: 64, height: 64, fx: 100, fy: 100, cx: 32, cy: 32,
                     rotation: .identity, translation: SIMD3(0, 0, 5))
    #expect(cam.center == SIMD3(0, 0, -5))
}

@Test func coefficientCounts() {
    #expect((0...3).map { GaussianCloud.coefficientCount(degree: $0) } == [1, 4, 9, 16])
}

@Test func rejectsInconsistentLengths() {
    #expect(throws: GaussianCloudError.inconsistentLengths) {
        try GaussianCloud(shDegree: 0, positions: [0, 0, 0], logScales: [0, 0], rotations: [1, 0, 0, 0],
                          opacityLogits: [0], sh: [0, 0, 0])
    }
    #expect(throws: GaussianCloudError.invalidSHDegree(4)) {
        try GaussianCloud(shDegree: 4, positions: [], logScales: [], rotations: [], opacityLogits: [], sh: [])
    }
}

@Test func removingNonFiniteDropsBadGaussians() throws {
    // 3 Gaussians; #1 has NaN position, #2 has +Inf in sh
    let (cloud, removed) = try threeGaussiansWithBadOnes().removingNonFinite()
    #expect(removed == 2)
    #expect(cloud.count == 1)
    #expect(cloud.positions == [0, 0, 0])
}
