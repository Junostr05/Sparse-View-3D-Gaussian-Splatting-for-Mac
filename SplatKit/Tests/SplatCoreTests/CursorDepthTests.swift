import Testing
import SplatCore

/// Identity camera at the origin looking along +z: a 100 × 100 image with focal length 100.
/// The image centre (50, 50) is the ray along +z.
private func cdCamera() -> Camera {
    Camera(width: 100, height: 100, fx: 100, fy: 100, cx: 50, cy: 50,
           rotation: .identity, translation: .zero)
}

/// A cloud of opaque Gaussians (logit 2) at the given centres, with a per-point opacity logit.
private func cdCloud(_ points: [(SIMD3<Float>, Float)]) throws -> GaussianCloud {
    let n: Int = points.count
    var positions: [Float] = []
    var rotations: [Float] = []
    var logits: [Float] = []
    for (point, logit) in points {
        positions += [point.x, point.y, point.z]
        rotations += [1, 0, 0, 0]
        logits.append(logit)
    }
    return try GaussianCloud(shDegree: 0,
                             positions: positions,
                             logScales: [Float](repeating: -3, count: 3 * n),
                             rotations: rotations,
                             opacityLogits: logits,
                             sh: [Float](repeating: 0, count: 3 * n))
}

/// A small plane patch at depth `z` around the image centre, within a few tenths of a pixel.
private func cdPlane(z: Float, logit: Float = 2) -> [(SIMD3<Float>, Float)] {
    var points: [(SIMD3<Float>, Float)] = []
    for i in 0..<4 {
        for j in 0..<4 {
            let x: Float = 0.005 * Float(i)
            let y: Float = 0.005 * Float(j)
            points.append((SIMD3<Float>(x, y, z), logit))
        }
    }
    return points
}

@Suite struct CursorDepthTests {
    @Test func picksThePlaneDepthAndWorldPoint() throws {
        let cloud = try cdCloud(cdPlane(z: 5))
        let pick = CursorDepth.pick(cloud: cloud, camera: cdCamera(), pixel: SIMD2(50, 50))
        let found = try #require(pick)
        #expect(abs(found.depth - 5) < 1e-5)
        // The ray through the image centre is the +z axis, so the world point is (0, 0, depth).
        #expect(abs(found.worldPoint.x) < 1e-5)
        #expect(abs(found.worldPoint.y) < 1e-5)
        #expect(abs(found.worldPoint.z - 5) < 1e-5)
    }

    @Test func floatersInFrontDoNotPullTheDepthForward() throws {
        // 5 floaters at depth 2 among 16 plane points at depth 5: the 25th percentile of 21 is the plane.
        let cloud = try cdCloud(cdPlane(z: 5) + [(SIMD3<Float>(0, 0, 2), 2),
                                                 (SIMD3<Float>(0, 0, 2), 2),
                                                 (SIMD3<Float>(0, 0, 2), 2),
                                                 (SIMD3<Float>(0, 0, 2), 2),
                                                 (SIMD3<Float>(0, 0, 2), 2)])
        let found = try #require(CursorDepth.pick(cloud: cloud, camera: cdCamera(), pixel: SIMD2(50, 50)))
        #expect(abs(found.depth - 5) < 1e-5)
    }

    @Test func pointsBehindTheSurfaceDoNotChangeTheDepth() throws {
        let cloud = try cdCloud(cdPlane(z: 5) + (0..<10).map { _ in (SIMD3<Float>(0, 0, 9), Float(2)) })
        let found = try #require(CursorDepth.pick(cloud: cloud, camera: cdCamera(), pixel: SIMD2(50, 50)))
        #expect(abs(found.depth - 5) < 1e-5)
    }

    @Test func transparentGaussiansAreIgnored() throws {
        // Sigmoid(-3) is about 0.047, below the 0.3 minimum: these floaters must not count.
        let cloud = try cdCloud(cdPlane(z: 5) + (0..<10).map { _ in (SIMD3<Float>(0, 0, 2), Float(-3)) })
        let found = try #require(CursorDepth.pick(cloud: cloud, camera: cdCamera(), pixel: SIMD2(50, 50)))
        #expect(abs(found.depth - 5) < 1e-5)
    }

    @Test func emptyRegionGivesNil() throws {
        let cloud = try cdCloud(cdPlane(z: 5))
        #expect(CursorDepth.pick(cloud: cloud, camera: cdCamera(), pixel: SIMD2(10, 10)) == nil)
    }

    @Test func fewerThanThreeCandidatesGiveNil() throws {
        let cloud = try cdCloud([(SIMD3<Float>(0, 0, 5), 2), (SIMD3<Float>(0.001, 0, 5), 2)])
        #expect(CursorDepth.pick(cloud: cloud, camera: cdCamera(), pixel: SIMD2(50, 50)) == nil)
    }

    @Test func emptyCloudGivesNil() throws {
        let cloud = try cdCloud([])
        #expect(CursorDepth.pick(cloud: cloud, camera: cdCamera(), pixel: SIMD2(50, 50)) == nil)
    }

    @Test func pointsBehindTheCameraAreIgnored() throws {
        let cloud = try cdCloud((0..<5).map { _ in (SIMD3<Float>(0, 0, -5), Float(2)) })
        #expect(CursorDepth.pick(cloud: cloud, camera: cdCamera(), pixel: SIMD2(50, 50)) == nil)
    }

    @Test func pointsOutsideTheRadiusAreIgnored() throws {
        // x = 1 at depth 5 projects 20 pixels away from the centre: outside the 12 pixel radius.
        let far: [(SIMD3<Float>, Float)] = (0..<5).map { _ in (SIMD3<Float>(1, 0, 5), Float(2)) }
        let cloud = try cdCloud(cdPlane(z: 5) + far)
        let found = try #require(CursorDepth.pick(cloud: cloud, camera: cdCamera(), pixel: SIMD2(50, 50)))
        #expect(abs(found.depth - 5) < 1e-5)
        // With only the far points, nothing is picked.
        let onlyFar = try cdCloud(far)
        #expect(CursorDepth.pick(cloud: onlyFar, camera: cdCamera(), pixel: SIMD2(50, 50)) == nil)
    }
}
