import Foundation
import Testing
import SplatCore

/// Projects a world point with OpenCV convention: `p_cam = R·p_world + t`, then pinhole.
/// Deliberately independent of any library projection helper.
private func project(_ world: SIMD3<Float>, _ cam: Camera) -> SIMD3<Float> {
    let p = cam.rotation * world + cam.translation
    return SIMD3(cam.fx * p.x / p.z + cam.cx, cam.fy * p.y / p.z + cam.cy, p.z)
}

private func makeCloud(_ points: [SIMD3<Float>]) throws -> GaussianCloud {
    let n = points.count
    var positions: [Float] = []
    positions.reserveCapacity(3 * n)
    for p in points {
        positions += [p.x, p.y, p.z]
    }
    return try GaussianCloud(
        shDegree: 0,
        positions: positions,
        logScales: [Float](repeating: 0, count: 3 * n),
        rotations: (0..<n).flatMap { _ in [Float(1), 0, 0, 0] },
        opacityLogits: [Float](repeating: 0, count: n),
        sh: [Float](repeating: 0, count: n * 3)
    )
}

private let halfPi = Float.pi / 2

@Test func targetProjectsToImageCenter() {
    for upAxis in UpAxis.allCases {
        for yaw: Float in [-1.2, 0, 0.7, 2.5] {
            for pitch: Float in [-1.0, 0, 0.3, 1.2] {
                let orbit = OrbitCamera(target: SIMD3(0.5, -0.25, 2), distance: 7,
                                        yaw: yaw, pitch: pitch, upAxis: upAxis,
                                        minDistance: 0.07, maxDistance: 700)
                let cam = orbit.camera(width: 640, height: 480)
                let p = project(orbit.target, cam)
                #expect(approxEqual(SIMD2(p.x, p.y), SIMD2(320, 240), tol: 1e-3))
                #expect(p.z > 0)
            }
        }
    }
}

@Test func worldUpPointsUpOnScreen() {
    for upAxis in UpAxis.allCases {
        let orbit = OrbitCamera(target: .zero, distance: 5, yaw: 0.4, pitch: 0.3,
                                upAxis: upAxis, minDistance: 0.05, maxDistance: 500)
        let cam = orbit.camera(width: 640, height: 480)
        let up = upAxis.vector
        let base = project(orbit.target, cam)
        let raised = project(orbit.target + up, cam)
        #expect(raised.y < base.y, "\(upAxis) should point up on screen")
    }
}

@Test func pitchIsClamped() {
    var orbit = OrbitCamera(target: .zero, distance: 5, yaw: 0, pitch: 0,
                            upAxis: .negativeY, minDistance: 0.05, maxDistance: 500)
    orbit.orbit(deltaYaw: 0, deltaPitch: 10)
    #expect(approxEqual(orbit.pitch, halfPi - 0.01, tol: 1e-6))
    orbit.orbit(deltaYaw: 0, deltaPitch: -20)
    #expect(approxEqual(orbit.pitch, -(halfPi - 0.01), tol: 1e-6))
}

@Test func zoomIsClamped() {
    var orbit = OrbitCamera(target: .zero, distance: 5, yaw: 0, pitch: 0,
                            upAxis: .negativeY, minDistance: 0.5, maxDistance: 50)
    orbit.zoom(factor: 1e6)
    #expect(orbit.distance == orbit.maxDistance)
    orbit.zoom(factor: 1e-9)
    #expect(orbit.distance == orbit.minDistance)
}

@Test func panKeepsDistance() {
    var orbit = OrbitCamera(target: SIMD3(1, 2, 3), distance: 6, yaw: 0.9, pitch: 0.4,
                            upAxis: .negativeY, minDistance: 0.06, maxDistance: 600)
    let original = orbit
    let beforeTarget = original.target

    // Dragging right by 10 px: the scene follows the cursor, so the old target
    // must now appear exactly 10 px right of the image center.
    orbit.pan(deltaPixels: SIMD2(10, 0), viewportHeight: 480)
    #expect(orbit.distance == original.distance)
    #expect(orbit.target != beforeTarget)
    let afterRight = project(beforeTarget, orbit.camera(width: 640, height: 480))
    #expect(approxEqual(SIMD2(afterRight.x, afterRight.y), SIMD2(330, 240), tol: 1e-2))

    // Dragging down by 10 px: the old target appears 10 px below the center.
    var orbitDown = original
    orbitDown.pan(deltaPixels: SIMD2(0, 10), viewportHeight: 480)
    #expect(orbitDown.distance == original.distance)
    let afterDown = project(beforeTarget, orbitDown.camera(width: 640, height: 480))
    #expect(approxEqual(SIMD2(afterDown.x, afterDown.y), SIMD2(320, 250), tol: 1e-2))
}

@Test func framingContainsBoundsCorners() {
    let bounds = Bounds(min: SIMD3(-1, -2, -3), max: SIMD3(2, 1, 0.5))
    for upAxis in UpAxis.allCases {
        let orbit = OrbitCamera.framing(bounds, upAxis: upAxis)
        #expect(orbit.target == bounds.center)
        let cam = orbit.camera(width: 640, height: 480)
        for i in 0..<8 {
            let corner = SIMD3<Float>(
                i & 1 == 0 ? bounds.min.x : bounds.max.x,
                i & 2 == 0 ? bounds.min.y : bounds.max.y,
                i & 4 == 0 ? bounds.min.z : bounds.max.z
            )
            let p = project(corner, cam)
            #expect(p.z > 0, "corner \(i) must be in front of the camera")
            #expect(p.x >= 0 && p.x < 640, "corner \(i) x=\(p.x) outside image")
            #expect(p.y >= 0 && p.y < 480, "corner \(i) y=\(p.y) outside image")
        }
    }
}

@Test func robustBoundsIgnoresOutliers() throws {
    var points: [SIMD3<Float>] = []
    for i in 0..<1000 {
        points.append(SIMD3(Float(i % 10) / 10, Float((i / 10) % 10) / 10, Float(i / 100) / 10))
    }
    for _ in 0..<5 {
        points.append(SIMD3(repeating: 1e4))
    }
    let cloud = try makeCloud(points)
    let bounds = cloud.robustBounds()
    #expect(bounds.max.x < 2)
    #expect(bounds.max.y < 2)
    #expect(bounds.max.z < 2)
    #expect(bounds.min.x >= 0)
}

@Test func emptyCloudBoundsAreZero() throws {
    let cloud = try makeCloud([])
    #expect(cloud.robustBounds() == Bounds(min: .zero, max: .zero))
}
