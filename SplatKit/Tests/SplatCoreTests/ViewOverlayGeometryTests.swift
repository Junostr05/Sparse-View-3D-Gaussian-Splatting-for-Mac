import Foundation
import Testing
import SplatCore

/// Camera at the origin looking down +z, with the given image size and focal length (pixels).
private func overlayCamera(width: Int, height: Int, focal: Float) -> Camera {
    Camera(width: width, height: height, fx: focal, fy: focal,
           cx: Float(width) / 2, cy: Float(height) / 2,
           rotation: .identity, translation: SIMD3<Float>(0, 0, 0))
}

private let overlayBounds: Bounds = Bounds(min: SIMD3<Float>(-1, -2, -3), max: SIMD3<Float>(1, 4, 3))

@Test func gridSpacingIsAPowerOfTenNearTheTargetLineCount() {
    #expect(OverlayGeometry.gridSpacing(sceneSize: 15) == 1)
    #expect(OverlayGeometry.gridSpacing(sceneSize: 150) == 10)
    #expect(approxEqual(OverlayGeometry.gridSpacing(sceneSize: 1.5), 0.1, tol: 1e-7))
    #expect(OverlayGeometry.gridSpacing(sceneSize: 0) == 1)
    #expect(OverlayGeometry.gridSpacing(sceneSize: .nan) == 1)
}

@Test func groundGridSitsAtTheLowestBoundsPointAlongTheUpAxis() {
    // Up is -Y: the lowest point of the bounds is the largest y, which is 4.
    let negativeY: [OverlaySegment] = OverlayGeometry.groundGrid(bounds: overlayBounds, upAxis: .negativeY)
    #expect(!negativeY.isEmpty)
    #expect(negativeY.allSatisfy { $0.start.y == 4 && $0.end.y == 4 })

    // Up is +Y: the lowest point is the smallest y, which is -2.
    let positiveY: [OverlaySegment] = OverlayGeometry.groundGrid(bounds: overlayBounds, upAxis: .positiveY)
    #expect(positiveY.allSatisfy { $0.start.y == -2 && $0.end.y == -2 })

    // Up is +Z: the lowest point is the smallest z, which is -3.
    let positiveZ: [OverlaySegment] = OverlayGeometry.groundGrid(bounds: overlayBounds, upAxis: .positiveZ)
    #expect(positiveZ.allSatisfy { $0.start.z == -3 && $0.end.z == -3 })
}

@Test func groundGridWithoutBoundsGoesThroughThePivot() {
    let grid: [OverlaySegment] = OverlayGeometry.groundGrid(bounds: nil, upAxis: .positiveY,
                                                            fallbackPivot: SIMD3<Float>(0, 5, 0))
    #expect(!grid.isEmpty)
    #expect(grid.allSatisfy { $0.start.y == 5 && $0.end.y == 5 })
}

@Test func groundGridSpacingAndAxesFollowTheScene() {
    // Scene extent is 6 along z and reach is 3, so the spacing is 1. The grid keeps at least five lines each side.
    let grid: [OverlaySegment] = OverlayGeometry.groundGrid(bounds: overlayBounds, upAxis: .negativeY)
    let linesAlongX: [OverlaySegment] = grid.filter { $0.start.x != $0.end.x }
    let zPositions: [Float] = linesAlongX.map { $0.start.z }.sorted()
    #expect(zPositions == [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    // The line through the origin along X is the red axis.
    let axisX: [OverlaySegment] = linesAlongX.filter { $0.start.z == 0 }
    #expect(axisX.count == 1)
    #expect(axisX.first?.role == .axisX)
    // Up is -Y, so the second horizontal axis is Z and the line through the origin along Z is blue.
    let linesAlongZ: [OverlaySegment] = grid.filter { $0.start.z != $0.end.z }
    #expect(linesAlongZ.filter { $0.start.x == 0 }.first?.role == .axisZ)
}

@Test func groundGridMarksEveryTenthLineAsMajor() {
    // Reach 100 gives a spacing of 10 and lines at x = ±100 are the tenth lines.
    let bounds: Bounds = Bounds(min: SIMD3<Float>(-100, 0, -100), max: SIMD3<Float>(100, 1, 100))
    let grid: [OverlaySegment] = OverlayGeometry.groundGrid(bounds: bounds, upAxis: .positiveY)
    let linesAlongX: [OverlaySegment] = grid.filter { $0.start.x != $0.end.x }
    let major: [OverlaySegment] = linesAlongX.filter { $0.role == .majorGrid }
    #expect(major.contains { $0.start.z == 100 })
    #expect(major.contains { $0.start.z == -100 })
    #expect(!major.contains { $0.start.z == 5 })
    #expect(linesAlongX.contains { $0.start.z == 10 && $0.role == .minorGrid })
}

@Test func nearPlaneClippingKeepsTheVisiblePartOfASegment() {
    let camera: Camera = overlayCamera(width: 200, height: 100, focal: 100)
    let crossing: OverlaySegment = OverlaySegment(start: SIMD3<Float>(-1, 0, -1),
                                                  end: SIMD3<Float>(1, 0, 3),
                                                  role: .camera(index: 0))
    let projected: [ProjectedOverlaySegment] = OverlayGeometry.project(
        segments: [crossing], camera: camera, viewSize: SIMD2<Float>(200, 100))
    #expect(projected.count == 1)
    guard let line = projected.first else { return }
    // The end at z = 3 projects to u = 100 · 1 / 3 + 100. The start moves to z = 0.01, where x = -0.495.
    #expect(approxEqual(line.end, SIMD2<Float>(100 + 100.0 / 3.0, 50), tol: 0.01))
    #expect(approxEqual(line.start.x, 100 * (-0.495) / 0.01 + 100, tol: 1))
    #expect(line.alpha == 1)
}

@Test func segmentBehindTheCameraIsDropped() {
    let camera: Camera = overlayCamera(width: 200, height: 100, focal: 100)
    let behind: OverlaySegment = OverlaySegment(start: SIMD3<Float>(0, 0, -2),
                                                end: SIMD3<Float>(0, 0, -1),
                                                role: .axisY)
    let projected: [ProjectedOverlaySegment] = OverlayGeometry.project(
        segments: [behind], camera: camera, viewSize: SIMD2<Float>(200, 100))
    #expect(projected.isEmpty)
}

@Test func gridLinesFadeWithDistanceFromTheEye() {
    let camera: Camera = overlayCamera(width: 200, height: 100, focal: 100)
    let near: OverlaySegment = OverlaySegment(start: SIMD3<Float>(-1, 0, 2), end: SIMD3<Float>(1, 0, 2),
                                              role: .minorGrid)
    let far: OverlaySegment = OverlaySegment(start: SIMD3<Float>(-1, 0, 50), end: SIMD3<Float>(1, 0, 50),
                                             role: .minorGrid)
    let axis: OverlaySegment = OverlaySegment(start: SIMD3<Float>(-1, 0, 50), end: SIMD3<Float>(1, 0, 50),
                                              role: .axisX)
    let projected: [ProjectedOverlaySegment] = OverlayGeometry.project(
        segments: [near, far, axis], camera: camera, viewSize: SIMD2<Float>(200, 100), fadeDistance: 20)
    #expect(projected.count == 3)
    #expect(projected[0].alpha > projected[1].alpha)
    #expect(projected[1].alpha == 0.1)
    #expect(projected[2].alpha == 1)
}

@Test func frustumApexAndCornersForAnIdentityCamera() {
    let camera: Camera = overlayCamera(width: 100, height: 100, focal: 50)
    let size: Float = 2
    let segments: [OverlaySegment] = OverlayGeometry.cameraFrusta(cameras: [camera], size: size)
    // Four rays, four rectangle edges and one up tick.
    #expect(segments.count == 9)
    #expect(segments.allSatisfy { $0.role == .camera(index: 0) })
    // The field of view is 90 degrees here, so the rectangle corners sit at (±size, ±size, size).
    let corners: [SIMD3<Float>] = [
        SIMD3<Float>(-2, -2, 2), SIMD3<Float>(2, -2, 2), SIMD3<Float>(2, 2, 2), SIMD3<Float>(-2, 2, 2),
    ]
    for corner in corners {
        #expect(segments.contains { approxEqual($0.start, .zero, tol: 1e-6) && approxEqual($0.end, corner, tol: 1e-5) })
    }
}

@Test func frustumRectangleFollowsTheAspectRatio() {
    // Width 200 and height 100 with focal 100 gives half sizes of `size` and `size / 2`.
    let camera: Camera = overlayCamera(width: 200, height: 100, focal: 100)
    let segments: [OverlaySegment] = OverlayGeometry.cameraFrusta(cameras: [camera], size: 4)
    let corner: SIMD3<Float> = SIMD3<Float>(4, 2, 4)
    #expect(segments.contains { approxEqual($0.end, corner, tol: 1e-5) })
}

@Test func frustumIndexFollowsTheCameraOrder() {
    let first: Camera = overlayCamera(width: 100, height: 100, focal: 50)
    let second: Camera = overlayCamera(width: 100, height: 100, focal: 25)
    let segments: [OverlaySegment] = OverlayGeometry.cameraFrusta(cameras: [first, second], size: 1)
    #expect(segments.count == 18)
    #expect(segments.last?.role == .camera(index: 1))
}

@Test func gizmoDirectionsFollowTheCameraRotation() {
    let identity: [OverlayGizmoAxis] = OverlayGeometry.axesGizmo(rotation: .identity, length: 30)
    #expect(identity.count == 3)
    #expect(approxEqual(identity[0].offset, SIMD2<Float>(30, 0), tol: 1e-6))
    #expect(approxEqual(identity[1].offset, SIMD2<Float>(0, 30), tol: 1e-6))
    #expect(approxEqual(identity[2].offset, SIMD2<Float>(0, 0), tol: 1e-6))
    #expect(identity[2].depth == 1)

    // A view along +x rotated so that world z points to the right of the screen and world x points away.
    let rotated: Matrix3 = Matrix3(rows: [SIMD3<Float>(0, 0, 1),
                                          SIMD3<Float>(0, 1, 0),
                                          SIMD3<Float>(-1, 0, 0)])
    let arms: [OverlayGizmoAxis] = OverlayGeometry.axesGizmo(rotation: rotated, length: 30)
    #expect(approxEqual(arms[2].offset, SIMD2<Float>(30, 0), tol: 1e-6))
    #expect(approxEqual(arms[0].offset, SIMD2<Float>(0, 0), tol: 1e-6))
    #expect(arms[0].depth == -1)
}

@Test func projectPointRejectsPointsAtOrBehindTheNearPlane() {
    let camera: Camera = overlayCamera(width: 200, height: 100, focal: 100)
    #expect(OverlayGeometry.projectPoint(SIMD3<Float>(0, 0, -1), camera: camera) == nil)
    let visible: SIMD2<Float>? = OverlayGeometry.projectPoint(SIMD3<Float>(1, 0, 2), camera: camera)
    #expect(visible.map { approxEqual($0, SIMD2<Float>(150, 50), tol: 1e-4) } == true)
}
