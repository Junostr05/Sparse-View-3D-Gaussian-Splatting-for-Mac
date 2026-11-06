import Foundation
import Testing
import SplatCore

@Suite struct PseudoViewTests {

    // MARK: - Fixtures

    private func pvNormalized(_ v: SIMD3<Float>) -> SIMD3<Float> {
        let length: Float = (v.x * v.x + v.y * v.y + v.z * v.z).squareRoot()
        return v / length
    }

    private func pvCross(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> SIMD3<Float> {
        let x: Float = a.y * b.z - a.z * b.y
        let y: Float = a.z * b.x - a.x * b.z
        let z: Float = a.x * b.y - a.y * b.x
        return SIMD3<Float>(x, y, z)
    }

    /// Camera at `center` looking at `target` in OpenCV convention (image y points along world −y
    /// by construction of the down vector below). The target must not lie straight above or below.
    private func lookAtCamera(center: SIMD3<Float>, target: SIMD3<Float>,
                              width: Int = 32, height: Int = 24, focal: Float = 20) -> Camera {
        let forward: SIMD3<Float> = pvNormalized(target - center)
        let down: SIMD3<Float> = SIMD3<Float>(0, 1, 0)
        let right: SIMD3<Float> = pvNormalized(pvCross(down, forward))
        let imageDown: SIMD3<Float> = pvCross(forward, right)
        let rotation = Matrix3(rows: [right, imageDown, forward])
        let translation: SIMD3<Float> = -(rotation * center)
        return Camera(width: width, height: height,
                      fx: focal, fy: focal,
                      cx: Float(width) * 0.5, cy: Float(height) * 0.5,
                      rotation: rotation, translation: translation)
    }

    /// Analytic camera-z depth of the plane z = 0 seen through pixel (x, y). The camera must be
    /// on the z < 0 side and see the plane.
    private func planeDepth(camera: Camera, x: Int, y: Int) -> Float {
        let u: Float = Float(x) + 0.5
        let v: Float = Float(y) + 0.5
        let direction = SIMD3<Float>((u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, 1)
        let worldDirection: SIMD3<Float> = camera.rotation.transposed * direction
        return -camera.center.z / worldDirection.z
    }

    private func blackImage(width: Int, height: Int) -> RGBImage {
        RGBImage(width: width, height: height,
                 pixels: [SIMD3<Float>](repeating: SIMD3<Float>(0, 0, 0), count: width * height))
    }

    /// View with exact metric depth of the plane z = 0.
    private func planeView(name: String, center: SIMD3<Float>, target: SIMD3<Float>) -> TrainingView {
        let camera = lookAtCamera(center: center, target: target)
        var depth: [Float] = []
        for y in 0..<camera.height {
            for x in 0..<camera.width {
                depth.append(planeDepth(camera: camera, x: x, y: y))
            }
        }
        let prior = DepthPrior(width: camera.width, height: camera.height, correlationTarget: depth,
                               domain: .depth, metricDepth: depth, isMeasuredMetric: true)
        return TrainingView(name: name, camera: camera,
                            image: blackImage(width: camera.width, height: camera.height), depth: prior)
    }

    /// View whose metric depth is given per pixel by `valueAt`.
    private func depthView(name: String, camera: Camera, valueAt: (Int, Int) -> Float) -> TrainingView {
        var depth: [Float] = []
        for y in 0..<camera.height {
            for x in 0..<camera.width {
                depth.append(valueAt(x, y))
            }
        }
        let prior = DepthPrior(width: camera.width, height: camera.height, correlationTarget: depth,
                               domain: .depth, metricDepth: depth, isMeasuredMetric: true)
        return TrainingView(name: name, camera: camera,
                            image: blackImage(width: camera.width, height: camera.height), depth: prior)
    }

    private func isLowCoverage(_ rejection: PseudoViewRejection) -> Bool {
        if case .lowCoverage = rejection { return true }
        return false
    }

    private func approxMatrix(_ a: Matrix3, _ b: Matrix3, tol: Float) -> Bool {
        approxEqual(a.c0, b.c0, tol: tol) && approxEqual(a.c1, b.c1, tol: tol) && approxEqual(a.c2, b.c2, tol: tol)
    }

    // MARK: - Pairs

    @Test func pairsAreNearestNeighboursAndUnique() {
        let azimuths: [Double] = [-30, -20, 0, 25]
        var views: [TrainingView] = []
        for (index, degrees) in azimuths.enumerated() {
            let radians: Double = degrees * Double.pi / 180
            let center = SIMD3<Float>(Float(3.5 * sin(radians)), 1, Float(3.5 * cos(radians)))
            let camera = lookAtCamera(center: center, target: SIMD3<Float>(0, 0, 0))
            views.append(depthView(name: "view_\(index)", camera: camera) { _, _ in 5 })
        }
        let found = PseudoViews.pairs(views)
        #expect(found == [SIMD2<Int>(0, 1), SIMD2<Int>(1, 2), SIMD2<Int>(2, 3)])
    }

    @Test func viewsWithoutMetricDepthProduceNoPseudoViews() {
        let camera = lookAtCamera(center: SIMD3<Float>(-1, 0, -5), target: SIMD3<Float>(-1, 0, 0))
        let other = lookAtCamera(center: SIMD3<Float>(1, 0, -5), target: SIMD3<Float>(1, 0, 0))
        let image = blackImage(width: 32, height: 24)
        let noDepth = TrainingView(name: "a", camera: camera, image: image, depth: nil)
        let inverseOnly = TrainingView(
            name: "b", camera: other, image: image,
            depth: DepthPrior(width: 32, height: 24,
                              correlationTarget: [Float](repeating: 0.5, count: 32 * 24),
                              domain: .inverseDepth))
        let views = [noDepth, inverseOnly]
        #expect(PseudoViews.pairs(views).isEmpty)
        let result = PseudoViews.make(views, settings: PseudoViewSettings(), near: 0.01)
        #expect(result.views.isEmpty)
        #expect(result.rejected.isEmpty)
    }

    // MARK: - Interpolation

    @Test func interpolationEndpointsMatchParents() {
        let a = lookAtCamera(center: SIMD3<Float>(-2, 1, -4), target: SIMD3<Float>(0, 0, 0))
        let b = lookAtCamera(center: SIMD3<Float>(2, 0.5, -3), target: SIMD3<Float>(0, 0, 0))
        let start = PseudoViews.interpolate(a, b, t: 0, resolutionDivisor: 1)
        #expect(approxMatrix(start.rotation, a.rotation, tol: 1e-5))
        #expect(approxEqual(start.center, a.center, tol: 1e-5))
        let end = PseudoViews.interpolate(a, b, t: 1, resolutionDivisor: 1)
        #expect(approxMatrix(end.rotation, b.rotation, tol: 1e-5))
        #expect(approxEqual(end.center, b.center, tol: 1e-5))
    }

    @Test func interpolatedCentreAndOrientationAreMidway() {
        // Level cameras (same height) looking at the origin: the rotation slerp midpoint is the
        // horizontal bisector, so its optical axis runs through the origin. Cameras at different
        // heights do not satisfy this, because slerp of rotations is not the chord bisector there.
        let a = lookAtCamera(center: SIMD3<Float>(-2, 0, -4), target: SIMD3<Float>(0, 0, 0))
        let b = lookAtCamera(center: SIMD3<Float>(2, 0, -4), target: SIMD3<Float>(0, 0, 0))
        let mid = PseudoViews.interpolate(a, b, t: 0.5, resolutionDivisor: 1)
        #expect(approxEqual(mid.center, SIMD3<Float>(0, 0, -4), tol: 1e-5))
        // The optical axis (third row of R) must point at the origin from the midway centre.
        let axis = SIMD3<Float>(mid.rotation.c0.z, mid.rotation.c1.z, mid.rotation.c2.z)
        let toOrigin: SIMD3<Float> = pvNormalized(-mid.center)
        let dot: Float = axis.x * toOrigin.x + axis.y * toOrigin.y + axis.z * toOrigin.z
        let angleDegrees: Double = Double(acos(min(1, Double(dot)))) * 180 / Double.pi
        #expect(angleDegrees < 0.5)
    }

    @Test func interpolatedIntrinsicsAreHalvedWithDivisor() {
        let a = lookAtCamera(center: SIMD3<Float>(-1, 0, -5), target: SIMD3<Float>(0, 0, 0))
        let b = lookAtCamera(center: SIMD3<Float>(1, 0, -5), target: SIMD3<Float>(0, 0, 0))
        let half = PseudoViews.interpolate(a, b, t: 0.5, resolutionDivisor: 2)
        #expect(half.width == 16)
        #expect(half.height == 12)
        #expect(abs(half.fx - 10) < 1e-5)
        #expect(abs(half.cx - 8) < 1e-5)
    }

    // MARK: - Warping

    @Test func warpedDepthReproducesPlane() {
        let parentA = planeView(name: "a", center: SIMD3<Float>(-0.5, 0, -5), target: SIMD3<Float>(0, 0, 0))
        let parentB = planeView(name: "b", center: SIMD3<Float>(0.5, 0, -5), target: SIMD3<Float>(0, 0, 0))
        let mid = lookAtCamera(center: SIMD3<Float>(0, 0, -5), target: SIMD3<Float>(0, 0, 0))
        let warped = PseudoViews.warpedDepth(into: mid, from: [parentA, parentB], near: 0.01)
        #expect(warped.count == mid.width * mid.height)
        var valid = 0
        for y in 0..<mid.height {
            for x in 0..<mid.width {
                let value: Float = warped[y * mid.width + x]
                guard value > 0 else { continue }
                valid += 1
                let analytic: Float = planeDepth(camera: mid, x: x, y: y)
                #expect(abs(value - analytic) <= 1e-3 * analytic)
            }
        }
        #expect(valid > mid.width * mid.height / 2)
    }

    // MARK: - Rejection and acceptance

    @Test func rejectsWideAnglePairs() {
        // Optical axes 90° apart: a looks along +z, b along +x.
        let a = depthView(name: "a",
                          camera: lookAtCamera(center: SIMD3<Float>(0, 0, -5), target: SIMD3<Float>(0, 0, 0))) { _, _ in 5 }
        let b = depthView(name: "b",
                          camera: lookAtCamera(center: SIMD3<Float>(5, 0, 0), target: SIMD3<Float>(10, 0, 0))) { _, _ in 5 }
        let settings = PseudoViewSettings()
        let result = PseudoViews.make([a, b], settings: settings, near: 0.01)
        #expect(result.views.isEmpty)
        #expect(result.rejected == [PseudoViewRejection](repeating: .wideAngle, count: settings.perPair))
    }

    @Test func rejectsCameraInsideGeometry() {
        // Both parents see a near surface (depth 1) on the right-hand columns and a far wall (depth 10)
        // elsewhere. The warped midway view is then dominated by the wall, but its 5th percentile
        // falls below 0.2 × the parents' median depth.
        let parentA = depthView(name: "a",
                                camera: lookAtCamera(center: SIMD3<Float>(-1, 0, -5), target: SIMD3<Float>(-1, 0, 0))) { x, _ in
            x >= 20 ? 1 : 10
        }
        let parentB = depthView(name: "b",
                                camera: lookAtCamera(center: SIMD3<Float>(1, 0, -5), target: SIMD3<Float>(1, 0, 0))) { x, _ in
            x >= 20 ? 1 : 10
        }
        let result = PseudoViews.make([parentA, parentB], settings: PseudoViewSettings(), near: 0.01)
        #expect(result.views.isEmpty)
        #expect(result.rejected.count == PseudoViewSettings().perPair)
        #expect(result.rejected.allSatisfy { $0 == .tooClose })
    }

    @Test func coverageBelowThresholdIsRejected() {
        // Metric depth only in a 4×4 corner: the warped midway view is almost empty.
        let parentA = depthView(name: "a",
                                camera: lookAtCamera(center: SIMD3<Float>(-1, 0, -5), target: SIMD3<Float>(-1, 0, 0))) { x, y in
            (x < 4 && y < 4) ? 5 : 0
        }
        let parentB = depthView(name: "b",
                                camera: lookAtCamera(center: SIMD3<Float>(1, 0, -5), target: SIMD3<Float>(1, 0, 0))) { x, y in
            (x < 4 && y < 4) ? 5 : 0
        }
        let settings = PseudoViewSettings()
        let result = PseudoViews.make([parentA, parentB], settings: settings, near: 0.01)
        #expect(result.views.isEmpty)
        #expect(result.rejected.count == settings.perPair)
        #expect(result.rejected.allSatisfy { isLowCoverage($0) })
    }

    @Test func acceptedPairProducesPseudoViews() {
        // Happy path: two exact planar views 1 unit apart give three half-resolution pseudo-views.
        let parentA = planeView(name: "a", center: SIMD3<Float>(-0.5, 0, -5), target: SIMD3<Float>(0, 0, 0))
        let parentB = planeView(name: "b", center: SIMD3<Float>(0.5, 0, -5), target: SIMD3<Float>(0, 0, 0))
        let result = PseudoViews.make([parentA, parentB], settings: PseudoViewSettings(), near: 0.01)
        #expect(result.rejected.isEmpty)
        #expect(result.views.count == 3)
        let ts: [Float] = result.views.map { $0.t }
        #expect(ts == [0.25, 0.5, 0.75])
        for view in result.views {
            #expect(view.parents == SIMD2<Int>(0, 1))
            #expect(view.camera.width == 16)
            #expect(view.camera.height == 12)
            #expect(view.prior.domain == .depth)
            #expect(view.prior.isMeasuredMetric == false)
            #expect(view.prior.metricDepth?.count == 16 * 12)
            #expect(view.prior.correlationTarget == view.prior.metricDepth)
        }
    }

    // MARK: - Index

    @Test func indexIsDeterministicAndInRange() {
        for iteration in 0..<1_000 {
            let first = PseudoViews.index(iteration: iteration, count: 7, seed: 42)
            let second = PseudoViews.index(iteration: iteration, count: 7, seed: 42)
            #expect(first == second)
            #expect(first >= 0 && first < 7)
        }
        #expect(PseudoViews.index(iteration: 5, count: 1, seed: 1) == 0)
    }
}
