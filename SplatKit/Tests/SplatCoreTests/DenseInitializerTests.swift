import Foundation
import Testing
import SplatCore

@Suite struct DenseInitializerTests {

    // MARK: - Fixtures

    /// Camera looking straight down at the plane z = 0 from `center` (world z up).
    private func lookDownCamera(center: SIMD3<Float>, width: Int, height: Int, focal: Float) -> Camera {
        let rotation = Matrix3(rows: [SIMD3<Float>(1, 0, 0),
                                      SIMD3<Float>(0, -1, 0),
                                      SIMD3<Float>(0, 0, -1)])
        let translation: SIMD3<Float> = -(rotation * center)
        return Camera(width: width, height: height,
                      fx: focal, fy: focal,
                      cx: Float(width) * 0.5, cy: Float(height) * 0.5,
                      rotation: rotation, translation: translation)
    }

    /// Analytic camera-z depth of the plane z = 0 seen through pixel (x, y).
    private func planeDepth(camera: Camera, x: Int, y: Int) -> Float {
        let u: Float = Float(x) + 0.5
        let v: Float = Float(y) + 0.5
        let direction = SIMD3<Float>((u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, 1)
        let worldDirection: SIMD3<Float> = camera.rotation.transposed * direction
        let center = camera.center
        return -center.z / worldDirection.z
    }

    private func imageColor(x: Int, y: Int, width: Int, height: Int) -> SIMD3<Float> {
        SIMD3<Float>(Float(x) / Float(width), Float(y) / Float(height), 0.5)
    }

    /// A 32×24 view of the plane z = 0 with analytic metric depth.
    private func planeView(name: String, center: SIMD3<Float>) -> TrainingView {
        let width = 32
        let height = 24
        let camera = lookDownCamera(center: center, width: width, height: height, focal: 30)
        var depth: [Float] = []
        var pixels: [SIMD3<Float>] = []
        depth.reserveCapacity(width * height)
        pixels.reserveCapacity(width * height)
        for y in 0..<height {
            for x in 0..<width {
                depth.append(planeDepth(camera: camera, x: x, y: y))
                pixels.append(imageColor(x: x, y: y, width: width, height: height))
            }
        }
        let prior = DepthPrior(width: width, height: height, correlationTarget: depth,
                               domain: .depth, metricDepth: depth)
        return TrainingView(name: name, camera: camera,
                            image: RGBImage(width: width, height: height, pixels: pixels),
                            depth: prior)
    }

    private func settings(maxPoints: Int = 150_000) -> DenseInitSettings {
        var value = DenseInitSettings()
        value.maxPoints = maxPoints
        return value
    }

    // MARK: - Geometry

    @Test func backProjectInvertsProject() {
        let rotation = Matrix3.rotation(quaternion: SIMD4<Float>(0.9, 0.2, -0.3, 0.1))
        let center = SIMD3<Float>(0.4, -0.7, 2.5)
        let camera = Camera(width: 32, height: 24, fx: 28, fy: 27, cx: 15.5, cy: 11.0,
                            rotation: rotation, translation: -(rotation * center))
        var generator = SeededGenerator(seed: 7)
        for _ in 0..<200 {
            let x = Int.random(in: 0..<32, using: &generator)
            let y = Int.random(in: 0..<24, using: &generator)
            let depth = Float.random(in: 1...10, using: &generator)
            let world = DepthGeometry.backProject(x: x, y: y, depth: depth, camera: camera)
            guard let projected = DepthGeometry.project(world, camera: camera, near: 0) else {
                Issue.record("projection of a back-projected point returned nil")
                return
            }
            #expect(abs(projected.u - (Float(x) + 0.5)) < 1e-4)
            #expect(abs(projected.v - (Float(y) + 0.5)) < 1e-4)
            #expect(abs(projected.z - depth) < 1e-4 * depth)
        }
    }

    @Test func projectRejectsPointsBehindNearPlane() {
        let camera = lookDownCamera(center: SIMD3<Float>(0, 0, 4), width: 32, height: 24, focal: 30)
        // Above the camera: camera-z is negative.
        #expect(DepthGeometry.project(SIMD3<Float>(0, 0, 10), camera: camera, near: 0) == nil)
        #expect(DepthGeometry.project(SIMD3<Float>(0, 0, 0), camera: camera, near: 0) != nil)
    }

    // MARK: - Dense initialization

    @Test func pointsLieOnTheSurface() {
        let dataset = TrainingDataset(views: [
            planeView(name: "a", center: SIMD3<Float>(-0.3, 0, 4)),
            planeView(name: "b", center: SIMD3<Float>(0.3, 0, 4)),
            planeView(name: "c", center: SIMD3<Float>(0, 0.3, 4)),
        ])
        guard case let (cloud, report)? = DenseInitializer.pointCloud(dataset, settings: settings(),
                                                                      maxCount: 1_000_000, seed: 1) else {
            Issue.record("expected a dense cloud")
            return
        }
        #expect(report.final > 0)
        #expect(cloud.positions.count == report.final)
        for position in cloud.positions {
            #expect(abs(position.z) < 1e-3)
        }
    }

    @Test func overlappingViewsAreDeduplicated() {
        let center = SIMD3<Float>(0, 0, 4)
        let single = TrainingDataset(views: [planeView(name: "a", center: center)])
        let double = TrainingDataset(views: [planeView(name: "a", center: center),
                                             planeView(name: "b", center: center)])
        // Large caps keep the stride at 1 for both runs, so the counts are directly comparable.
        guard case let (_, singleReport)? = DenseInitializer.pointCloud(single, settings: settings(),
                                                                        maxCount: 1_000_000, seed: 1),
              case let (_, doubleReport)? = DenseInitializer.pointCloud(double, settings: settings(),
                                                                        maxCount: 1_000_000, seed: 1) else {
            Issue.record("expected dense clouds")
            return
        }
        #expect(singleReport.afterDedup > 0)
        let ratio = Double(doubleReport.afterDedup) / Double(singleReport.afterDedup)
        #expect(abs(ratio - 1) <= 0.05)
    }

    @Test func freeSpaceViolationsAreRemoved() {
        // View 0 carries a floater: a patch of pixels that claims a surface at depth 2, in front
        // of the plane. View 1 sees the plane there, so the floater points are inconsistent.
        var floaterView = planeView(name: "floater", center: SIMD3<Float>(-0.3, 0, 4))
        var depth = floaterView.depth!.metricDepth!
        for y in 9..<15 {
            for x in 12..<20 {
                depth[y * 32 + x] = 2.0
            }
        }
        floaterView.depth = DepthPrior(width: 32, height: 24, correlationTarget: depth,
                                       domain: .depth, metricDepth: depth)
        let dataset = TrainingDataset(views: [
            floaterView,
            planeView(name: "plain", center: SIMD3<Float>(0.3, 0, 4)),
        ])
        guard case let (cloud, report)? = DenseInitializer.pointCloud(dataset, settings: settings(),
                                                                      maxCount: 1_000_000, seed: 1) else {
            Issue.record("expected a dense cloud")
            return
        }
        #expect(report.rejectedInconsistent > 0)
        // Only the plane survives: no point of the floater patch (height about 2) is kept.
        for position in cloud.positions {
            #expect(abs(position.z) < 1e-3)
        }
    }

    @Test func depthEdgesAreSkipped() {
        var view = planeView(name: "box", center: SIMD3<Float>(0, 0, 4))
        var depth = view.depth!.metricDepth!
        // A "box" in front of the plane: a rectangle at depth 2 with a sharp silhouette.
        for y in 8..<16 {
            for x in 10..<22 {
                depth[y * 32 + x] = 2.0
            }
        }
        view.depth = DepthPrior(width: 32, height: 24, correlationTarget: depth,
                                domain: .depth, metricDepth: depth)
        let dataset = TrainingDataset(views: [view])
        guard case let (cloud, report)? = DenseInitializer.pointCloud(dataset, settings: settings(),
                                                                      maxCount: 1_000_000, seed: 1) else {
            Issue.record("expected a dense cloud")
            return
        }
        // The rectangle boundary (96 − 60 interior pixels) alone gives 36 edge pixels.
        #expect(report.rejectedEdges >= 36)
        // Every kept point must project onto a pixel that is not a depth edge.
        for position in cloud.positions {
            guard let projected = DepthGeometry.project(position, camera: view.camera, near: 0) else {
                Issue.record("kept point projects behind the camera")
                continue
            }
            let x = Int(projected.u.rounded(.down))
            let y = Int(projected.v.rounded(.down))
            #expect(!DepthGeometry.isDepthEdge(depth, width: 32, height: 24, x: x, y: y, ratio: 1.05))
        }
    }

    @Test func capIsRespected() {
        let views = [
            planeView(name: "a", center: SIMD3<Float>(-0.3, 0, 4)),
            planeView(name: "b", center: SIMD3<Float>(0.3, 0, 4)),
        ]
        let dataset = TrainingDataset(views: views)

        guard case let (small, smallReport)? = DenseInitializer.pointCloud(dataset, settings: settings(maxPoints: 500),
                                                                           maxCount: 1_000_000, seed: 1) else {
            Issue.record("expected a dense cloud")
            return
        }
        #expect(small.positions.count <= 500)
        #expect(smallReport.final <= 500)

        // maxCount 600 leaves half of it (300) for the dense cloud.
        guard case let (halved, halvedReport)? = DenseInitializer.pointCloud(dataset, settings: settings(),
                                                                              maxCount: 600, seed: 1) else {
            Issue.record("expected a dense cloud")
            return
        }
        #expect(halved.positions.count <= 300)
        #expect(halvedReport.final <= 300)
    }

    @Test func fallsBackWithoutAlignedViews() {
        let plain = TrainingView(name: "plain",
                                 camera: lookDownCamera(center: SIMD3<Float>(0, 0, 4),
                                                        width: 32, height: 24, focal: 30),
                                 image: RGBImage(width: 32, height: 24,
                                                 pixels: [SIMD3<Float>](repeating: .zero, count: 32 * 24)),
                                 depth: nil)
        let dataset = TrainingDataset(views: [plain])
        var configuration = TrainingConfiguration()
        configuration.seed = 11
        #expect(DenseInitializer.pointCloud(dataset, settings: configuration.denseInit,
                                            maxCount: 1_000, seed: 11) == nil)

        let dense = DenseInitializer.initialize(dataset, configuration: configuration, maxCount: 1_000)
        let phase3 = GaussianInitializer.initialize(dataset, configuration: configuration, maxCount: 1_000)
        #expect(dense.count == phase3.count)
        #expect(dense.positions == phase3.positions)
        #expect(dense.logScales == phase3.logScales)
        #expect(dense.rotations == phase3.rotations)
        #expect(dense.opacityLogits == phase3.opacityLogits)
        #expect(dense.sh == phase3.sh)
    }

    @Test func denseInitIsDeterministic() {
        let dataset = TrainingDataset(views: [
            planeView(name: "a", center: SIMD3<Float>(-0.3, 0, 4)),
            planeView(name: "b", center: SIMD3<Float>(0.3, 0, 4)),
        ])
        // A small cap forces the seeded subset path.
        let first = DenseInitializer.pointCloud(dataset, settings: settings(maxPoints: 200),
                                                maxCount: 1_000_000, seed: 3)
        let second = DenseInitializer.pointCloud(dataset, settings: settings(maxPoints: 200),
                                                 maxCount: 1_000_000, seed: 3)
        #expect(first != nil)
        #expect(first?.0 == second?.0)
        #expect(first?.1 == second?.1)
    }
}
