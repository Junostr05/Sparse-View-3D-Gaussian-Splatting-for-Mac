import Foundation
import Metal
import Testing
import SplatCore
import SplatMetal
import SplatTraining

// Camera pose refinement, end to end on the GPU trainer: the training cameras are perturbed (0.7° rotation and 1.5 %
// of the scene radius translation) relative to the cameras that rendered the targets. Training from the ground-truth
// cloud with refinement on must lower the final loss and move the cameras towards the ground truth.
@Suite(.serialized, .enabled(if: TrainingTestEnvironment.hasMetal))
struct PoseRefinementTests {

    private struct Scene {
        var groundTruth: GaussianCloud
        var targetCameras: [Camera]
        var targets: [TrainingView]
        var perturbed: [Camera]
        var radius: Float
    }

    private func axisAngle(_ axis: SIMD3<Float>, _ angle: Float) -> SIMD4<Float> {
        let half = angle / 2
        return SIMD4(cos(half), sin(half) * axis.x, sin(half) * axis.y, sin(half) * axis.z)
    }

    private func unitVector(_ rng: inout SplitMix64) -> SIMD3<Float> {
        var v = SIMD3<Float>(Float.random(in: -1...1, using: &rng),
                             Float.random(in: -1...1, using: &rng),
                             Float.random(in: -1...1, using: &rng))
        while (v * v).sum() < 1e-4 {
            v = SIMD3(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                      Float.random(in: -1...1, using: &rng))
        }
        return v / (v * v).sum().squareRoot()
    }

    /// Four 48×48 views with small rotations and offsets around the origin, targets rendered from them.
    private func makeScene(context: MetalContext) throws -> Scene {
        let groundTruth = try makeRandomCloud(count: 120, seed: 7, shDegree: 1)
        let rotations: [Float] = [-0.12, -0.04, 0.05, 0.14]
        let shifts: [SIMD3<Float>] = [SIMD3(-0.3, 0.05, 0), SIMD3(-0.1, -0.1, 0.05),
                                      SIMD3(0.12, 0.08, -0.05), SIMD3(0.3, -0.05, 0)]
        var cameras: [Camera] = []
        for (angle, shift) in zip(rotations, shifts) {
            let rotation = Matrix3.rotation(quaternion: axisAngle(SIMD3(0, 1, 0), angle))
            cameras.append(Camera(width: 48, height: 48, fx: 48, fy: 48, cx: 24, cy: 24,
                                  rotation: rotation, translation: shift))
        }
        let views = try makeViews(context: context, groundTruth: groundTruth, cameras: cameras)
        // Scene radius as the trainer estimates it from the camera centres (no points).
        let centers = cameras.map { $0.center }
        let radius = SceneExtent.estimate(cameraCenters: centers, points: []).radius

        // Perturbation in the trainer's own model: R' = exp([ω]×)R, t' = exp([ω]×)t + τ, with |ω| = 0.7° and
        // |τ| = 1.5 % of the radius, random directions.
        var rng = SplitMix64(seed: 99)
        var perturbed: [Camera] = []
        for camera in cameras {
            let omega = unitVector(&rng) * Float(0.7 * Double.pi / 180)
            let tau = unitVector(&rng) * (0.015 * radius)
            perturbed.append(PoseRefinement.fold(camera, rotation: SIMD3<Double>(Double(omega.x), Double(omega.y),
                                                                                 Double(omega.z)),
                                                 translation: SIMD3<Double>(Double(tau.x), Double(tau.y),
                                                                            Double(tau.z))))
        }
        let trainingViews = zip(views, perturbed).map { view, camera in
            TrainingView(name: view.name, camera: camera, image: view.image)
        }
        return Scene(groundTruth: groundTruth, targetCameras: cameras, targets: trainingViews,
                     perturbed: perturbed, radius: radius)
    }

    private func configuration(refinement: Bool) -> TrainingConfiguration {
        var configuration = TrainingConfiguration.testBaseline()
        configuration.shDegree = 1
        configuration.shDegreeInterval = 0
        configuration.iterations = 300
        configuration.logInterval = 10
        configuration.densification.startIteration = 0
        configuration.densification.stopIteration = 0
        configuration.regularizers = []
        configuration.poseRefinement.isEnabled = refinement
        configuration.poseRefinement.startIteration = 0
        // Four views sampled round robin: each view gets about 75 updates, so the rate is set to reach the 0.7°
        // perturbation within the run.
        configuration.poseRefinement.rotationLearningRate = 1e-3
        return configuration
    }

    /// Mean image loss (the training loss, `ImageLoss` with the default λ) of `cloud` rendered from `cameras`.
    private func meanLoss(context: MetalContext, cloud: GaussianCloud, cameras: [Camera],
                          targets: [TrainingView]) throws -> Double {
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let settings = RasterSettings(shDegreeLimit: cloud.shDegree, background: .zero)
        var sum: Double = 0
        for (camera, target) in zip(cameras, targets) {
            let rendered = try renderer.renderImage(camera: camera, settings: settings)
            sum += ImageLoss.evaluate(rendered: rendered, target: target.image, lambda: 0.2).loss
        }
        return sum / Double(cameras.count)
    }

    /// Mean rotation angle (radians) and camera-centre distance between `estimated` and `reference`.
    private func poseError(_ estimated: [Camera], _ reference: [Camera]) -> (rotation: Double, center: Double) {
        var rotation = 0.0
        var center = 0.0
        for (e, r) in zip(estimated, reference) {
            let relative = e.rotation * r.rotation.transposed
            let trace = Double(relative.c0.x + relative.c1.y + relative.c2.z)
            rotation += acos(min(1, max(-1, (trace - 1) / 2)))
            let difference = e.center - r.center
            center += Double((difference * difference).sum().squareRoot())
        }
        return (rotation / Double(estimated.count), center / Double(estimated.count))
    }

    private func trainer(context: MetalContext, scene: Scene, refinement: Bool) throws -> GPUTrainer {
        let dataset = TrainingDataset(views: scene.targets)
        return try GPUTrainer(context: context, dataset: dataset, initialCloud: scene.groundTruth,
                              configuration: configuration(refinement: refinement))
    }

    @Test func refinementLowersLossAndMovesCamerasTowardGroundTruth() throws {
        let context = try MetalContext.makeDefault()
        let scene = try makeScene(context: context)
        let initial = poseError(scene.perturbed, scene.targetCameras)

        // Refinement off: the cameras stay perturbed.
        let off = try trainer(context: context, scene: scene, refinement: false)
        for _ in 0..<300 { try off.step() }
        let offCloud = try off.snapshot()
        let offLoss = try meanLoss(context: context, cloud: offCloud, cameras: scene.perturbed,
                                   targets: scene.targets)

        // Refinement on: the cameras are refined from the perturbed start.
        let on = try trainer(context: context, scene: scene, refinement: true)
        for _ in 0..<300 { try on.step() }
        let onCloud = try on.snapshot()
        let refined = try on.refinedCameras()
        let onLoss = try meanLoss(context: context, cloud: onCloud, cameras: refined, targets: scene.targets)
        let final = poseError(refined, scene.targetCameras)

        // Measured: loss 2.2e-2 → 3.9e-4 (off → on), rotation 0.70° → 0.024°, centre 0.0051 → 0.0010.
        #expect(onLoss < 0.5 * offLoss, "refined loss \(onLoss) is not clearly below the unrefined loss \(offLoss)")
        #expect(final.rotation < 0.5 * initial.rotation, "rotation error \(final.rotation) did not decrease")
        #expect(final.center < initial.center, "centre error \(final.center) did not decrease")
    }

    /// With refinement off the cameras are returned exactly as the dataset has them.
    @Test func refinedCamerasAreUnchangedWhenDisabled() throws {
        let context = try MetalContext.makeDefault()
        let scene = try makeScene(context: context)
        let off = try trainer(context: context, scene: scene, refinement: false)
        for _ in 0..<5 { try off.step() }
        #expect(try off.refinedCameras() == scene.perturbed)
    }

    /// The session reports the refined cameras of its final run, and none when refinement is off.
    @Test func sessionOutcomeCarriesRefinedCameras() async throws {
        let context = try MetalContext.makeDefault()
        let scene = try makeScene(context: context)
        let dataset = TrainingDataset(views: scene.targets)

        var on = configuration(refinement: true)
        on.iterations = 20
        let refinedOutcome = try await TrainingSession(dataset: dataset, initialCloud: scene.groundTruth,
                                                       configuration: on).runWithOutcome()
        #expect(refinedOutcome.refinedCameras?.count == scene.targets.count)

        var off = configuration(refinement: false)
        off.iterations = 20
        let plainOutcome = try await TrainingSession(dataset: dataset, initialCloud: scene.groundTruth,
                                                     configuration: off).runWithOutcome()
        #expect(plainOutcome.refinedCameras == nil)
    }
}
