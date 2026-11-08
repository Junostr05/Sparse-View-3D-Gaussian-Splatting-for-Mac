import Foundation
import Testing
import SplatCore
import SplatMetal
@testable import SplatTraining

// v1.5 normal prior in the GPU trainer: the term exists only when switched on and normals are attached, it
// changes the optimisation, and training with it stays finite. The loss and its gradient are checked against
// the CPU reference and finite differences in NormalLossTests / GPUNormalLossTests.

private func phase4TermCount(_ trainer: GPUTrainer) -> Int? {
    (Mirror(reflecting: trainer).descendant("phase4Terms") as? [any AuxiliaryLoss])?.count
}

/// The scene's training views with ground-truth metric priors and, optionally, normals of that ground truth.
private func dataset(_ scene: SyntheticScene, withNormals: Bool) -> TrainingDataset {
    var views: [TrainingView] = scene.train
    for index in views.indices {
        let depth: DepthMap = scene.trainDepth[index]
        var normals: [SIMD3<Float>]? = nil
        if withNormals {
            normals = NormalLoss.normals(depth: depth.values, width: depth.width, height: depth.height,
                                         intrinsics: NormalLossIntrinsics(camera: views[index].camera))
        }
        views[index].depth = DepthPrior(width: depth.width, height: depth.height, correlationTarget: depth.values,
                                        domain: .depth, metricDepth: depth.values, isMeasuredMetric: true,
                                        normals: normals)
    }
    return TrainingDataset(views: views, points: scene.points)
}

private func configuration(iterations: Int, normalPrior: Bool, weight: Float = 0.05) -> TrainingConfiguration {
    var configuration = TrainingConfiguration.testBaseline()
    configuration.appearance.isEnabled = false
    configuration.iterations = iterations
    configuration.shDegree = 0
    configuration.seed = 11
    configuration.logInterval = 1
    configuration.evalInterval = 0
    configuration.previewInterval = 0
    configuration.regularizers = [.depthPearson, .depthL1]
    configuration.densification.startIteration = 0
    configuration.densification.stopIteration = 0
    configuration.normalPrior.isEnabled = normalPrior
    configuration.normalPrior.weight = weight
    return configuration
}

@Suite(.serialized, .enabled(if: TrainingTestEnvironment.hasMetal))
struct GPUTrainerNormalPriorTests {

    @Test func termExistsOnlyWhenEnabledAndNormalsAreAttached() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 3, novelCount: 0, width: 32, height: 24,
                                        gaussians: 120, sparsePoints: 60, seed: 4)
        let withNormals = dataset(scene, withNormals: true)
        let without = dataset(scene, withNormals: false)
        let off = configuration(iterations: 1, normalPrior: false)
        let on = configuration(iterations: 1, normalPrior: true)
        let initial = GaussianInitializer.initialize(withNormals, configuration: off, maxCount: 300)

        let baseline = try #require(phase4TermCount(try GPUTrainer(context: context, dataset: withNormals,
                                                                   initialCloud: initial, configuration: off)))
        let enabled = try #require(phase4TermCount(try GPUTrainer(context: context, dataset: withNormals,
                                                                  initialCloud: initial, configuration: on)))
        let noNormals = try #require(phase4TermCount(try GPUTrainer(context: context, dataset: without,
                                                                    initialCloud: initial, configuration: on)))
        #expect(enabled == baseline + 1)
        #expect(noNormals == baseline)
    }

    @Test func normalPriorChangesTheOptimisationAndStaysFinite() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                        gaussians: 160, sparsePoints: 80, seed: 5)
        let data = dataset(scene, withNormals: true)
        let off = configuration(iterations: 20, normalPrior: false)
        let on = configuration(iterations: 20, normalPrior: true, weight: 1)
        let initial = GaussianInitializer.initialize(data, configuration: off, maxCount: 400)

        let plain = try GPUTrainer(context: context, dataset: data, initialCloud: initial, configuration: off)
        let normal = try GPUTrainer(context: context, dataset: data, initialCloud: initial, configuration: on)
        for _ in 0..<on.iterations {
            let a = try plain.step()
            let b = try normal.step()
            #expect(try #require(a.loss).loss.isFinite)
            #expect(try #require(b.loss).loss.isFinite)
        }
        let cloudA = try plain.snapshot()
        let cloudB = try normal.snapshot()
        #expect(cloudB.positions.allSatisfy { $0.isFinite })
        #expect(cloudB.logScales.allSatisfy { $0.isFinite })
        #expect(cloudB.opacityLogits.allSatisfy { $0.isFinite })
        var largest: Float = 0
        for index in 0..<min(cloudA.positions.count, cloudB.positions.count) {
            let delta: Float = cloudA.positions[index] - cloudB.positions[index]
            largest = max(largest, abs(delta))
        }
        // Run-to-run noise of the float atomics is ~1e-5 relative; the normal term must move far more.
        #expect(largest > 1e-3, "the normal prior did not change the positions (largest change \(largest))")
    }
}
