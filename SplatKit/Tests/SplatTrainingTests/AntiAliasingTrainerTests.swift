import Foundation
import Metal
import Testing
import SplatCore
import SplatMetal
@testable import SplatTraining

// Mip-Splatting anti-aliasing in the Metal trainer: the sampling rates are kept current, the snapshot bakes the 3D
// filter, and a short run with anti-aliasing on stays finite and improves. Runs only where Metal exists.
@Suite(.serialized, .enabled(if: TrainingTestEnvironment.hasMetal))
struct AntiAliasingTrainerTests {

    private func antiAliasedConfiguration(iterations: Int) -> TrainingConfiguration {
        var configuration = TrainingConfiguration()
        configuration.shDegree = 0
        configuration.iterations = iterations
        configuration.logInterval = 25
        configuration.evalInterval = 0
        configuration.previewInterval = 0
        configuration.seed = 5
        // Densification off: the window is empty, so the test does not depend on densify boundaries.
        configuration.densification.startIteration = 0
        configuration.densification.stopIteration = 0
        configuration.antiAliasing.isEnabled = true
        configuration.antiAliasing.samplingRateInterval = 20
        return configuration
    }

    private func allFinite(_ values: [Float]) -> Bool {
        values.allSatisfy { $0.isFinite }
    }

    @Test func snapshotBakesTheThreeDFilterWithCurrentSamplingRates() throws {
        let context = try MetalContext.makeDefault()
        let groundTruth = try makeRandomCloud(count: 300, seed: 11, shDegree: 0)
        let cameras = [makeCamera(width: 48, height: 48), makeCamera(width: 48, height: 48, shift: 0.5)]
        let views = try makeViews(context: context, groundTruth: groundTruth, cameras: cameras)
        let initial = try makeRandomCloud(count: 300, seed: 97, shDegree: 0)
        let configuration = antiAliasedConfiguration(iterations: 10)

        let trainer = try GPUTrainer(context: context, dataset: TrainingDataset(views: views),
                                     initialCloud: initial, configuration: configuration)
        // No step yet: the snapshot computes the sampling rates on the GPU and bakes them into the initial cloud.
        let baked = try trainer.snapshot()
        let rates = SamplingRate.compute(positions: initial.positions, cameras: cameras, near: 0.01)
        let v3 = configuration.antiAliasing.filter3DVariance
        let variances: [Float] = rates.map { nu in v3 / (nu * nu) }
        let expected = initial.bakingSmoothing3D(filterVariances: variances)

        #expect(baked.count == 300)
        #expect(maxAbsDifference(baked, initial) > 1e-3, "the bake must change the scales or opacities")
        let difference = maxAbsDifference(baked, expected)
        #expect(difference < 2e-3, "GPU bake differs from the CPU bake by \(difference)")
        #expect(allFinite(baked.logScales) && allFinite(baked.opacityLogits))
    }

    @Test func trainingWithAntiAliasingStaysFiniteAndImproves() throws {
        let context = try MetalContext.makeDefault()
        let groundTruth = try makeRandomCloud(count: 300, seed: 11, shDegree: 0)
        let views = try makeViews(context: context, groundTruth: groundTruth,
                                  cameras: [makeCamera(width: 48, height: 48),
                                            makeCamera(width: 48, height: 48, shift: 0.5)])
        let initial = try makeRandomCloud(count: 300, seed: 97, shDegree: 0)
        let configuration = antiAliasedConfiguration(iterations: 120)

        let trainer = try GPUTrainer(context: context, dataset: TrainingDataset(views: views),
                                     initialCloud: initial, configuration: configuration)
        let before = try meanL1(context: context, cloud: try trainer.snapshot(), views: views)
        for _ in 0..<configuration.iterations {
            let report = try trainer.step()
            if let loss = report.loss {
                #expect(loss.loss.isFinite)
            }
        }
        #expect(trainer.gaussianCount == 300)
        let final = try trainer.snapshot()
        #expect(final.count == 300)
        #expect(allFinite(final.positions) && allFinite(final.logScales) && allFinite(final.rotations))
        #expect(allFinite(final.opacityLogits) && allFinite(final.sh))

        let after = try meanL1(context: context, cloud: final, views: views)
        print("trainingWithAntiAliasing: mean L1 before \(before), after \(after)")
        #expect(after <= 0.9 * before, "L1 went from \(before) to \(after)")
    }
}
