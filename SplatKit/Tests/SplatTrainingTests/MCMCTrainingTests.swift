import Foundation
import Metal
import Testing
import SplatCore
import SplatMetal
@testable import SplatTraining

// MCMC densification (v1.2) on the GPU trainer and on its shared buffers. Runs only where Metal exists. Scenes are
// small: the CI GPU is a slow paravirtual device. `.serialized`: the GPU work competes for one device.
// The GPU backward uses float atomics, so runs are not bit-identical; comparisons use tolerances.
@Suite(.serialized, .enabled(if: TrainingTestEnvironment.hasMetal))
struct MCMCTrainingTests {

    // MARK: - Helpers

    /// Tiny degree-0 scene as in TrainingSessionTests: a random ground truth rendered from two cameras.
    private func makeScene(initialCount: Int) throws -> (TrainingDataset, GaussianCloud) {
        let context = try MetalContext.makeDefault()
        let groundTruth = try makeRandomCloud(count: 150, seed: 41, shDegree: 0)
        let views = try makeViews(context: context, groundTruth: groundTruth,
                                  cameras: [makeCamera(width: 32, height: 32),
                                            makeCamera(width: 32, height: 32, shift: 0.2)])
        let initial = try makeRandomCloud(count: initialCount, seed: 42, shDegree: 0)
        return (TrainingDataset(views: views), initial)
    }

    /// MCMC on, classic densification on with zero threshold (so it would clone everything if it ran), and the
    /// Phase 4 regularizers off so that the count only changes through MCMC.
    private func mcmcConfiguration(iterations: Int, maxGaussians: Int) -> TrainingConfiguration {
        var configuration = TrainingConfiguration.testBaseline()
        configuration.shDegree = 0
        configuration.iterations = iterations
        configuration.logInterval = 10
        configuration.evalInterval = 0
        configuration.previewInterval = 0
        configuration.maxGaussians = maxGaussians
        configuration.regularizers = []
        configuration.densification.startIteration = 5
        configuration.densification.stopIteration = iterations
        configuration.densification.interval = 5
        configuration.densification.gradientThreshold = 0
        configuration.mcmc.isEnabled = true
        configuration.mcmc.startIteration = 10
        configuration.mcmc.stopIteration = iterations - 20
        configuration.mcmc.interval = 10
        configuration.mcmc.growthFraction = 0.5
        configuration.mcmc.deadOpacity = 0.3
        return configuration
    }

    private func isFinite(_ cloud: GaussianCloud) -> Bool {
        let values = cloud.positions + cloud.logScales + cloud.rotations + cloud.opacityLogits + cloud.sh
        return values.allSatisfy { $0.isFinite }
    }

    /// Degree-0 cloud whose Gaussians have distinct positions: `live` Gaussians at logit 2 (opacity ≈ 0.88) first,
    /// then `dead` Gaussians at logit −8 (opacity ≈ 3e-4).
    private func relocationCloud(live: Int, dead: Int) throws -> GaussianCloud {
        let count = live + dead
        var positions: [Float] = []
        var logits: [Float] = []
        for index in 0..<count {
            positions += [Float(index) * 0.1 - 1, Float(index % 7) * 0.2 - 0.6, 2 + Float(index) * 0.05]
            logits.append(index < live ? 2 : -8)
        }
        let logScales = [Float](repeating: Float(log(0.1)), count: 3 * count)
        var rotations: [Float] = []
        for _ in 0..<count {
            rotations += [1, 0, 0, 0]
        }
        let sh = [Float](repeating: 0.1, count: 3 * count)
        return try GaussianCloud(shDegree: 0, positions: positions, logScales: logScales,
                                 rotations: rotations, opacityLogits: logits, sh: sh)
    }

    /// Sets every float of the store's groups to `value`.
    private func fill(_ store: GaussianBuffers, with value: Float) {
        for buffer in store.groups {
            let count = buffer.length / MemoryLayout<Float>.size
            let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
            for index in 0..<count {
                pointer[index] = value
            }
        }
    }

    private func settings(growth: Float, deadOpacity: Float) -> MCMCSettings {
        var result = MCMCSettings()
        result.isEnabled = true
        result.growthFraction = growth
        result.deadOpacity = deadOpacity
        return result
    }

    // MARK: - Relocation on the shared buffers

    @Test func relocationKeepsCompositeOpacityOfEachGroup() throws {
        let context = try MetalContext.makeDefault()
        let cloud = try relocationCloud(live: 20, dead: 20)
        let parameters = try GaussianBuffers(context: context, cloud: cloud, capacity: 40)
        let adam = try AdamBuffers(context: context, capacity: 40, shDegree: 0)
        // The trainer keeps the moment counts equal to the parameter count (GPUAdam.encode does this).
        adam.m.count = 40
        adam.v.count = 40
        fill(adam.m, with: 1)
        fill(adam.v, with: 1)

        let result = try MCMCRelocator.run(parameters: parameters, m: adam.m, v: adam.v,
                                           settings: settings(growth: 0.5, deadOpacity: 0.005),
                                           maxGaussians: 80, seed: 3)
        #expect(result.dead == 20)
        #expect(result.grown == 20)
        #expect(result.grewCapacity)
        #expect(parameters.count == 60)
        #expect(adam.m.count == 60 && adam.v.count == 60)

        // Copies share a position with their source, so each group of equal positions is one source and its copies.
        let snapshot = parameters.snapshot()
        var groups: [String: [Int]] = [:]
        for index in 0..<snapshot.count {
            let key = "\(snapshot.positions[3 * index]),\(snapshot.positions[3 * index + 1]),\(snapshot.positions[3 * index + 2])"
            groups[key, default: []].append(index)
        }
        #expect(groups.count == 20)
        let liveOpacity: Double = 1 / (1 + exp(-2))
        for members in groups.values {
            var survival: Double = 1
            for index in members {
                let opacity = 1 / (1 + exp(-Double(snapshot.opacityLogits[index])))
                survival *= 1 - opacity
            }
            #expect(abs((1 - survival) - liveOpacity) < 1e-4, "composite \(1 - survival) for \(members.count) members")
        }

        // Adam moments: zero for every Gaussian of a group that was relocated, untouched otherwise.
        let momentM = adam.m.opacityLogits.contents().assumingMemoryBound(to: Float.self)
        let momentV = adam.v.opacityLogits.contents().assumingMemoryBound(to: Float.self)
        for members in groups.values {
            let expected: Float = members.count > 1 ? 0 : 1
            for index in members {
                #expect(momentM[index] == expected)
                #expect(momentV[index] == expected)
            }
        }
    }

    @Test func growthStopsAtTheBudget() throws {
        let context = try MetalContext.makeDefault()
        let cloud = try relocationCloud(live: 25, dead: 15)
        let parameters = try GaussianBuffers(context: context, cloud: cloud, capacity: 40)
        let adam = try AdamBuffers(context: context, capacity: 40, shDegree: 0)
        adam.m.count = 40
        adam.v.count = 40

        // Room for 30 more; growth asks for 40, so only 30 are added and the capacity grows to exactly the budget.
        let result = try MCMCRelocator.run(parameters: parameters, m: adam.m, v: adam.v,
                                           settings: settings(growth: 1, deadOpacity: 0.005),
                                           maxGaussians: 70, seed: 1)
        #expect(result.grown == 30)
        #expect(parameters.count == 70)
        #expect(parameters.capacity == 70)
        #expect(isFinite(parameters.snapshot()))

        // At the budget: no growth, and the count stays.
        let again = try MCMCRelocator.run(parameters: parameters, m: adam.m, v: adam.v,
                                          settings: settings(growth: 1, deadOpacity: 0.005),
                                          maxGaussians: 70, seed: 2)
        #expect(again.grown == 0)
        #expect(parameters.count == 70)
    }

    // MARK: - Trainer

    @Test func mcmcTrainingStaysWithinBudgetAndSkipsClassicDensification() throws {
        let (dataset, initial) = try makeScene(initialCount: 60)
        let configuration = mcmcConfiguration(iterations: 120, maxGaussians: 120)
        let trainer = try GPUTrainer(context: MetalContext.makeDefault(), dataset: dataset,
                                     initialCloud: initial, configuration: configuration)
        let startPSNR = try trainer.evaluatePSNR()

        var reports: [StepReport] = []
        for _ in 0..<configuration.iterations {
            reports.append(try trainer.step())
        }
        let endPSNR = try trainer.evaluatePSNR()
        let cloud = try trainer.snapshot()

        // Classic densification is replaced: it never reports a summary, although its threshold would clone.
        #expect(reports.allSatisfy { $0.densify == nil })
        #expect(trainer.gaussianCount <= configuration.maxGaussians)
        #expect(trainer.gaussianCount > initial.count)
        #expect(cloud.count == trainer.gaussianCount)
        #expect(isFinite(cloud))
        #expect(endPSNR > startPSNR, "PSNR \(startPSNR) -> \(endPSNR)")
    }

    @Test func mcmcGrowsCapacityWhenRelocationNeedsIt() throws {
        let (dataset, initial) = try makeScene(initialCount: 60)
        let configuration = mcmcConfiguration(iterations: 120, maxGaussians: 120)
        // Capacity equal to the initial count, so the first growth has to reallocate every buffer.
        let trainer = try GPUTrainer(context: MetalContext.makeDefault(), dataset: dataset,
                                     initialCloud: initial, configuration: configuration,
                                     capacityOverride: initial.count)
        for _ in 0..<configuration.iterations {
            _ = try trainer.step()
        }
        let cloud = try trainer.snapshot()
        #expect(trainer.gaussianCount == 120)
        #expect(isFinite(cloud))
    }

    @Test func disabledMCMCIgnoresItsSettings() throws {
        let (dataset, initial) = try makeScene(initialCount: 80)
        let context = try MetalContext.makeDefault()
        var baseline = TrainingConfiguration.testBaseline()
        baseline.shDegree = 0
        baseline.iterations = 40
        baseline.logInterval = 10
        baseline.evalInterval = 0
        baseline.previewInterval = 0
        baseline.maxGaussians = 200
        baseline.regularizers = []
        baseline.densification.startIteration = 5
        baseline.densification.stopIteration = 35
        baseline.densification.interval = 5
        baseline.densification.gradientThreshold = 0

        // Same run with MCMC switched off but every other MCMC field changed: nothing may change.
        var altered = baseline
        altered.mcmc = MCMCSettings()
        altered.mcmc.isEnabled = false
        altered.mcmc.startIteration = 0
        altered.mcmc.stopIteration = 1_000
        altered.mcmc.interval = 3
        altered.mcmc.growthFraction = 1
        altered.mcmc.deadOpacity = 0.9
        altered.mcmc.noiseLearningRate = 1
        altered.mcmc.opacityRegularizer = 5
        altered.mcmc.scaleRegularizer = 5

        func train(_ configuration: TrainingConfiguration) throws -> (GaussianCloud, [DensifySummary?], Int) {
            let trainer = try GPUTrainer(context: context, dataset: dataset, initialCloud: initial,
                                         configuration: configuration)
            var summaries: [DensifySummary?] = []
            for _ in 0..<configuration.iterations {
                summaries.append(try trainer.step().densify)
            }
            return (try trainer.snapshot(), summaries, trainer.gaussianCount)
        }
        let (cloudA, summariesA, countA) = try train(baseline)
        let (cloudB, summariesB, countB) = try train(altered)

        // Classic densification ran in both (it is what this test is about).
        #expect(summariesA.contains { $0 != nil })
        #expect(summariesA == summariesB)
        #expect(countA == countB)
        // The backward pass uses float atomics, so allow the run-to-run noise of the GPU.
        #expect(maxAbsDifference(cloudA, cloudB) < 1e-3)
    }
}
