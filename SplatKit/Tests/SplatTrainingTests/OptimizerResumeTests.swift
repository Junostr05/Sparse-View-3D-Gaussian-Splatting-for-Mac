import Foundation
import Metal
import Testing
import SplatCore
import SplatIO
import SplatMetal
@testable import SplatTraining

// Exact resume (v1.5): optimizer state export and restore on the Metal trainer, the autosave cadence and the
// fallback for a state that does not fit. Metal-gated; small scenes; `.serialized` like the other trainer suites.
// Not verified on Linux: compiled and run on macOS only.
@Suite(.serialized, .enabled(if: TrainingTestEnvironment.hasMetal))
struct OptimizerResumeTests {

    /// Two 48×48 views of a random ground truth and a random initial cloud of `count` Gaussians (SH degree 1).
    private func makeScene(count: Int) throws -> (TrainingDataset, GaussianCloud) {
        let context = try MetalContext.makeDefault()
        let groundTruth = try makeRandomCloud(count: count, seed: 61, shDegree: 1)
        let views = try makeViews(context: context, groundTruth: groundTruth,
                                  cameras: [makeCamera(width: 48, height: 48),
                                            makeCamera(width: 48, height: 48, shift: 0.5)])
        let initial = try makeRandomCloud(count: count, seed: 62, shDegree: 1)
        return (TrainingDataset(views: views), initial)
    }

    /// SH degree 1 for every step, no densification (its statistics are not part of the state), no autosave.
    private func makeConfiguration(iterations: Int) -> TrainingConfiguration {
        var configuration = TrainingConfiguration()
        configuration.shDegree = 1
        configuration.shDegreeInterval = 0
        configuration.iterations = iterations
        configuration.logInterval = 0
        configuration.evalInterval = 0
        configuration.previewInterval = 0
        configuration.autosaveInterval = 0
        configuration.seed = 7
        configuration.densification.startIteration = 0
        configuration.densification.stopIteration = 0
        return configuration
    }

    private func makeTrainer(_ dataset: TrainingDataset, _ cloud: GaussianCloud,
                             _ configuration: TrainingConfiguration) throws -> GPUTrainer {
        let context = try MetalContext.makeDefault()
        return try GPUTrainer(context: context, dataset: dataset, initialCloud: cloud, configuration: configuration)
    }

    private func steps(_ trainer: GPUTrainer, _ count: Int) throws {
        for _ in 0..<count {
            try trainer.step()
        }
    }

    private func bits(_ values: [Float]) -> [UInt32] {
        values.map(\.bitPattern)
    }

    /// The moments of `state` with every first and second moment of the Gaussians set to zero.
    private func zeroedMoments(_ state: OptimizerState) -> OptimizerState {
        var copy = state
        let zero = { (values: [Float]) -> [Float] in [Float](repeating: 0, count: values.count) }
        copy.firstMoment = GaussianMoments(positions: zero(state.firstMoment.positions),
                                           logScales: zero(state.firstMoment.logScales),
                                           rotations: zero(state.firstMoment.rotations),
                                           opacityLogits: zero(state.firstMoment.opacityLogits),
                                           sh: zero(state.firstMoment.sh))
        copy.secondMoment = GaussianMoments(positions: zero(state.secondMoment.positions),
                                            logScales: zero(state.secondMoment.logScales),
                                            rotations: zero(state.secondMoment.rotations),
                                            opacityLogits: zero(state.secondMoment.opacityLogits),
                                            sh: zero(state.secondMoment.sh))
        return copy
    }

    @Test
    func restoredMomentsAreTheExportedMomentsBitForBit() throws {
        let (dataset, initial) = try makeScene(count: 300)
        let configuration = makeConfiguration(iterations: 30)
        let first = try makeTrainer(dataset, initial, configuration)
        try steps(first, 30)
        let exported = try first.exportOptimizerState()
        #expect(exported.iteration == 30)
        #expect(exported.adamStep == 30)
        #expect(exported.appearance != nil)

        let second = try makeTrainer(dataset, try first.snapshot(), configuration)
        let reason = try second.restore(from: exported, continuing: false)
        #expect(reason == nil)
        #expect(second.iteration == 30)

        // Read back through the GPU: the moments and the appearance state are the exported ones, bit for bit.
        let readBack = try second.exportOptimizerState()
        #expect(bits(readBack.firstMoment.positions) == bits(exported.firstMoment.positions))
        #expect(bits(readBack.firstMoment.logScales) == bits(exported.firstMoment.logScales))
        #expect(bits(readBack.firstMoment.rotations) == bits(exported.firstMoment.rotations))
        #expect(bits(readBack.firstMoment.opacityLogits) == bits(exported.firstMoment.opacityLogits))
        #expect(bits(readBack.firstMoment.sh) == bits(exported.firstMoment.sh))
        #expect(bits(readBack.secondMoment.positions) == bits(exported.secondMoment.positions))
        #expect(bits(readBack.secondMoment.sh) == bits(exported.secondMoment.sh))
        #expect(readBack.appearance == exported.appearance)
        #expect(readBack.adamStep == exported.adamStep)
    }

    @Test
    func resumedRunMatchesTheUninterruptedRun() throws {
        let (dataset, initial) = try makeScene(count: 300)
        let total = 60
        let configuration = makeConfiguration(iterations: total)

        // One go, twice: the difference between the two is the GPU's own run-to-run noise (float atomics).
        let whole = try makeTrainer(dataset, initial, configuration)
        try steps(whole, total)
        let wholeCloud = try whole.snapshot()
        let repeatRun = try makeTrainer(dataset, initial, configuration)
        try steps(repeatRun, total)
        let noise = maxAbsDifference(wholeCloud, try repeatRun.snapshot())

        // 30 steps, export, restore into a new trainer, 30 more.
        let firstHalf = try makeTrainer(dataset, initial, configuration)
        try steps(firstHalf, 30)
        let exported = try firstHalf.exportOptimizerState()
        let resumed = try makeTrainer(dataset, try firstHalf.snapshot(), configuration)
        #expect(try resumed.restore(from: exported, continuing: false) == nil)
        try steps(resumed, 30)
        let resumedCloud = try resumed.snapshot()

        // Control: the same 30 steps with the moments zeroed (the iteration, sampler and schedule are unchanged).
        let zeroed = try makeTrainer(dataset, try firstHalf.snapshot(), configuration)
        #expect(try zeroed.restore(from: zeroedMoments(exported), continuing: false) == nil)
        try steps(zeroed, 30)
        let zeroedCloud = try zeroed.snapshot()

        let exactError = maxAbsDifference(wholeCloud, resumedCloud)
        let zeroedError = maxAbsDifference(wholeCloud, zeroedCloud)
        // Documented tolerance: the resumed run may differ from the whole run by the GPU's own run-to-run noise (float
        // atomics; measured 1.7e-6 on the test scene, the resume error measured 1.4e-6 to 1.7e-6), with a floor of 1e-5
        // for the float rounding of 60 Adam steps. The zeroed-moment control measured 1.3, about 8e5 times further away.
        let tolerance = max(10 * noise, 1e-5)
        #expect(exactError <= tolerance, "resume error \(exactError) exceeds tolerance \(tolerance) (noise \(noise))")
        #expect(exactError < zeroedError, "exact \(exactError) is not below the zeroed-moment control \(zeroedError)")
    }

    @Test
    func poseAndAppearanceStateRestoreExactly() throws {
        let (dataset, initial) = try makeScene(count: 200)
        var configuration = makeConfiguration(iterations: 30)
        configuration.poseRefinement.isEnabled = true
        configuration.poseRefinement.startIteration = 0
        configuration.poseRefinement.rotationLearningRate = 1e-3
        configuration.poseRefinement.translationLearningRate = 1e-2
        let first = try makeTrainer(dataset, initial, configuration)
        try steps(first, 30)
        let exported = try first.exportOptimizerState()
        let pose = try #require(exported.pose)
        #expect(pose.steps.contains { $0 > 0 })

        let second = try makeTrainer(dataset, try first.snapshot(), configuration)
        #expect(try second.restore(from: exported, continuing: false) == nil)
        let firstCameras = try first.refinedCameras()
        let secondCameras = try second.refinedCameras()
        for (a, b) in zip(firstCameras, secondCameras) {
            #expect(bits(a.rotation.c0.asArray) == bits(b.rotation.c0.asArray))
            #expect(bits(a.rotation.c1.asArray) == bits(b.rotation.c1.asArray))
            #expect(bits(a.rotation.c2.asArray) == bits(b.rotation.c2.asArray))
            #expect(bits(a.translation.asArray) == bits(b.translation.asArray))
        }
        let readBack = try second.exportOptimizerState()
        #expect(readBack.pose == pose)
        #expect(readBack.appearance == exported.appearance)
    }

    @Test
    func continuationCarriesTheLineageStepCount() async throws {
        let (dataset, initial) = try makeScene(count: 150)
        let first = try makeTrainer(dataset, initial, makeConfiguration(iterations: 20))
        try steps(first, 20)
        let exported = try first.exportOptimizerState()

        // A continuation round starts its own schedule at iteration 0 and continues the Adam step count.
        var roundConfiguration = makeConfiguration(iterations: 15)
        roundConfiguration.seed = 9
        let session = TrainingSession(dataset: dataset, initialCloud: try first.snapshot(),
                                      configuration: roundConfiguration, optimizerState: exported)
        let outcome = try await session.runWithOutcome()
        #expect(outcome.iterations == 15)
        let carried = try #require(outcome.optimizerState)
        #expect(carried.iteration == 15)
        #expect(carried.adamStep == exported.adamStep + 15)
    }

    @Test
    func unfittingStateFallsBackToFreshAdam() async throws {
        let (dataset, initial) = try makeScene(count: 150)
        let configuration = makeConfiguration(iterations: 10)
        let first = try makeTrainer(dataset, initial, configuration)
        try steps(first, 10)
        let exported = try first.exportOptimizerState()

        // A state for another Gaussian count (for example after deletions) is ignored, and the run still completes.
        let (_, other) = try makeScene(count: 120)
        let session = TrainingSession(dataset: dataset, initialCloud: other, configuration: configuration,
                                      optimizerState: exported)
        let outcome = try await session.runWithOutcome()
        #expect(outcome.iterations == 10)
        #expect(outcome.cloud.count == 120)
        // Fresh Adam: the step counter starts at zero for this run.
        #expect(try #require(outcome.optimizerState).adamStep == 10)
    }

    @Test
    func autosaveFiresEveryIntervalExceptTheLastIteration() async throws {
        let (dataset, initial) = try makeScene(count: 150)
        var configuration = makeConfiguration(iterations: 30)
        configuration.autosaveInterval = 10
        let (stream, continuation) = AsyncStream.makeStream(of: TrainingAutosave.self)
        let session = TrainingSession(dataset: dataset, initialCloud: initial, configuration: configuration,
                                      autosave: { snapshot in continuation.yield(snapshot) })
        let task = Task { try await session.run() }
        _ = try await task.value
        continuation.finish()

        var received: [TrainingAutosave] = []
        for await snapshot in stream {
            received.append(snapshot)
        }
        #expect(received.map(\.iteration) == [10, 20])
        for snapshot in received {
            #expect(snapshot.optimizerState.gaussianCount == snapshot.cloud.count)
            #expect(snapshot.optimizerState.iteration == snapshot.iteration)
            #expect(snapshot.optimizerState.adamStep == snapshot.iteration)
        }
    }
}

private extension SIMD3 where Scalar == Float {
    var asArray: [Float] { [x, y, z] }
}
