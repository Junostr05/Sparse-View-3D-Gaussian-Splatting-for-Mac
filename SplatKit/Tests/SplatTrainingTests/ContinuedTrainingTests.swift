import Foundation
import Testing
import SplatCore
import SplatMetal
import SplatTraining

// Continued training (v1.1): a second session starts from the cloud the first one produced, with the
// configuration of `TrainingConfiguration.continued`. Metal-gated; small scenes only.
@Suite(.serialized, .enabled(if: TrainingTestEnvironment.hasMetal))
struct ContinuedTrainingTests {

    private func makeScene() throws -> (TrainingDataset, GaussianCloud) {
        let context = try MetalContext.makeDefault()
        let groundTruth = try makeRandomCloud(count: 150, seed: 41, shDegree: 1)
        let views = try makeViews(context: context, groundTruth: groundTruth,
                                  cameras: [makeCamera(width: 32, height: 32),
                                            makeCamera(width: 32, height: 32, shift: 0.2)])
        let initial = try makeRandomCloud(count: 150, seed: 42, shDegree: 1)
        return (TrainingDataset(views: views), initial)
    }

    private func baseConfiguration() -> TrainingConfiguration {
        var configuration = TrainingConfiguration.testBaseline()
        configuration.shDegree = 1
        configuration.iterations = 60
        configuration.logInterval = 10
        configuration.previewInterval = 0
        configuration.evalInterval = 0
        configuration.regularizers = []
        configuration.densification.startIteration = 0
        configuration.densification.stopIteration = 0
        return configuration
    }

    private func run(_ dataset: TrainingDataset, _ cloud: GaussianCloud,
                     _ configuration: TrainingConfiguration) async throws -> GaussianCloud {
        let session = TrainingSession(dataset: dataset, initialCloud: cloud, configuration: configuration)
        let task = Task { try await session.run() }
        for await _ in session.progress {}
        return try await task.value
    }

    @Test
    func continuationWithoutGrowthKeepsCountAndShDegree() async throws {
        let (dataset, initial) = try makeScene()
        let base = baseConfiguration()
        let first = try await run(dataset, initial, base)

        var settings = ContinuationSettings()
        settings.roundIterations = 40
        let configuration = base.continued(from: first.count, shDegree: first.shDegree, round: 0, settings: settings)
        let second = try await run(dataset, first, configuration)

        #expect(second.count == first.count)
        #expect(second.shDegree == first.shDegree)
        #expect(second.removingNonFinite().removed == 0)
    }

    @Test
    func continuationWithGrowthStaysWithinTheCapAndFinite() async throws {
        let (dataset, initial) = try makeScene()
        let base = baseConfiguration()
        let first = try await run(dataset, initial, base)

        var settings = ContinuationSettings()
        settings.roundIterations = 300
        settings.addGaussians = true
        settings.growthFactor = 2
        var configuration = base.continued(from: first.count, shDegree: first.shDegree, round: 0, settings: settings)
        // A small scene has few large gradients; make growth likely so the cap is exercised.
        configuration.densification.gradientThreshold = 1e-7
        let second = try await run(dataset, first, configuration)

        #expect(second.count <= configuration.maxGaussians)
        #expect(second.count >= 1)
        #expect(second.shDegree == first.shDegree)
        #expect(second.removingNonFinite().removed == 0)
    }
}
