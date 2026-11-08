import Foundation
import Testing
import SplatCore
import SplatMetal
import SplatTraining

// Headless training session: progress stream, cancellation and pause. Metal-gated; small scenes only.
// `.serialized`: the GPU work is synchronous and the suites share one GPU; see GPUTrainerTests.
// The progress stream keeps only the newest 4 events, so counts below use lower bounds and the
// terminal state (always the newest event) is what is asserted exactly.
// Not verified on Linux: compiled and run on macOS CI only.
@Suite(.serialized, .enabled(if: TrainingTestEnvironment.hasMetal))
struct TrainingSessionTests {

    /// A small dataset with two 32×32 views of a random ground truth, and a random initial cloud.
    private func makeScene(initialCount: Int) throws -> (TrainingDataset, GaussianCloud) {
        let context = try MetalContext.makeDefault()
        let groundTruth = try makeRandomCloud(count: 150, seed: 41, shDegree: 0)
        let views = try makeViews(context: context, groundTruth: groundTruth,
                                  cameras: [makeCamera(width: 32, height: 32),
                                            makeCamera(width: 32, height: 32, shift: 0.2)])
        let initial = try makeRandomCloud(count: initialCount, seed: 42, shDegree: 0)
        return (TrainingDataset(views: views), initial)
    }

    @Test
    func streamReportsProgressAndFinishes() async throws {
        let (dataset, initial) = try makeScene(initialCount: 150)
        var configuration = TrainingConfiguration.testBaseline()
        configuration.shDegree = 0
        configuration.iterations = 40
        configuration.logInterval = 10
        configuration.previewInterval = 20
        configuration.evalInterval = 0
        configuration.densification.startIteration = 0
        configuration.densification.stopIteration = 0

        let session = TrainingSession(dataset: dataset, initialCloud: initial, configuration: configuration)
        let task = Task { try await session.run() }
        var events: [TrainingProgress] = []
        for await event in session.progress {
            events.append(event)
        }
        let cloud = try await task.value

        let running = events.filter { $0.state == .running }
        #expect(running.count >= 1)
        let last = try #require(events.last)
        #expect(last.state == .finished)
        #expect(last.iteration == 40)
        #expect(last.preview != nil)
        #expect(cloud.count == initial.count)
    }

    @Test
    func cancellationLeavesConsistentModel() async throws {
        // Densifies at 10, 20, ... The run is far longer than the test needs, so the cancellation (after
        // the third progress event) is what ends it. Densification is forced by a zero gradient threshold.
        let (dataset, initial) = try makeScene(initialCount: 100)
        var configuration = TrainingConfiguration.testBaseline()
        configuration.shDegree = 0
        configuration.iterations = 100_000
        configuration.logInterval = 5
        configuration.previewInterval = 0
        configuration.evalInterval = 0
        configuration.maxGaussians = 150
        configuration.densification.startIteration = 10
        configuration.densification.stopIteration = 100_000
        configuration.densification.interval = 10
        configuration.densification.gradientThreshold = 0

        let session = TrainingSession(dataset: dataset, initialCloud: initial, configuration: configuration)
        let task = Task { try await session.run() }
        var last: TrainingProgress?
        var runningEvents = 0
        for await event in session.progress {
            last = event
            if event.state == .running {
                runningEvents += 1
                // `>=`: the stream may drop old events when the consumer lags.
                if runningEvents >= 3 {
                    task.cancel()
                }
            }
        }
        let cloud = try await task.value

        #expect(last?.state == .cancelled)
        // Re-validating through the throwing initializer checks that every array length agrees.
        let rebuilt = try GaussianCloud(shDegree: cloud.shDegree, positions: cloud.positions,
                                        logScales: cloud.logScales, rotations: cloud.rotations,
                                        opacityLogits: cloud.opacityLogits, sh: cloud.sh)
        #expect(rebuilt.count == cloud.count)
        #expect(cloud.count <= configuration.maxGaussians)
        #expect(cloud.count >= 1)
    }

    @Test
    func pauseStopsIterating() async throws {
        let (dataset, initial) = try makeScene(initialCount: 100)
        var configuration = TrainingConfiguration.testBaseline()
        configuration.shDegree = 0
        // Long enough that the run is still going while the test pauses it.
        configuration.iterations = 400
        configuration.logInterval = 5
        configuration.previewInterval = 0
        configuration.evalInterval = 0
        configuration.densification.startIteration = 0
        configuration.densification.stopIteration = 0

        let session = TrainingSession(dataset: dataset, initialCloud: initial, configuration: configuration)
        let task = Task { try await session.run() }
        var iterator = session.progress.makeAsyncIterator()

        // Wait for the first running event, then pause.
        var first: TrainingProgress?
        while let event = await iterator.next() {
            if event.state == .running {
                first = event
                break
            }
        }
        #expect(first != nil)
        await session.pause()

        // The run reports `.paused` at an iteration boundary.
        var sawPaused = false
        while let event = await iterator.next() {
            if event.state == .paused {
                sawPaused = true
                break
            }
        }
        #expect(sawPaused)

        // The iteration must not move while paused: read it, wait, read it again.
        let pausedAt = await session.currentIteration
        try await Task.sleep(for: .milliseconds(200))
        let stillPausedAt = await session.currentIteration
        #expect(stillPausedAt == pausedAt, "iteration moved from \(pausedAt) to \(stillPausedAt) while paused")

        await session.resume()

        // After resuming, the iteration advances past the paused value.
        var advanced: TrainingProgress?
        while let event = await iterator.next() {
            if event.state == .running && event.iteration > pausedAt {
                advanced = event
                break
            }
        }
        #expect(advanced != nil, "training did not advance after resume")

        while await iterator.next() != nil {}
        let cloud = try await task.value
        #expect(cloud.count == initial.count)
    }
}
