import Foundation
import Synchronization
import Testing
import SplatCore
import SplatMetal
import SplatTraining

// Leave-one-out early stopping in `TrainingSession` (P11): run 1 holds one view out, run 2 retrains
// from a fresh initial cloud on all views. Metal-gated; small scenes only (32×24, ≤ 600 iterations).
// `.serialized`: the GPU work is synchronous and the suites share one GPU; see GPUTrainerTests.
// Not verified on Linux: compiled and run on macOS only.
@Suite(.serialized, .enabled(if: TrainingTestEnvironment.hasMetal))
struct LeaveOneOutSessionTests {

    /// Camera offsets in the plane, so the camera centres are not collinear (no PCA tie).
    private static let offsets: [(Float, Float)] = [(0, 0), (0.2, 0.05), (0.4, -0.1), (0.1, 0.3)]

    private func makeCamera(x: Float, y: Float) -> Camera {
        Camera(width: 32, height: 24, fx: 24, fy: 24, cx: 16, cy: 12,
               rotation: .identity, translation: SIMD3<Float>(x, y, 0))
    }

    /// A dataset of `viewCount` views (named view0, view1, …) and a random initial cloud.
    private func makeScene(viewCount: Int) throws -> (TrainingDataset, GaussianCloud) {
        let context = try MetalContext.makeDefault()
        let groundTruth = try makeRandomCloud(count: 150, seed: 41, shDegree: 0)
        var cameras: [Camera] = []
        for index in 0..<viewCount {
            let offset: (Float, Float) = Self.offsets[index]
            cameras.append(makeCamera(x: offset.0, y: offset.1))
        }
        let views = try makeViews(context: context, groundTruth: groundTruth, cameras: cameras)
        let initial = try makeRandomCloud(count: 150, seed: 42, shDegree: 0)
        return (TrainingDataset(views: views), initial)
    }

    /// Phase 3 style configuration with densification off and no PSNR evaluation; early stopping at
    /// every 100 iterations with patience 2 and no stop before `minimumIterations`.
    private func makeConfiguration(iterations: Int, minimumIterations: Int = 200) -> TrainingConfiguration {
        var configuration = TrainingConfiguration()
        configuration.shDegree = 0
        configuration.iterations = iterations
        configuration.logInterval = 20
        configuration.previewInterval = 0
        configuration.evalInterval = 0
        configuration.densification.startIteration = 0
        configuration.densification.stopIteration = 0
        configuration.earlyStopping.isEnabled = true
        configuration.earlyStopping.evalInterval = 100
        configuration.earlyStopping.patience = 2
        configuration.earlyStopping.minimumIterations = minimumIterations
        return configuration
    }

    /// Runs the session and collects every event the stream delivers, in order.
    private func runCollecting(_ session: TrainingSession) async throws -> (TrainingOutcome, [TrainingProgress]) {
        let task = Task { try await session.runWithOutcome() }
        var events: [TrainingProgress] = []
        for await event in session.progress {
            events.append(event)
        }
        let outcome = try await task.value
        return (outcome, events)
    }

    /// Distinct phases in the order they were first observed.
    private func phaseSequence(_ events: [TrainingProgress]) -> [TrainingPhase] {
        var sequence: [TrainingPhase] = []
        for event in events where sequence.last != event.phase {
            sequence.append(event.phase)
        }
        return sequence
    }

    @Test
    func threeViewsRunSingle() async throws {
        let (dataset, initial) = try makeScene(viewCount: 3)
        let session = TrainingSession(dataset: dataset, initialCloud: initial,
                                      configuration: makeConfiguration(iterations: 200))
        let (outcome, events) = try await runCollecting(session)

        #expect(events.allSatisfy { $0.phase == .single })
        #expect(outcome.heldOutView == nil)
        #expect(outcome.bestIteration == nil)
        #expect(outcome.heldOut == nil)
        #expect(outcome.iterations == 200)
        #expect(events.last?.state == .finished)
    }

    @Test
    func selectionThenFinal() async throws {
        let (dataset, initial) = try makeScene(viewCount: 4)
        let session = TrainingSession(dataset: dataset, initialCloud: initial,
                                      configuration: makeConfiguration(iterations: 600))
        let (outcome, events) = try await runCollecting(session)

        let sequence = phaseSequence(events)
        print("LeaveOneOut phase sequence (observed, distinct): \(sequence)")
        let heldOutName: String = try #require(outcome.heldOutView)
        let best: Int = try #require(outcome.bestIteration)
        let metrics: HeldOutMetrics = try #require(outcome.heldOut)
        let last = try #require(events.last)
        print("LeaveOneOut held-out \(heldOutName): bestIteration \(best), PSNR \(metrics.psnr), SSIM \(metrics.ssim), stop \(outcome.iterations)")

        #expect(last.state == .finished)
        #expect(last.phase == .final(stopIteration: outcome.iterations))
        #expect(last.totalIterations == outcome.iterations)
        #expect(outcome.iterations <= 600)
        #expect(last.bestIteration == best)
        // No selection event may come after the first final event.
        if let firstFinal = events.firstIndex(where: { if case .final = $0.phase { return true } else { return false } }) {
            let lateSelection = events[firstFinal...].contains { if case .selection = $0.phase { return true } else { return false } }
            #expect(!lateSelection)
        }
        #expect(outcome.cloud.positions.allSatisfy { $0.isFinite })
        #expect(outcome.cloud.count == initial.count)
    }

    @Test
    func heldOutViewExcludedFromInitializer() async throws {
        let (dataset, initial) = try makeScene(viewCount: 4)
        let calls = Mutex<[[String]]>([])
        let initializer: @Sendable (TrainingDataset) -> GaussianCloud = { reduced in
            calls.withLock { $0.append(reduced.views.map(\.name)) }
            return initial
        }
        let session = TrainingSession(dataset: dataset, configuration: makeConfiguration(iterations: 400),
                                      initializer: initializer)
        let (outcome, _) = try await runCollecting(session)

        let seen = calls.withLock { $0 }
        #expect(seen.count == 2)
        let heldOutName: String = try #require(outcome.heldOutView)
        #expect(seen.count == 2 && seen[0].count == 3 && seen[1].count == 4)
        #expect(seen.count == 2 && !seen[0].contains(heldOutName))
        #expect(seen.count == 2 && seen[1].contains(heldOutName))
    }

    @Test
    func cancellationInSelectionReturnsModel() async throws {
        let (dataset, initial) = try makeScene(viewCount: 4)
        // No early stop can happen (minimumIterations is far above the run), so cancellation ends run 1.
        var configuration = makeConfiguration(iterations: 100_000, minimumIterations: 100_000)
        configuration.logInterval = 5
        let session = TrainingSession(dataset: dataset, initialCloud: initial, configuration: configuration)
        let task = Task { try await session.runWithOutcome() }
        var last: TrainingProgress?
        var selectionEvents = 0
        for await event in session.progress {
            last = event
            if case .selection = event.phase, event.state == .running {
                selectionEvents += 1
                // `>=`: the stream may drop old events when the consumer lags.
                if selectionEvents >= 3 {
                    task.cancel()
                }
            }
        }
        let outcome = try await task.value

        #expect(last?.state == .cancelled)
        if case .selection = last?.phase {} else {
            Issue.record("cancelled event phase is \(String(describing: last?.phase)), expected selection")
        }
        #expect(outcome.heldOutView != nil)
        #expect(outcome.iterations < 100_000)
        #expect(outcome.cloud.count == initial.count)
        #expect(outcome.cloud.positions.allSatisfy { $0.isFinite })
    }

    @Test
    func overrideNameIsHeldOut() async throws {
        let (dataset, initial) = try makeScene(viewCount: 4)
        var configuration = makeConfiguration(iterations: 300)
        configuration.heldOutViews = ["view2"]
        let calls = Mutex<[[String]]>([])
        let initializer: @Sendable (TrainingDataset) -> GaussianCloud = { reduced in
            calls.withLock { $0.append(reduced.views.map(\.name)) }
            return initial
        }
        let session = TrainingSession(dataset: dataset, configuration: configuration, initializer: initializer)
        let (outcome, _) = try await runCollecting(session)

        #expect(outcome.heldOutView == "view2")
        let seen = calls.withLock { $0 }
        #expect(seen.count == 2)
        #expect(seen.count == 2 && !seen[0].contains("view2"))
    }
}
