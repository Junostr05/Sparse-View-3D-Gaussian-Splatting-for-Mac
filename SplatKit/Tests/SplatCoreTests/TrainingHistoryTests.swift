import Foundation
import Testing
import SplatCore

private func sample(_ iteration: Int, loss: Double? = nil, psnr: Double? = nil) -> TrainingHistory.Sample {
    TrainingHistory.Sample(iteration: iteration,
                           loss: loss,
                           psnr: psnr,
                           gaussianCount: 1000,
                           iterationsPerSecond: 0)
}

private func isClose(_ a: Double, _ b: Double, tolerance: Double = 1e-9) -> Bool {
    abs(a - b) <= tolerance
}

@Suite struct TrainingHistoryTests {
    @Test func startsEmptyWithoutEstimate() {
        let history = TrainingHistory()
        #expect(history.isEmpty)
        #expect(history.latest == nil)
        #expect(history.elapsedSeconds == 0)
        #expect(history.smoothedIterationsPerSecond == 0)
        #expect(history.estimatedRemainingSeconds(totalIterations: 1000) == nil)
    }

    @Test func recordsSamplesWithLossAndPSNR() {
        var history = TrainingHistory()
        history.append(sample(0, loss: nil, psnr: nil), at: 0)
        history.append(sample(100, loss: 0.17, psnr: 21.5), at: 10)
        #expect(history.points.count == 2)
        #expect(history.points[0].loss == nil)
        #expect(history.points[1].loss == 0.17)
        #expect(history.points[1].psnr == 21.5)
        #expect(history.points[1].gaussianCount == 1000)
        #expect(history.latest?.iteration == 100)
    }

    @Test func pausedTimeIsExcludedFromElapsed() {
        var history = TrainingHistory()
        history.append(sample(0), at: 0)
        history.append(sample(100), at: 10)
        history.pause(at: 10)
        // Events while paused record samples but add no active time.
        history.append(sample(100), at: 60)
        history.resume(at: 110)
        history.append(sample(200), at: 120)
        #expect(isClose(history.elapsedSeconds, 20))
        #expect(isClose(history.points.last?.elapsedSeconds ?? -1, 20))
        #expect(history.isPaused == false)
    }

    @Test func pauseAndResumeAreIdempotent() {
        var history = TrainingHistory()
        history.append(sample(0), at: 0)
        history.pause(at: 5)
        history.pause(at: 8)
        history.resume(at: 20)
        history.resume(at: 25)
        history.append(sample(10), at: 30)
        // 5 s before the pause, then 10 s after the resume at 20.
        #expect(isClose(history.elapsedSeconds, 15))
    }

    @Test func smoothedRateFollowsIterationDeltas() {
        var history = TrainingHistory()
        history.append(sample(0), at: 0)
        history.append(sample(100), at: 10)
        // The first delta seeds the average: 100 iterations in 10 s.
        #expect(isClose(history.smoothedIterationsPerSecond, 10))
        history.append(sample(300), at: 20)
        // Instantaneous 20 it/s, smoothed with weight 0.2 against the previous 10 it/s.
        let expected: Double = 0.2 * 20 + 0.8 * 10
        #expect(isClose(history.smoothedIterationsPerSecond, expected))
    }

    @Test func etaIsRemainingOverSmoothedRate() {
        var history = TrainingHistory()
        history.append(sample(0), at: 0)
        history.append(sample(100), at: 10)
        history.append(sample(300), at: 20)
        let rate: Double = history.smoothedIterationsPerSecond
        let eta: Double? = history.estimatedRemainingSeconds(totalIterations: 1000)
        #expect(eta != nil)
        #expect(isClose(eta ?? -1, 700 / rate))
    }

    @Test func etaIsNilForEndlessRunsAndZeroRate() {
        var history = TrainingHistory()
        history.append(sample(0), at: 0)
        // One point has no rate yet.
        #expect(history.estimatedRemainingSeconds(totalIterations: 1000) == nil)
        history.append(sample(100), at: 10)
        #expect(history.estimatedRemainingSeconds(totalIterations: nil) == nil)
        #expect(history.estimatedRemainingSeconds(totalIterations: 1000) != nil)
    }

    @Test func etaIsZeroWhenTotalReached() {
        var history = TrainingHistory()
        history.append(sample(0), at: 0)
        history.append(sample(100), at: 10)
        #expect(history.estimatedRemainingSeconds(totalIterations: 100) == 0)
        #expect(history.estimatedRemainingSeconds(totalIterations: 50) == 0)
    }

    @Test func repeatedIterationReplacesLastPoint() {
        var history = TrainingHistory()
        history.append(sample(0), at: 0)
        history.append(sample(100, loss: 0.5), at: 10)
        history.append(sample(100, loss: 0.4), at: 11)
        #expect(history.points.count == 2)
        #expect(history.points.last?.loss == 0.4)
        #expect(isClose(history.points.last?.elapsedSeconds ?? -1, 11))
    }

    @Test func restartedIterationsAddPointWithoutRateChange() {
        var history = TrainingHistory()
        history.append(sample(0), at: 0)
        history.append(sample(100), at: 10)
        let rate: Double = history.smoothedIterationsPerSecond
        // A second round restarting at a smaller iteration: recorded, but no negative rate.
        history.append(sample(50), at: 12)
        #expect(history.points.count == 3)
        #expect(history.smoothedIterationsPerSecond == rate)
    }

    @Test func capacityIsKeptByDecimatingTheOlderHalf() {
        var history = TrainingHistory()
        for iteration in 0..<TrainingHistory.capacity {
            history.append(sample(iteration), at: Double(iteration))
        }
        #expect(history.points.count == TrainingHistory.capacity)
        #expect(history.points.first?.iteration == 0)

        // The next point triggers decimation: 750 older points keep every other one (375), the newer 750 stay.
        history.append(sample(TrainingHistory.capacity), at: Double(TrainingHistory.capacity))
        #expect(history.points.count == 1126)
        #expect(history.points.first?.iteration == 0)
        #expect(history.points.last?.iteration == TrainingHistory.capacity)
        let newest: [Int] = history.points.suffix(750).map(\.iteration)
        #expect(newest == Array(751...TrainingHistory.capacity))
    }

    @Test func longRunStaysWithinCapacityAndOrdered() {
        var history = TrainingHistory()
        for iteration in 0..<5000 {
            history.append(sample(iteration), at: Double(iteration) * 0.5)
        }
        #expect(history.points.count <= TrainingHistory.capacity)
        #expect(history.points.last?.iteration == 4999)
        let iterations: [Int] = history.points.map(\.iteration)
        #expect(iterations == iterations.sorted())
        #expect(isClose(history.elapsedSeconds, 2499.5))
    }

    @Test func resetClearsEverything() {
        var history = TrainingHistory()
        history.append(sample(0), at: 0)
        history.append(sample(100), at: 10)
        history.pause(at: 12)
        history.reset()
        #expect(history.isEmpty)
        #expect(history.elapsedSeconds == 0)
        #expect(history.smoothedIterationsPerSecond == 0)
        #expect(history.isPaused == false)
        history.append(sample(5), at: 100)
        #expect(history.elapsedSeconds == 0)
    }
}
