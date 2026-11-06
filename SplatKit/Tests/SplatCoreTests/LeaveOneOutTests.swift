import Foundation
import Testing
@testable import SplatCore

// MARK: - Helpers

private func loRandomImage(width: Int, height: Int, seed: UInt64) -> RGBImage {
    var rng = SeededGenerator(seed: seed)
    var pixels: [SIMD3<Float>] = []
    pixels.reserveCapacity(width * height)
    for _ in 0..<(width * height) {
        let r: Float = Float.random(in: 0...1, using: &rng)
        let g: Float = Float.random(in: 0...1, using: &rng)
        let b: Float = Float.random(in: 0...1, using: &rng)
        pixels.append(SIMD3<Float>(r, g, b))
    }
    return RGBImage(width: width, height: height, pixels: pixels)
}

/// Camera centres on a circle of radius 3.5 in the x–z plane at height 1, at the given azimuths (degrees).
private func arcCenters(degrees: [Double]) -> [SIMD3<Float>] {
    var centers: [SIMD3<Float>] = []
    for degree in degrees {
        let radians: Double = degree * Double.pi / 180
        let x: Float = Float(3.5 * sin(radians))
        let z: Float = Float(3.5 * cos(radians))
        centers.append(SIMD3<Float>(x, 1, z))
    }
    return centers
}

/// `count` azimuths evenly spread over a 110° arc, from −55° to +55°.
private func spreadDegrees(_ count: Int) -> [Double] {
    var degrees: [Double] = []
    for i in 0..<count {
        let t: Double = Double(i) / Double(count - 1)
        degrees.append(-55 + 110 * t)
    }
    return degrees
}

private func viewNames(_ count: Int) -> [String] {
    var names: [String] = []
    for i in 0..<count {
        names.append("view_\(i)")
    }
    return names
}

// MARK: - SSIM metric

@Test func ssimMatchesLossSSIM() {
    let a = loRandomImage(width: 16, height: 12, seed: 11)
    let b = loRandomImage(width: 16, height: 12, seed: 12)
    let reference: Double = ImageLoss.evaluate(rendered: a, target: b, lambda: 0.2).ssim
    let value: Double = ImageMetrics.ssim(a, b)
    #expect(abs(value - reference) < 1e-12)
}

@Test func ssimOfIdenticalImagesIsOne() {
    let image = loRandomImage(width: 16, height: 12, seed: 13)
    #expect(ImageMetrics.ssim(image, image) > 1 - 1e-9)
}

// MARK: - Held-out selection

@Test func selectsInteriorCameraOnArc() {
    let policy = EarlyStoppingPolicy()
    let five = HeldOutSelection.select(names: viewNames(5),
                                       cameraCenters: arcCenters(degrees: spreadDegrees(5)),
                                       override: nil, policy: policy)
    #expect(five == 2)
    let four = HeldOutSelection.select(names: viewNames(4),
                                       cameraCenters: arcCenters(degrees: spreadDegrees(4)),
                                       override: nil, policy: policy)
    #expect(four == 1)
}

@Test func neverHoldsOutAnExtremeCamera() {
    // Shuffled order: view_0 (−55°) and view_5 (+55°) are the extremes along the arc.
    let degrees: [Double] = [-55, -33, -11, 11, 33, 55]
    let order: [Int] = [3, 0, 5, 1, 4, 2]
    var names: [String] = []
    var centers: [SIMD3<Float>] = []
    for index in order {
        names.append("view_\(index)")
        centers.append(arcCenters(degrees: [degrees[index]])[0])
    }
    let chosen: Int? = HeldOutSelection.select(names: names, cameraCenters: centers,
                                               override: nil, policy: EarlyStoppingPolicy())
    #expect(chosen != nil)
    guard let chosen else { return }
    #expect(names[chosen] != "view_0")
    #expect(names[chosen] != "view_5")
}

@Test func disabledBelowFourViews() {
    let selected: Int? = HeldOutSelection.select(names: viewNames(3),
                                                 cameraCenters: arcCenters(degrees: spreadDegrees(3)),
                                                 override: nil, policy: EarlyStoppingPolicy())
    #expect(selected == nil)
}

@Test func coincidentCentresReturnNil() {
    let centers = [SIMD3<Float>](repeating: SIMD3<Float>(1, 2, 3), count: 5)
    let selected: Int? = HeldOutSelection.select(names: viewNames(5), cameraCenters: centers,
                                                 override: nil, policy: EarlyStoppingPolicy())
    #expect(selected == nil)
}

@Test func overrideWins() {
    let names = ["view_00", "view_01", "view_02", "view_03", "view_04"]
    let selected: Int? = HeldOutSelection.select(names: names,
                                                 cameraCenters: arcCenters(degrees: spreadDegrees(5)),
                                                 override: "view_03", policy: EarlyStoppingPolicy())
    #expect(selected == 3)
}

// MARK: - Leave-one-out scheduler

@Test func stopsAfterPatienceWithoutImprovement() {
    // Rising SSIM for four evaluations, then flat. Patience 4 must stop the run after the
    // fourth evaluation without a new best EMA, and never before minimumIterations.
    var scheduler = LeaveOneOutScheduler(policy: EarlyStoppingPolicy(), maxIterations: 20_000)
    var stopIteration: Int? = nil
    for k in 1...40 {
        let iteration: Int = 250 * k
        let ssim: Double = k <= 4 ? 0.2 * Double(k) : 0.8
        let metrics = HeldOutMetrics(psnr: 20, ssim: ssim)
        if scheduler.record(iteration: iteration, metrics: metrics) == .stop {
            stopIteration = iteration
            break
        }
    }
    let stop: Int = stopIteration ?? -1
    #expect(stop >= 1_500)
    #expect(scheduler.bestIteration != nil)
    if let best = scheduler.bestIteration {
        #expect(stop == best + 4 * 250)
    }
}

@Test func singleSpikeDoesNotStop() {
    // Steady improvement of 0.02 per evaluation with one lucky evaluation at iteration 1250.
    var scheduler = LeaveOneOutScheduler(policy: EarlyStoppingPolicy(), maxIterations: 20_000)
    var stopped = false
    for k in 1...40 {
        var ssim: Double = 0.3 + 0.02 * Double(k)
        if k == 5 {
            ssim += 0.1
        }
        let metrics = HeldOutMetrics(psnr: 20, ssim: ssim)
        if scheduler.record(iteration: 250 * k, metrics: metrics) == .stop {
            stopped = true
            break
        }
    }
    #expect(!stopped)
    let best: Int = scheduler.bestIteration ?? 0
    #expect(best > 1_250)
}

@Test func noStopBeforeMinimumIterations() {
    // Flat SSIM: patience is exhausted at iteration 1 250, but the stop must wait for 1 500.
    var scheduler = LeaveOneOutScheduler(policy: EarlyStoppingPolicy(), maxIterations: 20_000)
    var firstStop: Int? = nil
    for k in 1...20 {
        let iteration: Int = 250 * k
        let metrics = HeldOutMetrics(psnr: 20, ssim: 0.5)
        if scheduler.record(iteration: iteration, metrics: metrics) == .stop, firstStop == nil {
            firstStop = iteration
        }
    }
    #expect(firstStop == 1_500)
}

@Test func finalIterationsScalesByViewRatio() {
    var scheduler = LeaveOneOutScheduler(policy: EarlyStoppingPolicy(), maxIterations: 5_000)
    _ = scheduler.record(iteration: 3_000, metrics: HeldOutMetrics(psnr: 20, ssim: 0.5))
    #expect(scheduler.finalIterations(viewCount: 4) == 4_000)

    var clamped = LeaveOneOutScheduler(policy: EarlyStoppingPolicy(), maxIterations: 3_500)
    _ = clamped.record(iteration: 3_000, metrics: HeldOutMetrics(psnr: 20, ssim: 0.5))
    #expect(clamped.finalIterations(viewCount: 4) == 3_500)

    let unrecorded = LeaveOneOutScheduler(policy: EarlyStoppingPolicy(), maxIterations: 5_000)
    #expect(unrecorded.finalIterations(viewCount: 4) == 5_000)
}
