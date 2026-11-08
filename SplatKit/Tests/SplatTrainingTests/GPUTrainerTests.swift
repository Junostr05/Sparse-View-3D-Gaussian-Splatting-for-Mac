import Foundation
import Metal
import Testing
import SplatCore
import SplatMetal
@testable import SplatTraining

// Metal training loop. Runs only where Metal exists (macOS CI). Scenes are small by design: the CI GPU
// is a slow paravirtual device. `.serialized`: the synchronous GPU work blocks threads and competes for
// the one GPU, so suites must not run in parallel. Wall-clock assertions are generous on purpose.
// Not verified on Linux: compiled and run on macOS CI only.
@Suite(.serialized, .enabled(if: TrainingTestEnvironment.hasMetal))
struct GPUTrainerTests {

    /// Densification off: the window is empty, so no statistics are collected and no boundary runs.
    private func noDensification(_ configuration: inout TrainingConfiguration) {
        configuration.densification.startIteration = 0
        configuration.densification.stopIteration = 0
    }

    /// Repeats one Gaussian's values for `count` Gaussians (`values.count` floats per Gaussian).
    private func perGaussian(_ values: [Float], count: Int) -> [Float] {
        var result: [Float] = []
        result.reserveCapacity(values.count * count)
        for _ in 0..<count {
            result.append(contentsOf: values)
        }
        return result
    }

    /// Ground truth for the CPU comparison: three degree-0 Gaussians in front of a 24×24 camera.
    /// Scales are unequal and the rotations are not identity, so every parameter gradient is non-zero
    /// on the first step. (With identical scales and identity rotations the true rotation gradient is
    /// exactly zero, and Float noise in the GPU gradient would move the rotations while the Double CPU
    /// reference does not.)
    private func comparisonGroundTruth() throws -> GaussianCloud {
        let positions: [Float] = [0, 0, 2, 0.4, 0.2, 3, -0.4, -0.2, 2.5]
        let logScaleA: Float = Float(log(0.3))
        let logScaleB: Float = Float(log(0.2))
        let logScaleC: Float = Float(log(0.15))
        let logScales: [Float] = perGaussian([logScaleA, logScaleB, logScaleC], count: 3)
        let rotations: [Float] = perGaussian([0.9, 0.2, -0.3, 0.1], count: 3)
        let opacityLogits: [Float] = [1.0, 0.5, 0.8]
        let sh: [Float] = [0.8, -0.3, 0.1, -0.5, 0.6, 0.2, 0.3, 0.1, -0.4]
        return try GaussianCloud(shDegree: 0, positions: positions, logScales: logScales,
                                 rotations: rotations, opacityLogits: opacityLogits, sh: sh)
    }

    /// Starting point for the CPU comparison: the ground truth's positions shifted, different scales and
    /// rotations, and DC colours at 0.5 (SH DC = 0). The same comparison is run on the CPU and the GPU.
    private func comparisonInitialCloud() throws -> GaussianCloud {
        let positions: [Float] = [0.05, -0.05, 2, 0.45, 0.15, 3, -0.35, -0.25, 2.5]
        let logScaleA: Float = Float(log(0.2))
        let logScaleB: Float = Float(log(0.25))
        let logScaleC: Float = Float(log(0.18))
        let logScales: [Float] = perGaussian([logScaleA, logScaleB, logScaleC], count: 3)
        let rotations: [Float] = perGaussian([0.8, -0.1, 0.3, 0.2], count: 3)
        let opacityLogits: [Float] = [1.0, 0.5, 0.8]
        let sh: [Float] = [Float](repeating: 0, count: 9)
        return try GaussianCloud(shDegree: 0, positions: positions, logScales: logScales,
                                 rotations: rotations, opacityLogits: opacityLogits, sh: sh)
    }

    @Test
    func overfitsSmallScene() throws {
        let context = try MetalContext.makeDefault()
        let groundTruth = try makeRandomCloud(count: 500, seed: 11, shDegree: 0)
        // Baseline 1.0 between the cameras: scene radius ≈ 0.55, so position learning rates move positions.
        let views = try makeViews(context: context, groundTruth: groundTruth,
                                  cameras: [makeCamera(width: 64, height: 64),
                                            makeCamera(width: 64, height: 64, shift: 1.0)])
        let initial = try makeRandomCloud(count: 500, seed: 97, shDegree: 0)
        var configuration = TrainingConfiguration.testBaseline()
        // The CPU reference trainer has no appearance compensation (on by default in the GPU trainer).
        configuration.appearance.isEnabled = false
        configuration.shDegree = 0
        configuration.iterations = 300
        configuration.logInterval = 50
        configuration.evalInterval = 0
        configuration.previewInterval = 0
        configuration.seed = 3
        noDensification(&configuration)
        let dataset = TrainingDataset(views: views)

        let trainer = try GPUTrainer(context: context, dataset: dataset, initialCloud: initial,
                                     configuration: configuration)
        let before = try meanL1(context: context, cloud: initial, views: views)
        let started = ContinuousClock.now
        for _ in 0..<configuration.iterations {
            try trainer.step()
        }
        let elapsed = started.duration(to: ContinuousClock.now)
        let after = try meanL1(context: context, cloud: try trainer.snapshot(), views: views)
        print("overfitsSmallScene: mean L1 before \(before), after \(after), elapsed \(elapsed)")

        #expect(after <= 0.5 * before, "L1 went from \(before) to \(after)")
        #expect(elapsed < .seconds(120), "training took \(elapsed)")
    }

    @Test
    func overfitWithDensificationStaysWithinCap() throws {
        let context = try MetalContext.makeDefault()
        let groundTruth = try makeRandomCloud(count: 500, seed: 11, shDegree: 0)
        let views = try makeViews(context: context, groundTruth: groundTruth,
                                  cameras: [makeCamera(width: 48, height: 48),
                                            makeCamera(width: 48, height: 48, shift: 0.2)])
        let initial = try makeRandomCloud(count: 500, seed: 97, shDegree: 0)
        var configuration = TrainingConfiguration.testBaseline()
        // The CPU reference trainer has no appearance compensation (on by default in the GPU trainer).
        configuration.appearance.isEnabled = false
        configuration.shDegree = 0
        configuration.iterations = 200
        configuration.logInterval = 50
        configuration.evalInterval = 0
        configuration.previewInterval = 0
        configuration.maxGaussians = 800
        configuration.densification.startIteration = 50
        configuration.densification.stopIteration = 150
        configuration.densification.interval = 50
        let dataset = TrainingDataset(views: views)

        let trainer = try GPUTrainer(context: context, dataset: dataset, initialCloud: initial,
                                     configuration: configuration)
        var densifications = 0
        var largestCount = 0
        for _ in 0..<configuration.iterations {
            let report = try trainer.step()
            if report.densify != nil {
                densifications += 1
            }
            largestCount = max(largestCount, trainer.gaussianCount)
            #expect(trainer.gaussianCount <= trainer.maxGaussians)
        }
        // Boundaries 50 and 100 (the window ends before 150).
        #expect(densifications == 2)
        #expect(trainer.gaussianCount != 500, "densification did not change the count")
        #expect(largestCount <= 800)

        let cloud = try trainer.snapshot()
        #expect(cloud.count == trainer.gaussianCount)
        #expect(cloud.positions.allSatisfy { $0.isFinite })
        #expect(cloud.logScales.allSatisfy { $0.isFinite })
        #expect(cloud.rotations.allSatisfy { $0.isFinite })
        #expect(cloud.opacityLogits.allSatisfy { $0.isFinite })
        #expect(cloud.sh.allSatisfy { $0.isFinite })
    }

    /// Forces a densify that grows the capacity (D2 would not reach it in a small scene), so the
    /// reallocation of parameters, moments, statistics and gradient buffers is exercised.
    @Test
    func densifyGrowthRebuildsGradientBuffers() throws {
        let context = try MetalContext.makeDefault()
        let groundTruth = try makeRandomCloud(count: 100, seed: 51, shDegree: 0)
        let views = try makeViews(context: context, groundTruth: groundTruth,
                                  cameras: [makeCamera(width: 32, height: 32),
                                            makeCamera(width: 32, height: 32, shift: 0.2)])
        let initial = try makeRandomCloud(count: 100, seed: 52, shDegree: 0)
        var configuration = TrainingConfiguration.testBaseline()
        // The CPU reference trainer has no appearance compensation (on by default in the GPU trainer).
        configuration.appearance.isEnabled = false
        configuration.shDegree = 0
        configuration.iterations = 60
        configuration.logInterval = 10
        configuration.evalInterval = 0
        configuration.previewInterval = 0
        configuration.maxGaussians = 400
        configuration.densification.startIteration = 10
        configuration.densification.stopIteration = 50
        configuration.densification.interval = 10
        configuration.densification.gradientThreshold = 0

        let trainer = try GPUTrainer(context: context, dataset: TrainingDataset(views: views),
                                     initialCloud: initial, configuration: configuration,
                                     capacityOverride: 100)
        var growths = 0
        for _ in 0..<configuration.iterations {
            let report = try trainer.step()
            if report.densify?.grewCapacity == true {
                growths += 1
            }
            #expect(trainer.gaussianCount <= 400)
        }
        #expect(growths >= 1, "no densify grew the capacity")
        let cloud = try trainer.snapshot()
        #expect(cloud.count == trainer.gaussianCount)
        #expect(cloud.positions.allSatisfy { $0.isFinite })
    }

    @Test
    func matchesCPUTrainerForFirstSteps() throws {
        let context = try MetalContext.makeDefault()
        let groundTruth = try comparisonGroundTruth()
        let camera = makeCamera(width: 24, height: 24)
        let settings = RasterSettings(shDegreeLimit: 0, background: .zero)
        let target = CPURasterizer.render(groundTruth, camera: camera, settings: settings)
        let dataset = TrainingDataset(views: [TrainingView(name: "comparison", camera: camera, image: target)])
        let initial = try comparisonInitialCloud()
        var configuration = TrainingConfiguration.testBaseline()
        // The CPU reference trainer has no appearance compensation (on by default in the GPU trainer).
        configuration.appearance.isEnabled = false
        configuration.shDegree = 0
        configuration.iterations = 5
        configuration.logInterval = 1
        configuration.evalInterval = 0
        configuration.previewInterval = 0
        configuration.seed = 1
        noDensification(&configuration)
        // Default regularizers (all Phase 4 switches on). No depth prior exists in this scenario, so the
        // only active terms are anisotropy and the scale cap, which both trainers apply in the same order.

        var cpu = try CPUTrainer(dataset: dataset, initialCloud: initial, configuration: configuration)
        for _ in 0..<5 {
            _ = try cpu.step()
        }
        let gpu = try GPUTrainer(context: context, dataset: dataset, initialCloud: initial,
                                 configuration: configuration)
        for _ in 0..<5 {
            try gpu.step()
        }
        let result = try gpu.snapshot()

        #expect(result.count == cpu.cloud.count)
        #expect(allClose(result.positions, cpu.cloud.positions), "positions differ from the CPU reference")
        #expect(allClose(result.logScales, cpu.cloud.logScales), "log scales differ from the CPU reference")
        #expect(allClose(result.rotations, cpu.cloud.rotations), "rotations differ from the CPU reference")
        #expect(allClose(result.opacityLogits, cpu.cloud.opacityLogits), "opacity differs from the CPU reference")
        #expect(allClose(result.sh, cpu.cloud.sh), "SH differs from the CPU reference")
    }

    /// |a − b| ≤ 1e-4 + 1e-3·|b| element-wise.
    private func allClose(_ a: [Float], _ b: [Float]) -> Bool {
        guard a.count == b.count else { return false }
        for i in 0..<a.count {
            let tolerance: Float = 1e-4 + 1e-3 * abs(b[i])
            let delta: Float = abs(a[i] - b[i])
            if delta > tolerance {
                return false
            }
        }
        return true
    }

    @Test
    func viewWithNoVisibleGaussiansIsHarmless() throws {
        let context = try MetalContext.makeDefault()
        let cloud = try makeRandomCloud(count: 200, seed: 5, shDegree: 0)
        // Translated 100 units sideways: every Gaussian projects far outside the 32×32 image.
        let away = makeCamera(width: 32, height: 32, shift: 100)
        let blank = RGBImage(width: 32, height: 32, pixels: [SIMD3<Float>](repeating: .zero, count: 32 * 32))
        let dataset = TrainingDataset(views: [TrainingView(name: "away", camera: away, image: blank)])
        var configuration = TrainingConfiguration.testBaseline()
        // The CPU reference trainer has no appearance compensation (on by default in the GPU trainer).
        configuration.appearance.isEnabled = false
        configuration.shDegree = 0
        configuration.iterations = 2
        configuration.logInterval = 1
        configuration.evalInterval = 0
        configuration.previewInterval = 0
        noDensification(&configuration)
        // The anisotropy gradient does not depend on visibility, so with the Phase 4 defaults a first Adam step
        // would move the log-scales of this invisible cloud (as on the CPU). This test checks the Phase 3
        // visibility path, so the regularizers are off.
        configuration.regularizers = []

        let trainer = try GPUTrainer(context: context, dataset: dataset, initialCloud: cloud,
                                     configuration: configuration)
        let report = try trainer.step()

        #expect(report.visibleGaussians == 0)
        let loss = try #require(report.loss)
        #expect(loss.loss.isFinite)
        // Zero gradients and zero moments give a zero Adam update on the first step.
        let after = try trainer.snapshot()
        #expect(maxAbsDifference(after, cloud) <= 1e-7)
    }

    @Test
    func trainsViewsWithDifferentResolutions() throws {
        let context = try MetalContext.makeDefault()
        let groundTruth = try makeRandomCloud(count: 150, seed: 21, shDegree: 0)
        let views = try makeViews(context: context, groundTruth: groundTruth,
                                  cameras: [makeCamera(width: 64, height: 48),
                                            makeCamera(width: 32, height: 32, shift: 0.1)])
        let initial = try makeRandomCloud(count: 150, seed: 22, shDegree: 0)
        var configuration = TrainingConfiguration.testBaseline()
        // The CPU reference trainer has no appearance compensation (on by default in the GPU trainer).
        configuration.appearance.isEnabled = false
        configuration.shDegree = 0
        configuration.iterations = 60
        configuration.logInterval = 20
        configuration.evalInterval = 0
        configuration.previewInterval = 0
        noDensification(&configuration)

        let trainer = try GPUTrainer(context: context, dataset: TrainingDataset(views: views),
                                     initialCloud: initial, configuration: configuration)
        let psnrBefore = try trainer.evaluatePSNR()
        for _ in 0..<configuration.iterations {
            try trainer.step()
        }
        let psnrAfter = try trainer.evaluatePSNR()
        print("trainsViewsWithDifferentResolutions: PSNR before \(psnrBefore) dB, after \(psnrAfter) dB")
        #expect(psnrAfter > psnrBefore, "PSNR did not improve: \(psnrBefore) → \(psnrAfter)")
    }

    /// The NaN target is in the only view. The loss of step 0 is NaN; the check happens at the start of
    /// step 1 (the previous step's loss), so the SECOND `step()` throws with index 0, and every later
    /// call (step, snapshot, evaluatePSNR) throws the same error.
    @Test
    func nonFiniteLossStopsWithError() throws {
        let context = try MetalContext.makeDefault()
        let groundTruth = try makeRandomCloud(count: 100, seed: 31, shDegree: 0)
        var views = try makeViews(context: context, groundTruth: groundTruth,
                                  cameras: [makeCamera(width: 32, height: 32)])
        views[0].image.pixels[0] = SIMD3<Float>(Float.nan, 0, 0)
        let initial = try makeRandomCloud(count: 100, seed: 32, shDegree: 0)
        var configuration = TrainingConfiguration.testBaseline()
        // The CPU reference trainer has no appearance compensation (on by default in the GPU trainer).
        configuration.appearance.isEnabled = false
        configuration.shDegree = 0
        configuration.iterations = 100
        configuration.logInterval = 10
        configuration.evalInterval = 0
        configuration.previewInterval = 0
        noDensification(&configuration)

        let trainer = try GPUTrainer(context: context, dataset: TrainingDataset(views: views),
                                     initialCloud: initial, configuration: configuration)
        try trainer.step()
        do {
            try trainer.step()
            Issue.record("expected a nonFiniteLoss error for step 0 at the second call")
        } catch {
            // `step()` throws typed `TrainingError`, so `error` already has that type here.
            #expect(error == .nonFiniteLoss(iteration: 0))
        }
        do {
            try trainer.step()
            Issue.record("the failure must be sticky for step()")
        } catch {
            #expect(error == .nonFiniteLoss(iteration: 0))
        }
        do {
            _ = try trainer.snapshot()
            Issue.record("the failure must be sticky for snapshot()")
        } catch {
            #expect(error == .nonFiniteLoss(iteration: 0))
        }
    }
}
