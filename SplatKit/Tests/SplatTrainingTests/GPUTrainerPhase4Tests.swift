import Foundation
import Testing
import SplatCore
import SplatMetal
@testable import SplatTraining

// Phase 4 Task 17: GPUTrainer with the staged auxiliary terms (depth loss, anisotropy, scale cap) and
// isolation pruning. Runs only where Metal exists (macOS). The step order under test is the normative
// order of the plan, mirrored from CPUTrainer: render (depth mode for views with a prior), image and
// depth loss, backward with statistics, anisotropy, Adam, scale cap, densification, isolation pruning.
//
// Floating-point note: the rasterizer's backward pass accumulates with float atomics, whose order is not
// deterministic. Two runs of the same GPU configuration differ by up to ~1e-5 relative after 30 steps,
// so GPU-versus-reference comparisons use relative tolerances, not bit equality. The CPU reference is
// deterministic (Double accumulation), so it is the normative definition of every expected value here.

// MARK: - Helpers

/// Metric prior of a depth map: depth domain, L1 enabled (measured depth).
private func metricPrior(_ depth: DepthMap) -> DepthPrior {
    DepthPrior(width: depth.width, height: depth.height, correlationTarget: depth.values,
               domain: .depth, metricDepth: depth.values, isMeasuredMetric: true)
}

/// The scene's dataset with every training view carrying its ground-truth metric prior.
private func datasetWithPriors(_ scene: SyntheticScene) -> TrainingDataset {
    var views: [TrainingView] = scene.train
    for index in views.indices {
        views[index].depth = metricPrior(scene.trainDepth[index])
    }
    return TrainingDataset(views: views, points: scene.points)
}

/// The same prior with every 7th pixel NaN in both the target and the metric depth.
private func priorWithNonFinitePixels(_ depth: DepthMap) -> DepthPrior {
    var target: [Float] = depth.values
    var metric: [Float] = depth.values
    for index in stride(from: 0, to: depth.values.count, by: 7) {
        target[index] = Float.nan
        metric[index] = Float.nan
    }
    return DepthPrior(width: depth.width, height: depth.height, correlationTarget: target,
                      domain: .depth, metricDepth: metric, isMeasuredMetric: true)
}

/// Phase 4 configuration with densification off and a loss read at every step.
private func phase4Configuration(iterations: Int, regularizers: Set<RegularizerKind>) -> TrainingConfiguration {
    var configuration = TrainingConfiguration()
    configuration.iterations = iterations
    configuration.shDegree = 0
    configuration.seed = 11
    configuration.logInterval = 1
    configuration.evalInterval = 0
    configuration.previewInterval = 0
    configuration.regularizers = regularizers
    configuration.densification.startIteration = 0
    configuration.densification.stopIteration = 0
    return configuration
}

/// Same Gaussians with every log-scale replaced by `(log 0.02, log 0.3, log 0.3)`: axis ratio 15.
/// With the default scene extent the scale cap (about `0.05 · radius`, log about −2) is below the
/// `log 0.3` axes, so the cap clamps them in the first step; the ratio 15 exceeds `maxAxisRatio` 10, so the
/// anisotropy gradient is non-zero from the first step.
private func elongated(_ base: GaussianCloud) throws -> GaussianCloud {
    var logScales: [Float] = []
    logScales.reserveCapacity(base.logScales.count)
    for _ in 0..<base.count {
        logScales.append(log(0.02))
        logScales.append(log(0.3))
        logScales.append(log(0.3))
    }
    return try GaussianCloud(shDegree: base.shDegree, positions: base.positions, logScales: logScales,
                             rotations: base.rotations, opacityLogits: base.opacityLogits, sh: base.sh)
}

/// Same Gaussians with `extra` appended: isotropic scale 0.12, opacity 0.9, DC colour grey.
private func appending(_ base: GaussianCloud, positions extra: [SIMD3<Float>]) throws -> GaussianCloud {
    var flatPositions = base.positions
    var logScales = base.logScales
    var rotations = base.rotations
    var logits = base.opacityLogits
    var sh = base.sh
    for p in extra {
        flatPositions.append(p.x)
        flatPositions.append(p.y)
        flatPositions.append(p.z)
        for _ in 0..<3 {
            logScales.append(log(0.12))
        }
        rotations.append(contentsOf: [1, 0, 0, 0])
        logits.append(log(9))
        sh.append(contentsOf: [0, 0, 0])
    }
    return try GaussianCloud(shDegree: base.shDegree, positions: flatPositions, logScales: logScales,
                             rotations: rotations, opacityLogits: logits, sh: sh)
}

/// The ground truth with 20 floaters on a sphere of radius 10 (far outside the cameras' view of the sphere).
private func cloudWithFloaters(_ base: GaussianCloud) throws -> GaussianCloud {
    var floaters: [SIMD3<Float>] = []
    for i in 0..<20 {
        let y: Float = 1 - 2 * (Float(i) + 0.5) / 20
        let ring: Float = (1 - y * y).squareRoot()
        let phi: Float = Float(i) * 2.39996323
        floaters.append(10 * SIMD3<Float>(ring * cos(phi), y, ring * sin(phi)))
    }
    return try appending(base, positions: floaters)
}

/// Number of built-in Phase 4 terms the trainer encodes. The property is private, so it is read through
/// `Mirror`; nil when the property does not exist.
private func builtInTermCount(_ trainer: GPUTrainer) -> Int? {
    (Mirror(reflecting: trainer).descendant("phase4Terms") as? [any AuxiliaryLoss])?.count
}

/// Element-wise closeness with an absolute floor and a relative part: |a − b| ≤ floor + relative·|b|.
/// Returns the number of violating elements; the first one is described in `message`.
private func closenessViolations(_ actual: [Float], _ expected: [Float], relative: Float, floor: Float) -> (count: Int, message: String) {
    guard actual.count == expected.count else {
        return (1, "count \(actual.count) against \(expected.count)")
    }
    var violations = 0
    var message = ""
    for i in actual.indices {
        let delta: Float = abs(actual[i] - expected[i])
        let allowed: Float = floor + relative * abs(expected[i])
        if delta > allowed {
            if violations == 0 {
                message = "index \(i): \(actual[i]) against \(expected[i])"
            }
            violations += 1
        }
    }
    return (violations, message)
}

// MARK: - Tests

@Suite(.serialized, .enabled(if: TrainingTestEnvironment.hasMetal))
struct GPUTrainerPhase4Tests {

    /// With `regularizers == []` the trainer must reproduce the Phase 3 step. The reference is the CPU trainer
    /// with the same configuration (deterministic, Double accumulation), not recorded values, so the test does
    /// not depend on the GPU model. The GPU tolerance is relative (2e-4) because of the float-atomic backward
    /// noise described in the file comment. The built-in Phase 4 terms must be empty, which is checked directly.
    @Test
    func phase3ConfigurationIsUnchanged() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                        gaussians: 160, sparsePoints: 80, seed: 5)
        let dataset = scene.dataset
        let configuration = phase4Configuration(iterations: 30, regularizers: [])
        let initial = GaussianInitializer.initialize(dataset, configuration: configuration, maxCount: 400)

        var cpu = try CPUTrainer(dataset: dataset, initialCloud: initial, configuration: configuration)
        var reference: [Double] = []
        for _ in 0..<configuration.iterations {
            let result = try cpu.step()
            #expect(result.depth.training == nil)
            #expect(result.depth.isolationPruned == 0)
            reference.append(result.loss)
        }

        let trainer = try GPUTrainer(context: context, dataset: dataset, initialCloud: initial,
                                     configuration: configuration)
        #expect(builtInTermCount(trainer) == 0, "regularizers == [] must leave no built-in Phase 4 term")

        var losses: [Double] = []
        for _ in 0..<configuration.iterations {
            let report = try trainer.step()
            #expect(report.depthLoss == nil)
            #expect(report.isolationPruned == nil)
            losses.append(try #require(report.loss).loss)
        }
        #expect(losses.count == reference.count)
        for index in 0..<min(losses.count, reference.count) {
            let tolerance: Double = 2e-4 * abs(reference[index])
            #expect(abs(losses[index] - reference[index]) <= tolerance,
                    "step \(index): loss \(losses[index]) against the CPU reference \(reference[index])")
        }
        #expect(trainer.gaussianCount == cpu.cloud.count)
        #expect(trainer.gaussianCount == 80)
    }

    /// All features on and no depth prior anywhere: the trainer runs, no depth term reports a value, and
    /// nothing becomes non-finite.
    @Test
    func trainsWithoutAnyDepthPrior() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                        gaussians: 160, sparsePoints: 80, seed: 5)
        let configuration = phase4Configuration(iterations: 40, regularizers: Set(RegularizerKind.allCases))
        let initial = GaussianInitializer.initialize(scene.dataset, configuration: configuration, maxCount: 400)
        let trainer = try GPUTrainer(context: context, dataset: scene.dataset, initialCloud: initial,
                                     configuration: configuration)

        for _ in 0..<configuration.iterations {
            let report = try trainer.step()
            #expect(report.depthLoss == nil, "no view has a depth prior, so no depth loss is reported")
            let loss = try #require(report.loss)
            #expect(loss.loss.isFinite)
        }
        let cloud = try trainer.snapshot()
        #expect(cloud.count > 0)
        #expect(cloud.positions.allSatisfy { $0.isFinite })
        #expect(cloud.logScales.allSatisfy { $0.isFinite })
        #expect(cloud.opacityLogits.allSatisfy { $0.isFinite })
        #expect(cloud.sh.allSatisfy { $0.isFinite })
    }

    /// First five steps with metric priors and all features on (densification off): the image loss and
    /// the depth loss of every step agree with `CPUTrainer` to relative 1e-3.
    @Test
    func depthLossMatchesCPUTrainerFirstSteps() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                        gaussians: 160, sparsePoints: 80, seed: 5)
        let dataset = datasetWithPriors(scene)
        let configuration = phase4Configuration(iterations: 5, regularizers: Set(RegularizerKind.allCases))
        let initial = GaussianInitializer.initialize(dataset, configuration: configuration, maxCount: 400)

        var cpu = try CPUTrainer(dataset: dataset, initialCloud: initial, configuration: configuration)
        var cpuImage: [Double] = []
        var cpuDepth: [Double] = []
        for _ in 0..<5 {
            let result = try cpu.step()
            cpuImage.append(result.loss)
            cpuDepth.append(try #require(result.depth.training).loss)
        }

        let gpu = try GPUTrainer(context: context, dataset: dataset, initialCloud: initial,
                                 configuration: configuration)
        for index in 0..<5 {
            let report = try gpu.step()
            let image = try #require(report.loss).loss
            let depth = try #require(report.depthLoss).loss
            print("DEPTHPARITY step \(index): image gpu \(image) cpu \(cpuImage[index]); depth gpu \(depth) cpu \(cpuDepth[index])")
            #expect(abs(image - cpuImage[index]) <= 1e-3 * abs(cpuImage[index]),
                    "step \(index): image loss \(image) against CPU \(cpuImage[index])")
            #expect(abs(depth - cpuDepth[index]) <= 1e-3 * abs(cpuDepth[index]),
                    "step \(index): depth loss \(depth) against CPU \(cpuDepth[index])")
        }
    }

    /// Stage order of the parameter terms: the anisotropy gradient enters before Adam, the scale cap is a
    /// projection after Adam. Five GPU steps on an elongated cloud with both terms must reproduce the CPU
    /// parameters (positions, log-scales, rotations, opacities, SH), and both terms must change the
    /// parameters in step 1 compared with `regularizers == []`, so a wrong order cannot pass unnoticed.
    @Test
    func anisotropyAndScaleCapMatchCPUParameters() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                        gaussians: 160, sparsePoints: 80, seed: 6)
        let dataset = scene.dataset
        let configuration = phase4Configuration(iterations: 5, regularizers: [.anisotropy, .scaleCap])
        let initial = try elongated(GaussianInitializer.initialize(dataset, configuration: configuration,
                                                                   maxCount: 400))

        // Both terms act in step 1: each one alone changes the parameters relative to no regularizer.
        var plainConfiguration = configuration
        plainConfiguration.regularizers = []
        var anisotropyConfiguration = configuration
        anisotropyConfiguration.regularizers = [.anisotropy]
        var capConfiguration = configuration
        capConfiguration.regularizers = [.scaleCap]
        var plain = try CPUTrainer(dataset: dataset, initialCloud: initial, configuration: plainConfiguration)
        var anisotropyOnly = try CPUTrainer(dataset: dataset, initialCloud: initial,
                                            configuration: anisotropyConfiguration)
        var capOnly = try CPUTrainer(dataset: dataset, initialCloud: initial, configuration: capConfiguration)
        _ = try plain.step()
        _ = try anisotropyOnly.step()
        _ = try capOnly.step()
        let anisotropyChangesStepOne = closenessViolations(anisotropyOnly.cloud.logScales, plain.cloud.logScales,
                                                           relative: 0, floor: 1e-6).count > 0
        let capChangesStepOne = closenessViolations(capOnly.cloud.logScales, plain.cloud.logScales,
                                                    relative: 0, floor: 1e-6).count > 0
        #expect(anisotropyChangesStepOne, "the anisotropy term must change the log-scales in step 1")
        #expect(capChangesStepOne, "the scale cap must change the log-scales in step 1")

        // Five steps, CPU against GPU.
        var cpu = try CPUTrainer(dataset: dataset, initialCloud: initial, configuration: configuration)
        for _ in 0..<configuration.iterations {
            _ = try cpu.step()
        }
        let gpu = try GPUTrainer(context: context, dataset: dataset, initialCloud: initial,
                                 configuration: configuration)
        for _ in 0..<configuration.iterations {
            try gpu.step()
        }
        let result = try gpu.snapshot()
        #expect(result.count == cpu.cloud.count)

        // Relative 1e-3 with an absolute floor of 1e-4 (the float-atomic noise is far below this for positions,
        // log-scales, opacities and SH). A wrong stage order moves log-scales by about the Adam step (5e-3) or by
        // the whole cap difference, well above the tolerance.
        //
        // Rotations get a wider bound, and no Phase 4 term touches them, so they are a sanity bound only. Adam turns
        // a noise-level GPU gradient into a full ±lr step where the Double CPU gradient is much smaller, so
        // rotations drift by up to about lr per step even with `regularizers == []` (measured on this scene:
        // 4.4e-3 after 5 steps without any Phase 4 term). The bound is 2 · steps · lr.
        let relative: Float = 1e-3
        let floor: Float = 1e-4
        let rotationFloor: Float = 2 * Float(configuration.iterations) * configuration.learningRates.rotation
        let checks: [(String, [Float], [Float], Float)] = [
            ("positions", result.positions, cpu.cloud.positions, floor),
            ("log-scales", result.logScales, cpu.cloud.logScales, floor),
            ("rotations", result.rotations, cpu.cloud.rotations, rotationFloor),
            ("opacities", result.opacityLogits, cpu.cloud.opacityLogits, floor),
            ("SH", result.sh, cpu.cloud.sh, floor),
        ]
        for (name, actual, expected, allowedFloor) in checks {
            let outcome = closenessViolations(actual, expected, relative: relative, floor: allowedFloor)
            #expect(outcome.count == 0, "\(name): \(outcome.count) values outside tolerance, first at \(outcome.message)")
        }
    }

    /// Mirrors the CPU test: 20 floaters far from the sphere are removed at the first isolation boundary
    /// (iteration 10), and the rest of the model is kept. The steps after the boundary run as well: the
    /// losses stay finite and the Gaussian count stays constant.
    @Test
    func isolationPruningRemovesInjectedFloaters() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                        gaussians: 400, sparsePoints: 100, seed: 8)
        let base = scene.groundTruth
        let cloud = try cloudWithFloaters(base)

        var configuration = phase4Configuration(iterations: 12, regularizers: [.isolationPrune])
        configuration.artifacts.isolationStartIteration = 10
        configuration.artifacts.isolationInterval = 10
        let trainer = try GPUTrainer(context: context, dataset: scene.dataset, initialCloud: cloud,
                                     configuration: configuration)
        for _ in 0..<9 {
            let report = try trainer.step()
            #expect(report.isolationPruned == nil, "no isolation boundary before iteration 10")
        }
        #expect(trainer.gaussianCount == cloud.count, "no pruning before the first boundary")

        let boundary = try trainer.step()
        print("GPUISOLATION pruned at first boundary = \(boundary.isolationPruned ?? -1), count \(trainer.gaussianCount)")
        #expect(boundary.isolationPruned == 20)
        #expect(trainer.gaussianCount == base.count)
        let survivors = try trainer.snapshot()
        var farthest: Float = 0
        for i in 0..<survivors.count {
            let p = SIMD3<Float>(survivors.positions[3 * i], survivors.positions[3 * i + 1],
                                 survivors.positions[3 * i + 2])
            farthest = max(farthest, (p * p).sum().squareRoot())
        }
        #expect(farthest < 5, "floater still present at distance \(farthest)")

        // Iterations 11 and 12 run after the compaction: the next boundary (20) is past the run.
        for _ in 0..<2 {
            let report = try trainer.step()
            #expect(report.isolationPruned == nil, "no second isolation boundary inside 12 iterations")
            let loss = try #require(report.loss)
            #expect(loss.loss.isFinite, "loss after the prune must stay finite")
            #expect(trainer.gaussianCount == base.count, "the count must stay constant after the prune")
        }
        let after = try trainer.snapshot()
        #expect(after.count == base.count)
        #expect(after.positions.allSatisfy { $0.isFinite })
        #expect(after.logScales.allSatisfy { $0.isFinite })
        #expect(after.opacityLogits.allSatisfy { $0.isFinite })
        #expect(after.sh.allSatisfy { $0.isFinite })
    }

    /// A densification boundary and an isolation boundary on the SAME step (iteration 10), with the initial
    /// capacity forced to the initial count so that densification must grow the capacity first. The prune then
    /// runs on the grown store. The CPU reference runs the same setup: its densification and its isolation
    /// pruned count are compared with the GPU. Gradient threshold 0 makes every visible Gaussian a growth
    /// candidate, so the densify decisions depend on visibility and scale only, not on the float gradient value.
    @Test
    func densificationAndIsolationOnSameStepAfterCapacityGrowth() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                        gaussians: 400, sparsePoints: 100, seed: 8)
        let cloud = try cloudWithFloaters(scene.groundTruth)

        var configuration = phase4Configuration(iterations: 12, regularizers: [.isolationPrune])
        configuration.densification.startIteration = 5
        configuration.densification.stopIteration = 12
        configuration.densification.interval = 5
        configuration.densification.gradientThreshold = 0
        configuration.artifacts.isolationStartIteration = 10
        configuration.artifacts.isolationInterval = 10

        // CPU reference of the same setup: densify plan and isolation count at iteration 10.
        var cpu = try CPUTrainer(dataset: scene.dataset, initialCloud: cloud, configuration: configuration)
        var cpuDensify: DensifyPlan?
        var cpuPruned = -1
        for _ in 0..<configuration.iterations {
            let result = try cpu.step()
            if result.iteration == 10 {
                cpuDensify = result.densified
                cpuPruned = result.depth.isolationPruned
            }
        }
        let cpuPlan = try #require(cpuDensify)

        let trainer = try GPUTrainer(context: context, dataset: scene.dataset, initialCloud: cloud,
                                     configuration: configuration, capacityOverride: cloud.count)
        var gpuSummary: DensifySummary?
        var gpuPruned: Int?
        var previousCount = cloud.count
        for _ in 0..<configuration.iterations {
            let report = try trainer.step()
            if let loss = report.loss {
                #expect(loss.loss.isFinite, "image loss at iteration \(report.iteration) must stay finite")
            }
            if report.iteration == 10 {
                gpuSummary = report.densify
                gpuPruned = report.isolationPruned
            }
            if report.iteration > 10 {
                #expect(trainer.gaussianCount == previousCount, "no growth or pruning after iteration 10")
            }
            previousCount = trainer.gaussianCount
        }

        let summary = try #require(gpuSummary)
        #expect(summary.grewCapacity, "densification must grow the capacity in this setup")
        #expect(summary.after > summary.before, "densification must add Gaussians")
        // The two trainers must make the same discrete decisions in this setup; otherwise the pruned counts
        // are not comparable and the failure message below points at the densification, not the prune.
        #expect(summary.cloned == cpuPlan.cloned, "cloned: GPU \(summary.cloned) against CPU \(cpuPlan.cloned)")
        #expect(summary.split == cpuPlan.split, "split: GPU \(summary.split) against CPU \(cpuPlan.split)")
        #expect(summary.pruned == cpuPlan.pruned, "densify pruned: GPU \(summary.pruned) against CPU \(cpuPlan.pruned)")
        #expect(gpuPruned == cpuPruned, "isolation pruned: GPU \(gpuPruned ?? -1) against CPU \(cpuPruned)")
        #expect(cpuPruned > 0, "the isolation boundary must remove the injected floaters on the CPU reference")
        #expect(trainer.gaussianCount == summary.after - (gpuPruned ?? 0))
        print("SAMESTEP densify \(summary.before) -> \(summary.after) cloned \(summary.cloned) split \(summary.split), isolation pruned GPU \(gpuPruned ?? -1) CPU \(cpuPruned)")
    }

    /// Anisotropy and scale cap on an elongated start: every Gaussian respects the cap (checked on each axis)
    /// and the 95th percentile of the axis ratio is at most 12.
    @Test
    func anisotropyAndScaleCapApplied() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                        gaussians: 160, sparsePoints: 80, seed: 6)
        let configuration = phase4Configuration(iterations: 200, regularizers: [.anisotropy, .scaleCap])
        let initial = try elongated(GaussianInitializer.initialize(scene.dataset, configuration: configuration,
                                                                   maxCount: 400))
        let trainer = try GPUTrainer(context: context, dataset: scene.dataset, initialCloud: initial,
                                     configuration: configuration)
        for _ in 0..<configuration.iterations {
            try trainer.step()
        }
        let cloud = try trainer.snapshot()
        let extent = trainer.sceneExtent
        let fraction: Double = Double(configuration.artifacts.scaleCapFraction)

        var overCap = 0
        var ratios: [Double] = []
        for i in 0..<cloud.count {
            let px = Double(cloud.positions[3 * i])
            let py = Double(cloud.positions[3 * i + 1])
            let pz = Double(cloud.positions[3 * i + 2])
            let dx = px - Double(extent.center.x)
            let dy = py - Double(extent.center.y)
            let dz = pz - Double(extent.center.z)
            let distance = (dx * dx + dy * dy + dz * dz).squareRoot()
            let cap = log(fraction * max(Double(extent.radius), distance))
            var maxLog = Double(cloud.logScales[3 * i])
            var minLog = maxLog
            for axis in 0..<3 {
                let value = Double(cloud.logScales[3 * i + axis])
                if value > cap + 1e-4 {
                    overCap += 1
                }
                maxLog = max(maxLog, value)
                minLog = min(minLog, value)
            }
            ratios.append(exp(maxLog - minLog))
        }
        ratios.sort()
        let p95 = ratios[Int((0.95 * Double(ratios.count - 1)).rounded(.down))]
        print("GPUANISOTROPY 95th percentile axis ratio = \(p95), axes over cap = \(overCap)")
        #expect(overCap == 0, "\(overCap) log-scales above the scale cap")
        #expect(p95 <= 12, "95th percentile ratio \(p95)")
    }

    /// NaN prior pixels are invalid for the depth loss, so every step's depth loss stays finite and the
    /// trainer does not throw.
    @Test
    func nonFiniteDepthPriorIsHarmless() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                        gaussians: 160, sparsePoints: 80, seed: 5)
        var views: [TrainingView] = scene.train
        for index in views.indices {
            views[index].depth = priorWithNonFinitePixels(scene.trainDepth[index])
        }
        let dataset = TrainingDataset(views: views, points: scene.points)
        let configuration = phase4Configuration(iterations: 10, regularizers: [.depthPearson, .depthL1])
        let initial = GaussianInitializer.initialize(dataset, configuration: configuration, maxCount: 400)
        let trainer = try GPUTrainer(context: context, dataset: dataset, initialCloud: initial,
                                     configuration: configuration)
        var depthSteps = 0
        for _ in 0..<configuration.iterations {
            let report = try trainer.step()
            if let depth = report.depthLoss {
                depthSteps += 1
                #expect(depth.loss.isFinite)
            }
        }
        #expect(depthSteps > 0, "the depth term never reported a value")
        #expect(try trainer.snapshot().positions.allSatisfy { $0.isFinite })
    }

    /// A non-finite depth loss must stop the trainer with `.nonFiniteLoss` at the step that produced it.
    /// The route is a prior that is valid by the prior's own rules (finite, positive) but large enough that the
    /// GPU's float L1 sums overflow: every metric depth is 1e38, so |D − m| ≈ 1e38 per pixel. The CPU reference
    /// accumulates in Double and stays finite on the same input, which the test checks as a control. The GPU
    /// throw is therefore a float-range limit of the GPU path, not a fault of the data.
    @Test
    func overflowingDepthLossThrowsNonFiniteLoss() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                        gaussians: 160, sparsePoints: 80, seed: 5)
        var views: [TrainingView] = scene.train
        for index in views.indices {
            let depth: DepthMap = scene.trainDepth[index]
            let huge: [Float] = [Float](repeating: 1e38, count: depth.values.count)
            views[index].depth = DepthPrior(width: depth.width, height: depth.height,
                                            correlationTarget: depth.values, domain: .depth,
                                            metricDepth: huge, isMeasuredMetric: true)
        }
        let dataset = TrainingDataset(views: views, points: scene.points)
        let configuration = phase4Configuration(iterations: 3, regularizers: [.depthL1])
        let initial = GaussianInitializer.initialize(dataset, configuration: configuration, maxCount: 400)

        // Control: the CPU reference is finite on this input.
        var cpu = try CPUTrainer(dataset: dataset, initialCloud: initial, configuration: configuration)
        let cpuStep = try cpu.step()
        #expect(cpuStep.loss.isFinite)
        #expect(try #require(cpuStep.depth.training).loss.isFinite, "the CPU depth loss must stay finite")

        let trainer = try GPUTrainer(context: context, dataset: dataset, initialCloud: initial,
                                     configuration: configuration)
        do {
            try trainer.step()
            Issue.record("expected nonFiniteLoss(iteration: 0) from the overflowing depth loss")
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
