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
// so GPU-versus-reference comparisons use relative tolerances, not bit equality.

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

/// Phase 3 reference: per-step losses of the unmodified Phase 3 `GPUTrainer` on the scenario of
/// `phase3ConfigurationIsUnchanged`, recorded before Task 17 changed the trainer (macOS, this Mac).
private let phase3LossReference: [Double] = [
    0.249256528456, 0.239452477025, 0.231359400613, 0.221157304024, 0.216554478373,
    0.224202199481, 0.220134638084, 0.211408328576, 0.214144924969, 0.204102405792,
    0.203679624852, 0.188436119609, 0.193697429570, 0.193132292880, 0.198472118061,
    0.176168838647, 0.181764308142, 0.181772215819, 0.191321147497, 0.167082393996,
    0.165266611341, 0.187210156869, 0.171855001644, 0.167158199202, 0.183715619992,
    0.158418582039, 0.162770803286, 0.165855131213, 0.155441490297, 0.178924414526,
]

// MARK: - Tests

@Suite(.serialized, .enabled(if: TrainingTestEnvironment.hasMetal))
struct GPUTrainerPhase4Tests {

    /// With `regularizers == []` the trainer must reproduce the Phase 3 loss sequence. The tolerance is
    /// relative (2e-4) because of the float-atomic backward noise described in the file comment; the
    /// measured run-to-run spread on the reference scenario is below 5e-5 relative.
    @Test
    func phase3ConfigurationIsUnchanged() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                        gaussians: 160, sparsePoints: 80, seed: 5)
        let dataset = scene.dataset
        let configuration = phase4Configuration(iterations: 30, regularizers: [])
        let initial = GaussianInitializer.initialize(dataset, configuration: configuration, maxCount: 400)
        let trainer = try GPUTrainer(context: context, dataset: dataset, initialCloud: initial,
                                     configuration: configuration)

        var losses: [Double] = []
        for _ in 0..<configuration.iterations {
            let report = try trainer.step()
            #expect(report.depthLoss == nil)
            #expect(report.isolationPruned == nil)
            losses.append(try #require(report.loss).loss)
        }
        #expect(losses.count == phase3LossReference.count)
        for (index, expected) in phase3LossReference.enumerated() where index < losses.count {
            let tolerance: Double = 2e-4 * expected
            #expect(abs(losses[index] - expected) <= tolerance,
                    "step \(index): loss \(losses[index]) against the Phase 3 reference \(expected)")
        }
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

    /// 200 steps with anisotropy and the scale cap on an elongated start: every Gaussian respects the cap
    /// (checked on each axis) and the 95th percentile of the axis ratio is at most 12.
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

    /// Mirrors the CPU test: 20 floaters far from the sphere are removed at the first isolation boundary
    /// (iteration 10), and the rest of the model is kept.
    @Test
    func isolationPruningRemovesInjectedFloaters() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                        gaussians: 400, sparsePoints: 100, seed: 8)
        let base = scene.groundTruth
        var floaters: [SIMD3<Float>] = []
        for i in 0..<20 {
            let y: Float = 1 - 2 * (Float(i) + 0.5) / 20
            let ring: Float = (1 - y * y).squareRoot()
            let phi: Float = Float(i) * 2.39996323
            floaters.append(10 * SIMD3<Float>(ring * cos(phi), y, ring * sin(phi)))
        }
        let cloud = try appending(base, positions: floaters)

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
}
