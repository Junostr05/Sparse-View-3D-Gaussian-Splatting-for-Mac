import Foundation
import Testing
@testable import SplatCore

// Phase 4 Task 11: synthetic ground-truth scene and the CPU trainer with the Phase 4 terms.
// The step order under test is the normative order of the plan (sample, depth-aware render,
// image and depth loss, backward with statistics, pseudo-view, anisotropy, Adam, scale cap,
// densification, isolation pruning).

// MARK: - Helpers

/// 64-bit FNV-1a over 32-bit words.
private func fnv1a(_ words: [UInt32]) -> UInt64 {
    var hash: UInt64 = 0xcbf2_9ce4_8422_2325
    for w in words {
        hash ^= UInt64(w)
        hash = hash &* 0x0000_0100_0000_01b3
    }
    return hash
}

/// Bit patterns of every parameter array, in the order positions, logScales, rotations,
/// opacityLogits, sh.
private func cloudBits(_ cloud: GaussianCloud) -> [UInt32] {
    var bits: [UInt32] = []
    for v in cloud.positions { bits.append(v.bitPattern) }
    for v in cloud.logScales { bits.append(v.bitPattern) }
    for v in cloud.rotations { bits.append(v.bitPattern) }
    for v in cloud.opacityLogits { bits.append(v.bitPattern) }
    for v in cloud.sh { bits.append(v.bitPattern) }
    return bits
}

/// Metric prior of a depth map: the depth domain, with L1 enabled (measured depth).
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

/// Short configuration: degree-0 SH, densification off unless a test enables it.
private func makeConfiguration(iterations: Int, regularizers: Set<RegularizerKind>) -> TrainingConfiguration {
    var configuration = TrainingConfiguration()
    configuration.iterations = iterations
    configuration.shDegree = 0
    configuration.seed = 11
    configuration.regularizers = regularizers
    configuration.densification.stopIteration = 0
    return configuration
}

/// Same Gaussians with `extra` appended: isotropic scale, opacity 0.9, DC colour grey.
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

/// Pearson correlation in Double.
private func pearson(_ xs: [Double], _ ys: [Double]) -> Double {
    let n = Double(xs.count)
    guard xs.count > 1 else { return 0 }
    var mx: Double = 0
    var my: Double = 0
    for i in xs.indices {
        mx += xs[i]
        my += ys[i]
    }
    mx /= n
    my /= n
    var cov: Double = 0
    var vx: Double = 0
    var vy: Double = 0
    for i in xs.indices {
        let dx = xs[i] - mx
        let dy = ys[i] - my
        cov += dx * dy
        vx += dx * dx
        vy += dy * dy
    }
    guard vx > 0, vy > 0 else { return 0 }
    return cov / (vx.squareRoot() * vy.squareRoot())
}

/// Correlation of the cloud's normalised rendered depth with the ground truth over all held-out
/// novel views, on pixels where both are valid. Returns the correlation and the valid-pixel count.
private func novelDepthCorrelation(_ scene: SyntheticScene, cloud: GaussianCloud) -> (rho: Double, pixels: Int) {
    let settings = RasterSettings(shDegreeLimit: 0, background: .zero)
    var predictedValues: [Double] = []
    var truthValues: [Double] = []
    for view in scene.novel {
        let predicted: [Float] = CPURasterizer.renderWithDepth(cloud, camera: view.camera, settings: settings)
            .depth.normalized(minimumAlpha: 0.5)
        let truth: [Float] = CPURasterizer.renderWithDepth(scene.groundTruth, camera: view.camera, settings: settings)
            .depth.normalized(minimumAlpha: 0.5)
        for index in truth.indices where truth[index] > 0 && predicted[index] > 0 {
            predictedValues.append(Double(predicted[index]))
            truthValues.append(Double(truth[index]))
        }
    }
    return (pearson(predictedValues, truthValues), predictedValues.count)
}

// MARK: - Scene

@Test func syntheticSceneIsDeterministicAndConsistent() {
    // Dense enough that the front of the sphere saturates the alpha at the centre pixel.
    let first = SyntheticScene.make(viewCount: 4, novelCount: 2, width: 32, height: 24,
                                    gaussians: 1200, sparsePoints: 80, seed: 3)
    let second = SyntheticScene.make(viewCount: 4, novelCount: 2, width: 32, height: 24,
                                     gaussians: 1200, sparsePoints: 80, seed: 3)
    #expect(first.groundTruth.positions == second.groundTruth.positions)
    #expect(first.groundTruth.sh == second.groundTruth.sh)
    #expect(first.points == second.points)
    #expect(first.trainDepth == second.trainDepth)
    let firstImage: RGBImage = first.train[0].image
    let secondImage: RGBImage = second.train[0].image
    #expect(firstImage == secondImage)

    let other = SyntheticScene.make(viewCount: 4, novelCount: 2, width: 32, height: 24,
                                    gaussians: 1200, sparsePoints: 80, seed: 4)
    #expect(other.points != first.points)

    #expect(first.train.count == 4)
    #expect(first.novel.count == 2)
    #expect(first.trainDepth.count == 4)
    #expect(first.points.positions.count == 80)

    // The camera is 3.5 from the origin and the sphere has radius 1, so the front of the sphere
    // is 2.5 away on the central ray (the arc looks at the origin).
    let centre = (24 / 2) * 32 + 32 / 2
    let centreDepth: Float = first.trainDepth[0].values[centre]
    print("SCENE centre depth view0 = \(centreDepth)")
    #expect(abs(centreDepth - 2.5) <= 0.05, "centre depth \(centreDepth)")
}

// MARK: - Phase 3 order

/// With `regularizers == []` the Phase 4 settings must be inert. Checked within one process (a fingerprint recorded
/// on one platform fails on others because of libm differences): two runs that differ only in Phase 4 settings give
/// the same cloud bit for bit, and a repeated run is deterministic.
@Test func cpuTrainerWithoutDepthMatchesPhase3Order() throws {
    // Depth priors are attached, but no Phase 4 feature is enabled, so they must be ignored.
    let scene = SyntheticScene.make(viewCount: 4, novelCount: 2, width: 32, height: 24,
                                    gaussians: 120, sparsePoints: 60, seed: 3)
    let dataset = datasetWithPriors(scene)
    var configuration = TrainingConfiguration()
    configuration.iterations = 40
    configuration.shDegree = 0
    configuration.seed = 11
    configuration.regularizers = []
    configuration.densification.startIteration = 5
    configuration.densification.stopIteration = 30
    configuration.densification.interval = 5
    configuration.densification.gradientThreshold = 0
    configuration.maxGaussians = 400

    func run(_ configuration: TrainingConfiguration) throws -> (cloud: GaussianCloud, pseudoViews: Int) {
        let initial = GaussianInitializer.initialize(dataset, configuration: configuration,
                                                     maxCount: configuration.maxGaussians)
        var trainer = try CPUTrainer(dataset: dataset, initialCloud: initial, configuration: configuration)
        for _ in 0..<20 {
            let result = try trainer.step()
            #expect(result.depth.training == nil)
            #expect(result.depth.pseudo == nil)
            #expect(result.depth.isolationPruned == 0)
        }
        #expect(trainer.iteration == 20)
        return (trainer.cloud, trainer.pseudoViews.count)
    }

    let reference = try run(configuration)
    #expect(reference.pseudoViews == 0)
    let repeated = try run(configuration)
    #expect(cloudBits(reference.cloud) == cloudBits(repeated.cloud), "the run must be deterministic")

    var inert = configuration
    inert.depthPrior.pearsonWeight = 5
    inert.depthPrior.l1Weight = 5
    inert.pseudoViews.weight = 3
    inert.pseudoViews.startIteration = 0
    inert.artifacts.anisotropyWeight = 9
    inert.artifacts.scaleCapFraction = 0.001
    inert.artifacts.isolationStartIteration = 1
    inert.artifacts.isolationInterval = 1
    inert.denseInit.maxPoints = 5
    let changed = try run(inert)
    #expect(cloudBits(reference.cloud) == cloudBits(changed.cloud),
            "Phase 4 settings must not matter while no Phase 4 feature is enabled")
}

// MARK: - Depth prior

@Test func depthPriorImprovesNovelDepthCorrelation() throws {
    let scene = SyntheticScene.make(viewCount: 4, novelCount: 4, width: 32, height: 24,
                                    gaussians: 160, sparsePoints: 80, seed: 5)
    let extent = SceneExtent.estimate(cameraCenters: scene.train.map { $0.camera.center }, points: [])
    let initial = GaussianInitializer.random(count: 400, extent: extent, shDegree: 0,
                                             initialOpacity: 0.1, seed: 5)

    func train(dataset: TrainingDataset, regularizers: Set<RegularizerKind>) throws -> GaussianCloud {
        var configuration = makeConfiguration(iterations: 120, regularizers: regularizers)
        configuration.depthPrior.startIteration = 0
        var trainer = try CPUTrainer(dataset: dataset, initialCloud: initial, configuration: configuration)
        for _ in 0..<120 {
            _ = try trainer.step()
        }
        return trainer.cloud
    }

    let withoutPrior = try train(dataset: scene.dataset, regularizers: [])
    let withPrior = try train(dataset: datasetWithPriors(scene), regularizers: [.depthPearson, .depthL1])
    let baseline = novelDepthCorrelation(scene, cloud: withoutPrior)
    let improved = novelDepthCorrelation(scene, cloud: withPrior)
    print("DEPTH correlation without prior = \(baseline.rho) (\(baseline.pixels) px), with prior = \(improved.rho) (\(improved.pixels) px)")
    #expect(improved.rho > baseline.rho, "without \(baseline.rho), with \(improved.rho)")
}

@Test func depthPriorLossIsRecordedPerStep() throws {
    let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                    gaussians: 160, sparsePoints: 80, seed: 5)
    let dataset = datasetWithPriors(scene)
    var configuration = makeConfiguration(iterations: 5, regularizers: [.depthPearson, .depthL1])
    configuration.depthPrior.startIteration = 2
    let initial = GaussianInitializer.initialize(dataset, configuration: configuration, maxCount: 400)
    var trainer = try CPUTrainer(dataset: dataset, initialCloud: initial, configuration: configuration)
    var recorded: [Bool] = []
    for _ in 0..<5 {
        let result = try trainer.step()
        recorded.append(result.depth.training != nil)
        if let training = result.depth.training {
            #expect(training.loss.isFinite)
        }
    }
    // The prior starts at iteration 2: steps 0 and 1 render without it.
    #expect(recorded == [false, false, true, true, true])
}

// MARK: - Regularizers

@Test func anisotropyBoundsAxisRatio() throws {
    let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                    gaussians: 160, sparsePoints: 80, seed: 6)
    let configuration = makeConfiguration(iterations: 120, regularizers: [.anisotropy, .scaleCap])
    let initial = try elongated(GaussianInitializer.initialize(scene.dataset, configuration: configuration,
                                                               maxCount: 400))

    var trainer = try CPUTrainer(dataset: scene.dataset, initialCloud: initial, configuration: configuration)
    for _ in 0..<120 {
        _ = try trainer.step()
    }

    var ratios: [Double] = []
    let cloud = trainer.cloud
    for i in 0..<cloud.count {
        var maxLog = Double(cloud.logScales[3 * i])
        var minLog = maxLog
        for axis in 1..<3 {
            let value = Double(cloud.logScales[3 * i + axis])
            maxLog = max(maxLog, value)
            minLog = min(minLog, value)
        }
        ratios.append(exp(maxLog - minLog))
    }
    ratios.sort()
    let index = Int((0.95 * Double(ratios.count - 1)).rounded(.down))
    let p95 = ratios[index]
    print("ANISOTROPY 95th percentile axis ratio = \(p95), max = \(ratios.last ?? 0)")
    #expect(p95 <= 12, "95th percentile ratio \(p95)")
}

@Test func isolationPruningRemovesInjectedFloaters() throws {
    // 400 sphere Gaussians plus 50 ground Gaussians: the 5 % cap (23) fits the 20 floaters.
    let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                    gaussians: 400, sparsePoints: 100, seed: 8)
    let base = scene.groundTruth
    // 20 floaters spread on a sphere of radius 10 (Fibonacci directions): far from everything.
    var floaters: [SIMD3<Float>] = []
    for i in 0..<20 {
        let y: Float = 1 - 2 * (Float(i) + 0.5) / 20
        let ring: Float = (1 - y * y).squareRoot()
        let phi: Float = Float(i) * 2.39996323
        floaters.append(10 * SIMD3<Float>(ring * cos(phi), y, ring * sin(phi)))
    }
    let cloud = try appending(base, positions: floaters)

    var configuration = makeConfiguration(iterations: 12, regularizers: [.isolationPrune])
    configuration.artifacts.isolationStartIteration = 10
    configuration.artifacts.isolationInterval = 10
    var trainer = try CPUTrainer(dataset: scene.dataset, initialCloud: cloud, configuration: configuration)
    for _ in 0..<9 {
        let result = try trainer.step()
        #expect(result.depth.isolationPruned == 0)
    }
    #expect(trainer.cloud.count == cloud.count, "no pruning before the first boundary")

    let boundary = try trainer.step()
    print("ISOLATION pruned at first boundary = \(boundary.depth.isolationPruned), count \(trainer.cloud.count)")
    #expect(boundary.depth.isolationPruned == 20)
    #expect(trainer.cloud.count == base.count)
    var farthest: Float = 0
    for i in 0..<trainer.cloud.count {
        let p = SIMD3<Float>(trainer.cloud.positions[3 * i], trainer.cloud.positions[3 * i + 1],
                             trainer.cloud.positions[3 * i + 2])
        farthest = max(farthest, (p * p).sum().squareRoot())
    }
    #expect(farthest < 5, "floater still present at distance \(farthest)")
}

// MARK: - Pseudo-views

@Test func pseudoViewsContributeGradient() throws {
    let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 64, height: 48,
                                    gaussians: 160, sparsePoints: 80, seed: 7)
    let dataset = datasetWithPriors(scene)
    var pseudoConfiguration = makeConfiguration(iterations: 10, regularizers: [.pseudoViews])
    pseudoConfiguration.pseudoViews.startIteration = 0
    pseudoConfiguration.pseudoViews.interval = 1
    let initial = GaussianInitializer.initialize(dataset, configuration: pseudoConfiguration, maxCount: 400)

    var withPseudo = try CPUTrainer(dataset: dataset, initialCloud: initial, configuration: pseudoConfiguration)
    #expect(!withPseudo.pseudoViews.isEmpty)
    let result = try withPseudo.step()
    #expect(result.depth.pseudo != nil)
    #expect(result.depth.pseudo?.loss.isFinite == true)
    #expect(result.depth.training == nil, "depth prior terms are off")

    // Same step with no regularizers: the only difference may come from the pseudo-view gradient.
    let plainConfiguration = makeConfiguration(iterations: 10, regularizers: [])
    var plain = try CPUTrainer(dataset: dataset, initialCloud: initial, configuration: plainConfiguration)
    _ = try plain.step()
    #expect(withPseudo.cloud.positions != plain.cloud.positions,
            "pseudo-view step must change the update")
    print("PSEUDO pool size = \(withPseudo.pseudoViews.count), loss = \(result.depth.pseudo?.loss ?? -1)")
}

// MARK: - Stability

@Test func modelStaysFinite() throws {
    let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                    gaussians: 160, sparsePoints: 80, seed: 9)
    let dataset = datasetWithPriors(scene)
    var configuration = makeConfiguration(iterations: 150, regularizers: Set(RegularizerKind.allCases))
    configuration.depthPrior.startIteration = 0
    configuration.densification.startIteration = 10
    configuration.densification.stopIteration = 100
    configuration.densification.interval = 10
    configuration.maxGaussians = 300   // keeps densification within the debug-build budget
    configuration.pseudoViews.startIteration = 20
    configuration.pseudoViews.interval = 2
    configuration.artifacts.isolationStartIteration = 50
    configuration.artifacts.isolationInterval = 25
    let initial = GaussianInitializer.initialize(dataset, configuration: configuration, maxCount: 400)

    var trainer = try CPUTrainer(dataset: dataset, initialCloud: initial, configuration: configuration)
    var pseudoSteps = 0
    for _ in 0..<150 {
        let result = try trainer.step()
        if result.depth.pseudo != nil {
            pseudoSteps += 1
        }
    }
    let cloud = trainer.cloud
    let allFinite = cloud.positions.allSatisfy { $0.isFinite }
        && cloud.logScales.allSatisfy { $0.isFinite }
        && cloud.rotations.allSatisfy { $0.isFinite }
        && cloud.opacityLogits.allSatisfy { $0.isFinite }
        && cloud.sh.allSatisfy { $0.isFinite }
    #expect(allFinite)
    #expect(pseudoSteps > 0)
    #expect(trainer.iteration == 150)
}
