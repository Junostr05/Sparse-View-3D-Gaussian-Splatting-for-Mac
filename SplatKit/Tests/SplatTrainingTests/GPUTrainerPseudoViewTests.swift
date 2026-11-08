import Foundation
import Testing
import SplatCore
import SplatMetal
@testable import SplatTraining

// Phase 4 Task 18: pseudo-views and held-out evaluation on the GPU trainer. Runs only where Metal exists.
// The pseudo step under test is CPUTrainer step 4 (P9): the training part of the iteration (render, image
// and depth loss, backward with statistics) is committed first, then the pseudo-view forward, its depth loss
// (slot 1, Pearson weight `pseudoViews.weight`, no L1), a backward with a zero image gradient that ADDS into the
// gradients without statistics, and then anisotropy, Adam and the scale cap.
//
// Floating-point note: the rasterizer's backward pass accumulates with float atomics, so GPU gradients differ
// from the CPU reference by a small relative amount. Comparisons use a tolerance relative to the largest
// reference magnitude of each array.

// MARK: - Helpers

/// Metric prior of a depth map: depth domain, L1 enabled (measured depth).
private func pseudoMetricPrior(_ depth: DepthMap) -> DepthPrior {
    DepthPrior(width: depth.width, height: depth.height, correlationTarget: depth.values,
               domain: .depth, metricDepth: depth.values, isMeasuredMetric: true)
}

/// The scene's dataset with every training view carrying its ground-truth metric prior.
private func pseudoDataset(_ scene: SyntheticScene) -> TrainingDataset {
    var views: [TrainingView] = scene.train
    for index in views.indices {
        views[index].depth = pseudoMetricPrior(scene.trainDepth[index])
    }
    return TrainingDataset(views: views, points: scene.points)
}

/// Configuration with densification and isolation off, a loss read at every step, and pseudo-views from
/// iteration 0 every step (so a single step is a pseudo step).
private func pseudoConfiguration(iterations: Int, regularizers: Set<RegularizerKind>) -> TrainingConfiguration {
    var configuration = TrainingConfiguration.testBaseline()
    configuration.iterations = iterations
    configuration.shDegree = 0
    configuration.seed = 11
    configuration.logInterval = 1
    configuration.evalInterval = 0
    configuration.previewInterval = 0
    configuration.regularizers = regularizers
    configuration.densification.startIteration = 0
    configuration.densification.stopIteration = 0
    configuration.pseudoViews.startIteration = 0
    configuration.pseudoViews.interval = 1
    return configuration
}

/// Largest absolute difference between `a` and `b`, and the largest absolute value of `b`.
private func compareArrays(_ actual: [Float], _ expected: [Float]) -> (difference: Double, scale: Double) {
    precondition(actual.count == expected.count, "arrays must have the same count")
    var difference: Double = 0
    var scale: Double = 0
    for index in expected.indices {
        let e: Double = Double(expected[index])
        let a: Double = Double(actual[index])
        difference = max(difference, abs(a - e))
        scale = max(scale, abs(e))
    }
    return (difference, scale)
}

/// Element-wise sum of two gradient sets with the same layout.
private func sumGradients(_ a: GaussianGradients, _ b: GaussianGradients) -> GaussianGradients {
    func add(_ x: [Float], _ y: [Float]) -> [Float] {
        var out: [Float] = x
        for index in out.indices {
            out[index] += y[index]
        }
        return out
    }
    return GaussianGradients(positions: add(a.positions, b.positions),
                             logScales: add(a.logScales, b.logScales),
                             rotations: add(a.rotations, b.rotations),
                             opacityLogits: add(a.opacityLogits, b.opacityLogits),
                             sh: add(a.sh, b.sh))
}

/// Largest absolute value over every parameter array of a gradient set.
private func maxMagnitude(_ g: GaussianGradients) -> Double {
    var largest: Double = 0
    for array in [g.positions, g.logScales, g.rotations, g.opacityLogits, g.sh] {
        for value in array {
            largest = max(largest, abs(Double(value)))
        }
    }
    return largest
}

/// Element-wise `factor * g`.
private func scaleGradients(_ g: GaussianGradients, by factor: Float) -> GaussianGradients {
    func scale(_ x: [Float]) -> [Float] {
        x.map { $0 * factor }
    }
    return GaussianGradients(positions: scale(g.positions),
                             logScales: scale(g.logScales),
                             rotations: scale(g.rotations),
                             opacityLogits: scale(g.opacityLogits),
                             sh: scale(g.sh))
}

/// Names of the parameter arrays whose largest absolute difference from `reference` is more than `factor` times the
/// comparison tolerance of that array: 1e-3 of the array's scale, floored at 1e-3 of the largest reference magnitude.
private func arraysBeyondTolerance(_ actual: GaussianGradients, _ reference: GaussianGradients,
                                   factor: Double) -> [String] {
    let globalScale: Double = maxMagnitude(reference)
    let pairs: [(String, [Float], [Float])] = [
        ("positions", actual.positions, reference.positions),
        ("logScales", actual.logScales, reference.logScales),
        ("rotations", actual.rotations, reference.rotations),
        ("opacityLogits", actual.opacityLogits, reference.opacityLogits),
        ("sh", actual.sh, reference.sh),
    ]
    var names: [String] = []
    for (name, a, r) in pairs {
        let comparison = compareArrays(a, r)
        let tolerance: Double = 1e-3 * max(comparison.scale, 1e-3 * globalScale)
        if comparison.difference > factor * tolerance {
            names.append(name)
        }
    }
    return names
}

// MARK: - Tests

@Suite(.serialized, .enabled(if: TrainingTestEnvironment.hasMetal))
struct GPUTrainerPseudoViewTests {

    /// Four metric views give between 3 and 9 accepted pseudo-views (at most three candidates per unique pair,
    /// and the pool matches the CPU reference exactly).
    @Test
    func pseudoPoolBuiltFromMetricViews() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                        gaussians: 160, sparsePoints: 80, seed: 5)
        let dataset = pseudoDataset(scene)
        let configuration = pseudoConfiguration(iterations: 2, regularizers: [.depthPearson, .depthL1, .pseudoViews])
        let initial = GaussianInitializer.initialize(dataset, configuration: configuration, maxCount: 400)

        let cpu = try CPUTrainer(dataset: dataset, initialCloud: initial, configuration: configuration)
        let trainer = try GPUTrainer(context: context, dataset: dataset, initialCloud: initial,
                                     configuration: configuration)
        #expect((3...9).contains(trainer.pseudoViewCount), "pool of \(trainer.pseudoViewCount) pseudo-views")
        #expect(trainer.pseudoViewCount == cpu.pseudoViews.count)
    }

    /// Without metric views there is no pool, and the pseudo step never reports a value.
    @Test
    func noPoolWithoutMetricViews() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                        gaussians: 160, sparsePoints: 80, seed: 5)
        let configuration = pseudoConfiguration(iterations: 2, regularizers: [.pseudoViews])
        let initial = GaussianInitializer.initialize(scene.dataset, configuration: configuration, maxCount: 400)
        let trainer = try GPUTrainer(context: context, dataset: scene.dataset, initialCloud: initial,
                                     configuration: configuration)
        #expect(trainer.pseudoViewCount == 0)
        let report = try trainer.step()
        #expect(report.pseudoDepthLoss == nil)
    }

    /// One step at iteration 0 (a pseudo step): the gradient read back from the GPU equals the CPU training-only
    /// gradient plus the CPU pseudo-only gradient, relative 1e-3. The pseudo part must be a real contribution: at
    /// least 1e-2 of the training gradient, and large enough that dropping it or counting it twice fails the check.
    @Test
    func pseudoIterationAccumulatesGradient() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                        gaussians: 160, sparsePoints: 80, seed: 5)
        let dataset = pseudoDataset(scene)
        let configuration = pseudoConfiguration(iterations: 1, regularizers: [.depthPearson, .depthL1, .pseudoViews])
        let initial = GaussianInitializer.initialize(dataset, configuration: configuration, maxCount: 400)

        // CPU references on the initial cloud (the trainer's first step sees exactly this cloud).
        var sampler = ViewSampler(count: dataset.views.count, seed: configuration.seed)
        let viewIndex: Int = sampler.next()
        let view: TrainingView = dataset.views[viewIndex]
        let settings = RasterSettings(shDegreeLimit: configuration.activeSHDegree(iteration: 0),
                                      background: configuration.background)
        let rendered = CPURasterizer.renderWithDepth(initial, camera: view.camera, settings: settings)
        let imageLoss = ImageLoss.evaluate(rendered: rendered.image, target: view.image,
                                           lambda: configuration.lambdaDSSIM)
        let trainingDepth = DepthLoss.evaluate(depth: rendered.depth, prior: view.depth!,
                                               weights: DepthLossWeights(settings: configuration.depthPrior,
                                                                         nearPlane: 0.01))
        let trainingOnly = CPURasterizer.backwardWithScreenSpace(initial, camera: view.camera, settings: settings,
                                                                 imageGradient: imageLoss.gradient,
                                                                 depthGradient: trainingDepth.gradient).gradients

        let pool = PseudoViews.make(dataset.views, settings: configuration.pseudoViews, near: 0.01).views
        let pseudo = pool[PseudoViews.index(iteration: 0, count: pool.count, seed: configuration.seed)]
        let pseudoRender = CPURasterizer.renderWithDepth(initial, camera: pseudo.camera, settings: settings)
        let pseudoWeights = DepthLossWeights(pearson: configuration.pseudoViews.weight, l1: 0,
                                             alphaMask: configuration.depthPrior.alphaMask,
                                             minimumPixels: configuration.depthPrior.minimumPixels,
                                             nearPlane: 0.01)
        let pseudoDepth = DepthLoss.evaluate(depth: pseudoRender.depth, prior: pseudo.prior, weights: pseudoWeights)
        let zeroImage = [SIMD3<Float>](repeating: .zero, count: pseudo.camera.width * pseudo.camera.height)
        let pseudoOnly = CPURasterizer.backwardWithScreenSpace(initial, camera: pseudo.camera, settings: settings,
                                                               imageGradient: zeroImage,
                                                               depthGradient: pseudoDepth.gradient).gradients
        let expected = sumGradients(trainingOnly, pseudoOnly)

        // GPU: one step, then read the gradients the step left behind (Adam does not modify them).
        let trainer = try GPUTrainer(context: context, dataset: dataset, initialCloud: initial,
                                     configuration: configuration)
        let report = try trainer.step()
        let pseudoGPU = try #require(report.pseudoDepthLoss)
        #expect(pseudoGPU.loss.isFinite)
        #expect(abs(pseudoGPU.loss - pseudoDepth.loss) <= 1e-3 * max(abs(pseudoDepth.loss), 1e-6),
                "pseudo depth loss \(pseudoGPU.loss) against CPU \(pseudoDepth.loss)")

        let readout = try trainer.testingReadout()
        let arrays: [(String, [Float], [Float])] = [
            ("positions", readout.gradients.positions, expected.positions),
            ("logScales", readout.gradients.logScales, expected.logScales),
            ("rotations", readout.gradients.rotations, expected.rotations),
            ("opacityLogits", readout.gradients.opacityLogits, expected.opacityLogits),
            ("sh", readout.gradients.sh, expected.sh),
        ]
        // Floor: an array whose reference is numerically zero (e.g. rotations of a near-isotropic Gaussian) is judged
        // against 1e-3 of the largest gradient of the whole step, so float noise on zeros does not fail the test.
        let globalScale: Double = maxMagnitude(expected)
        for (name, actual, reference) in arrays {
            let comparison = compareArrays(actual, reference)
            #expect(comparison.difference <= 1e-3 * max(comparison.scale, 1e-3 * globalScale),
                    "\(name): max difference \(comparison.difference) against scale \(comparison.scale), global \(globalScale)")
        }

        // The pseudo part must change the gradient, otherwise the agreement above proves nothing.
        let pseudoMagnitude = maxMagnitude(pseudoOnly)
        let trainingMagnitude = maxMagnitude(trainingOnly)
        let ratio = pseudoMagnitude / trainingMagnitude
        #expect(ratio >= 1e-2, "pseudo gradient is negligible: \(pseudoMagnitude) against \(trainingMagnitude)")

        // A dropped pseudo gradient must fail: the GPU result has to differ from the training-only reference by more
        // than 10x the comparison tolerance in at least one array.
        let droppedDifferences = arraysBeyondTolerance(readout.gradients, trainingOnly, factor: 10)
        #expect(!droppedDifferences.isEmpty, "GPU gradient equals the training-only gradient: pseudo part dropped")

        // A double-counted pseudo gradient must fail the same way against training + 2 x pseudo.
        let doubled = sumGradients(trainingOnly, scaleGradients(pseudoOnly, by: 2))
        let doubledDifferences = arraysBeyondTolerance(readout.gradients, doubled, factor: 10)
        #expect(!doubledDifferences.isEmpty, "GPU gradient equals training + 2 x pseudo: pseudo part double-counted")
    }

    /// Six steps with pseudo-views every second iteration, a log boundary every second step and a densification
    /// boundary at completed iteration 3, which is the pseudo iteration with index 2. A step reports the pseudo depth
    /// loss only when it synchronizes; the value is then non-nil exactly on the pseudo iterations. No step may throw,
    /// and every loss must be finite.
    @Test
    func pseudoStepsAcrossLogAndDensifyBoundaries() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                        gaussians: 160, sparsePoints: 80, seed: 5)
        let dataset = pseudoDataset(scene)
        var configuration = pseudoConfiguration(iterations: 6, regularizers: [.depthPearson, .depthL1, .pseudoViews])
        configuration.logInterval = 2
        configuration.pseudoViews.startIteration = 0
        configuration.pseudoViews.interval = 2
        configuration.densification.startIteration = 1
        configuration.densification.stopIteration = 100
        configuration.densification.interval = 3
        let initial = GaussianInitializer.initialize(dataset, configuration: configuration, maxCount: 400)
        let trainer = try GPUTrainer(context: context, dataset: dataset, initialCloud: initial,
                                     configuration: configuration)
        #expect(trainer.pseudoViewCount > 0)

        let schedule = configuration.pseudoViews
        var reportedPseudoSteps = 0
        var unreportedPseudoSteps = 0
        for index in 0..<configuration.iterations {
            let report = try trainer.step()
            let completed: Int = index + 1
            let isPseudo: Bool = index >= schedule.startIteration && (index - schedule.startIteration) % schedule.interval == 0
            let isLogStep: Bool = completed % configuration.logInterval == 0
            let isLastStep: Bool = completed >= configuration.iterations
            let isDensifyStep: Bool = completed >= 1 && completed < 100 && completed % 3 == 0
            let synchronizes: Bool = isLogStep || isLastStep || isDensifyStep

            #expect(report.iteration == completed)
            #expect((report.densify != nil) == isDensifyStep, "step \(index): densify summary mismatch")
            if synchronizes {
                #expect(report.loss != nil, "step \(index) synchronized but reported no image loss")
                #expect((report.pseudoDepthLoss != nil) == isPseudo,
                        "step \(index): pseudo depth loss reported \(report.pseudoDepthLoss != nil), expected \(isPseudo)")
                if isPseudo {
                    reportedPseudoSteps += 1
                }
            } else {
                #expect(report.loss == nil && report.pseudoDepthLoss == nil,
                        "step \(index) did not synchronize but reported a loss")
                if isPseudo {
                    unreportedPseudoSteps += 1
                }
            }
            if let loss = report.loss {
                #expect(loss.loss.isFinite, "step \(index): image loss \(loss.loss)")
            }
            if let depth = report.depthLoss {
                #expect(depth.loss.isFinite, "step \(index): depth loss \(depth.loss)")
            }
            if let pseudo = report.pseudoDepthLoss {
                #expect(pseudo.loss.isFinite, "step \(index): pseudo depth loss \(pseudo.loss)")
            }
        }
        // Both halves of the rule must be exercised: pseudo steps that report, and pseudo steps that do not.
        #expect(reportedPseudoSteps > 0)
        #expect(unreportedPseudoSteps > 0)
    }

    /// The pseudo step must not change the densification statistics: a step with pseudo-views leaves the
    /// statistics equal to those of the same step without them (the training part is identical).
    @Test
    func pseudoViewsDoNotTouchStatistics() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                        gaussians: 160, sparsePoints: 80, seed: 5)
        let dataset = pseudoDataset(scene)

        func statisticsAfterOneStep(_ regularizers: Set<RegularizerKind>) throws -> ([Float], [UInt32]) {
            var configuration = pseudoConfiguration(iterations: 1, regularizers: regularizers)
            configuration.densification.startIteration = 0
            configuration.densification.stopIteration = 100
            configuration.densification.interval = 1_000
            let initial = GaussianInitializer.initialize(dataset, configuration: configuration, maxCount: 400)
            let trainer = try GPUTrainer(context: context, dataset: dataset, initialCloud: initial,
                                         configuration: configuration)
            try trainer.step()
            let readout = try trainer.testingReadout()
            return (readout.gradientNormSums, readout.visibleCounts)
        }

        let without = try statisticsAfterOneStep([.depthPearson, .depthL1])
        let with = try statisticsAfterOneStep([.depthPearson, .depthL1, .pseudoViews])
        #expect(without.1 == with.1, "visible counts changed by the pseudo step")
        #expect(without.1.contains { $0 > 0 }, "the training view must have visible Gaussians")
        #expect(without.0.count == with.0.count)
        var largest: Float = 0
        for value in without.0 {
            largest = max(largest, abs(value))
        }
        for index in without.0.indices {
            #expect(abs(without.0[index] - with.0[index]) <= 1e-3 * max(largest, 1e-9),
                    "gradient norm sum \(index) changed by the pseudo step")
        }
    }

    /// `evaluate(view:)` matches the CPU metric definitions applied to the GPU rendering of the same cloud.
    @Test
    func evaluateMatchesCPUMetrics() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 4, novelCount: 2, width: 32, height: 24,
                                        gaussians: 160, sparsePoints: 80, seed: 5)
        let configuration = pseudoConfiguration(iterations: 3, regularizers: [])
        let initial = GaussianInitializer.initialize(scene.dataset, configuration: configuration, maxCount: 400)
        let trainer = try GPUTrainer(context: context, dataset: scene.dataset, initialCloud: initial,
                                     configuration: configuration)
        for _ in 0..<configuration.iterations {
            try trainer.step()
        }

        let held: TrainingView = scene.novel[0]
        let metrics = try trainer.evaluate(view: held)

        let cloud = try trainer.snapshot()
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let settings = RasterSettings(shDegreeLimit: configuration.shDegree, background: configuration.background)
        let rendered = try renderer.renderImage(camera: held.camera, settings: settings)
        let expectedPSNR: Double = ImageMetrics.psnr(rendered, held.image)
        let expectedSSIM: Double = ImageMetrics.ssim(rendered, held.image)
        #expect(abs(metrics.psnr - expectedPSNR) <= 1e-9 * max(1, abs(expectedPSNR)),
                "PSNR \(metrics.psnr) against \(expectedPSNR)")
        #expect(abs(metrics.ssim - expectedSSIM) <= 1e-9, "SSIM \(metrics.ssim) against \(expectedSSIM)")
        #expect(metrics.ssim <= 1 && metrics.ssim > -1)
    }

    /// With `regularizers == []` no depth buffers are reserved even when the views carry priors: the budget is
    /// exactly the Phase 3 budget (image bytes only).
    @Test
    func phase3BudgetWhenNoRegularizers() throws {
        let context = try MetalContext.makeDefault()
        let scene = SyntheticScene.make(viewCount: 4, novelCount: 0, width: 32, height: 24,
                                        gaussians: 160, sparsePoints: 80, seed: 5)
        let dataset = pseudoDataset(scene)
        let configuration = pseudoConfiguration(iterations: 2, regularizers: [])
        let initial = GaussianInitializer.initialize(dataset, configuration: configuration, maxCount: 400)
        let trainer = try GPUTrainer(context: context, dataset: dataset, initialCloud: initial,
                                     configuration: configuration)
        let viewSizes: [SIMD2<Int>] = dataset.views.map { SIMD2<Int>($0.camera.width, $0.camera.height) }
        let expected: Int = TrainingMemoryBudget.maxGaussians(
            workingSetBytes: context.device.recommendedMaxWorkingSetSize,
            shDegree: configuration.shDegree,
            reservedBytes: TrainingMemoryBudget.reservedImageBytes(viewSizes: viewSizes),
            configuredMax: configuration.maxGaussians)
        #expect(trainer.maxGaussians == expected)
    }
}
