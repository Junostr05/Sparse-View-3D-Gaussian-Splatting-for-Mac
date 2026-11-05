import Foundation
import Testing
@testable import SplatCore

// MARK: - Toy scene

/// Degree-0 Gaussians at the given positions with the given colours (DC term only).
private func makeCloud(positions: [SIMD3<Float>], colors: [SIMD3<Float>],
                       logScale: Float = log(0.25), opacity: Float = 0.8) -> GaussianCloud {
    var flatPositions: [Float] = []
    var logScales: [Float] = []
    var rotations: [Float] = []
    var logits: [Float] = []
    var sh: [Float] = []
    let logit = log(opacity / (1 - opacity))
    for i in 0..<positions.count {
        let p = positions[i]
        flatPositions += [p.x, p.y, p.z]
        logScales += [logScale, logScale, logScale]
        rotations += [1, 0, 0, 0]
        logits.append(logit)
        let c = colors[i]
        // colour = 0.5 + c0 · dc
        sh += [(c.x - 0.5) / 0.28209479177387814,
               (c.y - 0.5) / 0.28209479177387814,
               (c.z - 0.5) / 0.28209479177387814]
    }
    return try! GaussianCloud(shDegree: 0, positions: flatPositions, logScales: logScales,
                              rotations: rotations, opacityLogits: logits, sh: sh)
}

/// Ground truth: three coloured Gaussians in front of the camera.
private func groundTruth() -> GaussianCloud {
    makeCloud(positions: [SIMD3(-0.3, 0.2, 2), SIMD3(0.25, -0.1, 2.5), SIMD3(0, 0, 3)],
              colors: [SIMD3(0.9, 0.2, 0.2), SIMD3(0.2, 0.9, 0.2), SIMD3(0.2, 0.2, 0.9)])
}

/// Initial guess: ground-truth positions shifted by (0.05, −0.05, 0), colours 0.5.
private func shiftedGrayStart() -> GaussianCloud {
    makeCloud(positions: [SIMD3(-0.25, 0.15, 2), SIMD3(0.30, -0.15, 2.5), SIMD3(0.05, -0.05, 3)],
              colors: [SIMD3(0.5, 0.5, 0.5), SIMD3(0.5, 0.5, 0.5), SIMD3(0.5, 0.5, 0.5)])
}

private func camera(translationX: Float = 0) -> Camera {
    Camera(width: 24, height: 24, fx: 30, fy: 30, cx: 12, cy: 12,
           rotation: .identity, translation: SIMD3(translationX, 0, 0))
}

private func view(_ name: String, cloud: GaussianCloud, camera: Camera) -> TrainingView {
    let settings = RasterSettings(shDegreeLimit: 0, background: .zero)
    let image = CPURasterizer.render(cloud, camera: camera, settings: settings)
    return TrainingView(name: name, camera: camera, image: image)
}

private func sameCloud(_ a: GaussianCloud, _ b: GaussianCloud) -> Bool {
    a.count == b.count
        && a.positions == b.positions
        && a.logScales == b.logScales
        && a.rotations == b.rotations
        && a.opacityLogits == b.opacityLogits
        && a.sh == b.sh
}

// MARK: - Tests

@Test func lossDecreasesOnToyScene() throws {
    let truth = groundTruth()
    let dataset = TrainingDataset(views: [view("gt", cloud: truth, camera: camera())])
    var configuration = TrainingConfiguration()
    configuration.iterations = 150
    configuration.shDegree = 0
    configuration.densification.startIteration = 0
    configuration.densification.stopIteration = 0   // densification disabled

    var trainer = try CPUTrainer(dataset: dataset, initialCloud: shiftedGrayStart(),
                                 configuration: configuration)
    var losses: [Double] = []
    for _ in 0..<150 {
        losses.append(try trainer.step().loss)
    }

    let first = losses[0]
    let tailMean = losses.suffix(10).reduce(0, +) / 10
    #expect(losses.allSatisfy { $0.isFinite })
    #expect(tailMean <= 0.6 * first, "first \(first), mean of last 10 \(tailMean)")
    #expect(trainer.iteration == 150)
}

@Test func trainingIsDeterministic() throws {
    let truth = groundTruth()
    let views = [view("a", cloud: truth, camera: camera()),
                 view("b", cloud: truth, camera: camera(translationX: 0.2))]
    let dataset = TrainingDataset(views: views)
    var configuration = TrainingConfiguration()
    configuration.iterations = 20
    configuration.shDegree = 0
    configuration.seed = 7
    configuration.densification.startIteration = 5
    configuration.densification.stopIteration = 30
    configuration.densification.interval = 5
    configuration.densification.gradientThreshold = 0
    configuration.maxGaussians = 12

    var first = try CPUTrainer(dataset: dataset, initialCloud: shiftedGrayStart(),
                               configuration: configuration)
    var second = try CPUTrainer(dataset: dataset, initialCloud: shiftedGrayStart(),
                                configuration: configuration)
    for _ in 0..<20 {
        _ = try first.step()
        _ = try second.step()
    }
    #expect(sameCloud(first.cloud, second.cloud))
    #expect(first.cloud.count > 3, "densification should have run")
}

@Test func densificationRunsAtBoundaries() throws {
    let dataset = TrainingDataset(views: [view("gt", cloud: groundTruth(), camera: camera())])
    var configuration = TrainingConfiguration()
    configuration.iterations = 30
    configuration.shDegree = 0
    configuration.densification.startIteration = 5
    configuration.densification.stopIteration = 30
    configuration.densification.interval = 5
    configuration.densification.gradientThreshold = 0
    configuration.maxGaussians = 12

    var trainer = try CPUTrainer(dataset: dataset, initialCloud: shiftedGrayStart(),
                                 configuration: configuration)
    var densifiedAt: [Int] = []
    var counts: [Int] = [trainer.cloud.count]
    for _ in 0..<30 {
        let result = try trainer.step()
        if result.densified != nil {
            densifiedAt.append(result.iteration)
        }
        #expect(trainer.cloud.count <= 12)
        counts.append(trainer.cloud.count)
    }

    #expect(densifiedAt == [5, 10, 15, 20, 25])
    // The first boundary (after iteration 5) must actually grow the model.
    #expect(counts[5] > counts[4])
}

@Test func nonFiniteTargetIsReported() throws {
    let cam = camera()
    var target = view("gt", cloud: groundTruth(), camera: cam)
    target.image.pixels[5] = SIMD3(Float.nan, 0, 0)
    let dataset = TrainingDataset(views: [target])
    var configuration = TrainingConfiguration()
    configuration.shDegree = 0
    configuration.densification.stopIteration = 0

    let start = shiftedGrayStart()
    var trainer = try CPUTrainer(dataset: dataset, initialCloud: start, configuration: configuration)
    #expect(throws: CPUTrainerError.nonFiniteLoss(iteration: 0)) {
        try trainer.step()
    }
    // A failed step must leave the model untouched.
    #expect(trainer.iteration == 0)
    #expect(sameCloud(trainer.cloud, start))
}

@Test func emptyDatasetIsRejected() {
    let dataset = TrainingDataset(views: [])
    #expect(throws: CPUTrainerError.noViews) {
        _ = try CPUTrainer(dataset: dataset, initialCloud: shiftedGrayStart(),
                           configuration: TrainingConfiguration())
    }
}
