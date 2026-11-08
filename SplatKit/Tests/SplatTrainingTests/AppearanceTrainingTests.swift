import Foundation
import Metal
import Testing
import SplatCore
import SplatMetal
@testable import SplatTraining

// End-to-end appearance compensation (v1.2) on the Metal trainer. Two views of one ground-truth cloud; the second
// view's target is the same image multiplied by 1.3 and shifted by 0.05. With compensation the second view's gain
// must learn the change, and the training loss must end lower than without it. Runs only where Metal exists.
@Suite(.serialized, .enabled(if: TrainingTestEnvironment.hasMetal))
struct AppearanceTrainingTests {

    private func makeConfiguration(appearance: Bool, startIteration: Int = 0) -> TrainingConfiguration {
        var configuration = TrainingConfiguration.testBaseline()
        configuration.shDegree = 0
        configuration.iterations = 400
        configuration.logInterval = 0
        configuration.evalInterval = 0
        configuration.previewInterval = 0
        configuration.seed = 3
        configuration.densification.startIteration = 0
        configuration.densification.stopIteration = 0
        configuration.regularizers = []
        configuration.appearance.isEnabled = appearance
        configuration.appearance.learningRate = 0.01
        configuration.appearance.startIteration = startIteration
        return configuration
    }

    /// Two training views of `groundTruth`; the second target is 1.3 · render + 0.05.
    private func makeDataset(context: MetalContext) throws -> TrainingDataset {
        let groundTruth = try makeRandomCloud(count: 500, seed: 11, shDegree: 0)
        let cameras = [makeCamera(width: 64, height: 64), makeCamera(width: 64, height: 64, shift: 1.0)]
        let renders = try makeViews(context: context, groundTruth: groundTruth, cameras: cameras)
        let brighter: RGBImage = AppearanceTransform.apply(
            renders[1].image, AppearanceParameters(gain: SIMD3<Float>(repeating: 1.3),
                                                   bias: SIMD3<Float>(repeating: 0.05)))
        let views = [renders[0], TrainingView(name: "view1", camera: cameras[1], image: brighter)]
        return TrainingDataset(views: views)
    }

    private func initialCloud() throws -> GaussianCloud {
        try makeRandomCloud(count: 500, seed: 97, shDegree: 0)
    }

    /// Trains to the end and returns the trainer's appearance parameters and cloud.
    private func train(context: MetalContext, dataset: TrainingDataset, configuration: TrainingConfiguration)
        throws -> (parameters: [AppearanceParameters], cloud: GaussianCloud) {
        let trainer = try GPUTrainer(context: context, dataset: dataset, initialCloud: try initialCloud(),
                                     configuration: configuration)
        for _ in 0..<configuration.iterations {
            try trainer.step()
        }
        let parameters = try trainer.appearanceParameters()
        let cloud = try trainer.snapshot()
        return (parameters, cloud)
    }

    /// Mean CPU image loss over the views of `cloud`, rendered with Metal, with the compensation applied when given.
    private func trainingLoss(context: MetalContext, cloud: GaussianCloud, views: [TrainingView],
                              parameters: [AppearanceParameters]?, lambda: Float) throws -> Double {
        let renderer = try SplatRenderer(context: context, cloud: cloud)
        let settings = RasterSettings(shDegreeLimit: cloud.shDegree, background: .zero)
        var sum: Double = 0
        for (index, view) in views.enumerated() {
            var rendered: RGBImage = try renderer.renderImage(camera: view.camera, settings: settings)
            if let parameters {
                rendered = AppearanceTransform.apply(rendered, parameters[index])
            }
            sum += ImageLoss.evaluate(rendered: rendered, target: view.image, lambda: lambda).loss
        }
        return sum / Double(views.count)
    }

    private func meanGain(_ parameters: AppearanceParameters) -> Float {
        (parameters.gain.x + parameters.gain.y + parameters.gain.z) / 3
    }

    @Test func secondViewLearnsItsGainAndLowersTheLoss() throws {
        let context = try MetalContext.makeDefault()
        let dataset = try makeDataset(context: context)
        let onConfiguration = makeConfiguration(appearance: true)
        let offConfiguration = makeConfiguration(appearance: false)

        let on = try train(context: context, dataset: dataset, configuration: onConfiguration)
        let off = try train(context: context, dataset: dataset, configuration: offConfiguration)

        // The brighter view's gain moves clearly above the reference view's gain.
        let gainReference: Float = meanGain(on.parameters[0])
        let gainBrighter: Float = meanGain(on.parameters[1])
        #expect(gainBrighter > gainReference + 0.1,
                "gain of the brighter view \(gainBrighter) should exceed the reference view's \(gainReference) by 0.1")
        #expect(on.parameters[1].bias.x > 0, "bias of the brighter view should move towards 0.05")

        // With the learned compensation the loss is lower than without it.
        let lossOn = try trainingLoss(context: context, cloud: on.cloud, views: dataset.views,
                                      parameters: on.parameters, lambda: onConfiguration.lambdaDSSIM)
        let lossOff = try trainingLoss(context: context, cloud: off.cloud, views: dataset.views,
                                       parameters: nil, lambda: offConfiguration.lambdaDSSIM)
        #expect(lossOn < lossOff, "loss with appearance \(lossOn) should be below \(lossOff) without it")
    }

    @Test func disabledAndNeverActiveRunsMatchTheBaseline() throws {
        let context = try MetalContext.makeDefault()
        let dataset = try makeDataset(context: context)
        // Disabled: no appearance model exists, so every parameter reads back as the identity.
        let disabled = try train(context: context, dataset: dataset,
                                 configuration: makeConfiguration(appearance: false))
        #expect(disabled.parameters.allSatisfy { $0 == .identity })

        // Enabled, but the start iteration is never reached: no compensation runs, so the parameters stay identity.
        // (The GPU trainer is not bit-deterministic run to run, so the clouds are not compared exactly.)
        let neverActive = try train(context: context, dataset: dataset,
                                    configuration: makeConfiguration(appearance: true, startIteration: 400))
        #expect(neverActive.parameters.allSatisfy { $0 == .identity })
    }
}
