import Metal
import SplatCore
import SplatMetal

// MCMC per-step terms of the GPU trainer (v1.2). They are built and encoded only when `mcmc.isEnabled`, and
// they follow the staged order of `AuxiliaryLoss`: the regularizers add to the gradients before Adam, the noise
// changes the positions after Adam.

/// Opacity and scale L1 gradients (`parameterGradient`), on every step.
final class MCMCRegularizerTerm: AuxiliaryLoss {
    let stage: AuxiliaryStage = .parameterGradient
    private let gpu: GPUMCMC
    private let opacityWeight: Float
    private let scaleWeight: Float

    init(gpu: GPUMCMC, opacityWeight: Float, scaleWeight: Float) {
        self.gpu = gpu
        self.opacityWeight = opacityWeight
        self.scaleWeight = scaleWeight
    }

    func encode(_ context: AuxiliaryContext) throws(TrainingError) {
        try metalCall { () throws(MetalError) -> Void in
            try self.gpu.encodeRegularizers(parameters: context.parameters, gradients: context.gradients,
                                            opacityWeight: self.opacityWeight, scaleWeight: self.scaleWeight,
                                            commandBuffer: context.commandBuffer)
        }
    }
}

/// SGLD position noise (`postOptimizer`), for iterations in `startIteration..<stopIteration`. The step is the
/// noise learning rate times the position learning rate of the iteration, as the Adam step uses it.
final class MCMCNoiseTerm: AuxiliaryLoss {
    let stage: AuxiliaryStage = .postOptimizer
    private let gpu: GPUMCMC
    private let settings: MCMCSettings
    private let configuration: TrainingConfiguration

    init(gpu: GPUMCMC, configuration: TrainingConfiguration) {
        self.gpu = gpu
        self.settings = configuration.mcmc
        self.configuration = configuration
    }

    func encode(_ context: AuxiliaryContext) throws(TrainingError) {
        let index = context.iteration
        guard index >= settings.startIteration, index < settings.stopIteration else {
            return
        }
        let rates = configuration.learningRates(iteration: index, sceneRadius: context.sceneExtent.radius)
        let step: Float = settings.noiseLearningRate * rates.position
        // Counter-based seed: the configuration seed plus the iteration; each Gaussian is the counter index.
        let seed = UInt32(truncatingIfNeeded: configuration.seed) &+ UInt32(truncatingIfNeeded: index)
        try metalCall { () throws(MetalError) -> Void in
            try self.gpu.encodeNoise(parameters: context.parameters, step: step, seed: seed,
                                     commandBuffer: context.commandBuffer)
        }
    }
}

enum MCMCTerms {
    /// The per-step terms for a configuration with MCMC on. Empty when `mcmc.isEnabled` is false.
    static func make(context: MetalContext, configuration: TrainingConfiguration) throws(TrainingError) -> [any AuxiliaryLoss] {
        guard configuration.mcmc.isEnabled else {
            return []
        }
        let gpu = try metalCall { () throws(MetalError) -> GPUMCMC in
            try GPUMCMC(context: context)
        }
        return [
            MCMCRegularizerTerm(gpu: gpu, opacityWeight: configuration.mcmc.opacityRegularizer,
                                scaleWeight: configuration.mcmc.scaleRegularizer),
            MCMCNoiseTerm(gpu: gpu, configuration: configuration),
        ]
    }
}
