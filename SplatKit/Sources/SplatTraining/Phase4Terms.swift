import Metal
import SplatCore
import SplatMetal

// Phase 4 auxiliary terms of the GPU trainer (Task 17, decision P12). Each term only encodes work into the
// step's command buffer at its stage; the order of the stages inside one step is documented in `GPUTrainer`.
// Terms never wait for the GPU and never read GPU buffers on the CPU. Values that the CPU needs are read by
// the trainer at its wait points.

/// Where an auxiliary term is encoded inside one training step (P12).
public enum AuxiliaryStage: Sendable {
    /// After the photometric loss and before the backward pass. The term may write the depth gradient.
    case pixelGradient
    /// After the backward pass and before Adam. The term adds into the parameter gradients.
    case parameterGradient
    /// After Adam. The term changes the parameters in place and has no gradient.
    case postOptimizer
}

/// What an auxiliary term may read or write for one step. Every buffer belongs to the trainer and stays valid
/// until the step's command buffer has completed.
public struct AuxiliaryContext {
    public let viewIndex: Int
    public let iteration: Int
    /// False for every step in Task 17. Pseudo-view steps (Task 18) set it.
    public let isPseudoView: Bool
    public let camera: Camera
    /// `float2 (Draw, A)` per pixel of this view's depth forward pass. Nil when the view is rendered without depth.
    public let renderedDepth: MTLBuffer?
    /// `float2 (∂L/∂Draw, ∂L/∂A)` per pixel. Written by `.pixelGradient` terms. Nil when there is no depth pass.
    public let depthGradient: MTLBuffer?
    public let parameters: GaussianBuffers
    public let gradients: GradientBuffers
    public let sceneExtent: SceneExtent
    /// Open (uncommitted) until the trainer commits the step.
    public let commandBuffer: MTLCommandBuffer
}

/// An extra term encoded at one stage of every training step. The Phase 3 protocol had no external conformers,
/// so this replaces it (Task 17).
public protocol AuxiliaryLoss: AnyObject {
    var stage: AuxiliaryStage { get }
    func encode(_ context: AuxiliaryContext) throws(TrainingError)
}

/// Depth prior loss (Phase 4 P7) of a training view, at the `.pixelGradient` stage. Owns the GPU loss object,
/// one uploaded prior per view (nil for a view without one) and the value of the most recent encode.
final class DepthLossTerm: AuxiliaryLoss {
    let stage: AuxiliaryStage = .pixelGradient
    /// Slot of the training view in `GPUDepthLoss` (slot 1 belongs to pseudo-views, Task 18).
    private static let trainingSlot = 0

    private let loss: GPUDepthLoss
    private let priors: [GPUDepthPrior?]
    private let weights: DepthLossWeights
    /// True from an encode until the trainer reads its value at a wait point.
    private var hasPendingValue = false

    /// - Parameter priors: one entry per dataset view, in dataset order.
    init(loss: GPUDepthLoss, priors: [GPUDepthPrior?], weights: DepthLossWeights) {
        self.loss = loss
        self.priors = priors
        self.weights = weights
    }

    /// Whether the view at `index` has a depth prior.
    func hasPrior(view index: Int) -> Bool {
        priors[index] != nil
    }

    /// Encodes the loss of the view's rendered depth and writes its gradient. Does nothing when the context
    /// has no depth pass (the view is rendered without depth in this step).
    func encode(_ context: AuxiliaryContext) throws(TrainingError) {
        guard let renderedDepth = context.renderedDepth,
              let depthGradient = context.depthGradient,
              let prior = priors[context.viewIndex] else {
            return
        }
        try metalCall { () throws(MetalError) -> Void in
            try self.loss.encode(renderedDepth: renderedDepth, prior: prior, weights: self.weights,
                                 slot: DepthLossTerm.trainingSlot, depthGradient: depthGradient,
                                 commandBuffer: context.commandBuffer)
        }
        hasPendingValue = true
    }

    /// The value of the most recent encode, or nil when no encode is pending. Precondition: the command buffer
    /// of that encode has completed (the trainer calls this only after its wait).
    func takePendingValue() -> DepthLossValue? {
        guard hasPendingValue else {
            return nil
        }
        hasPendingValue = false
        return loss.readValue(slot: DepthLossTerm.trainingSlot)
    }
}

/// Anisotropy gradient (Phase 4 P10) on the log-scales, at the `.parameterGradient` stage. Adds into the
/// gradients of the step, which the backward pass has just overwritten.
final class AnisotropyTerm: AuxiliaryLoss {
    let stage: AuxiliaryStage = .parameterGradient
    private let regularizers: GPURegularizers
    private let weight: Float
    private let maxRatio: Float

    init(regularizers: GPURegularizers, weight: Float, maxRatio: Float) {
        self.regularizers = regularizers
        self.weight = weight
        self.maxRatio = maxRatio
    }

    func encode(_ context: AuxiliaryContext) throws(TrainingError) {
        try metalCall { () throws(MetalError) -> Void in
            try self.regularizers.encodeAnisotropy(parameters: context.parameters, gradients: context.gradients,
                                                   weight: self.weight, maxRatio: self.maxRatio,
                                                   commandBuffer: context.commandBuffer)
        }
    }
}

/// Scale cap projection (Phase 4 P10) at the `.postOptimizer` stage: clamps the log-scales after Adam, in place.
final class ScaleCapTerm: AuxiliaryLoss {
    let stage: AuxiliaryStage = .postOptimizer
    private let regularizers: GPURegularizers
    private let fraction: Float

    init(regularizers: GPURegularizers, fraction: Float) {
        self.regularizers = regularizers
        self.fraction = fraction
    }

    func encode(_ context: AuxiliaryContext) throws(TrainingError) {
        try metalCall { () throws(MetalError) -> Void in
            try self.regularizers.encodeScaleCap(parameters: context.parameters, fraction: self.fraction,
                                                 extent: context.sceneExtent, commandBuffer: context.commandBuffer)
        }
    }
}
