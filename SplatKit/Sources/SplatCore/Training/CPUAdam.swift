import Foundation

/// Adam moments share the SoA layout of the gradients (and of the cloud).
public typealias GaussianMoments = GaussianGradients

/// Optimizer state for one `GaussianCloud`: first and second moments plus the global step counter.
public struct AdamState: Sendable, Equatable {
    public var m: GaussianMoments
    public var v: GaussianMoments
    public var step: Int

    /// Zero moments and step 0, sized for `cloud`.
    public init(zeroLike cloud: GaussianCloud) {
        m = GaussianMoments(zeroLike: cloud)
        v = GaussianMoments(zeroLike: cloud)
        step = 0
    }
}

/// CPU reference for the Adam optimizer (PyTorch formulation with bias correction).
/// The Metal kernel in Phase 3 Task 12 copies the arithmetic order used here.
public enum CPUAdam {

    /// Updates `parameters` in place. Float `i` belongs to Gaussian `i / stride`; floats with
    /// `i % stride < headCount` use `headRate`, the others `tailRate`.
    public static func update(_ parameters: inout [Float],
                              gradients: [Float],
                              m: inout [Float],
                              v: inout [Float],
                              stride: Int,
                              headCount: Int,
                              headRate: Float,
                              tailRate: Float,
                              step: Int,
                              hyper: AdamHyperparameters) {
        precondition(step >= 1, "Adam step counter starts at 1")
        precondition(stride > 0 && headCount >= 0 && headCount <= stride)
        precondition(gradients.count == parameters.count)
        precondition(m.count == parameters.count && v.count == parameters.count)

        let t = Double(step)
        let bc1: Float = Float(1 - pow(Double(hyper.beta1), t))
        let bc2: Float = Float(1 - pow(Double(hyper.beta2), t))
        let invSqrtBC2: Float = 1 / bc2.squareRoot()
        let beta1 = hyper.beta1
        let beta2 = hyper.beta2
        let epsilon = hyper.epsilon

        for i in 0..<parameters.count {
            let rate: Float = (i % stride) < headCount ? headRate : tailRate
            let stepSize: Float = rate / bc1
            let g: Float = gradients[i]

            let mNew: Float = beta1 * m[i] + (1 - beta1) * g
            let vNew: Float = beta2 * v[i] + (1 - beta2) * g * g
            m[i] = mNew
            v[i] = vNew

            let denominator: Float = vNew.squareRoot() * invSqrtBC2 + epsilon
            parameters[i] = parameters[i] - stepSize * mNew / denominator
        }
    }

    /// Increments `state.step`, then updates all five groups (SH uses stride 3K, headCount 3,
    /// so DC and rest coefficients get different rates). Quaternions are not renormalized.
    public static func step(_ cloud: GaussianCloud,
                            gradients: GaussianGradients,
                            state: inout AdamState,
                            rates: ResolvedLearningRates,
                            hyper: AdamHyperparameters) -> GaussianCloud {
        state.step += 1
        let step = state.step
        let k = GaussianCloud.coefficientCount(degree: cloud.shDegree)

        var positions = cloud.positions
        var logScales = cloud.logScales
        var rotations = cloud.rotations
        var opacityLogits = cloud.opacityLogits
        var sh = cloud.sh

        update(&positions, gradients: gradients.positions,
               m: &state.m.positions, v: &state.v.positions,
               stride: 3, headCount: 3, headRate: rates.position, tailRate: rates.position,
               step: step, hyper: hyper)
        update(&logScales, gradients: gradients.logScales,
               m: &state.m.logScales, v: &state.v.logScales,
               stride: 3, headCount: 3, headRate: rates.logScale, tailRate: rates.logScale,
               step: step, hyper: hyper)
        update(&rotations, gradients: gradients.rotations,
               m: &state.m.rotations, v: &state.v.rotations,
               stride: 4, headCount: 4, headRate: rates.rotation, tailRate: rates.rotation,
               step: step, hyper: hyper)
        update(&opacityLogits, gradients: gradients.opacityLogits,
               m: &state.m.opacityLogits, v: &state.v.opacityLogits,
               stride: 1, headCount: 1, headRate: rates.opacity, tailRate: rates.opacity,
               step: step, hyper: hyper)
        update(&sh, gradients: gradients.sh,
               m: &state.m.sh, v: &state.v.sh,
               stride: 3 * k, headCount: 3, headRate: rates.shDC, tailRate: rates.shRest,
               step: step, hyper: hyper)

        return GaussianCloud(
            validated: cloud.shDegree,
            positions: positions,
            logScales: logScales,
            rotations: rotations,
            opacityLogits: opacityLogits,
            sh: sh
        )
    }
}
