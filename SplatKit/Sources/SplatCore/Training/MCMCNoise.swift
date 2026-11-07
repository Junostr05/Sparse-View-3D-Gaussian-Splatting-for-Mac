import Foundation

// MCMC per-step terms (v1.2): the SGLD position noise and the opacity and scale L1 regularizers. The GPU
// kernels in Shaders/MCMC.metal mirror these functions. Noise uses Float arithmetic like the kernel.

public enum MCMCNoise {
    /// Steepness `k` of the noise gate `1 / (1 + exp(k (o − x0)))`.
    public static let gateSteepness: Float = 100
    /// Opacity at which the gate is one half. This is the paper's `sigmoid(k ((1 − o) − 0.995))`: the noise is
    /// strong only for almost transparent Gaussians and vanishes for opaque ones.
    public static let gateOpacity: Float = 0.005

    /// Noise gate `M(o) = 1 / (1 + exp(k (o − x0)))`. At `o = 0` it is about 0.62 (its maximum), at `o = x0` one half,
    /// and it is practically zero for opacities above 0.1.
    public static func gate(opacity: Float) -> Float {
        1 / (1 + exp(gateSteepness * (opacity - gateOpacity)))
    }

    /// `Σ ε` with `Σ = R S² Rᵀ`, the covariance of a Gaussian with log-scales `logScale` and quaternion
    /// `rotation` (w, x, y, z). Computed as `R (S² (Rᵀ ε))`.
    public static func covarianceTimes(logScale: SIMD3<Float>, rotation: SIMD4<Float>,
                                       epsilon: SIMD3<Float>) -> SIMD3<Float> {
        let matrix = Matrix3.rotation(quaternion: rotation)
        // Rᵀ ε: the components are the dot products of the columns with ε.
        let projected = SIMD3<Float>((matrix.c0 * epsilon).sum(),
                                     (matrix.c1 * epsilon).sum(),
                                     (matrix.c2 * epsilon).sum())
        let scale = SIMD3<Float>(exp(logScale.x), exp(logScale.y), exp(logScale.z))
        let scaled = projected * (scale * scale)
        return matrix * scaled
    }

    /// Position displacement of one Gaussian: `step · M(o) · Σ ε`, where `step` is the noise learning rate times
    /// the position learning rate of the iteration.
    public static func displacement(logScale: SIMD3<Float>, rotation: SIMD4<Float>, opacityLogit: Float,
                                    step: Float, epsilon: SIMD3<Float>) -> SIMD3<Float> {
        let opacity = MCMCRelocation.sigmoid(opacityLogit)
        let factor = step * gate(opacity: opacity)
        return covarianceTimes(logScale: logScale, rotation: rotation, epsilon: epsilon) * factor
    }

    /// Adds the noise displacement to every position in place. `epsilon(i)` returns the standard normal sample
    /// of Gaussian `i`, so the caller can inject it (tests) or use the counter-based generator (trainer).
    public static func apply(positions: inout [Float], logScales: [Float], rotations: [Float],
                             opacityLogits: [Float], step: Float, epsilon: (Int) -> SIMD3<Float>) {
        let n = opacityLogits.count
        precondition(positions.count == 3 * n && logScales.count == 3 * n && rotations.count == 4 * n,
                     "parameter arrays do not match \(n) Gaussians")
        for i in 0..<n {
            let logScale = SIMD3<Float>(logScales[3 * i], logScales[3 * i + 1], logScales[3 * i + 2])
            let rotation = SIMD4<Float>(rotations[4 * i], rotations[4 * i + 1],
                                        rotations[4 * i + 2], rotations[4 * i + 3])
            let offset = displacement(logScale: logScale, rotation: rotation, opacityLogit: opacityLogits[i],
                                      step: step, epsilon: epsilon(i))
            positions[3 * i] += offset.x
            positions[3 * i + 1] += offset.y
            positions[3 * i + 2] += offset.z
        }
    }
}

/// L1 regularizers of MCMC (v1.2), added to the parameter gradients before Adam on every step:
/// `loss = opacityWeight · mean_i σ(l_i) + scaleWeight · mean_i Σ_a exp(s_ia)`, with `l` the opacity logits and
/// `s` the log-scales. Both terms are L1 because σ and exp are positive. The mean is over Gaussians.
public enum MCMCRegularizers {
    /// Loss value (Double) of the two terms.
    public static func loss(opacityLogits: [Float], logScales: [Float],
                            opacityWeight: Float, scaleWeight: Float) -> Double {
        let n = opacityLogits.count
        guard n > 0 else { return 0 }
        var opacitySum: Double = 0
        for logit in opacityLogits {
            opacitySum += 1 / (1 + exp(-Double(logit)))
        }
        var scaleSum: Double = 0
        for value in logScales {
            scaleSum += exp(Double(value))
        }
        let count = Double(n)
        return Double(opacityWeight) * opacitySum / count + Double(scaleWeight) * scaleSum / count
    }

    /// Adds the gradients of `loss` into `gradients` (accumulating) and returns the loss value.
    /// `d/dl_i = opacityWeight / n · σ(l_i)(1 − σ(l_i))`, `d/ds_ia = scaleWeight / n · exp(s_ia)`.
    @discardableResult
    public static func addGradients(opacityLogits: [Float], logScales: [Float], opacityWeight: Float,
                                    scaleWeight: Float, into gradients: inout GaussianGradients) -> Double {
        let n = opacityLogits.count
        precondition(logScales.count == 3 * n, "log-scales do not match \(n) Gaussians")
        precondition(gradients.opacityLogits.count == n && gradients.logScales.count == 3 * n,
                     "gradient buffers do not match \(n) Gaussians")
        guard n > 0 else { return 0 }
        let invCount: Double = 1 / Double(n)
        for i in 0..<n {
            let sigma: Double = 1 / (1 + exp(-Double(opacityLogits[i])))
            gradients.opacityLogits[i] += Float(Double(opacityWeight) * invCount * sigma * (1 - sigma))
        }
        for index in 0..<(3 * n) {
            let slope: Double = Double(scaleWeight) * invCount * exp(Double(logScales[index]))
            gradients.logScales[index] += Float(slope)
        }
        return loss(opacityLogits: opacityLogits, logScales: logScales,
                    opacityWeight: opacityWeight, scaleWeight: scaleWeight)
    }
}
