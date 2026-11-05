import Foundation
import Testing
import SplatCore

@Suite struct CPUAdamTests {

    /// Double-precision re-derivation of the documented update, used as the oracle.
    private func referenceUpdate(parameter: Double, gradient: Double, m: Double, v: Double,
                                 rate: Double, step: Int,
                                 hyper: AdamHyperparameters) -> (p: Double, m: Double, v: Double) {
        let beta1 = Double(hyper.beta1)
        let beta2 = Double(hyper.beta2)
        let t = Double(step)
        let bc1 = 1 - pow(beta1, t)
        let bc2 = 1 - pow(beta2, t)
        let mNew = beta1 * m + (1 - beta1) * gradient
        let vNew = beta2 * v + (1 - beta2) * gradient * gradient
        let stepSize = rate / bc1
        let denom = sqrt(vNew) / sqrt(bc2) + Double(hyper.epsilon)
        let pNew = parameter - stepSize * mNew / denom
        return (pNew, mNew, vNew)
    }

    @Test func firstStepMovesBySignTimesRate() {
        var parameters: [Float] = [0, 0, 0]
        let gradients: [Float] = [0.5, -2, 1e-3]
        var m = [Float](repeating: 0, count: 3)
        var v = [Float](repeating: 0, count: 3)
        let rate: Float = 0.01

        CPUAdam.update(&parameters, gradients: gradients, m: &m, v: &v,
                       stride: 3, headCount: 3, headRate: rate, tailRate: rate,
                       step: 1, hyper: AdamHyperparameters())

        // At t = 1 the bias-corrected update is rate · g / |g| = rate · sign(g).
        let expected: [Float] = [-rate, rate, -rate]
        for i in 0..<3 {
            let relativeError = abs(parameters[i] - expected[i]) / abs(expected[i])
            #expect(relativeError < 1e-5, "index \(i): \(parameters[i]) vs \(expected[i])")
        }
    }

    @Test func matchesHandComputedSecondStep() {
        let hyper = AdamHyperparameters()
        let rate: Float = 0.1
        let g1: Float = 1.0
        let g2: Float = 0.5

        var parameters: [Float] = [0]
        var m: [Float] = [0]
        var v: [Float] = [0]

        CPUAdam.update(&parameters, gradients: [g1], m: &m, v: &v,
                       stride: 1, headCount: 1, headRate: rate, tailRate: rate,
                       step: 1, hyper: hyper)
        CPUAdam.update(&parameters, gradients: [g2], m: &m, v: &v,
                       stride: 1, headCount: 1, headRate: rate, tailRate: rate,
                       step: 2, hyper: hyper)

        // Oracle in Double: step 1 then step 2.
        let first = referenceUpdate(parameter: 0, gradient: Double(g1), m: 0, v: 0,
                                    rate: Double(rate), step: 1, hyper: hyper)
        let second = referenceUpdate(parameter: first.p, gradient: Double(g2), m: first.m, v: first.v,
                                     rate: Double(rate), step: 2, hyper: hyper)

        // Hand check: step 2 has m = 0.14, v = 0.001249, bc1 = 0.19, bc2 = 0.001999,
        // so the second displacement is about -0.0932 and p ends near -0.1932.
        #expect(abs(parameters[0] - Float(second.p)) <= 1e-6 * abs(Float(second.p)))
        #expect(abs(Float(m[0]) - Float(second.m)) <= 1e-6)
        #expect(abs(Float(v[0]) - Float(second.v)) <= 1e-6 * Float(second.v))
        #expect(abs(parameters[0] - (-0.1932)) < 1e-3)
    }

    @Test func headAndTailRatesApplyPerStride() {
        // Two Gaussians with 6 floats each; the first 3 of each use headRate.
        var parameters = [Float](repeating: 0, count: 12)
        let gradients = [Float](repeating: 1, count: 12)
        var m = [Float](repeating: 0, count: 12)
        var v = [Float](repeating: 0, count: 12)
        let headRate: Float = 0.02
        let tailRate: Float = 0.005

        CPUAdam.update(&parameters, gradients: gradients, m: &m, v: &v,
                       stride: 6, headCount: 3, headRate: headRate, tailRate: tailRate,
                       step: 1, hyper: AdamHyperparameters())

        for i in 0..<12 {
            let expected: Float = (i % 6) < 3 ? -headRate : -tailRate
            #expect(abs(parameters[i] - expected) < 1e-6, "index \(i)")
        }
    }

    @Test func zeroGradientWithZeroMomentsIsNoOp() {
        var parameters: [Float] = [0.3, -1.2, 4]
        var m = [Float](repeating: 0, count: 3)
        var v = [Float](repeating: 0, count: 3)

        CPUAdam.update(&parameters, gradients: [0, 0, 0], m: &m, v: &v,
                       stride: 3, headCount: 3, headRate: 0.1, tailRate: 0.1,
                       step: 5, hyper: AdamHyperparameters())

        #expect(parameters == [0.3, -1.2, 4])
        #expect(m == [0, 0, 0])
        #expect(v == [0, 0, 0])
    }

    @Test func stepUpdatesEveryGroupWithItsRate() throws {
        // Degree-1 cloud: K = 4, so each Gaussian has 12 SH floats.
        let cloud = try GaussianCloud(
            shDegree: 1,
            positions: [Float](repeating: 0, count: 6),
            logScales: [Float](repeating: 0, count: 6),
            rotations: [1, 0, 0, 0, 1, 0, 0, 0],
            opacityLogits: [0, 0],
            sh: [Float](repeating: 0, count: 24)
        )
        let gradients = GaussianGradients(
            positions: [Float](repeating: 1, count: 6),
            logScales: [Float](repeating: 1, count: 6),
            rotations: [Float](repeating: 1, count: 8),
            opacityLogits: [1, 1],
            sh: [Float](repeating: 1, count: 24)
        )
        var state = AdamState(zeroLike: cloud)
        let rates = ResolvedLearningRates(position: 0.4, logScale: 0.3, rotation: 0.2,
                                          opacity: 0.1, shDC: 0.05, shRest: 0.01)

        let updated = CPUAdam.step(cloud, gradients: gradients, state: &state,
                                   rates: rates, hyper: AdamHyperparameters())

        #expect(state.step == 1)
        // Unit gradients at t = 1 move every parameter by exactly -rate.
        for i in 0..<6 {
            #expect(abs(updated.positions[i] - (-0.4)) < 1e-5)
            #expect(abs(updated.logScales[i] - (-0.3)) < 1e-5)
        }
        for i in 0..<8 {
            #expect(abs(updated.rotations[i] - (cloud.rotations[i] - 0.2)) < 1e-5)
        }
        #expect(abs(updated.opacityLogits[0] - (-0.1)) < 1e-5)
        #expect(abs(updated.opacityLogits[1] - (-0.1)) < 1e-5)
        // SH layout per Gaussian: [k][c] with 4 coefficients × 3 channels; the first 3 floats are DC.
        for i in 0..<24 {
            let indexInGaussian = i % 12
            let expected: Float = indexInGaussian < 3 ? -0.05 : -0.01
            #expect(abs(updated.sh[i] - expected) < 1e-5, "sh index \(i)")
        }
        #expect(updated.shDegree == 1)
        #expect(updated.count == 2)
    }
}
