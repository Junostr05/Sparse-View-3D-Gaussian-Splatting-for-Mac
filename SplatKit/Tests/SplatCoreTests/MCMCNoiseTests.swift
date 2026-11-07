import Foundation
import Testing
@testable import SplatCore

// CPU reference of the MCMC per-step terms (v1.2): the noise gate, the covariance-shaped noise with injected ε,
// and the opacity and scale L1 regularizers checked by finite differences.

@Suite struct MCMCNoiseTests {

    @Test func gateValues() {
        // The paper's gate is sigmoid(100 ((1 - o) - 0.995)): at o = 0 it is 1 / (1 + e^-0.5), its maximum.
        #expect(abs(MCMCNoise.gate(opacity: 0) - 1 / (1 + Float(exp(-0.5)))) < 1e-6)
        // One half at o = 0.005.
        #expect(abs(MCMCNoise.gate(opacity: 0.005) - 0.5) < 1e-6)
        // Opaque Gaussians get practically no noise.
        #expect(MCMCNoise.gate(opacity: 0.5) < 1e-6)
        #expect(MCMCNoise.gate(opacity: 1) < 1e-6)
        // Monotonically decreasing in opacity.
        #expect(MCMCNoise.gate(opacity: 0.001) > MCMCNoise.gate(opacity: 0.01))
        #expect(MCMCNoise.gate(opacity: 0.01) > MCMCNoise.gate(opacity: 0.05))
    }

    @Test func covarianceTimesIdentityRotationIsScaledSquared() {
        // Identity rotation and s = 0.1 on every axis: Σ = 0.01 I, so Σε = 0.01 ε.
        let logScale = SIMD3<Float>(repeating: log(0.1))
        let epsilon = SIMD3<Float>(1.5, -2, 0.25)
        let result = MCMCNoise.covarianceTimes(logScale: logScale, rotation: SIMD4(1, 0, 0, 0), epsilon: epsilon)
        #expect(abs(result.x - 0.015) < 1e-6)
        #expect(abs(result.y + 0.02) < 1e-6)
        #expect(abs(result.z - 0.0025) < 1e-6)
    }

    @Test func covarianceTimesRotatedMatchesExplicitCovariance() {
        // A 90° rotation about z maps x to y and y to −x, so Σ = R diag(s²) Rᵀ = diag(s_y², s_x², s_z²).
        // With s = (0.1, 0.2, 0.3) that is diag(0.04, 0.01, 0.09) and Σ(1, 2, 3) = (0.04, 0.02, 0.27).
        let logScale = SIMD3<Float>(log(0.1), log(0.2), log(0.3))
        let half = Float(0.5).squareRoot()
        let rotation = SIMD4<Float>(half, 0, 0, half)
        let result = MCMCNoise.covarianceTimes(logScale: logScale, rotation: rotation,
                                               epsilon: SIMD3<Float>(1, 2, 3))
        #expect(abs(result.x - 0.04) < 1e-6)
        #expect(abs(result.y - 0.02) < 1e-6)
        #expect(abs(result.z - 0.27) < 1e-6)
    }

    @Test func displacementUsesStepAndGate() {
        let logScale = SIMD3<Float>(repeating: log(0.1))
        let epsilon = SIMD3<Float>(1, 1, 1)
        // Logit -6 is opacity ≈ 0.0025, a nearly transparent Gaussian: the displacement is step · gate · Σε.
        let displaced = MCMCNoise.displacement(logScale: logScale, rotation: SIMD4(1, 0, 0, 0),
                                               opacityLogit: -6, step: 3, epsilon: epsilon)
        let gate: Float = MCMCNoise.gate(opacity: Float(0.0024726232))
        #expect(gate > 0.5)
        #expect(abs(displaced.x - 3 * gate * 0.01) < 1e-7)
        // Logit 0 is opacity 0.5: the gate removes the noise of an opaque Gaussian.
        let opaque = MCMCNoise.displacement(logScale: logScale, rotation: SIMD4(1, 0, 0, 0),
                                            opacityLogit: 0, step: 3, epsilon: epsilon)
        #expect(abs(opaque.x) < 1e-6)
    }

    @Test func applyAddsDisplacementWithInjectedEpsilon() {
        // Two Gaussians, identity rotations, both with s = 0.1 and opacity ≈ 0.0025 (logit -6).
        var positions: [Float] = [1, 2, 3, -1, 0, 0.5]
        let logScales = [Float](repeating: log(0.1), count: 6)
        let rotations: [Float] = [1, 0, 0, 0, 1, 0, 0, 0]
        let logits: [Float] = [-6, -6]
        let epsilons = [SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 2, 0)]
        MCMCNoise.apply(positions: &positions, logScales: logScales, rotations: rotations,
                        opacityLogits: logits, step: 1) { epsilons[$0] }
        let gate: Float = MCMCNoise.gate(opacity: Float(0.0024726232))
        #expect(abs(positions[0] - (1 + gate * 0.01)) < 1e-6)
        #expect(abs(positions[1] - 2) < 1e-6)
        #expect(abs(positions[4] - (0 + gate * 0.02)) < 1e-6)
    }

    // MARK: - Regularizers

    private func regularizerInputs() -> (logits: [Float], logScales: [Float]) {
        let logits: [Float] = [-1.2, 0.3, 2.0]
        let logScales: [Float] = [-1.0, -0.5, 0.2, -2.0, -2.2, -0.9, 0.1, 0.0, -0.3]
        return (logits, logScales)
    }

    @Test func regularizerGradientMatchesFiniteDifferences() {
        let inputs = regularizerInputs()
        let opacityWeight: Float = 0.7
        let scaleWeight: Float = 0.4
        var gradients = GaussianGradients(positions: [Float](repeating: 0, count: 9),
                                          logScales: [Float](repeating: 0, count: 9),
                                          rotations: [Float](repeating: 0, count: 12),
                                          opacityLogits: [Float](repeating: 0, count: 3),
                                          sh: [])
        let loss = MCMCRegularizers.addGradients(opacityLogits: inputs.logits, logScales: inputs.logScales,
                                                 opacityWeight: opacityWeight, scaleWeight: scaleWeight,
                                                 into: &gradients)
        let reference = MCMCRegularizers.loss(opacityLogits: inputs.logits, logScales: inputs.logScales,
                                              opacityWeight: opacityWeight, scaleWeight: scaleWeight)
        #expect(abs(loss - reference) < 1e-12)

        let h: Float = 1e-2
        for i in 0..<inputs.logits.count {
            var up = inputs.logits
            var down = inputs.logits
            up[i] += h
            down[i] -= h
            let numeric = (MCMCRegularizers.loss(opacityLogits: up, logScales: inputs.logScales,
                                                 opacityWeight: opacityWeight, scaleWeight: scaleWeight)
                - MCMCRegularizers.loss(opacityLogits: down, logScales: inputs.logScales,
                                        opacityWeight: opacityWeight, scaleWeight: scaleWeight)) / Double(2 * h)
            #expect(abs(Double(gradients.opacityLogits[i]) - numeric) < 2e-4, "opacity \(i)")
        }
        for i in 0..<inputs.logScales.count {
            var up = inputs.logScales
            var down = inputs.logScales
            up[i] += h
            down[i] -= h
            let numeric = (MCMCRegularizers.loss(opacityLogits: inputs.logits, logScales: up,
                                                 opacityWeight: opacityWeight, scaleWeight: scaleWeight)
                - MCMCRegularizers.loss(opacityLogits: inputs.logits, logScales: down,
                                        opacityWeight: opacityWeight, scaleWeight: scaleWeight)) / Double(2 * h)
            #expect(abs(Double(gradients.logScales[i]) - numeric) < 2e-4, "scale \(i)")
        }
    }

    @Test func regularizerGradientAccumulates() {
        let inputs = regularizerInputs()
        var gradients = GaussianGradients(positions: [Float](repeating: 0, count: 9),
                                          logScales: [Float](repeating: 0.5, count: 9),
                                          rotations: [Float](repeating: 0, count: 12),
                                          opacityLogits: [Float](repeating: 0.25, count: 3),
                                          sh: [])
        MCMCRegularizers.addGradients(opacityLogits: inputs.logits, logScales: inputs.logScales,
                                      opacityWeight: 0, scaleWeight: 0, into: &gradients)
        #expect(gradients.logScales.allSatisfy { $0 == 0.5 })
        #expect(gradients.opacityLogits.allSatisfy { $0 == 0.25 })
    }
}
