import Foundation
import Testing
@testable import SplatCore

// CPU reference of per-image appearance compensation. Every analytic gradient is checked against central finite
// differences of the full objective: the L1 + D-SSIM image loss of the compensated render plus the identity pull.

private struct AppearanceRNG: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

private func appearanceImage(width: Int, height: Int, seed: UInt64,
                             range: ClosedRange<Float> = 0.05...0.95) -> RGBImage {
    var rng = AppearanceRNG(state: seed)
    var pixels: [SIMD3<Float>] = []
    pixels.reserveCapacity(width * height)
    for _ in 0..<(width * height) {
        let r = Float.random(in: range, using: &rng)
        let g = Float.random(in: range, using: &rng)
        let b = Float.random(in: range, using: &rng)
        pixels.append(SIMD3<Float>(r, g, b))
    }
    return RGBImage(width: width, height: height, pixels: pixels)
}

/// Objective of one view: image loss of the compensated render plus the identity pull.
private func objective(rendered: RGBImage, target: RGBImage, parameters: AppearanceParameters,
                       identityWeight: Float, lambda: Float) -> Double {
    let compensated = AppearanceTransform.apply(rendered, parameters)
    let loss = ImageLoss.evaluate(rendered: compensated, target: target, lambda: lambda).loss
    return loss + AppearanceTransform.identityPenalty(parameters, weight: identityWeight)
}

/// Analytic gradient of `objective` with respect to the gains and biases, assembled from the reference backward and
/// the identity gradient.
private func analyticParameterGradient(rendered: RGBImage, target: RGBImage, parameters: AppearanceParameters,
                                       identityWeight: Float, lambda: Float) -> [Float] {
    let compensated = AppearanceTransform.apply(rendered, parameters)
    let loss = ImageLoss.evaluate(rendered: compensated, target: target, lambda: lambda)
    let backward = AppearanceTransform.backward(rendered: rendered, imageGradient: loss.gradient,
                                                parameters: parameters)
    let identity = AppearanceTransform.identityGradient(parameters, weight: identityWeight)
    let gain: SIMD3<Float> = backward.parameterGradient.gain + identity.gain
    let bias: SIMD3<Float> = backward.parameterGradient.bias + identity.bias
    return [gain.x, gain.y, gain.z, bias.x, bias.y, bias.z]
}

@Suite struct AppearanceTransformTests {

    @Test func identityLeavesImageUnchanged() {
        let image = appearanceImage(width: 5, height: 4, seed: 1)
        let result = AppearanceTransform.apply(image, .identity)
        #expect(result == image)
        #expect(AppearanceTransform.identityPenalty(.identity, weight: 0.5) == 0)
        let gradient = AppearanceTransform.identityGradient(.identity, weight: 0.5)
        #expect(gradient == AppearanceGradient.zero)
    }

    @Test func affineMapIsPerChannel() {
        let image = RGBImage(width: 1, height: 2,
                             pixels: [SIMD3<Float>(0.2, 0.4, 0.6), SIMD3<Float>(1, 0, 0.5)])
        let parameters = AppearanceParameters(gain: SIMD3<Float>(1.3, 2, 0.5), bias: SIMD3<Float>(0.05, -0.1, 0))
        let result = AppearanceTransform.apply(image, parameters)
        #expect(abs(result.pixels[0].x - (0.2 * 1.3 + 0.05)) < 1e-6)
        #expect(abs(result.pixels[0].y - (0.4 * 2 - 0.1)) < 1e-6)
        #expect(abs(result.pixels[0].z - 0.3) < 1e-6)
        #expect(abs(result.pixels[1].x - 1.35) < 1e-6)
        #expect(abs(result.pixels[1].y + 0.1) < 1e-6)
        #expect(abs(result.pixels[1].z - 0.25) < 1e-6)
    }

    @Test func parameterGradientMatchesFiniteDifferences() {
        // Render and target in disjoint ranges: no residual sits near the L1 kink, where a finite difference is biased.
        let rendered = appearanceImage(width: 6, height: 5, seed: 7, range: 0.05...0.4)
        let target = appearanceImage(width: 6, height: 5, seed: 9, range: 0.6...0.95)
        let parameters = AppearanceParameters(gain: SIMD3<Float>(1.2, 0.85, 1.1),
                                              bias: SIMD3<Float>(0.04, -0.03, 0.02))
        let identityWeight: Float = 0.05
        let lambda: Float = 0.2
        let analytic = analyticParameterGradient(rendered: rendered, target: target, parameters: parameters,
                                                 identityWeight: identityWeight, lambda: lambda)
        // Small step: the L1 term has a kink where a pixel's residual crosses zero, and a larger step can straddle it.
        let step: Float = 1e-3
        for index in 0..<6 {
            var plus = parameters.flat
            var minus = parameters.flat
            plus[index] += step
            minus[index] -= step
            let up = objective(rendered: rendered, target: target, parameters: AppearanceParameters(flat: plus),
                               identityWeight: identityWeight, lambda: lambda)
            let down = objective(rendered: rendered, target: target, parameters: AppearanceParameters(flat: minus),
                                 identityWeight: identityWeight, lambda: lambda)
            let numeric = (up - down) / Double(2 * step)
            let error = abs(numeric - Double(analytic[index]))
            #expect(error <= 2e-3 * max(1, abs(numeric)), "parameter \(index): analytic \(analytic[index]), numeric \(numeric)")
        }
    }

    @Test func renderedGradientMatchesFiniteDifferences() {
        let rendered = appearanceImage(width: 6, height: 5, seed: 21)
        let target = appearanceImage(width: 6, height: 5, seed: 23)
        let parameters = AppearanceParameters(gain: SIMD3<Float>(0.9, 1.25, 1.05),
                                              bias: SIMD3<Float>(-0.02, 0.03, 0.01))
        let lambda: Float = 0.2
        let loss = ImageLoss.evaluate(rendered: AppearanceTransform.apply(rendered, parameters), target: target,
                                      lambda: lambda)
        let backward = AppearanceTransform.backward(rendered: rendered, imageGradient: loss.gradient,
                                                    parameters: parameters)
        let step: Float = 1e-2
        // A spread of pixels and channels, including the first and last pixel.
        let probes: [(pixel: Int, channel: Int)] = [(0, 0), (7, 1), (13, 2), (21, 0), (29, 2)]
        for probe in probes {
            var up = rendered
            var down = rendered
            up.pixels[probe.pixel][probe.channel] += step
            down.pixels[probe.pixel][probe.channel] -= step
            let upLoss = ImageLoss.evaluate(rendered: AppearanceTransform.apply(up, parameters), target: target,
                                            lambda: lambda).loss
            let downLoss = ImageLoss.evaluate(rendered: AppearanceTransform.apply(down, parameters), target: target,
                                              lambda: lambda).loss
            let numeric: Double = (upLoss - downLoss) / Double(2 * step)
            let analytic: Float = backward.renderedGradient[probe.pixel][probe.channel]
            let error: Double = abs(numeric - Double(analytic))
            #expect(error <= 2e-3 * max(1e-3, abs(numeric)),
                    "pixel \(probe.pixel) channel \(probe.channel): analytic \(analytic), numeric \(numeric)")
        }
    }

    @Test func identityPenaltyGradientMatchesFiniteDifferences() {
        let parameters = AppearanceParameters(gain: SIMD3<Float>(1.4, 0.6, 1.0), bias: SIMD3<Float>(0.3, -0.2, 0.1))
        let weight: Float = 0.7
        let analytic: [Float] = AppearanceTransform.identityGradient(parameters, weight: weight).flat
        let step: Float = 1e-3
        let values: [Float] = parameters.flat
        for index in 0..<6 {
            var plus = values
            var minus = values
            plus[index] += step
            minus[index] -= step
            let up = AppearanceTransform.identityPenalty(AppearanceParameters(flat: plus), weight: weight)
            let down = AppearanceTransform.identityPenalty(AppearanceParameters(flat: minus), weight: weight)
            let numeric: Double = (up - down) / Double(2 * step)
            #expect(abs(numeric - Double(analytic[index])) <= 1e-4,
                    "index \(index): analytic \(analytic[index]), numeric \(numeric)")
        }
    }

    @Test func adamStateMovesTowardsTheTargetGain() {
        // A constant gradient that pushes the gain of channel 0 down: the gain must fall, and the identity pull
        // must keep the other channels near 1 when their gradient is zero.
        var state = AppearanceState()
        let gradient = AppearanceGradient(gain: SIMD3<Float>(0.5, 0, 0), bias: .zero)
        let hyper = AdamHyperparameters()
        for _ in 0..<20 {
            state.apply(gradient, identityWeight: 0, learningRate: 0.01, hyper: hyper)
        }
        #expect(state.step == 20)
        #expect(state.parameters.gain.x < 1 - 0.1)
        #expect(state.parameters.gain.y == 1)
        #expect(state.parameters.bias == .zero)
    }
}
