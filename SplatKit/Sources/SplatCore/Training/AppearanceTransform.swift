import Foundation

// Per-image appearance compensation (v1.2), CPU reference. Each training view v has a gain g_v and a bias b_v per
// colour channel, and the image loss sees A = R ⊙ g_v + b_v instead of the render R. The closed-form backward:
//   dL/dR   = G ⊙ g_v                    G = dL/dA is the image loss gradient; it feeds the rasterizer backward
//   dL/dg_v = Σ_pixels G ⊙ R
//   dL/db_v = Σ_pixels G
//   identity pull λ·mean_c((g_c − 1)² + b_c²) adds  dL/dg_c = (2λ/3)(g_c − 1)  and  dL/db_c = (2λ/3)·b_c.
// The Metal kernels in SplatMetal/Shaders/Appearance.metal mirror this file.

/// Gain and bias of one view, per colour channel. The identity leaves the render unchanged.
public struct AppearanceParameters: Sendable, Equatable {
    public var gain: SIMD3<Float>
    public var bias: SIMD3<Float>

    public static let identity = AppearanceParameters(gain: SIMD3<Float>(repeating: 1),
                                                      bias: SIMD3<Float>(repeating: 0))

    public init(gain: SIMD3<Float>, bias: SIMD3<Float>) {
        self.gain = gain
        self.bias = bias
    }

    /// Buffer layout shared with the Metal kernels: gain r, g, b, then bias r, g, b.
    var flat: [Float] {
        [gain.x, gain.y, gain.z, bias.x, bias.y, bias.z]
    }

    init(flat: [Float]) {
        precondition(flat.count == 6, "appearance parameters have 6 values")
        self.init(gain: SIMD3<Float>(flat[0], flat[1], flat[2]),
                  bias: SIMD3<Float>(flat[3], flat[4], flat[5]))
    }
}

/// Gradient of the loss with respect to one view's gain and bias, summed over the view's pixels.
public struct AppearanceGradient: Sendable, Equatable {
    /// dL/dg = Σ G ⊙ R.
    public var gain: SIMD3<Float>
    /// dL/db = Σ G.
    public var bias: SIMD3<Float>

    public static let zero = AppearanceGradient(gain: SIMD3<Float>(repeating: 0), bias: SIMD3<Float>(repeating: 0))

    public init(gain: SIMD3<Float>, bias: SIMD3<Float>) {
        self.gain = gain
        self.bias = bias
    }

    var flat: [Float] {
        [gain.x, gain.y, gain.z, bias.x, bias.y, bias.z]
    }
}

/// Result of `AppearanceTransform.backward`.
public struct AppearanceBackward: Sendable, Equatable {
    /// dL/dR per pixel, row-major: G ⊙ gain. This replaces the image gradient of the rasterizer backward.
    public var renderedGradient: [SIMD3<Float>]
    public var parameterGradient: AppearanceGradient
}

public enum AppearanceTransform {

    /// A = R ⊙ gain + bias for every pixel, without clamping.
    public static func apply(_ image: RGBImage, _ parameters: AppearanceParameters) -> RGBImage {
        var result = image
        let gain: SIMD3<Float> = parameters.gain
        let bias: SIMD3<Float> = parameters.bias
        for i in 0..<image.pixels.count {
            let rendered: SIMD3<Float> = image.pixels[i]
            let scaled: SIMD3<Float> = rendered * gain
            result.pixels[i] = scaled + bias
        }
        return result
    }

    /// Backward of `apply`. `imageGradient` is G = dL/dA, one entry per pixel of `rendered`. Sums are accumulated
    /// in `Double`.
    public static func backward(rendered: RGBImage, imageGradient: [SIMD3<Float>],
                                parameters: AppearanceParameters) -> AppearanceBackward {
        precondition(imageGradient.count == rendered.pixels.count,
                     "image gradient must have one entry per pixel")
        let gain: SIMD3<Float> = parameters.gain
        var renderedGradient = [SIMD3<Float>](repeating: SIMD3<Float>(repeating: 0), count: rendered.pixels.count)
        var gainSums: [Double] = [0, 0, 0]
        var biasSums: [Double] = [0, 0, 0]
        for i in 0..<rendered.pixels.count {
            let g: SIMD3<Float> = imageGradient[i]
            let r: SIMD3<Float> = rendered.pixels[i]
            let scaled: SIMD3<Float> = g * gain
            renderedGradient[i] = scaled
            for c in 0..<3 {
                gainSums[c] += Double(g[c]) * Double(r[c])
                biasSums[c] += Double(g[c])
            }
        }
        let gainGradient = SIMD3<Float>(Float(gainSums[0]), Float(gainSums[1]), Float(gainSums[2]))
        let biasGradient = SIMD3<Float>(Float(biasSums[0]), Float(biasSums[1]), Float(biasSums[2]))
        return AppearanceBackward(renderedGradient: renderedGradient,
                                  parameterGradient: AppearanceGradient(gain: gainGradient, bias: biasGradient))
    }

    /// Identity pull λ·mean_c((g_c − 1)² + b_c²) of one view, in `Double`.
    public static func identityPenalty(_ parameters: AppearanceParameters, weight: Float) -> Double {
        var sum: Double = 0
        for c in 0..<3 {
            let gainOffset: Double = Double(parameters.gain[c]) - 1
            let bias: Double = Double(parameters.bias[c])
            sum += gainOffset * gainOffset + bias * bias
        }
        return Double(weight) * sum / 3
    }

    /// Gradient of `identityPenalty`: (2λ/3)(g − 1) for the gains and (2λ/3)·b for the biases.
    public static func identityGradient(_ parameters: AppearanceParameters, weight: Float) -> AppearanceGradient {
        let scale: Float = 2 * weight / 3
        let gain: SIMD3<Float> = parameters.gain - SIMD3<Float>(repeating: 1)
        let gainGradient: SIMD3<Float> = gain * scale
        let biasGradient: SIMD3<Float> = parameters.bias * scale
        return AppearanceGradient(gain: gainGradient, bias: biasGradient)
    }
}

/// CPU mirror of one view's optimizer state on the GPU: parameters, Adam moments and the view's own step count.
/// Each view is updated only on the steps that sample it, so its bias correction uses its own count.
public struct AppearanceState: Sendable, Equatable {
    public private(set) var parameters: AppearanceParameters
    public private(set) var firstMoment: [Float]
    public private(set) var secondMoment: [Float]
    /// Adam steps taken by this view.
    public private(set) var step: Int

    /// Zero moments and step 0, starting from `parameters` (the identity by default).
    public init(parameters: AppearanceParameters = .identity) {
        self.parameters = parameters
        firstMoment = [Float](repeating: 0, count: 6)
        secondMoment = [Float](repeating: 0, count: 6)
        step = 0
    }

    /// One Adam step on the photometric `gradient` plus the identity pull. Uses `CPUAdam.update` with a single rate
    /// for the six values, so it is the reference for the Metal update kernel.
    public mutating func apply(_ gradient: AppearanceGradient, identityWeight: Float, learningRate: Float,
                               hyper: AdamHyperparameters) {
        let identity = AppearanceTransform.identityGradient(parameters, weight: identityWeight)
        let total = AppearanceGradient(gain: gradient.gain + identity.gain, bias: gradient.bias + identity.bias)
        var flatParameters: [Float] = parameters.flat
        var m: [Float] = firstMoment
        var v: [Float] = secondMoment
        step += 1
        CPUAdam.update(&flatParameters, gradients: total.flat, m: &m, v: &v, stride: 6, headCount: 6,
                       headRate: learningRate, tailRate: learningRate, step: step, hyper: hyper)
        parameters = AppearanceParameters(flat: flatParameters)
        firstMoment = m
        secondMoment = v
    }
}
