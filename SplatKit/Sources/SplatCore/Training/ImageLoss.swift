import Foundation

/// Result of evaluating the training image loss for one view.
public struct ImageLossResult: Sendable {
    /// `(1 − λ)·l1 + λ·(1 − ssim)`.
    public var loss: Double
    /// Mean absolute error over W·H·3 values.
    public var l1: Double
    /// Mean SSIM over W·H·3 values.
    public var ssim: Double
    /// dL/drendered, row-major, one entry per pixel.
    public var gradient: [SIMD3<Float>]

    public init(loss: Double, l1: Double, ssim: Double, gradient: [SIMD3<Float>]) {
        self.loss = loss
        self.l1 = l1
        self.ssim = ssim
        self.gradient = gradient
    }
}

/// CPU reference for the training loss `L = (1−λ)·mean|x−y| + λ·(1 − mean SSIM)`.
///
/// SSIM follows Wang et al. 2004 with an 11-tap Gaussian window (σ = 1.5), zero padding
/// and same-size output, applied per channel. The backward pass is the closed-form
/// derivative documented in the Phase 3 plan (Task 3); the Metal kernels mirror it.
public enum ImageLoss {
    public static let windowRadius = 5

    /// 11 normalized taps, `exp(−(i−5)²/(2·1.5²))`.
    public static let window: [Float] = makeWindow()

    /// C₁ = (0.01)².
    public static let c1: Float = 0.0001
    /// C₂ = (0.03)².
    public static let c2: Float = 0.0009

    private static func makeWindow() -> [Float] {
        let sigma: Double = 1.5
        var taps: [Double] = []
        for i in 0...(2 * windowRadius) {
            let d: Double = Double(i - windowRadius)
            let exponent: Double = -(d * d) / (2 * sigma * sigma)
            taps.append(exp(exponent))
        }
        let total: Double = taps.reduce(0, +)
        return taps.map { Float($0 / total) }
    }

    public static func evaluate(rendered: RGBImage, target: RGBImage, lambda: Float) -> ImageLossResult {
        precondition(rendered.width == target.width && rendered.height == target.height,
                     "rendered and target images must have the same size")
        let width = rendered.width
        let height = rendered.height
        let count = width * height
        let n: Double = Double(count * 3)
        let lam: Double = Double(lambda)
        let c1d: Double = Double(c1)
        let c2d: Double = Double(c2)
        let taps: [Double] = window.map { Double($0) }
        let ssimGradientScale: Double = -lam / n
        let l1GradientScale: Double = (1 - lam) / n

        var gradient = [SIMD3<Float>](repeating: .zero, count: count)
        var l1Sum: Double = 0
        var ssimSum: Double = 0

        for c in 0..<3 {
            var xs = [Double](repeating: 0, count: count)
            var ys = [Double](repeating: 0, count: count)
            var xx = [Double](repeating: 0, count: count)
            var yy = [Double](repeating: 0, count: count)
            var xy = [Double](repeating: 0, count: count)
            for i in 0..<count {
                let xv: Double = Double(rendered.pixels[i][c])
                let yv: Double = Double(target.pixels[i][c])
                xs[i] = xv
                ys[i] = yv
                xx[i] = xv * xv
                yy[i] = yv * yv
                xy[i] = xv * yv
            }

            let muX = blur(xs, width: width, height: height, taps: taps)
            let muY = blur(ys, width: width, height: height, taps: taps)
            let eXX = blur(xx, width: width, height: height, taps: taps)
            let eYY = blur(yy, width: width, height: height, taps: taps)
            let eXY = blur(xy, width: width, height: height, taps: taps)

            var m0 = [Double](repeating: 0, count: count)
            var m1 = [Double](repeating: 0, count: count)
            var m2 = [Double](repeating: 0, count: count)

            for i in 0..<count {
                let mx: Double = muX[i]
                let my: Double = muY[i]
                let sxx: Double = eXX[i] - mx * mx
                let syy: Double = eYY[i] - my * my
                let sxy: Double = eXY[i] - mx * my
                let a1: Double = 2 * mx * my + c1d
                let a2: Double = 2 * sxy + c2d
                let b1: Double = mx * mx + my * my + c1d
                let b2: Double = sxx + syy + c2d
                let numerator: Double = a1 * a2
                let denominator: Double = b1 * b2
                let s: Double = numerator / denominator
                ssimSum += s
                l1Sum += abs(xs[i] - ys[i])

                // Closed-form SSIM backward, g = −λ/N (see plan Task 3). The divisions by a1 and a2 are cancelled
                // analytically (s = a1·a2 / (b1·b2)): a2 = 2σxy + C2 crosses zero for anticorrelated windows, and the
                // forms s·2μy/a2 and 2s/a2 are 0·∞ = NaN there. Same arithmetic as `ssimMapsV` in Loss.metal.
                let muTerm: Double = 2 * my * (a2 - a1) / denominator + s * (2 * mx / b2 - 2 * mx / b1)
                m0[i] = ssimGradientScale * muTerm
                m1[i] = ssimGradientScale * (-s / b2)
                m2[i] = ssimGradientScale * (2 * a1 / denominator)
            }

            let g0 = blur(m0, width: width, height: height, taps: taps)
            let g1 = blur(m1, width: width, height: height, taps: taps)
            let g2 = blur(m2, width: width, height: height, taps: taps)

            for i in 0..<count {
                let d: Double = xs[i] - ys[i]
                let sign: Double = d > 0 ? 1 : (d < 0 ? -1 : 0)
                let l1Part: Double = l1GradientScale * sign
                let ssimPart: Double = g0[i] + 2 * xs[i] * g1[i] + ys[i] * g2[i]
                gradient[i][c] = Float(l1Part + ssimPart)
            }
        }

        let l1: Double = l1Sum / n
        let ssim: Double = ssimSum / n
        let loss: Double = (1 - lam) * l1 + lam * (1 - ssim)
        return ImageLossResult(loss: loss, l1: l1, ssim: ssim, gradient: gradient)
    }

    /// Separable zero-padded blur of a row-major plane. The kernel is symmetric, so this
    /// operator is its own adjoint.
    static func blur(_ src: [Double], width: Int, height: Int, taps: [Double]) -> [Double] {
        var tmp = [Double](repeating: 0, count: width * height)
        for y in 0..<height {
            let row = y * width
            for x in 0..<width {
                var acc: Double = 0
                for k in 0..<taps.count {
                    let sx = x + k - windowRadius
                    if sx >= 0 && sx < width {
                        acc += taps[k] * src[row + sx]
                    }
                }
                tmp[row + x] = acc
            }
        }
        var out = [Double](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                var acc: Double = 0
                for k in 0..<taps.count {
                    let sy = y + k - windowRadius
                    if sy >= 0 && sy < height {
                        acc += taps[k] * tmp[sy * width + x]
                    }
                }
                out[y * width + x] = acc
            }
        }
        return out
    }
}

/// Image quality metrics over RGB images with values in [0, 1].
public enum ImageMetrics {
    /// Mean squared error over all W·H·3 values, accumulated in `Double`.
    public static func mse(_ a: RGBImage, _ b: RGBImage) -> Double {
        precondition(a.width == b.width && a.height == b.height,
                     "images must have the same size")
        let count = a.pixels.count
        var sum: Double = 0
        for i in 0..<count {
            let dx: Double = Double(a.pixels[i].x) - Double(b.pixels[i].x)
            let dy: Double = Double(a.pixels[i].y) - Double(b.pixels[i].y)
            let dz: Double = Double(a.pixels[i].z) - Double(b.pixels[i].z)
            sum += dx * dx + dy * dy + dz * dz
        }
        return sum / Double(count * 3)
    }

    /// Peak signal-to-noise ratio in dB for a peak value of 1. Identical images give `+infinity`.
    public static func psnr(_ a: RGBImage, _ b: RGBImage) -> Double {
        let error: Double = mse(a, b)
        if error == 0 {
            return .infinity
        }
        return 10 * log10(1 / error)
    }

    /// Mean SSIM over W·H·3 values with the same window and constants as `ImageLoss`.
    /// Equal to `ImageLoss.evaluate(rendered: a, target: b, lambda: _).ssim`, without the gradient.
    public static func ssim(_ a: RGBImage, _ b: RGBImage) -> Double {
        precondition(a.width == b.width && a.height == b.height,
                     "images must have the same size")
        let width: Int = a.width
        let height: Int = a.height
        let count: Int = width * height
        let n: Double = Double(count * 3)
        let c1d: Double = Double(ImageLoss.c1)
        let c2d: Double = Double(ImageLoss.c2)
        let taps: [Double] = ImageLoss.window.map { Double($0) }

        var ssimSum: Double = 0
        for c in 0..<3 {
            var xs = [Double](repeating: 0, count: count)
            var ys = [Double](repeating: 0, count: count)
            var xx = [Double](repeating: 0, count: count)
            var yy = [Double](repeating: 0, count: count)
            var xy = [Double](repeating: 0, count: count)
            for i in 0..<count {
                let xv: Double = Double(a.pixels[i][c])
                let yv: Double = Double(b.pixels[i][c])
                xs[i] = xv
                ys[i] = yv
                xx[i] = xv * xv
                yy[i] = yv * yv
                xy[i] = xv * yv
            }

            let muX = ImageLoss.blur(xs, width: width, height: height, taps: taps)
            let muY = ImageLoss.blur(ys, width: width, height: height, taps: taps)
            let eXX = ImageLoss.blur(xx, width: width, height: height, taps: taps)
            let eYY = ImageLoss.blur(yy, width: width, height: height, taps: taps)
            let eXY = ImageLoss.blur(xy, width: width, height: height, taps: taps)

            for i in 0..<count {
                let mx: Double = muX[i]
                let my: Double = muY[i]
                let sxx: Double = eXX[i] - mx * mx
                let syy: Double = eYY[i] - my * my
                let sxy: Double = eXY[i] - mx * my
                let a1: Double = 2 * mx * my + c1d
                let a2: Double = 2 * sxy + c2d
                let b1: Double = mx * mx + my * my + c1d
                let b2: Double = sxx + syy + c2d
                let numerator: Double = a1 * a2
                let denominator: Double = b1 * b2
                ssimSum += numerator / denominator
            }
        }
        return ssimSum / n
    }
}
