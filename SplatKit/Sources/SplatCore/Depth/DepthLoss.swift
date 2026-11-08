import Foundation

/// Weights and gates of the per-view depth loss (Phase 4 P7).
public struct DepthLossWeights: Sendable, Equatable {
    /// λ_P, weight of `1 − ρ`.
    public var pearson: Float
    /// λ_L1, weight of the normalised L1 term (measured metric depth only).
    public var l1: Float
    /// Pixels with rendered opacity `A` at or below this are excluded.
    public var alphaMask: Float
    /// Fewer valid pixels than this makes the Pearson term degenerate.
    public var minimumPixels: Int
    /// Lower clamp for depth before the inverse-depth transform.
    public var nearPlane: Float

    public init(pearson: Float, l1: Float, alphaMask: Float, minimumPixels: Int, nearPlane: Float) {
        self.pearson = pearson
        self.l1 = l1
        self.alphaMask = alphaMask
        self.minimumPixels = minimumPixels
        self.nearPlane = nearPlane
    }

    public init(settings: DepthPriorSettings, nearPlane: Float) {
        self.init(pearson: settings.pearsonWeight,
                  l1: settings.l1Weight,
                  alphaMask: settings.alphaMask,
                  minimumPixels: settings.minimumPixels,
                  nearPlane: nearPlane)
    }
}

/// Output of one depth loss evaluation.
public struct DepthLossResult: Sendable {
    /// `λ_P·(1 − ρ)` (0 when degenerate) plus `λ_L1·l1`.
    public var loss: Double
    /// ρ, 0 when degenerate.
    public var pearson: Double
    /// Normalised L1 value before weighting, 0 when not applicable.
    public var l1: Double
    /// Number of pixels in the mask.
    public var validPixels: Int
    /// True when the Pearson term was skipped (too few pixels or a near-constant signal).
    public var degenerate: Bool
    /// Per pixel `(∂L/∂Draw, ∂L/∂A)`, zero outside the mask.
    public var gradient: [SIMD2<Float>]

    public init(loss: Double, pearson: Double, l1: Double, validPixels: Int, degenerate: Bool,
                gradient: [SIMD2<Float>]) {
        self.loss = loss
        self.pearson = pearson
        self.l1 = l1
        self.validPixels = validPixels
        self.degenerate = degenerate
        self.gradient = gradient
    }
}

/// CPU reference for the Phase 4 depth prior loss: a Pearson term in the prior's domain plus an
/// optional L1 term against measured metric depth. Accumulates in `Double`.
///
/// With per-pixel weights `w` (`DepthPrior.weights`, v1.5) both terms are weighted: the Pearson
/// correlation uses weighted means, variances and covariance (`W = Σ w` over the mask), and the L1
/// term is `Σ w·|D − m| / Σ w` over the measured pixels. Without weights every `w` is 1 and the
/// result is exactly the unweighted loss.
public enum DepthLoss {
    public static func evaluate(depth: DepthRender, prior: DepthPrior,
                                weights: DepthLossWeights) -> DepthLossResult {
        let draw: [Double] = depth.raw.map { Double($0) }
        let alpha: [Double] = depth.alpha.map { Double($0) }
        return evaluateDouble(draw: draw, alpha: alpha, prior: prior, weights: weights)
    }

    /// Double-precision core, also used by the finite-difference tests.
    static func evaluateDouble(draw: [Double], alpha: [Double], prior: DepthPrior,
                               weights: DepthLossWeights) -> DepthLossResult {
        let count: Int = draw.count
        precondition(alpha.count == count, "draw and alpha must have the same count")
        precondition(prior.correlationTarget.count == count,
                     "prior and depth must have the same pixel count")
        if let metric = prior.metricDepth {
            precondition(metric.count == count, "metric depth must match the pixel count")
        }
        if let pixelWeights = prior.weights {
            precondition(pixelWeights.count == count, "depth weights must match the pixel count")
        }

        let near: Double = Double(weights.nearPlane)
        let alphaMask: Double = Double(weights.alphaMask)
        let lambdaP: Double = Double(weights.pearson)
        let lambdaL1: Double = Double(weights.l1)

        // Mask M: opacity above the threshold (and strictly positive, so D = Draw / A is
        // finite) and a finite positive prior target.
        var maskIndices: [Int] = []
        maskIndices.reserveCapacity(count)
        var depthValues = [Double](repeating: 0, count: count)
        for p in 0..<count {
            let a: Double = alpha[p]
            let target: Double = Double(prior.correlationTarget[p])
            if a > 0 && a > alphaMask && target.isFinite && target > 0 {
                maskIndices.append(p)
                depthValues[p] = draw[p] / a
            }
        }
        let n: Int = maskIndices.count

        // Pearson inputs in the prior's domain, and dx/dD per masked pixel.
        var xs = [Double](repeating: 0, count: n)
        var ys = [Double](repeating: 0, count: n)
        var dxdD = [Double](repeating: 0, count: n)
        var ws = [Double](repeating: 1, count: n)
        for k in 0..<n {
            let p: Int = maskIndices[k]
            let d: Double = depthValues[p]
            ys[k] = Double(prior.correlationTarget[p])
            if let pixelWeights = prior.weights {
                ws[k] = Double(pixelWeights[p])
            }
            switch prior.domain {
            case .depth:
                xs[k] = d
                dxdD[k] = 1
            case .inverseDepth:
                let clamped: Double = max(d, near)
                xs[k] = 1 / clamped
                dxdD[k] = d > near ? -1 / (d * d) : 0
            }
        }

        // dL/dD per masked pixel, accumulated from the Pearson and L1 terms.
        var gD = [Double](repeating: 0, count: n)
        var pearsonTerm: Double = 0
        var rho: Double = 0
        var degenerate: Bool = true

        var weightSum: Double = 0
        for k in 0..<n {
            weightSum += ws[k]
        }

        if n > 0 && n >= weights.minimumPixels && weightSum > 0 {
            let nd: Double = weightSum
            var sumX: Double = 0
            var sumY: Double = 0
            for k in 0..<n {
                sumX += ws[k] * xs[k]
                sumY += ws[k] * ys[k]
            }
            let mx: Double = sumX / nd
            let my: Double = sumY / nd

            var varX: Double = 0
            var varY: Double = 0
            var covXY: Double = 0
            for k in 0..<n {
                let dx: Double = xs[k] - mx
                let dy: Double = ys[k] - my
                varX += ws[k] * (dx * dx)
                varY += ws[k] * (dy * dy)
                covXY += ws[k] * (dx * dy)
            }
            varX /= nd
            varY /= nd
            covXY /= nd
            let sx: Double = varX.squareRoot()
            let sy: Double = varY.squareRoot()

            let flatX: Bool = !(sx > 1e-6 * (abs(mx) + 1e-6))
            let flatY: Bool = !(sy > 1e-6 * (abs(my) + 1e-6))
            if !flatX && !flatY {
                degenerate = false
                let r: Double = covXY / (sx * sy)
                rho = r
                pearsonTerm = lambdaP * (1 - r)
                for k in 0..<n {
                    let xhat: Double = (xs[k] - mx) / sx
                    let yhat: Double = (ys[k] - my) / sy
                    let dTermDx: Double = -lambdaP * ws[k] * (yhat - r * xhat) / (nd * sx)
                    gD[k] = dTermDx * dxdD[k]
                }
            }
        }

        // Optional L1 against measured metric depth, normalised by the median.
        var l1Value: Double = 0
        var loss: Double = pearsonTerm
        let l1Normalizer: Double = Double(prior.l1Normalizer)
        if prior.isMeasuredMetric, lambdaL1 > 0, l1Normalizer > 0, let metric = prior.metricDepth {
            var n1: Int = 0
            var weightSum1: Double = 0
            for k in 0..<n {
                let m: Double = Double(metric[maskIndices[k]])
                if m > 0 {
                    n1 += 1
                    weightSum1 += ws[k]
                }
            }
            if n1 > 0 && weightSum1 > 0 {
                let n1d: Double = weightSum1
                var sumAbs: Double = 0
                for k in 0..<n {
                    let p: Int = maskIndices[k]
                    let m: Double = Double(metric[p])
                    if m <= 0 {
                        continue
                    }
                    let diff: Double = depthValues[p] - m
                    sumAbs += ws[k] * abs(diff)
                    let sign: Double = signOfDifference(diff)
                    gD[k] += lambdaL1 * ws[k] * sign / (n1d * l1Normalizer)
                }
                l1Value = sumAbs / n1d / l1Normalizer
                loss += lambdaL1 * l1Value
            }
        }

        // Chain rule to (∂L/∂Draw, ∂L/∂A) with D = Draw / A.
        var gradient = [SIMD2<Float>](repeating: .zero, count: count)
        for k in 0..<n {
            let p: Int = maskIndices[k]
            let a: Double = alpha[p]
            let d: Double = depthValues[p]
            let g: Double = gD[k]
            let gradDraw: Double = g / a
            let gradAlpha: Double = -g * d / a
            gradient[p] = SIMD2<Float>(Float(gradDraw), Float(gradAlpha))
        }

        return DepthLossResult(loss: loss,
                               pearson: rho,
                               l1: l1Value,
                               validPixels: n,
                               degenerate: degenerate,
                               gradient: gradient)
    }

    private static func signOfDifference(_ diff: Double) -> Double {
        if diff > 0 { return 1 }
        if diff < 0 { return -1 }
        return 0
    }
}
