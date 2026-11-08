import Foundation
import Testing
@testable import SplatCore

// v1.5 confidence weighting of the depth loss: per-pixel weights in `DepthPrior`, the weighted Pearson and
// L1 terms of the CPU reference, checked against finite differences and against the unweighted loss.

private struct WeightedDepthRNG: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// A `side × side` view: rendered `(Draw, A)` with opacity in [0.5, 1) (the 0.3 mask stays fixed under small
/// perturbations), a noisy affine target in the requested domain with a few invalid entries, measured metric
/// depth and random weights in [0.1, 5).
private struct WeightedCase {
    var draw: [Float]
    var alpha: [Float]
    var target: [Float]
    var metric: [Float]
    var weights: [Float]
}

private func makeWeightedCase(side: Int, seed: UInt64, domain: DepthDomain) -> WeightedCase {
    var rng = WeightedDepthRNG(state: seed)
    var result = WeightedCase(draw: [], alpha: [], target: [], metric: [], weights: [])
    for i in 0..<(side * side) {
        let depthValue: Double = 1 + 3 * Double.random(in: 0..<1, using: &rng)
        let opacity: Double = 0.5 + 0.5 * Double.random(in: 0..<1, using: &rng)
        let noise: Double = Double.random(in: -1..<1, using: &rng)
        var targetValue: Double
        switch domain {
        case .depth:
            targetValue = 2 * depthValue + 0.5 + 0.3 * noise
        case .inverseDepth:
            targetValue = 1 / depthValue + 0.05 * noise
        }
        if i % 13 == 0 {
            targetValue = -1
        }
        let drawValue: Double = opacity * depthValue
        let metricValue: Double = 1.1 * depthValue + 0.05 + 0.2 * noise
        let weightValue: Double = 0.1 + 4.9 * Double.random(in: 0..<1, using: &rng)
        result.draw.append(Float(drawValue))
        result.alpha.append(Float(opacity))
        result.target.append(Float(targetValue))
        result.metric.append(Float(metricValue))
        result.weights.append(Float(weightValue))
    }
    return result
}

private let lossWeights = DepthLossWeights(pearson: 0.05, l1: 0.1, alphaMask: 0.3, minimumPixels: 64,
                                           nearPlane: 0.01)

@Suite struct WeightedDepthLossTests {

    @Test func priorWeightsAreNormalisedToMeanOneOverValidTargets() throws {
        let target: [Float] = [1, 2, -1, 3, .nan, 4]
        let weights: [Float] = [2, 4, 100, .infinity, 7, -3]
        let prior = DepthPrior(width: 6, height: 1, correlationTarget: target, domain: .depth, weights: weights)
        let normalised = try #require(prior.weights)
        // Valid targets: indices 0, 1, 3, 5 with cleaned weights 2, 4, 0, 0 (sum 6, mean 1.5).
        #expect(abs(normalised[0] - 2 / 1.5) < 1e-6)
        #expect(abs(normalised[1] - 4 / 1.5) < 1e-6)
        #expect(normalised[3] == 0)
        #expect(normalised[5] == 0)
        var sum: Float = 0
        for index in [0, 1, 3, 5] {
            sum += normalised[index]
        }
        #expect(abs(sum / 4 - 1) < 1e-6)
    }

    @Test func zeroOrMissingWeightsMeanUnweighted() {
        let target: [Float] = [1, 2, 3]
        let none = DepthPrior(width: 3, height: 1, correlationTarget: target, domain: .depth)
        #expect(none.weights == nil)
        let zeros = DepthPrior(width: 3, height: 1, correlationTarget: target, domain: .depth, weights: [0, 0, 0])
        #expect(zeros.weights == nil)
    }

    @Test func uniformWeightsGiveTheUnweightedLoss() {
        for domain in [DepthDomain.depth, DepthDomain.inverseDepth] {
            let input = makeWeightedCase(side: 12, seed: 31, domain: domain)
            let render = DepthRender(raw: input.draw, alpha: input.alpha)
            let plain = DepthPrior(width: 12, height: 12, correlationTarget: input.target, domain: domain,
                                   metricDepth: input.metric, isMeasuredMetric: true)
            let uniform = DepthPrior(width: 12, height: 12, correlationTarget: input.target, domain: domain,
                                     metricDepth: input.metric, isMeasuredMetric: true,
                                     weights: [Float](repeating: 3.5, count: 144))
            let a = DepthLoss.evaluate(depth: render, prior: plain, weights: lossWeights)
            let b = DepthLoss.evaluate(depth: render, prior: uniform, weights: lossWeights)
            #expect(!a.degenerate)
            #expect(abs(a.loss - b.loss) <= 1e-12 + 1e-9 * abs(a.loss))
            #expect(abs(a.pearson - b.pearson) <= 1e-9)
            #expect(abs(a.l1 - b.l1) <= 1e-9)
            for p in 0..<144 {
                #expect(abs(a.gradient[p].x - b.gradient[p].x) <= 1e-7 * abs(a.gradient[p].x) + 1e-12)
                #expect(abs(a.gradient[p].y - b.gradient[p].y) <= 1e-7 * abs(a.gradient[p].y) + 1e-12)
            }
        }
    }

    /// A pixel of weight 0 contributes nothing: the loss equals that of a prior whose target is invalid
    /// there, and its gradient is zero.
    @Test func zeroWeightRemovesAPixel() {
        let input = makeWeightedCase(side: 10, seed: 41, domain: .depth)
        let render = DepthRender(raw: input.draw, alpha: input.alpha)
        var weights: [Float] = input.weights
        weights[7] = 0
        let weighted = DepthPrior(width: 10, height: 10, correlationTarget: input.target, domain: .depth,
                                  metricDepth: input.metric, isMeasuredMetric: true, weights: weights)
        var reducedTarget: [Float] = input.target
        reducedTarget[7] = -1
        var reducedMetric: [Float] = input.metric
        reducedMetric[7] = 0
        // Same normalisation constant: only relative weights matter, so the raw weights are reused.
        let reduced = DepthPrior(width: 10, height: 10, correlationTarget: reducedTarget, domain: .depth,
                                 metricDepth: reducedMetric, isMeasuredMetric: true, weights: weights)
        let a = DepthLoss.evaluate(depth: render, prior: weighted, weights: lossWeights)
        let b = DepthLoss.evaluate(depth: render, prior: reduced, weights: lossWeights)
        #expect(a.validPixels == b.validPixels + 1)
        #expect(a.gradient[7] == .zero)
        // The L1 normaliser (median of the metric depth) differs by one sample, so compare the Pearson term only.
        #expect(abs(a.pearson - b.pearson) <= 1e-9)
    }

    @Test func weightedGradientMatchesFiniteDifferences() {
        for domain in [DepthDomain.depth, DepthDomain.inverseDepth] {
            let side: Int = 10
            let input = makeWeightedCase(side: side, seed: 51, domain: domain)
            let prior = DepthPrior(width: side, height: side, correlationTarget: input.target, domain: domain,
                                   metricDepth: input.metric, isMeasuredMetric: true, weights: input.weights)
            let render = DepthRender(raw: input.draw, alpha: input.alpha)
            let analytic = DepthLoss.evaluate(depth: render, prior: prior, weights: lossWeights)
            #expect(!analytic.degenerate)
            #expect(analytic.l1 > 0)

            let drawD: [Double] = input.draw.map { Double($0) }
            let alphaD: [Double] = input.alpha.map { Double($0) }
            let h: Double = 1e-4
            var checked: Int = 0
            for p in 0..<drawD.count {
                var dPlus: [Double] = drawD
                var dMinus: [Double] = drawD
                dPlus[p] += h
                dMinus[p] -= h
                let lossPlus: Double = DepthLoss.evaluateDouble(draw: dPlus, alpha: alphaD, prior: prior,
                                                                weights: lossWeights).loss
                let lossMinus: Double = DepthLoss.evaluateDouble(draw: dMinus, alpha: alphaD, prior: prior,
                                                                 weights: lossWeights).loss
                let fdDraw: Double = (lossPlus - lossMinus) / (2 * h)

                var aPlus: [Double] = alphaD
                var aMinus: [Double] = alphaD
                aPlus[p] += h
                aMinus[p] -= h
                let lossAPlus: Double = DepthLoss.evaluateDouble(draw: drawD, alpha: aPlus, prior: prior,
                                                                 weights: lossWeights).loss
                let lossAMinus: Double = DepthLoss.evaluateDouble(draw: drawD, alpha: aMinus, prior: prior,
                                                                  weights: lossWeights).loss
                let fdAlpha: Double = (lossAPlus - lossAMinus) / (2 * h)

                // The L1 term has kinks at D = m; skip pixels within the step of one.
                let d: Double = drawD[p] / alphaD[p]
                let m: Double = Double(input.metric[p])
                if abs(d - m) < 1e-2 {
                    continue
                }
                checked += 1
                let gotDraw: Double = Double(analytic.gradient[p].x)
                let gotAlpha: Double = Double(analytic.gradient[p].y)
                #expect(abs(gotDraw - fdDraw) <= 1e-4 * abs(fdDraw) + 1e-8, "\(domain) pixel \(p) draw")
                #expect(abs(gotAlpha - fdAlpha) <= 1e-4 * abs(fdAlpha) + 1e-8, "\(domain) pixel \(p) alpha")
            }
            #expect(checked > 80)
        }
    }

    /// Weights change the loss: down-weighting pixels whose target disagrees raises the correlation.
    @Test func downWeightingOutliersRaisesCorrelation() {
        let side: Int = 10
        var draw: [Float] = []
        var alpha: [Float] = []
        var target: [Float] = []
        var weights: [Float] = []
        for i in 0..<(side * side) {
            let d: Float = 1 + Float(i % 17) * 0.1
            draw.append(d)
            alpha.append(1)
            let outlier: Bool = i % 5 == 0
            target.append(outlier ? 10 - 3 * d : 2 * d + 1)
            weights.append(outlier ? 0.01 : 1)
        }
        let render = DepthRender(raw: draw, alpha: alpha)
        let plain = DepthPrior(width: side, height: side, correlationTarget: target, domain: .depth)
        let weighted = DepthPrior(width: side, height: side, correlationTarget: target, domain: .depth,
                                  weights: weights)
        let a = DepthLoss.evaluate(depth: render, prior: plain, weights: lossWeights)
        let b = DepthLoss.evaluate(depth: render, prior: weighted, weights: lossWeights)
        #expect(b.pearson > a.pearson)
        #expect(b.pearson > 0.99)
    }

    @Test func resamplingAndRotationCarryWeights() throws {
        let map = DepthMap(width: 4, height: 2, kind: .relativeDepth,
                           values: [1, 1, 1, 1, 1, 1, 1, 1], isEstimated: true,
                           weights: [1, 2, 3, 4, 5, 6, 7, 8])
        let rotated = map.rotatedClockwise()
        #expect(rotated.weights?.count == 8)
        // Clockwise: output (x, y) takes source (x = y, y = H - 1 - x); output (0, 0) is source (0, 1).
        #expect(rotated.weights?[0] == 5)
        let back = rotated.rotatedCounterClockwise()
        #expect(back.weights == map.weights)

        let up = try map.resampled(width: 8, height: 4)
        let upWeights = try #require(up.weights)
        #expect(upWeights.count == 32)
        #expect(upWeights[0] == 1)
        #expect(upWeights[31] == 8)

        let plain = DepthMap(width: 4, height: 2, kind: .relativeDepth, values: [1, 1, 1, 1, 1, 1, 1, 1])
        #expect(try plain.resampled(width: 8, height: 4).weights == nil)
    }
}
