import Foundation
import Testing
@testable import SplatCore

// MARK: - Deterministic inputs

private struct DepthLossTestRNG: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// A `side × side` view with rendered depth `D` in [1, 4) and opacity 1 everywhere.
private struct SyntheticView {
    var depth: [Double]
    var alpha: [Double]
}

private func syntheticView(side: Int, seed: UInt64) -> SyntheticView {
    var rng = DepthLossTestRNG(state: seed)
    var depth: [Double] = []
    var alpha: [Double] = []
    for _ in 0..<(side * side) {
        let u: Double = Double.random(in: 0..<1, using: &rng)
        depth.append(1 + 3 * u)
        alpha.append(1)
    }
    return SyntheticView(depth: depth, alpha: alpha)
}

private func signOfDouble(_ v: Double) -> Double {
    if v > 0 { return 1 }
    if v < 0 { return -1 }
    return 0
}

// MARK: - DepthLoss

@Test func perfectCorrelationHasZeroLoss() {
    let view = syntheticView(side: 10, seed: 11)
    var target: [Float] = []
    for d in view.depth {
        target.append(Float(3 * d + 1))
    }
    let prior = DepthPrior(width: 10, height: 10, correlationTarget: target, domain: .depth)
    let render = DepthRender(raw: view.depth.map { Float($0) }, alpha: view.alpha.map { Float($0) })
    let weights = DepthLossWeights(pearson: 0.05, l1: 0.1, alphaMask: 0.3, minimumPixels: 64, nearPlane: 0.01)

    let result = DepthLoss.evaluate(depth: render, prior: prior, weights: weights)
    #expect(!result.degenerate)
    #expect(abs(result.pearson - 1) < 1e-6)
    #expect(abs(result.loss) < 1e-6)
    for g in result.gradient {
        #expect(abs(g.x) < 1e-6)
        #expect(abs(g.y) < 1e-6)
    }
}

@Test func anticorrelationHasLossTwiceWeight() {
    let view = syntheticView(side: 10, seed: 12)
    var target: [Float] = []
    for d in view.depth {
        target.append(Float(-d + 10))
    }
    let prior = DepthPrior(width: 10, height: 10, correlationTarget: target, domain: .depth)
    let render = DepthRender(raw: view.depth.map { Float($0) }, alpha: view.alpha.map { Float($0) })
    let weights = DepthLossWeights(pearson: 0.05, l1: 0.1, alphaMask: 0.3, minimumPixels: 64, nearPlane: 0.01)

    let result = DepthLoss.evaluate(depth: render, prior: prior, weights: weights)
    #expect(abs(result.pearson + 1) < 1e-6)
    #expect(abs(result.loss - 2 * 0.05) < 1e-6)
}

@Test func pearsonDomainFollowsPrior() {
    let view = syntheticView(side: 10, seed: 13)
    var inverseTarget: [Float] = []
    for d in view.depth {
        inverseTarget.append(Float(1 / d))
    }
    let render = DepthRender(raw: view.depth.map { Float($0) }, alpha: view.alpha.map { Float($0) })
    let weights = DepthLossWeights(pearson: 0.05, l1: 0.1, alphaMask: 0.3, minimumPixels: 64, nearPlane: 0.01)

    let inversePrior = DepthPrior(width: 10, height: 10, correlationTarget: inverseTarget,
                                  domain: .inverseDepth)
    let inverseResult = DepthLoss.evaluate(depth: render, prior: inversePrior, weights: weights)
    #expect(abs(inverseResult.pearson - 1) < 1e-9)

    let depthPrior = DepthPrior(width: 10, height: 10, correlationTarget: inverseTarget, domain: .depth)
    let depthResult = DepthLoss.evaluate(depth: render, prior: depthPrior, weights: weights)
    #expect(depthResult.pearson < 0.99)
}

@Test func matchesPearsonGolden() throws {
    let golden = try PearsonGolden.load("pearson")
    let weights = DepthLossWeights(pearson: golden.lambdaPearson, l1: golden.lambdaL1,
                                   alphaMask: golden.alphaMask, minimumPixels: golden.minimumPixels,
                                   nearPlane: golden.nearPlane)
    let render = DepthRender(raw: golden.draw, alpha: golden.alpha)
    let pixelCount = golden.width * golden.height
    #expect(golden.draw.count == pixelCount)
    #expect(golden.cases.count == 2)

    for c in golden.cases {
        let domain: DepthDomain = c.domain == "depth" ? .depth : .inverseDepth
        let prior = DepthPrior(width: golden.width, height: golden.height, correlationTarget: c.target,
                               domain: domain, metricDepth: golden.metricDepth,
                               isMeasuredMetric: golden.isMeasuredMetric)
        #expect(abs(Double(prior.l1Normalizer) - Double(golden.l1Normalizer)) < 1e-5)

        let result = DepthLoss.evaluate(depth: render, prior: prior, weights: weights)
        #expect(result.validPixels == c.validPixels)
        #expect(result.degenerate == c.degenerate)
        #expect(abs(result.loss - c.loss) < 1e-6)
        #expect(abs(result.pearson - c.pearson) < 1e-6)
        #expect(abs(result.l1 - c.l1) < 1e-6)

        #expect(result.gradient.count == pixelCount)
        var peak: Double = 0
        for g in c.gradDraw {
            peak = max(peak, abs(Double(g)))
        }
        for g in c.gradA {
            peak = max(peak, abs(Double(g)))
        }
        let floorValue: Double = 1e-6 * peak
        for p in 0..<pixelCount {
            let refDraw: Double = Double(c.gradDraw[p])
            let refA: Double = Double(c.gradA[p])
            let gotDraw: Double = Double(result.gradient[p].x)
            let gotA: Double = Double(result.gradient[p].y)
            #expect(abs(gotDraw - refDraw) <= 1e-5 * abs(refDraw) + floorValue)
            #expect(abs(gotA - refA) <= 1e-5 * abs(refA) + floorValue)
        }
    }
}

@Test func gradientMatchesFiniteDifferences() {
    for domain in [DepthDomain.depth, DepthDomain.inverseDepth] {
        let view = syntheticView(side: 10, seed: 14)
        var rng = DepthLossTestRNG(state: 15)
        // Opacity in [0.5, 1): the mask stays fixed under the ±h perturbations below.
        var alphaValues: [Double] = []
        var drawValues: [Double] = []
        var target: [Double] = []
        var metric: [Float] = []
        for i in 0..<view.depth.count {
            let a: Double = 0.5 + 0.5 * Double.random(in: 0..<1, using: &rng)
            let d: Double = view.depth[i]
            alphaValues.append(a)
            drawValues.append(a * d)
            let noise: Double = 0.2 * Double.random(in: -1..<1, using: &rng)
            target.append(2 * d + 0.5 + noise)
            metric.append(Float(1.1 * d + 0.05))
        }
        let targetFloat: [Float] = target.map { Float($0) }
        let prior = DepthPrior(width: 10, height: 10, correlationTarget: targetFloat, domain: domain,
                               metricDepth: metric, isMeasuredMetric: true)
        let weights = DepthLossWeights(pearson: 0.05, l1: 0.1, alphaMask: 0.3, minimumPixels: 64,
                                       nearPlane: 0.01)

        // Analytic gradient from the Float API, with the same inputs rounded to Float.
        let roundedDraw: [Float] = drawValues.map { Float($0) }
        let roundedAlpha: [Float] = alphaValues.map { Float($0) }
        let render = DepthRender(raw: roundedDraw, alpha: roundedAlpha)
        let analytic = DepthLoss.evaluate(depth: render, prior: prior, weights: weights)
        #expect(!analytic.degenerate)

        // Central differences in Double on exactly the same (rounded) inputs.
        let drawD: [Double] = roundedDraw.map { Double($0) }
        let alphaD: [Double] = roundedAlpha.map { Double($0) }
        let h: Double = 1e-4
        for p in 0..<drawD.count {
            var dPlus = drawD
            var dMinus = drawD
            dPlus[p] += h
            dMinus[p] -= h
            let lossPlus = DepthLoss.evaluateDouble(draw: dPlus, alpha: alphaD, prior: prior,
                                                    weights: weights).loss
            let lossMinus = DepthLoss.evaluateDouble(draw: dMinus, alpha: alphaD, prior: prior,
                                                     weights: weights).loss
            let fdDraw: Double = (lossPlus - lossMinus) / (2 * h)

            var aPlus = alphaD
            var aMinus = alphaD
            aPlus[p] += h
            aMinus[p] -= h
            let lossAPlus = DepthLoss.evaluateDouble(draw: drawD, alpha: aPlus, prior: prior,
                                                     weights: weights).loss
            let lossAMinus = DepthLoss.evaluateDouble(draw: drawD, alpha: aMinus, prior: prior,
                                                      weights: weights).loss
            let fdAlpha: Double = (lossAPlus - lossAMinus) / (2 * h)

            let gotDraw: Double = Double(analytic.gradient[p].x)
            let gotAlpha: Double = Double(analytic.gradient[p].y)
            #expect(abs(gotDraw - fdDraw) <= 1e-4 * abs(fdDraw) + 1e-8)
            #expect(abs(gotAlpha - fdAlpha) <= 1e-4 * abs(fdAlpha) + 1e-8)
        }
    }
}

@Test func constantRenderedDepthIsDegenerateNotNaN() {
    let side = 10
    let count = side * side
    var target: [Float] = []
    var raw: [Float] = []
    var alpha: [Float] = []
    for i in 0..<count {
        target.append(Float(1 + 0.01 * Double(i)))
        raw.append(2)
        alpha.append(1)
    }
    let prior = DepthPrior(width: side, height: side, correlationTarget: target, domain: .depth)
    let render = DepthRender(raw: raw, alpha: alpha)
    let weights = DepthLossWeights(pearson: 0.05, l1: 0.1, alphaMask: 0.3, minimumPixels: 64, nearPlane: 0.01)

    let result = DepthLoss.evaluate(depth: render, prior: prior, weights: weights)
    #expect(result.degenerate)
    #expect(result.loss.isFinite)
    #expect(result.loss == 0)
    #expect(result.pearson == 0)
    #expect(result.validPixels == count)
    for g in result.gradient {
        #expect(g.x == 0)
        #expect(g.y == 0)
        #expect(!g.x.isNaN && !g.y.isNaN)
    }
}

@Test func emptyMaskIsDegenerate() {
    let side = 10
    let count = side * side
    var target: [Float] = []
    var raw: [Float] = []
    var alpha: [Float] = []
    for i in 0..<count {
        target.append(Float(1 + 0.05 * Double(i)))
        raw.append(0.1)
        alpha.append(0.1)   // below alphaMask 0.3 everywhere
    }
    let prior = DepthPrior(width: side, height: side, correlationTarget: target, domain: .depth)
    let render = DepthRender(raw: raw, alpha: alpha)
    let weights = DepthLossWeights(pearson: 0.05, l1: 0.1, alphaMask: 0.3, minimumPixels: 64, nearPlane: 0.01)

    let result = DepthLoss.evaluate(depth: render, prior: prior, weights: weights)
    #expect(result.degenerate)
    #expect(result.validPixels == 0)
    #expect(result.loss == 0)
    #expect(result.pearson == 0)
    #expect(result.gradient.count == count)
    for g in result.gradient {
        #expect(g == .zero)
    }
}

@Test func l1OnlyForMeasuredMetric() {
    let view = syntheticView(side: 10, seed: 16)
    var target: [Float] = []
    var metric: [Float] = []
    for d in view.depth {
        target.append(Float(2 * d + 1))
        metric.append(Float(1.3 * d))
    }
    let render = DepthRender(raw: view.depth.map { Float($0) }, alpha: view.alpha.map { Float($0) })
    let weights = DepthLossWeights(pearson: 0.05, l1: 0.1, alphaMask: 0.3, minimumPixels: 64, nearPlane: 0.01)

    let relative = DepthPrior(width: 10, height: 10, correlationTarget: target, domain: .depth,
                              metricDepth: metric, isMeasuredMetric: false)
    let relativeResult = DepthLoss.evaluate(depth: render, prior: relative, weights: weights)
    #expect(relativeResult.l1 == 0)

    let measured = DepthPrior(width: 10, height: 10, correlationTarget: target, domain: .depth,
                              metricDepth: metric, isMeasuredMetric: true)
    let measuredResult = DepthLoss.evaluate(depth: render, prior: measured, weights: weights)
    #expect(measuredResult.l1 > 0)
    #expect(abs(measuredResult.loss - (relativeResult.loss + 0.1 * measuredResult.l1)) < 1e-9)
}
