import Foundation
import Testing
@testable import SplatCore

// v1.5 surface-normal consistency loss: CPU reference against analytic plane normals and finite differences.

private struct NormalTestRNG: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

private let testIntrinsics = NormalLossIntrinsics(fx: 20, fy: 22, cx: 6.3, cy: 4.8)

/// Camera-z depth of the plane `n·X = d` along the ray of pixel (x, y).
private func planeDepth(x: Int, y: Int, normal: SIMD3<Double>, offset: Double) -> Double {
    let ray: SIMD3<Double> = NormalLoss.ray(x: x, y: y, intrinsics: testIntrinsics)
    return offset / NormalLoss.dot(normal, ray)
}

private func unit(_ v: SIMD3<Double>) -> SIMD3<Double> {
    v / NormalLoss.dot(v, v).squareRoot()
}

@Suite struct NormalLossTests {

    @Test func renderedNormalsOfAPlaneAreItsNormalTowardsTheCamera() {
        let width: Int = 12
        let height: Int = 9
        // A slanted plane in front of the camera; its normal towards the camera has negative z.
        let towardsCamera: SIMD3<Double> = unit(SIMD3<Double>(0.3, -0.2, -1))
        var depth: [Float] = []
        for y in 0..<height {
            for x in 0..<width {
                depth.append(Float(planeDepth(x: x, y: y, normal: towardsCamera, offset: -3)))
            }
        }
        #expect(depth.allSatisfy { $0 > 0 })
        let normals: [SIMD3<Float>] = NormalLoss.normals(depth: depth, width: width, height: height,
                                                         intrinsics: testIntrinsics)
        for y in 0..<(height - 1) {
            for x in 0..<(width - 1) {
                let n: SIMD3<Float> = normals[y * width + x]
                #expect(abs(Double(n.x) - towardsCamera.x) < 1e-4)
                #expect(abs(Double(n.y) - towardsCamera.y) < 1e-4)
                #expect(abs(Double(n.z) - towardsCamera.z) < 1e-4)
            }
        }
        // The last column and row have no stencil.
        #expect(normals[width - 1] == .zero)
        #expect(normals[(height - 1) * width] == .zero)
    }

    @Test func matchingPriorHasZeroLossAndOppositePriorTwiceTheWeight() {
        let width: Int = 10
        let height: Int = 8
        let towardsCamera: SIMD3<Double> = unit(SIMD3<Double>(-0.1, 0.25, -1))
        var draw: [Double] = []
        var alpha: [Double] = []
        for y in 0..<height {
            for x in 0..<width {
                alpha.append(0.8)
                draw.append(0.8 * planeDepth(x: x, y: y, normal: towardsCamera, offset: -2))
            }
        }
        let prior: SIMD3<Float> = SIMD3<Float>(Float(towardsCamera.x), Float(towardsCamera.y), Float(towardsCamera.z))
        let weights = NormalLossWeights(weight: 0.5, alphaMask: 0.3, minimumPixels: 10)
        let same = NormalLoss.evaluateDouble(draw: draw, alpha: alpha,
                                             normals: [SIMD3<Float>](repeating: prior, count: width * height),
                                             width: width, height: height, intrinsics: testIntrinsics, weights: weights)
        #expect(!same.degenerate)
        #expect(same.validPixels == (width - 1) * (height - 1))
        #expect(abs(same.loss) < 1e-6)
        #expect(abs(same.meanCosine - 1) < 1e-6)
        let opposite = NormalLoss.evaluateDouble(draw: draw, alpha: alpha,
                                                 normals: [SIMD3<Float>](repeating: -prior, count: width * height),
                                                 width: width, height: height, intrinsics: testIntrinsics,
                                                 weights: weights)
        #expect(abs(opposite.loss - 1.0) < 1e-6)
    }

    @Test func maskAndMissingPriorExcludeStencils() {
        let width: Int = 6
        let height: Int = 5
        var draw: [Double] = [Double](repeating: 2, count: width * height)
        var alpha: [Double] = [Double](repeating: 1, count: width * height)
        var normals: [SIMD3<Float>] = [SIMD3<Float>](repeating: SIMD3<Float>(0, 0, -1), count: width * height)
        // Pixel (2, 2) below the opacity mask removes the stencils at (2, 2), (1, 2) and (2, 1).
        alpha[2 * width + 2] = 0.2
        draw[2 * width + 2] = 0.4
        // No prior at (0, 0).
        normals[0] = .zero
        let result = NormalLoss.evaluateDouble(draw: draw, alpha: alpha, normals: normals, width: width, height: height,
                                               intrinsics: testIntrinsics,
                                               weights: NormalLossWeights(weight: 1, alphaMask: 0.3, minimumPixels: 1))
        #expect(result.validPixels == (width - 1) * (height - 1) - 4)
        #expect(result.gradient[2 * width + 2] == .zero)
        let few = NormalLoss.evaluateDouble(draw: draw, alpha: alpha, normals: normals, width: width, height: height,
                                            intrinsics: testIntrinsics,
                                            weights: NormalLossWeights(weight: 1, alphaMask: 0.3, minimumPixels: 100))
        #expect(few.degenerate)
        #expect(few.loss == 0)
        #expect(few.gradient.allSatisfy { $0 == .zero })
    }

    @Test func gradientMatchesFiniteDifferences() {
        let width: Int = 9
        let height: Int = 7
        var rng = NormalTestRNG(state: 77)
        var draw: [Double] = []
        var alpha: [Double] = []
        var normals: [SIMD3<Float>] = []
        for _ in 0..<(width * height) {
            // Opacity in [0.5, 1): the 0.3 mask stays fixed under the ±h perturbations.
            let a: Double = 0.5 + 0.5 * Double.random(in: 0..<1, using: &rng)
            let d: Double = 2 + Double.random(in: 0..<1, using: &rng)
            alpha.append(Double(Float(a)))
            draw.append(Double(Float(a * d)))
            let raw: SIMD3<Double> = SIMD3<Double>(Double.random(in: -0.5..<0.5, using: &rng),
                                                   Double.random(in: -0.5..<0.5, using: &rng), -1)
            let m: SIMD3<Double> = unit(raw)
            normals.append(SIMD3<Float>(Float(m.x), Float(m.y), Float(m.z)))
        }
        let weights = NormalLossWeights(weight: 0.05, alphaMask: 0.3, minimumPixels: 8)
        let analytic = NormalLoss.evaluateDouble(draw: draw, alpha: alpha, normals: normals, width: width,
                                                 height: height, intrinsics: testIntrinsics, weights: weights)
        #expect(!analytic.degenerate)
        let h: Double = 1e-6
        var largest: Double = 0
        for p in 0..<draw.count {
            var plus: [Double] = draw
            var minus: [Double] = draw
            plus[p] += h
            minus[p] -= h
            let lossPlus: Double = NormalLoss.evaluateDouble(draw: plus, alpha: alpha, normals: normals, width: width,
                                                             height: height, intrinsics: testIntrinsics,
                                                             weights: weights).loss
            let lossMinus: Double = NormalLoss.evaluateDouble(draw: minus, alpha: alpha, normals: normals, width: width,
                                                              height: height, intrinsics: testIntrinsics,
                                                              weights: weights).loss
            let fdDraw: Double = (lossPlus - lossMinus) / (2 * h)

            var aPlus: [Double] = alpha
            var aMinus: [Double] = alpha
            aPlus[p] += h
            aMinus[p] -= h
            let lossAPlus: Double = NormalLoss.evaluateDouble(draw: draw, alpha: aPlus, normals: normals, width: width,
                                                              height: height, intrinsics: testIntrinsics,
                                                              weights: weights).loss
            let lossAMinus: Double = NormalLoss.evaluateDouble(draw: draw, alpha: aMinus, normals: normals,
                                                               width: width, height: height,
                                                               intrinsics: testIntrinsics, weights: weights).loss
            let fdAlpha: Double = (lossAPlus - lossAMinus) / (2 * h)

            let gotDraw: Double = Double(analytic.gradient[p].x)
            let gotAlpha: Double = Double(analytic.gradient[p].y)
            largest = max(largest, abs(fdDraw))
            #expect(abs(gotDraw - fdDraw) <= 1e-4 * abs(fdDraw) + 1e-8, "pixel \(p) draw: \(gotDraw) vs \(fdDraw)")
            #expect(abs(gotAlpha - fdAlpha) <= 1e-4 * abs(fdAlpha) + 1e-8, "pixel \(p) alpha: \(gotAlpha) vs \(fdAlpha)")
        }
        // The gradient is not trivially zero.
        #expect(largest > 1e-4)
    }

    @Test func normalPriorSettingsAreOffByDefaultAndDecode() throws {
        #expect(!TrainingConfiguration().normalPrior.isEnabled)
        let decoded = try JSONDecoder().decode(TrainingConfiguration.self, from: Data("{\"iterations\": 10}".utf8))
        #expect(decoded.normalPrior == NormalPriorSettings())
        let stored = try JSONDecoder().decode(NormalPriorSettings.self, from: Data("{\"isEnabled\": true}".utf8))
        #expect(stored.isEnabled)
        #expect(stored.weight == NormalPriorSettings().weight)
        var configuration = TrainingConfiguration()
        configuration.normalPrior.isEnabled = true
        configuration.normalPrior.weight = 0.01
        let roundTrip = try JSONDecoder().decode(TrainingConfiguration.self, from: try JSONEncoder().encode(configuration))
        #expect(roundTrip == configuration)
    }
}
