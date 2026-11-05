import Foundation
import Testing
@testable import SplatCore

// MARK: - Deterministic test images

private struct LossTestRNG: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

private func randomImage(width: Int, height: Int, seed: UInt64) -> RGBImage {
    var rng = LossTestRNG(state: seed)
    var pixels: [SIMD3<Float>] = []
    pixels.reserveCapacity(width * height)
    for _ in 0..<(width * height) {
        let r = Float.random(in: 0...1, using: &rng)
        let g = Float.random(in: 0...1, using: &rng)
        let b = Float.random(in: 0...1, using: &rng)
        pixels.append(SIMD3<Float>(r, g, b))
    }
    return RGBImage(width: width, height: height, pixels: pixels)
}

private func signOf(_ v: Float) -> Float {
    if v > 0 { return 1 }
    if v < 0 { return -1 }
    return 0
}

// MARK: - ImageLoss

@Test func identicalImagesHaveZeroLossAndUnitSSIM() {
    let image = randomImage(width: 13, height: 9, seed: 1)
    let result = ImageLoss.evaluate(rendered: image, target: image, lambda: 0.2)
    #expect(result.loss < 1e-7)
    #expect(result.ssim > 1 - 1e-6)
}

@Test func windowIsNormalizedAndSymmetric() {
    let w = ImageLoss.window
    #expect(w.count == 11)
    #expect(abs(w.reduce(0, +) - 1) < 1e-6)
    for i in 0..<w.count {
        #expect(w[i] == w[w.count - 1 - i])
    }
}

@Test func l1OnlyGradientIsSign() {
    var rendered = randomImage(width: 7, height: 5, seed: 2)
    let target = randomImage(width: 7, height: 5, seed: 4)
    // Exact ties exercise sign(0) = 0.
    rendered.pixels[0] = target.pixels[0]
    rendered.pixels[5] = target.pixels[5]
    let result = ImageLoss.evaluate(rendered: rendered, target: target, lambda: 0)
    let n = Double(7 * 5 * 3)
    #expect(result.gradient.count == rendered.pixels.count)
    for i in 0..<rendered.pixels.count {
        let d = rendered.pixels[i] - target.pixels[i]
        let expected = SIMD3<Float>(
            Float(Double(signOf(d.x)) / n),
            Float(Double(signOf(d.y)) / n),
            Float(Double(signOf(d.z)) / n)
        )
        #expect(approxEqual(result.gradient[i], expected, tol: 1e-8))
    }
    #expect(result.gradient[0] == .zero)
    #expect(result.gradient[5] == .zero)
}

@Test func lossGradientMatchesFiniteDifferences() {
    let rendered = randomImage(width: 12, height: 10, seed: 3)
    let target = randomImage(width: 12, height: 10, seed: 5)
    let lambda: Float = 0.2
    let base = ImageLoss.evaluate(rendered: rendered, target: target, lambda: lambda)
    var rng = LossTestRNG(state: 11)
    let h: Float = 1e-3
    let pixelCount = 12 * 10
    for _ in 0..<40 {
        let pixel = Int.random(in: 0..<pixelCount, using: &rng)
        let channel = Int.random(in: 0..<3, using: &rng)
        let x = rendered.pixels[pixel][channel]
        let xPlus = x + h
        let xMinus = x - h
        var plusImage = rendered
        var minusImage = rendered
        plusImage.pixels[pixel][channel] = xPlus
        minusImage.pixels[pixel][channel] = xMinus
        // Use the exact Float steps actually applied.
        let step = Double(xPlus) - Double(xMinus)
        let lossPlus = ImageLoss.evaluate(rendered: plusImage, target: target, lambda: lambda).loss
        let lossMinus = ImageLoss.evaluate(rendered: minusImage, target: target, lambda: lambda).loss
        let fd = (lossPlus - lossMinus) / step
        let analytic = Double(base.gradient[pixel][channel])
        #expect(abs(fd - analytic) <= 2e-2 * abs(analytic) + 1e-6)
    }
}

@Test func ssimIsTranslationSensitive() {
    let width = 16
    let height = 16
    let image = randomImage(width: width, height: height, seed: 7)
    var shifted = image
    for y in 0..<height {
        for x in 0..<width {
            shifted.pixels[y * width + x] = image.pixels[y * width + (x + 2) % width]
        }
    }
    let result = ImageLoss.evaluate(rendered: shifted, target: image, lambda: 0.2)
    #expect(result.ssim < 0.9)
}

// MARK: - ImageMetrics

@Test func psnrOfKnownMSE() {
    let target = RGBImage(width: 4, height: 3,
                          pixels: Array(repeating: SIMD3<Float>(0.5, 0.5, 0.5), count: 12))
    let offset = RGBImage(width: 4, height: 3,
                          pixels: Array(repeating: SIMD3<Float>(0.6, 0.6, 0.6), count: 12))
    #expect(abs(ImageMetrics.mse(offset, target) - 0.01) < 1e-6)
    #expect(abs(ImageMetrics.psnr(offset, target) - 20) < 1e-3)
    #expect(ImageMetrics.psnr(target, target) == .infinity)
}
