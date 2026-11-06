import Foundation
import Testing
@testable import SplatCore

// Phase 4 Task 4: rendered depth on the CPU reference rasterizer.
// Draw = Σ αᵢ Tᵢ zᵢ and A = 1 − T_final per pixel; the backward input is (∂L/∂Draw, ∂L/∂A).
// Shared helpers (SplitMix64, Group, values, perturbed, Scene, translatedScene, rotatedScene,
// stackedScene) come from CPUBackwardTests.swift.

// MARK: - Helpers

/// Bit patterns of every pixel channel, row-major.
private func bitPatterns(_ image: RGBImage) -> [UInt32] {
    var bits: [UInt32] = []
    bits.reserveCapacity(3 * image.pixels.count)
    for p in image.pixels {
        bits.append(p.x.bitPattern)
        bits.append(p.y.bitPattern)
        bits.append(p.z.bitPattern)
    }
    return bits
}

/// Bit patterns of every gradient array, in the order positions, logScales, rotations,
/// opacityLogits, sh.
private func gradientBits(_ g: GaussianGradients) -> [UInt32] {
    var bits: [UInt32] = []
    for v in g.positions { bits.append(v.bitPattern) }
    for v in g.logScales { bits.append(v.bitPattern) }
    for v in g.rotations { bits.append(v.bitPattern) }
    for v in g.opacityLogits { bits.append(v.bitPattern) }
    for v in g.sh { bits.append(v.bitPattern) }
    return bits
}

/// 64-bit FNV-1a over 32-bit words.
private func fnv1a(_ words: [UInt32]) -> UInt64 {
    var hash: UInt64 = 0xcbf2_9ce4_8422_2325
    for w in words {
        hash ^= UInt64(w)
        hash = hash &* 0x0000_0100_0000_01b3
    }
    return hash
}

/// Interleaved (x, y) pairs from a flat row-major W·H·2 array.
private func pairs(_ flat: [Float]) -> [SIMD2<Float>] {
    var out: [SIMD2<Float>] = []
    out.reserveCapacity(flat.count / 2)
    var i = 0
    while i + 1 < flat.count {
        out.append(SIMD2<Float>(flat[i], flat[i + 1]))
        i += 2
    }
    return out
}

/// Seeded random (∂L/∂Draw, ∂L/∂A) per pixel.
private func depthGradientField(count: Int, seed: UInt64) -> [SIMD2<Float>] {
    var rng = SplitMix64(state: seed)
    var out: [SIMD2<Float>] = []
    out.reserveCapacity(count)
    for _ in 0..<count {
        let a: Float = Float.random(in: -1...1, using: &rng)
        let b: Float = Float.random(in: -1...1, using: &rng)
        out.append(SIMD2<Float>(a, b))
    }
    return out
}

/// Gating decisions (culling, tile rect, σ, α, termination), as in `CPUBackwardTests`.
private func gatingSignature(_ cloud: GaussianCloud, camera: Camera, settings: RasterSettings) -> [[Int]] {
    let sorted = CPURasterizer.projectedAndSorted(cloud, camera: camera, settings: settings)
    return CPURasterizer.blendedSamples(sorted, camera: camera, settings: settings)
        .map { pixel in pixel.samples.map { sorted[$0.entry].index } }
}

/// Σ gD·Draw + gA·A accumulated in Double.
private func depthLoss(_ cloud: GaussianCloud, camera: Camera, settings: RasterSettings,
                       depthGradient: [SIMD2<Float>]) -> Double {
    let (_, depth) = CPURasterizer.renderWithDepth(cloud, camera: camera, settings: settings)
    var total = 0.0
    for i in 0..<depthGradient.count {
        total += Double(depthGradient[i].x) * Double(depth.raw[i])
        total += Double(depthGradient[i].y) * Double(depth.alpha[i])
    }
    return total
}

// MARK: - Forward

@Test(arguments: ["single", "multi", "rotated", "occluded", "depth_single", "depth_occluded"])
func renderWithDepthImageMatchesRender(name: String) throws {
    let golden = try GoldenCase.load(name)
    let cloud = try golden.cloud()
    let camera = golden.camera()
    let settings = golden.settings()

    let plain = CPURasterizer.render(cloud, camera: camera, settings: settings)
    let (image, depth) = CPURasterizer.renderWithDepth(cloud, camera: camera, settings: settings)

    #expect(bitPatterns(image) == bitPatterns(plain), "image must be bit-identical for \(name)")
    #expect(depth.raw.count == camera.width * camera.height)
    #expect(depth.alpha.count == camera.width * camera.height)
}

@Test func singleOpaqueGaussianDepthIsItsZ() {
    // One large opaque Gaussian centred at camera-space z = 3, identity camera at the origin.
    let cloud = try! GaussianCloud(shDegree: 0, positions: [0, 0, 3], logScales: [0, 0, 0],
                                   rotations: [1, 0, 0, 0], opacityLogits: [8],
                                   sh: [0.3, 0.3, 0.3])
    let camera = Camera(width: 64, height: 64, fx: 40, fy: 40, cx: 32, cy: 32,
                        rotation: .identity, translation: .zero)
    let (_, depth) = CPURasterizer.renderWithDepth(cloud, camera: camera, settings: .init())

    let centre = 32 * 64 + 32
    let normalized = depth.normalized(minimumAlpha: 1e-3)
    let value: Float = normalized[centre]
    #expect(abs(value - 3) <= 1e-4, "normalized depth at centre \(value)")
    let alpha: Float = depth.alpha[centre]
    // Half a pixel off the mean the falloff keeps α just below the 0.999 clamp.
    #expect(alpha > 0.99 && alpha <= 0.999, "alpha at centre \(alpha)")
}

@Test(arguments: ["depth_single", "depth_occluded"])
func depthMatchesGoldens(name: String) throws {
    let golden = try GoldenCase.load(name)
    let cloud = try golden.cloud()
    let (_, depth) = CPURasterizer.renderWithDepth(cloud, camera: golden.camera(),
                                                   settings: golden.settings())
    let expectedRaw = try #require(golden.renderedDepth)
    let expectedAlpha = try #require(golden.renderedAlpha)
    #expect(depth.raw.count == expectedRaw.count)
    #expect(depth.alpha.count == expectedAlpha.count)

    var rawWorst: Float = 0
    var alphaWorst: Float = 0
    for i in 0..<min(depth.raw.count, expectedRaw.count) {
        rawWorst = max(rawWorst, abs(depth.raw[i] - expectedRaw[i]))
        alphaWorst = max(alphaWorst, abs(depth.alpha[i] - expectedAlpha[i]))
    }
    print("DEPTH GOLDEN \(name): max |raw diff| \(rawWorst), max |alpha diff| \(alphaWorst)")
    #expect(rawWorst <= 1e-5, "raw max abs diff \(rawWorst) for \(name)")
    #expect(alphaWorst <= 1e-5, "alpha max abs diff \(alphaWorst) for \(name)")
}

// MARK: - Backward

@Test(arguments: ["depth_single", "depth_occluded"])
func depthBackwardMatchesGoldens(name: String) throws {
    let golden = try GoldenCase.load(name)
    let cloud = try golden.cloud()
    let depthGradient = pairs(try #require(golden.depthGradient))
    let gradients = CPURasterizer.backwardWithScreenSpace(
        cloud, camera: golden.camera(), settings: golden.settings(),
        imageGradient: golden.imageGradientPixels, depthGradient: depthGradient).gradients

    func check(_ label: String, _ actual: [Float], _ expected: [Float]) {
        #expect(actual.count == expected.count, "\(name).\(label): length")
        guard actual.count == expected.count else { return }
        var worst: Float = 0
        var worstIndex = 0
        for i in 0..<actual.count {
            // Allowed: abs 1e-6 + rel 1e-4 of the golden value.
            let allowed: Float = 1e-6 + 1e-4 * abs(expected[i])
            let excess = abs(actual[i] - expected[i]) - allowed
            if excess > worst { worst = excess; worstIndex = i }
        }
        #expect(worst <= 0, "\(name).\(label)[\(worstIndex)]: cpu \(actual[worstIndex]) golden \(expected[worstIndex])")
    }
    check("positions", gradients.positions, golden.gradients.positions)
    check("logScales", gradients.logScales, golden.gradients.logScales)
    check("rotations", gradients.rotations, golden.gradients.rotations)
    check("opacityLogits", gradients.opacityLogits, golden.gradients.opacityLogits)
    check("sh", gradients.sh, golden.gradients.sh)
}

/// Central finite differences of the depth loss on the existing random scenes (the golden scene
/// has too few gradients clear of the alpha-cutoff and tile discontinuities to check).
///
/// Tolerance: relative 2e-3 plus an absolute floor of 1e-3. The floor is the Float forward
/// noise of the loss divided by 2h (about 5e-4 measured on these scenes); without it,
/// gradients of order 0.05 cannot be checked at all. Parameters whose gating signature changes
/// under ±h are skipped, as in `CPUBackwardTests`.
@Test func depthBackwardMatchesFiniteDifferences() {
    let scenes: [(String, Scene)] = [("translated", translatedScene()),
                                     ("rotated", rotatedScene()),
                                     ("stacked", stackedScene())]
    let groups: [Group] = [.positions, .logScales, .rotations, .opacityLogits]
    let h: Float = 1e-3
    let threshold: Float = 0.05

    for (name, scene) in scenes {
        let cloud = scene.cloud
        let camera = scene.camera
        let settings = scene.settings
        let pixelCount = camera.width * camera.height
        let depthGradient = depthGradientField(count: pixelCount, seed: 17)
        let zeroImage = [SIMD3<Float>](repeating: .zero, count: pixelCount)

        let analytic = CPURasterizer.backwardWithScreenSpace(
            cloud, camera: camera, settings: settings,
            imageGradient: zeroImage, depthGradient: depthGradient).gradients
        let base = gatingSignature(cloud, camera: camera, settings: settings)

        for group in groups {
            var candidates: [Int] = []
            for (i, v) in values(analytic, group).enumerated() where abs(v) > threshold {
                candidates.append(i)
            }
            var rng = SplitMix64(state: 13)
            candidates.shuffle(using: &rng)

            var checked = 0, skipped = 0
            var worst = 0.0
            for i in candidates {
                if checked == 8 { break }
                let plus = perturbed(cloud, group, i, h)
                let minus = perturbed(cloud, group, i, -h)
                if gatingSignature(plus, camera: camera, settings: settings) != base
                    || gatingSignature(minus, camera: camera, settings: settings) != base {
                    skipped += 1
                    continue
                }
                // Central difference over the exact stored step (Float rounding of x ± h).
                let step: Double = Double(values(plus, group)[i]) - Double(values(minus, group)[i])
                let lossPlus = depthLoss(plus, camera: camera, settings: settings, depthGradient: depthGradient)
                let lossMinus = depthLoss(minus, camera: camera, settings: settings, depthGradient: depthGradient)
                let numeric = (lossPlus - lossMinus) / step
                let a = Double(values(analytic, group)[i])
                let error = abs(a - numeric)
                let allowed = 2e-3 * max(abs(a), abs(numeric)) + 1e-3
                worst = max(worst, error / max(abs(a), abs(numeric)))
                checked += 1
                #expect(error <= allowed, "\(name) \(group.rawValue) [\(i)] analytic \(a) numeric \(numeric)")
            }
            print("FD depth \(name) \(group.rawValue): checked \(checked), skipped \(skipped), max rel err \(worst)")
            #expect(checked >= 8, "\(name) \(group.rawValue): only \(checked) usable parameters")
        }
    }
}

@Test func combinedGradientIsSumOfParts() throws {
    let golden = try GoldenCase.load("depth_occluded")
    let cloud = try golden.cloud()
    let camera = golden.camera()
    let settings = golden.settings()
    let pixelCount = camera.width * camera.height
    let imageGradient = golden.imageGradientPixels
    let depthGradient = depthGradientField(count: pixelCount, seed: 29)
    let zeroImage = [SIMD3<Float>](repeating: .zero, count: pixelCount)

    let both = CPURasterizer.backwardWithScreenSpace(
        cloud, camera: camera, settings: settings,
        imageGradient: imageGradient, depthGradient: depthGradient).gradients
    let imageOnly = CPURasterizer.backwardWithScreenSpace(
        cloud, camera: camera, settings: settings,
        imageGradient: imageGradient, depthGradient: nil).gradients
    let depthOnly = CPURasterizer.backwardWithScreenSpace(
        cloud, camera: camera, settings: settings,
        imageGradient: zeroImage, depthGradient: depthGradient).gradients

    var largestDepthPart: Float = 0
    let all: [Group] = [.positions, .logScales, .rotations, .opacityLogits, .sh]
    for group in all {
        let b = values(both, group)
        let im = values(imageOnly, group)
        let dp = values(depthOnly, group)
        for i in 0..<b.count {
            let sum: Float = im[i] + dp[i]
            let diff: Float = abs(b[i] - sum)
            #expect(diff <= 1e-6, "\(group.rawValue)[\(i)]: combined \(b[i]) parts \(sum)")
            largestDepthPart = max(largestDepthPart, abs(dp[i]))
        }
    }
    #expect(largestDepthPart > 1e-3, "depth-only gradient must be non-trivial")
}

@Test func nilDepthGradientIsBitIdenticalToPhase3() throws {
    // Fingerprints recorded from the Phase 3 code (before Task 4) on the depth_occluded golden
    // scene: the forward image and the backward gradients for the golden image gradient.
    let phase3ImageHash: UInt64 = 6_367_145_117_700_868_717
    let phase3GradientHash: UInt64 = 3_997_033_113_602_015_524

    let golden = try GoldenCase.load("depth_occluded")
    let cloud = try golden.cloud()
    let camera = golden.camera()
    let settings = golden.settings()

    let (image, _) = CPURasterizer.renderWithDepth(cloud, camera: camera, settings: settings)
    #expect(fnv1a(bitPatterns(image)) == phase3ImageHash)

    let nilGradients = CPURasterizer.backwardWithScreenSpace(
        cloud, camera: camera, settings: settings,
        imageGradient: golden.imageGradientPixels, depthGradient: nil).gradients
    #expect(fnv1a(gradientBits(nilGradients)) == phase3GradientHash)

    let phase3Gradients = CPURasterizer.backward(
        cloud, camera: camera, settings: settings, imageGradient: golden.imageGradientPixels)
    #expect(phase3Gradients == nilGradients)
}
