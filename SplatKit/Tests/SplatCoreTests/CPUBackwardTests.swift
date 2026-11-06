import Foundation
import Testing
@testable import SplatCore

// MARK: - Deterministic scene construction

struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

private func rotationAbout(x ax: Float, y ay: Float) -> Matrix3 {
    let cx = cos(ax), sx = sin(ax), cy = cos(ay), sy = sin(ay)
    let rx = Matrix3(c0: SIMD3(1, 0, 0), c1: SIMD3(0, cx, sx), c2: SIMD3(0, -sx, cx))
    let ry = Matrix3(c0: SIMD3(cy, 0, -sy), c1: SIMD3(0, 1, 0), c2: SIMD3(sy, 0, cy))
    return ry * rx
}

private func randomCloud(count: Int, shDegree: Int, seed: UInt64) -> GaussianCloud {
    var rng = SplitMix64(state: seed)
    let k = GaussianCloud.coefficientCount(degree: shDegree)
    var positions: [Float] = [], logScales: [Float] = [], rotations: [Float] = []
    var logits: [Float] = [], sh: [Float] = []
    for _ in 0..<count {
        positions += [Float.random(in: -1.3...1.3, using: &rng),
                      Float.random(in: -1.0...1.0, using: &rng),
                      Float.random(in: -0.8...1.2, using: &rng)]
        for _ in 0..<3 { logScales.append(Float.random(in: -2.6 ... -1.1, using: &rng)) }
        // Deliberately not unit length: the gradient is w.r.t. the raw quaternion.
        let len = Float.random(in: 0.7...1.5, using: &rng)
        var q = (0..<4).map { _ in Float.random(in: -1...1, using: &rng) }
        let n = q.map { $0 * $0 }.reduce(0, +).squareRoot()
        q = q.map { $0 / n * len }
        rotations += q
        logits.append(Float.random(in: -1.5...2.0, using: &rng))
        for kk in 0..<k {
            for _ in 0..<3 {
                sh.append(Float.random(in: -0.6...0.6, using: &rng) * (kk == 0 ? 2 : 1))
            }
        }
    }
    return try! GaussianCloud(shDegree: shDegree, positions: positions, logScales: logScales,
                              rotations: rotations, opacityLogits: logits, sh: sh)
}

struct Scene {
    var cloud: GaussianCloud
    var camera: Camera
    var settings: RasterSettings
    var weights: [SIMD3<Float>]
}

private func randomWeights(count: Int, seed: UInt64) -> [SIMD3<Float>] {
    var rng = SplitMix64(state: seed)
    return (0..<count).map { _ in
        SIMD3(Float.random(in: -1...1, using: &rng),
              Float.random(in: -1...1, using: &rng),
              Float.random(in: -1...1, using: &rng))
    }
}

func translatedScene() -> Scene {
    let camera = Camera(width: 40, height: 28, fx: 40, fy: 40, cx: 20, cy: 14,
                        rotation: .identity, translation: SIMD3(0.2, -0.1, 4))
    return Scene(cloud: randomCloud(count: 30, shDegree: 3, seed: 11), camera: camera,
                 settings: .init(), weights: randomWeights(count: 40 * 28, seed: 5))
}

func rotatedScene() -> Scene {
    let rot = rotationAbout(x: 20 * .pi / 180, y: 20 * .pi / 180)
    let t = SIMD3<Float>(0.3, -0.2, 4)
    let camera = Camera(width: 40, height: 28, fx: 40, fy: 40, cx: 20, cy: 14,
                        rotation: rot, translation: t)
    return Scene(cloud: randomCloud(count: 30, shDegree: 3, seed: 23), camera: camera,
                 settings: .init(), weights: randomWeights(count: 40 * 28, seed: 6))
}

/// Opaque Gaussians stacked along the optical axis plus a few random ones, so that
/// the 0.999 alpha clamp and early termination both occur.
func stackedScene() -> Scene {
    let base = randomCloud(count: 30, shDegree: 1, seed: 31)
    var positions = base.positions, logits = base.opacityLogits
    var logScales = base.logScales
    for i in 0..<6 {
        positions[3 * i] = 0.05 * Float(i)
        positions[3 * i + 1] = 0.03 * Float(i)
        positions[3 * i + 2] = 0.2 * Float(i) - 0.5
        logits[i] = 9
        logScales[3 * i] = -1.2; logScales[3 * i + 1] = -1.2; logScales[3 * i + 2] = -1.2
    }
    let cloud = try! GaussianCloud(shDegree: 1, positions: positions, logScales: logScales,
                                   rotations: base.rotations, opacityLogits: logits, sh: base.sh)
    let camera = Camera(width: 32, height: 32, fx: 40, fy: 40, cx: 16.5, cy: 16.5,
                        rotation: .identity, translation: SIMD3(0, 0, 4))
    return Scene(cloud: cloud, camera: camera, settings: .init(),
                 weights: randomWeights(count: 32 * 32, seed: 8))
}

/// Large Gaussians whose centres lie beyond 1.3× the field of view, so the Jacobian clamp of
/// tx/ty is active.
private func clampedJacobianScene() -> Scene {
    let base = randomCloud(count: 20, shDegree: 2, seed: 41)
    var positions = base.positions, logScales = base.logScales, logits = base.opacityLogits
    var rng = SplitMix64(state: 3)
    for i in 0..<10 {
        // Camera-space x or y far outside the limits (|x/z| > 0.65 or |y/z| > 0.455).
        let side: Float = i % 2 == 0 ? 1 : -1
        if i % 4 < 2 {
            positions[3 * i] = side * Float.random(in: 3.6...4.4, using: &rng)
            positions[3 * i + 1] = Float.random(in: -1...1, using: &rng)
        } else {
            positions[3 * i] = Float.random(in: -1...1, using: &rng)
            positions[3 * i + 1] = side * Float.random(in: 2.4...3.0, using: &rng)
        }
        positions[3 * i + 2] = Float.random(in: -0.3...0.3, using: &rng)
        for j in 0..<3 { logScales[3 * i + j] = Float.random(in: -0.3...0.1, using: &rng) }
        logits[i] = Float.random(in: 0...1.5, using: &rng)
    }
    let cloud = try! GaussianCloud(shDegree: 2, positions: positions, logScales: logScales,
                                   rotations: base.rotations, opacityLogits: logits, sh: base.sh)
    let camera = Camera(width: 40, height: 28, fx: 40, fy: 40, cx: 20, cy: 14,
                        rotation: .identity, translation: SIMD3(0, 0, 4))
    return Scene(cloud: cloud, camera: camera, settings: .init(),
                 weights: randomWeights(count: 40 * 28, seed: 9))
}

// MARK: - Parameter access and finite differences

enum Group: String, CaseIterable {
    case positions, logScales, rotations, opacityLogits, sh
}

func values(_ c: GaussianCloud, _ g: Group) -> [Float] {
    switch g {
    case .positions: c.positions
    case .logScales: c.logScales
    case .rotations: c.rotations
    case .opacityLogits: c.opacityLogits
    case .sh: c.sh
    }
}

func values(_ gr: GaussianGradients, _ g: Group) -> [Float] {
    switch g {
    case .positions: gr.positions
    case .logScales: gr.logScales
    case .rotations: gr.rotations
    case .opacityLogits: gr.opacityLogits
    case .sh: gr.sh
    }
}

func perturbed(_ c: GaussianCloud, _ g: Group, _ index: Int, _ delta: Float) -> GaussianCloud {
    var p = c.positions, s = c.logScales, r = c.rotations, o = c.opacityLogits, h = c.sh
    switch g {
    case .positions: p[index] += delta
    case .logScales: s[index] += delta
    case .rotations: r[index] += delta
    case .opacityLogits: o[index] += delta
    case .sh: h[index] += delta
    }
    return try! GaussianCloud(shDegree: c.shDegree, positions: p, logScales: s, rotations: r,
                              opacityLogits: o, sh: h)
}

private func loss(_ scene: Scene, _ cloud: GaussianCloud) -> Double {
    let img = CPURasterizer.render(cloud, camera: scene.camera, settings: scene.settings)
    var total = 0.0
    for (p, w) in zip(img.pixels, scene.weights) {
        total += Double(p.x * w.x) + Double(p.y * w.y) + Double(p.z * w.z)
    }
    return total
}

/// Per-pixel lists of blended Gaussian indices; equal signatures mean no gating decision
/// (culling, tile rect, σ < 0, α < 1/255, termination) changed.
private func signature(_ scene: Scene, _ cloud: GaussianCloud) -> [[Int]] {
    let sorted = CPURasterizer.projectedAndSorted(cloud, camera: scene.camera, settings: scene.settings)
    return CPURasterizer.blendedSamples(sorted, camera: scene.camera, settings: scene.settings)
        .map { $0.samples.map { sorted[$0.entry].index } }
}

private struct FDResult {
    var analytic: Float
    var numeric: Double
    var skipped: Bool
}

private func finiteDifference(_ scene: Scene, _ g: Group, _ index: Int, analytic: Float) -> FDResult {
    let h: Float = g == .sh ? 1e-2 : 1e-3
    let plus = perturbed(scene.cloud, g, index, h)
    let minus = perturbed(scene.cloud, g, index, -h)
    let base = signature(scene, scene.cloud)
    let flipped = signature(scene, plus) != base || signature(scene, minus) != base
    let numeric = (loss(scene, plus) - loss(scene, minus)) / (2 * Double(h))
    return FDResult(analytic: analytic, numeric: numeric, skipped: flipped)
}

private func relativeError(_ r: FDResult) -> Double {
    let a = Double(r.analytic)
    return abs(a - r.numeric) / max(abs(a), abs(r.numeric))
}

private func analyticGradients(_ scene: Scene) -> GaussianGradients {
    CPURasterizer.backward(scene.cloud, camera: scene.camera, settings: scene.settings,
                           imageGradient: scene.weights)
}

// MARK: - Tests

/// Minimum |analytic| for a parameter to be finite-difference checked. Float rendering noise
/// in the loss is about 5e-4 after dividing by 2h, so smaller gradients cannot be verified
/// to 5 % relative accuracy.
private let fdThreshold: Float = 1e-2

@Test func sampleReplayMatchesRender() {
    for scene in [translatedScene(), rotatedScene(), stackedScene()] {
        let sorted = CPURasterizer.projectedAndSorted(scene.cloud, camera: scene.camera,
                                                      settings: scene.settings)
        let pixels = CPURasterizer.blendedSamples(sorted, camera: scene.camera, settings: scene.settings)
        let image = CPURasterizer.render(scene.cloud, camera: scene.camera, settings: scene.settings)
        for (p, ps) in zip(image.pixels, pixels) {
            var rgb = SIMD3<Float>.zero
            for s in ps.samples { rgb += sorted[s.entry].gaussian.color * s.alpha * s.transmittance }
            rgb += ps.finalTransmittance * scene.settings.background
            #expect(approxEqual(rgb, p, tol: 1e-6))
        }
    }
}

@Test func finiteDifferencesMatch() {
    for (name, scene) in [("translated", translatedScene()), ("rotated", rotatedScene())] {
        let grads = analyticGradients(scene)
        // All parameters as (group, index) with |analytic| > fdThreshold, chosen with a seeded shuffle.
        var candidates: [(Group, Int)] = []
        for g in Group.allCases {
            for (i, v) in values(grads, g).enumerated() where abs(v) > fdThreshold { candidates.append((g, i)) }
        }
        var rng = SplitMix64(state: 7)
        candidates.shuffle(using: &rng)
        var checked = 0, skipped = 0
        var maxErr = 0.0
        for (g, i) in candidates {
            if checked == 40 { break }
            let r = finiteDifference(scene, g, i, analytic: values(grads, g)[i])
            if r.skipped { skipped += 1; continue }
            checked += 1
            let e = relativeError(r)
            maxErr = max(maxErr, e)
            #expect(e <= 5e-2, "\(name) \(g) [\(i)] analytic \(r.analytic) numeric \(r.numeric)")
        }
        print("FD all-groups \(name): checked \(checked), skipped \(skipped), max rel err \(maxErr)")
        #expect(checked == 40)
    }
}

@Test func finiteDifferencesPerGroup() {
    for (name, scene) in [("translated", translatedScene()), ("rotated", rotatedScene()),
                          ("stacked", stackedScene())] {
        let grads = analyticGradients(scene)
        for g in Group.allCases {
            var candidates: [Int] = []
            for (i, v) in values(grads, g).enumerated() where abs(v) > fdThreshold { candidates.append(i) }
            var rng = SplitMix64(state: 99)
            candidates.shuffle(using: &rng)
            var checked = 0, skipped = 0
            var maxErr = 0.0
            for i in candidates {
                if checked == 8 { break }
                let r = finiteDifference(scene, g, i, analytic: values(grads, g)[i])
                if r.skipped { skipped += 1; continue }
                checked += 1
                let e = relativeError(r)
                maxErr = max(maxErr, e)
                #expect(e <= 5e-2, "\(name) \(g) [\(i)] analytic \(r.analytic) numeric \(r.numeric)")
            }
            print("FD \(name) \(g.rawValue): checked \(checked), skipped \(skipped), max rel err \(maxErr)")
            #expect(checked >= 8, "\(name) \(g): only \(checked) usable parameters")
        }
    }
}

@Test func finiteDifferencesWithClampedJacobian() {
    let scene = clampedJacobianScene()
    let cam = scene.camera
    var clampedCount = 0
    for i in 0..<10 {
        let p = SIMD3<Float>(scene.cloud.positions[3 * i], scene.cloud.positions[3 * i + 1],
                             scene.cloud.positions[3 * i + 2])
        let pc = cam.rotation * p + cam.translation
        // (w − cx)/fx + 0.3·tanFovX = 0.65, (h − cy)/fy + 0.3·tanFovY = 0.455
        if abs(pc.x / pc.z) > 0.66 || abs(pc.y / pc.z) > 0.46 { clampedCount += 1 }
        #expect(Projection.project(scene.cloud, index: i, camera: cam, settings: scene.settings) != nil)
    }
    #expect(clampedCount == 10)

    let grads = analyticGradients(scene)
    for g in Group.allCases {
        func gaussianIndex(_ i: Int) -> Int {
            switch g {
            case .positions, .logScales: i / 3
            case .rotations: i / 4
            case .opacityLogits: i
            case .sh: i / (9 * 3)
            }
        }
        var candidates = values(grads, g).enumerated()
            .filter { abs($0.element) > fdThreshold && gaussianIndex($0.offset) < 10 }
            .map(\.offset)
        var rng = SplitMix64(state: 5)
        candidates.shuffle(using: &rng)
        var checked = 0, skipped = 0
        var maxErr = 0.0
        for i in candidates {
            if checked == 8 { break }
            let r = finiteDifference(scene, g, i, analytic: values(grads, g)[i])
            if r.skipped { skipped += 1; continue }
            checked += 1
            let e = relativeError(r)
            maxErr = max(maxErr, e)
            #expect(e <= 5e-2, "clamped \(g) [\(i)] analytic \(r.analytic) numeric \(r.numeric)")
        }
        print("FD clamped-jacobian \(g.rawValue): checked \(checked), skipped \(skipped), max rel err \(maxErr)")
        #expect(checked >= 8, "clamped \(g): only \(checked) usable parameters")
    }
}

@Test func stackedSceneExercisesClampAndTermination() {
    let scene = stackedScene()
    let sorted = CPURasterizer.projectedAndSorted(scene.cloud, camera: scene.camera, settings: scene.settings)
    let pixels = CPURasterizer.blendedSamples(sorted, camera: scene.camera, settings: scene.settings)
    #expect(pixels.contains { $0.samples.contains { $0.clamped } })
    #expect(pixels.contains { $0.samples.count >= 1 && $0.finalTransmittance < 0.01 })
}

@Test func clampedSamplesHaveZeroOpacityGradient() throws {
    let cloud = try GaussianCloud(shDegree: 0, positions: [0, 0, 0],
                                  logScales: [-2.3, -2.3, -2.3], rotations: [1, 0, 0, 0],
                                  opacityLogits: [10], sh: [0.3, -0.2, 0.5])
    // mean2D lands exactly on the centre of pixel (32, 32).
    let camera = Camera(width: 64, height: 64, fx: 100, fy: 100, cx: 32.5, cy: 32.5,
                        rotation: .identity, translation: SIMD3(0, 0, 5))
    var gradient = [SIMD3<Float>](repeating: .zero, count: 64 * 64)
    gradient[32 * 64 + 32] = SIMD3(1, -0.5, 0.25)
    let g = CPURasterizer.backward(cloud, camera: camera, settings: .init(), imageGradient: gradient)
    #expect(g.opacityLogits == [0])
    #expect(g.positions == [0, 0, 0])
    #expect(g.logScales == [0, 0, 0])
    #expect(g.rotations == [0, 0, 0, 0])
    #expect(g.sh.contains { $0 != 0 })
}

@Test func higherSHBandsGetZeroGradientWhenLimited() {
    var scene = translatedScene()
    scene.settings.shDegreeLimit = 1
    let g = analyticGradients(scene)
    let k = 16
    var lowNonZero = false
    for i in 0..<scene.cloud.count {
        for kk in 0..<k {
            for c in 0..<3 {
                let v = g.sh[(i * k + kk) * 3 + c]
                if kk >= 4 { #expect(v == 0) } else if v != 0 { lowNonZero = true }
            }
        }
    }
    #expect(lowNonZero)
}

@Test func zeroImageGradientGivesZeroGradients() {
    let scene = translatedScene()
    let g = CPURasterizer.backward(scene.cloud, camera: scene.camera, settings: scene.settings,
                                   imageGradient: [SIMD3<Float>](repeating: .zero, count: 40 * 28))
    #expect(g == GaussianGradients(zeroLike: scene.cloud))
}

// MARK: - Golden comparison (PyTorch autograd)

@Test(arguments: ["single", "multi", "rotated", "occluded"])
func backwardMatchesGolden(name: String) throws {
    let golden = try GoldenCase.load(name)
    let cloud = try golden.cloud()
    let gradients = CPURasterizer.backward(
        cloud, camera: golden.camera(), settings: golden.settings(),
        imageGradient: golden.imageGradientPixels)

    func check(_ label: String, _ actual: [Float], _ expected: [Float]) {
        #expect(actual.count == expected.count, "\(name).\(label): length")
        guard actual.count == expected.count else { return }
        var worst: Float = 0
        var worstIndex = 0
        for i in 0..<actual.count {
            let allowed = 1e-4 + 1e-3 * abs(expected[i])
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
