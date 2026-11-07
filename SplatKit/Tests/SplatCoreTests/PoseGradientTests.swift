import Foundation
import Testing
@testable import SplatCore

// Finite-difference checks of the camera pose gradient (CPU reference) and of the exponential map, folding and
// Adam helpers. The scenes and the linear loss L = Σ weights · image follow CPUBackwardTests.

/// Pose components in the order (ω.x, ω.y, ω.z, τ.x, τ.y, τ.z).
private func poseVector(_ gradient: PoseGradient) -> [Double] {
    [gradient.rotation.x, gradient.rotation.y, gradient.rotation.z,
     gradient.translation.x, gradient.translation.y, gradient.translation.z]
}

/// `camera` with δ = `delta` folded in (components as in `poseVector`).
private func perturbedCamera(_ camera: Camera, delta: [Double]) -> Camera {
    PoseRefinement.fold(camera,
                        rotation: SIMD3(delta[0], delta[1], delta[2]),
                        translation: SIMD3(delta[3], delta[4], delta[5]))
}

private func linearLoss(_ cloud: GaussianCloud, camera: Camera, settings: RasterSettings,
                        weights: [SIMD3<Float>]) -> Double {
    let image = CPURasterizer.render(cloud, camera: camera, settings: settings)
    var total = 0.0
    for (p, w) in zip(image.pixels, weights) {
        total += Double(p.x * w.x) + Double(p.y * w.y) + Double(p.z * w.z)
    }
    return total
}

/// Per-pixel blended Gaussian indices. Equal signatures mean no gating decision changed under the perturbation.
private func blendSignature(_ cloud: GaussianCloud, camera: Camera, settings: RasterSettings) -> [[Int]] {
    let sorted = CPURasterizer.projectedAndSorted(cloud, camera: camera, settings: settings)
    return CPURasterizer.blendedSamples(sorted, camera: camera, settings: settings)
        .map { $0.samples.map { sorted[$0.entry].index } }
}

/// Compares the analytic pose gradient with central differences of `linearLoss`. Returns the indices (in
/// `poseVector` order) of the components that were checked.
private func checkPoseGradient(_ name: String, _ scene: Scene) -> [Int] {
    let analytic = poseVector(CPURasterizer.poseBackward(scene.cloud, camera: scene.camera,
                                                         settings: scene.settings,
                                                         imageGradient: scene.weights))
    let base = blendSignature(scene.cloud, camera: scene.camera, settings: scene.settings)
    var checked: [Int] = []
    for index in 0..<6 {
        let value = analytic[index]
        // Components with a small gradient cannot be resolved against the Float rendering noise.
        if abs(value) < 5e-2 { continue }
        // A step that flips a gating decision (a sample crossing the 1/255 alpha cutoff) makes the loss discontinuous,
        // so the largest step without a flip is used. Rotations about x and y flip at the larger steps, so the
        // smaller ones are needed for them.
        var numeric: Double?
        var slack = 0.0
        for h in [1e-4, 3e-5, 1e-5, 3e-6] {
            // The rounding noise of the loss is about 5e-3 and scales as 1/h, so after the division by 2h the noise of
            // the derivative is about 5e-3 · (1e-4 / h); the slack is twice that.
            slack = 1e-2 * (1e-4 / h)
            var plus = [Double](repeating: 0, count: 6)
            var minus = plus
            plus[index] = h
            minus[index] = -h
            let cameraPlus = perturbedCamera(scene.camera, delta: plus)
            let cameraMinus = perturbedCamera(scene.camera, delta: minus)
            let flipped = blendSignature(scene.cloud, camera: cameraPlus, settings: scene.settings) != base
                || blendSignature(scene.cloud, camera: cameraMinus, settings: scene.settings) != base
            if flipped { continue }
            numeric = (linearLoss(scene.cloud, camera: cameraPlus, settings: scene.settings, weights: scene.weights)
                - linearLoss(scene.cloud, camera: cameraMinus, settings: scene.settings, weights: scene.weights))
                / (2 * h)
            break
        }
        guard let numeric else { continue }
        checked.append(index)
        let tolerance = 5e-2 * max(abs(value), abs(numeric)) + slack
        #expect(abs(value - numeric) <= tolerance,
                "\(name) component \(index): analytic \(value), numeric \(numeric)")
    }
    return checked
}

/// Gaussians whose centres lie beyond the Jacobian clamp (1.3× the field of view) near the image border, so the
/// clamped tx/ty path feeds the pose gradient through the camera-space depth.
private func clampedScene() -> Scene {
    var rng = SplitMix64(state: 17)
    let count = 12
    var positions: [Float] = [], logScales: [Float] = [], rotations: [Float] = []
    var logits: [Float] = [], sh: [Float] = []
    for i in 0..<count {
        let side: Float = i % 2 == 0 ? 1 : -1
        if i % 4 < 2 {
            positions += [side * Float.random(in: 3.6...4.4, using: &rng),
                          Float.random(in: -1...1, using: &rng),
                          Float.random(in: -0.3...0.3, using: &rng)]
        } else {
            positions += [Float.random(in: -1...1, using: &rng),
                          side * Float.random(in: 2.4...3.0, using: &rng),
                          Float.random(in: -0.3...0.3, using: &rng)]
        }
        for _ in 0..<3 { logScales.append(Float.random(in: -0.3...0.1, using: &rng)) }
        rotations += [Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                      Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng)]
        logits.append(Float.random(in: 0...1.5, using: &rng))
        for k in 0..<4 {
            for _ in 0..<3 { sh.append(Float.random(in: -0.5...0.5, using: &rng) * (k == 0 ? 2 : 1)) }
        }
    }
    let cloud = try! GaussianCloud(shDegree: 1, positions: positions, logScales: logScales,
                                   rotations: rotations, opacityLogits: logits, sh: sh)
    let camera = Camera(width: 40, height: 28, fx: 40, fy: 40, cx: 20, cy: 14,
                        rotation: .identity, translation: SIMD3(0, 0, 4))
    var weightRng = SplitMix64(state: 4)
    let weights = (0..<(40 * 28)).map { _ in
        SIMD3(Float.random(in: -1...1, using: &weightRng), Float.random(in: -1...1, using: &weightRng),
              Float.random(in: -1...1, using: &weightRng))
    }
    return Scene(cloud: cloud, camera: camera, settings: .init(), weights: weights)
}

/// Scenes used by the gradient checks, with the minimum number of checked components each must yield.
private func gradientScenes() -> [(name: String, scene: Scene, minimum: Int)] {
    [("translated", translatedScene(), 3),
     // Several components of this scene flip a gating decision at both steps; the others are checked.
     ("rotated", rotatedScene(), 1),
     ("stacked", stackedScene(), 3),
     ("clamped", clampedScene(), 3)]
}

// MARK: - Gradient checks

@Test func poseGradientMatchesFiniteDifferencesTranslatedScene() {
    let (name, scene, minimum) = gradientScenes()[0]
    let checked = checkPoseGradient(name, scene)
    #expect(checked.count >= minimum, "only \(checked.count) pose components could be checked")
}

@Test func poseGradientMatchesFiniteDifferencesRotatedScene() {
    let (name, scene, minimum) = gradientScenes()[1]
    let checked = checkPoseGradient(name, scene)
    #expect(checked.count >= minimum, "only \(checked.count) pose components could be checked")
}

@Test func poseGradientMatchesFiniteDifferencesWithAlphaClamp() {
    let (name, scene, minimum) = gradientScenes()[2]
    let checked = checkPoseGradient(name, scene)
    #expect(checked.count >= minimum, "only \(checked.count) pose components could be checked")
}

@Test func poseGradientMatchesFiniteDifferencesWithClampedJacobian() {
    let (name, scene, minimum) = gradientScenes()[3]
    // Confirm the clamp is active for the test Gaussians (|x/z| > 0.65 or |y/z| > 0.455 at this camera).
    var clamped = 0
    for i in 0..<scene.cloud.count {
        let p = SIMD3<Float>(scene.cloud.positions[3 * i], scene.cloud.positions[3 * i + 1],
                             scene.cloud.positions[3 * i + 2])
        let pc = scene.camera.rotation * p + scene.camera.translation
        if abs(pc.x / pc.z) > 0.66 || abs(pc.y / pc.z) > 0.46 { clamped += 1 }
    }
    #expect(clamped == scene.cloud.count)
    let checked = checkPoseGradient(name, scene)
    #expect(checked.count >= minimum, "only \(checked.count) pose components could be checked")
}

/// Across the scenes every one of the six pose components (ω and τ) is checked at least once.
@Test func poseGradientCheckCoversEveryComponent() {
    var covered = Set<Int>()
    for (name, scene, _) in gradientScenes() {
        covered.formUnion(checkPoseGradient(name, scene))
    }
    #expect(covered == Set(0..<6), "components checked across scenes: \(covered.sorted())")
}

// MARK: - Exponential map, folding and Adam

@Test func exponentialMapIsRotation() {
    let omegas: [SIMD3<Double>] = [SIMD3(0.3, -0.7, 0.2), SIMD3(2.5, 0.1, -1.0), SIMD3(1e-9, 0, 0), SIMD3(0, 0, 0)]
    for omega in omegas {
        let r = PoseRefinement.exponentialMap(omega)
        let rtr = DoubleMatrix3(c0: SIMD3(r.c0.x, r.c1.x, r.c2.x), c1: SIMD3(r.c0.y, r.c1.y, r.c2.y),
                                c2: SIMD3(r.c0.z, r.c1.z, r.c2.z)) * r
        #expect(abs(rtr.c0.x - 1) < 1e-12 && abs(rtr.c1.y - 1) < 1e-12 && abs(rtr.c2.z - 1) < 1e-12)
        #expect(abs(rtr.c1.x) < 1e-12 && abs(rtr.c2.x) < 1e-12 && abs(rtr.c2.y) < 1e-12)
        let det = r.c0.x * (r.c1.y * r.c2.z - r.c2.y * r.c1.z)
            - r.c1.x * (r.c0.y * r.c2.z - r.c2.y * r.c0.z)
            + r.c2.x * (r.c0.y * r.c1.z - r.c1.y * r.c0.z)
        #expect(abs(det - 1) < 1e-12)
    }
}

@Test func exponentialMapRotatesAboutItsAxis() {
    // ω = (0, 0, θ) rotates the x axis to (cos θ, sin θ, 0) for [ω]× with the right-hand convention.
    let theta = 0.3
    let r = PoseRefinement.exponentialMap(SIMD3(0, 0, theta))
    let image = r * SIMD3<Double>(1, 0, 0)
    #expect(abs(image.x - cos(theta)) < 1e-12)
    #expect(abs(image.y - sin(theta)) < 1e-12)
    #expect(abs(image.z) < 1e-12)
}

@Test func foldAppliesRigidCorrectionToCameraPoints() {
    let camera = Camera(width: 40, height: 28, fx: 40, fy: 40, cx: 20, cy: 14,
                        rotation: .identity, translation: SIMD3(0.2, -0.1, 4))
    let omega = SIMD3<Double>(0.02, -0.01, 0.015)
    let tau = SIMD3<Double>(0.03, 0.01, -0.02)
    let folded = PoseRefinement.fold(camera, rotation: omega, translation: tau)
    let e = PoseRefinement.exponentialMap(omega)
    for point in [SIMD3<Float>(0.5, -0.3, 2.0), SIMD3<Float>(-1, 0.4, 3.5)] {
        let before = camera.rotation * point + camera.translation
        let after = folded.rotation * point + folded.translation
        let expected = e * SIMD3<Double>(Double(before.x), Double(before.y), Double(before.z)) + tau
        #expect(abs(Double(after.x) - expected.x) < 1e-6)
        #expect(abs(Double(after.y) - expected.y) < 1e-6)
        #expect(abs(Double(after.z) - expected.z) < 1e-6)
    }
}

@Test func foldKeepsRotationOrthonormal() {
    var camera = Camera(width: 40, height: 28, fx: 40, fy: 40, cx: 20, cy: 14,
                        rotation: PoseRefinement.exponentialMap(SIMD3(0.4, 0.1, -0.3)).matrix3,
                        translation: SIMD3(0.1, 0.2, 3))
    var rng = SplitMix64(state: 2)
    for _ in 0..<50 {
        let omega = SIMD3<Double>(Double.random(in: -0.01...0.01, using: &rng),
                                  Double.random(in: -0.01...0.01, using: &rng),
                                  Double.random(in: -0.01...0.01, using: &rng))
        let tau = SIMD3<Double>(Double.random(in: -0.01...0.01, using: &rng),
                                Double.random(in: -0.01...0.01, using: &rng),
                                Double.random(in: -0.01...0.01, using: &rng))
        camera = PoseRefinement.fold(camera, rotation: omega, translation: tau)
    }
    let r = camera.rotation
    let columns = [r.c0, r.c1, r.c2]
    for i in 0..<3 {
        for j in 0..<3 {
            let dot = (columns[i] * columns[j]).sum()
            #expect(abs(dot - (i == j ? 1 : 0)) < 1e-5, "column \(i)·\(j) = \(dot)")
        }
    }
}

/// δ followed by its inverse (−ω, −exp(−[ω]×)·τ) returns the original pose. Note that the inverse of the
/// translation correction is not −τ: p' = E·p + τ, so p = Eᵀ(p' − τ).
@Test func foldFollowedByInverseReturnsPose() {
    let camera = Camera(width: 40, height: 28, fx: 40, fy: 40, cx: 20, cy: 14,
                        rotation: PoseRefinement.exponentialMap(SIMD3(0.3, -0.2, 0.1)).matrix3,
                        translation: SIMD3(0.4, -0.3, 3.5))
    let omega = SIMD3<Double>(0.05, -0.02, 0.03)
    let tau = SIMD3<Double>(0.02, 0.04, -0.03)
    let inverseTau = -(PoseRefinement.exponentialMap(-omega) * tau)
    let folded = PoseRefinement.fold(camera, rotation: omega, translation: tau)
    let restored = PoseRefinement.fold(folded, rotation: -omega, translation: inverseTau)
    let original = [camera.rotation.c0, camera.rotation.c1, camera.rotation.c2]
    let returned = [restored.rotation.c0, restored.rotation.c1, restored.rotation.c2]
    for i in 0..<3 {
        let difference = original[i] - returned[i]
        #expect((difference * difference).sum() < 1e-10)
    }
    let translationDifference = camera.translation - restored.translation
    #expect((translationDifference * translationDifference).sum() < 1e-10)
}

@Test func adamFirstStepIsSignedStep() {
    var state = PoseAdamState()
    let gradient = PoseGradient(rotation: SIMD3(2e-3, -4e-3, 1e-3), translation: SIMD3(5e-2, 0, -1e-2))
    let delta = PoseRefinement.adamStep(&state, gradient: gradient, rotationRate: 2e-4,
                                        translationRate: 3e-3, hyper: AdamHyperparameters())
    #expect(state.step == 1)
    #expect(abs(delta.rotation.x + 2e-4) < 1e-9)
    #expect(abs(delta.rotation.y - 2e-4) < 1e-9)
    #expect(abs(delta.rotation.z + 2e-4) < 1e-9)
    #expect(abs(delta.translation.x + 3e-3) < 1e-9)
    #expect(abs(delta.translation.z - 3e-3) < 1e-9)
}
