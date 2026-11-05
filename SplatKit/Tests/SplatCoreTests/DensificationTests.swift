import Foundation
import Testing
@testable import SplatCore

// MARK: - Helpers

private struct Spec {
    var position: SIMD3<Float>
    var logScale: SIMD3<Float>
    var logit: Float
    /// (w, x, y, z), as stored in `GaussianCloud.rotations`.
    var rotation: SIMD4<Float> = SIMD4(1, 0, 0, 0)
}

/// Cloud with deterministic SH coefficients (`0.1·i + 0.01·k + 0.001·c`).
private func makeCloud(_ specs: [Spec], shDegree: Int = 0) -> GaussianCloud {
    let k = GaussianCloud.coefficientCount(degree: shDegree)
    var positions: [Float] = []
    var logScales: [Float] = []
    var rotations: [Float] = []
    var logits: [Float] = []
    var sh: [Float] = []
    for (i, s) in specs.enumerated() {
        positions += [s.position.x, s.position.y, s.position.z]
        logScales += [s.logScale.x, s.logScale.y, s.logScale.z]
        rotations += [s.rotation.x, s.rotation.y, s.rotation.z, s.rotation.w]
        logits.append(s.logit)
        for kk in 0..<k {
            for c in 0..<3 {
                sh.append(0.1 * Float(i) + 0.01 * Float(kk) + 0.001 * Float(c))
            }
        }
    }
    return try! GaussianCloud(shDegree: shDegree, positions: positions, logScales: logScales,
                              rotations: rotations, opacityLogits: logits, sh: sh)
}

/// Camera at the origin looking down +z, 40×28 pixels, principal point (20, 14).
private func originCamera(translation: SIMD3<Float> = .zero) -> Camera {
    Camera(width: 40, height: 28, fx: 40, fy: 40, cx: 20, cy: 14,
           rotation: .identity, translation: translation)
}

/// Smooth, non-symmetric image gradient so that screen-space sums do not cancel.
private func asymmetricImageGradient(width: Int, height: Int) -> [SIMD3<Float>] {
    var result: [SIMD3<Float>] = []
    result.reserveCapacity(width * height)
    for y in 0..<height {
        for x in 0..<width {
            let w: Float = 1 + 0.01 * Float(x) + 0.02 * Float(y)
            result.append(SIMD3<Float>(repeating: w))
        }
    }
    return result
}

/// Gradient set with distinct values per float, used to detect copies of moments.
private func marked(_ cloud: GaussianCloud, base: Float) -> GaussianGradients {
    func ramp(_ n: Int) -> [Float] {
        (0..<n).map { base + 0.01 * Float($0) }
    }
    return GaussianGradients(positions: ramp(cloud.positions.count),
                             logScales: ramp(cloud.logScales.count),
                             rotations: ramp(cloud.rotations.count),
                             opacityLogits: ramp(cloud.opacityLogits.count),
                             sh: ramp(cloud.sh.count))
}

private func row(_ values: [Float], _ i: Int, stride: Int) -> [Float] {
    Array(values[(i * stride)..<((i + 1) * stride)])
}

// MARK: - Screen-space gradients

@Test(arguments: ["single", "multi", "rotated", "occluded"])
func screenSpaceBackwardKeepsGradientsIdentical(name: String) throws {
    let golden = try GoldenCase.load(name)
    let cloud = try golden.cloud()
    let camera = golden.camera()
    let settings = golden.settings()
    let imageGradient = golden.imageGradientPixels

    let plain = CPURasterizer.backward(cloud, camera: camera, settings: settings,
                                       imageGradient: imageGradient)
    let screen = CPURasterizer.backwardWithScreenSpace(cloud, camera: camera, settings: settings,
                                                       imageGradient: imageGradient)
    #expect(screen.gradients == plain)
    #expect(screen.mean2D.count == cloud.count)
    #expect(screen.visible.count == cloud.count)
}

@Test func mean2DGradientMatchesPositionGradientOnAxis() {
    // Isotropic degree-0 Gaussian on the optical axis: dΣ2D/dx = 0 there, and the colour does
    // not depend on the direction, so the camera-space position gradient is only the mean2D term.
    let sigma: Float = 0.3
    let cloud = makeCloud([Spec(position: SIMD3(0, 0, 4),
                                logScale: SIMD3(repeating: log(sigma)), logit: 0)])
    let camera = originCamera()
    let imageGradient = asymmetricImageGradient(width: camera.width, height: camera.height)

    let result = CPURasterizer.backwardWithScreenSpace(cloud, camera: camera, settings: .init(),
                                                       imageGradient: imageGradient)
    #expect(result.visible == [true])

    let z: Float = 4
    let expectedX = result.gradients.positions[0] * z / camera.fx
    let expectedY = result.gradients.positions[1] * z / camera.fy
    #expect(abs(expectedX) > 1e-5)   // non-trivial: the check is not vacuous
    #expect(abs(result.mean2D[0].x - expectedX) <= 1e-4 * abs(expectedX))
    #expect(abs(result.mean2D[0].y - expectedY) <= 1e-4 * abs(expectedY) + 1e-7)
}

@Test func statisticsIgnoreInvisibleGaussians() {
    // The cloud lies behind the camera (camera-space z = −6 < near plane).
    let cloud = makeCloud([
        Spec(position: SIMD3(0, 0, -10), logScale: SIMD3(repeating: log(0.3)), logit: 0),
        Spec(position: SIMD3(0.5, 0, -10), logScale: SIMD3(repeating: log(0.3)), logit: 0),
    ])
    let camera = originCamera(translation: SIMD3(0, 0, 4))
    let imageGradient = asymmetricImageGradient(width: camera.width, height: camera.height)

    let result = CPURasterizer.backwardWithScreenSpace(cloud, camera: camera, settings: .init(),
                                                       imageGradient: imageGradient)
    #expect(result.visible == [false, false])

    var stats = DensificationStatistics(count: cloud.count)
    stats.accumulate(mean2D: result.mean2D, visible: result.visible,
                     width: camera.width, height: camera.height)
    #expect(stats == DensificationStatistics(count: cloud.count))
}

@Test func accumulateUsesHalfImageScaledNorms() {
    var stats = DensificationStatistics(count: 3)
    stats.accumulate(mean2D: [SIMD2(1, 0), SIMD2(0, 0.5), SIMD2(9, 9)],
                     visible: [true, true, false], width: 40, height: 28)
    #expect(stats.visibleCounts == [1, 1, 0])
    #expect(approxEqual(stats.gradientNormSums[0], 20, tol: 1e-5))   // |(1·20, 0)|
    #expect(approxEqual(stats.gradientNormSums[1], 7, tol: 1e-5))    // |(0, 0.5·14)|
    #expect(stats.gradientNormSums[2] == 0)

    stats.reset()
    #expect(stats == DensificationStatistics(count: 3))
}

// MARK: - Decision rules

@Test func actionTable() {
    let policy = DensificationPolicy()
    let radius: Float = 2
    let small = SIMD3<Float>(repeating: log(0.01))   // ≤ 0.01·R = 0.02 → clone when growing
    let large = SIMD3<Float>(repeating: log(0.05))   // > 0.02 → split when growing
    let huge = SIMD3<Float>(repeating: log(0.5))     // > 0.1·R = 0.2 → pruned after start
    let growing: Float = 0.004                       // average 0.001 over 4 views > 0.0002
    let transparent: Float = -6                      // opacity ≈ 0.0025 < 0.005

    func act(_ scale: SIMD3<Float>, logit: Float = 0, sum: Float = 0, count: UInt32 = 0,
             iteration: Int = 100) -> DensifyAction {
        Densification.action(logScale: scale, opacityLogit: logit, gradientNormSum: sum,
                             visibleCount: count, policy: policy, sceneRadius: radius,
                             iteration: iteration)
    }

    // Opacity prune wins regardless of gradient.
    #expect(act(small, logit: transparent, sum: growing, count: 4) == .prune)
    // Scale prune only strictly after the start iteration.
    #expect(act(huge, iteration: 3_001) == .prune)
    #expect(act(huge, iteration: 3_000) == .keep)
    // Scale prune takes precedence over growth.
    #expect(act(huge, sum: growing, count: 4, iteration: 3_001) == .prune)
    // Growth decision and clone/split by maximum scale.
    #expect(act(small, sum: growing, count: 4) == .clone)
    #expect(act(large, sum: growing, count: 4) == .split)
    // Boundary: average exactly at the threshold keeps (0.0008 / 4 == 0.0002 in Float).
    #expect(act(small, sum: 0.0008, count: 4) == .keep)
    // No visible views: average is zero.
    #expect(act(large, sum: 0.5, count: 0) == .keep)
}

// MARK: - Planning

@Test func planDemotesGrowthBeyondBudget() {
    let specs = (0..<10).map { i in
        Spec(position: SIMD3(Float(i), 0, 4), logScale: SIMD3(repeating: log(0.05)), logit: 0)
    }
    let cloud = makeCloud(specs)
    var stats = DensificationStatistics(count: cloud.count)
    for i in 0..<6 {
        stats.gradientNormSums[i] = 0.004
        stats.visibleCounts[i] = 4
    }

    let plan = Densification.plan(cloud, statistics: stats, policy: DensificationPolicy(),
                                  sceneRadius: 2, iteration: 100, maxGaussians: 13)
    // base = 10, allowed = 3: the three lowest candidate indices grow, 3…5 are demoted to keep.
    #expect(plan.actions == [.split, .split, .split, .keep, .keep, .keep, .keep, .keep, .keep, .keep])
    #expect(plan.outputCount == 13)
    #expect(plan.split == 3)
    #expect(plan.cloned == 0)
    #expect(plan.pruned == 0)
    #expect(plan.demoted == 3)
    #expect(plan.keptAll == false)
}

@Test func planNeverPrunesEverything() {
    let transparent: Float = log(1e-4 / (1 - 1e-4))
    let cloud = makeCloud((0..<3).map { i in
        Spec(position: SIMD3(Float(i), 0, 4), logScale: SIMD3(repeating: log(0.05)), logit: transparent)
    })
    let plan = Densification.plan(cloud, statistics: DensificationStatistics(count: 3),
                                  policy: DensificationPolicy(), sceneRadius: 2,
                                  iteration: 100, maxGaussians: 100)
    #expect(plan.actions == [.keep, .keep, .keep])
    #expect(plan.keptAll)
    #expect(plan.pruned == 0)
    #expect(plan.outputCount == cloud.count)
}

// MARK: - Applying a plan

@Test func applyLayoutAndMoments() {
    let cloud = makeCloud([
        Spec(position: SIMD3(0, 0, 3), logScale: SIMD3(repeating: log(0.05)), logit: 0.5),
        Spec(position: SIMD3(1, 0, 3), logScale: SIMD3(repeating: log(0.01)), logit: -0.5),
        Spec(position: SIMD3(2, 0, 3), logScale: SIMD3(repeating: log(0.02)), logit: -6),
        Spec(position: SIMD3(3, 0, 3), logScale: SIMD3(repeating: log(0.05)), logit: 1.5),
    ], shDegree: 1)
    let shStride = 3 * GaussianCloud.coefficientCount(degree: cloud.shDegree)   // 12 for degree 1
    var state = AdamState(zeroLike: cloud)
    state.m = marked(cloud, base: 1)
    state.v = marked(cloud, base: 2)
    state.step = 7
    let moments = state.m
    let squared = state.v

    // [keep, clone, prune, split] → 1 + 2 + 0 + 2 = 5 outputs in input order.
    let plan = DensifyPlan(actions: [.keep, .clone, .prune, .split], outputCount: 5,
                           pruned: 1, cloned: 1, split: 1, demoted: 0, keptAll: false)
    let out = Densification.apply(plan, to: cloud, state: &state, policy: DensificationPolicy(),
                                  seed: 7, iteration: 100)

    #expect(out.count == 5)
    #expect(out.opacityLogits == [cloud.opacityLogits[0], cloud.opacityLogits[1],
                                  cloud.opacityLogits[1], cloud.opacityLogits[3],
                                  cloud.opacityLogits[3]])
    // Keep and clone child 0 carry their moments; the other children start from zero.
    #expect(row(state.m.positions, 0, stride: 3) == row(moments.positions, 0, stride: 3))
    #expect(row(state.v.positions, 0, stride: 3) == row(squared.positions, 0, stride: 3))
    #expect(row(state.m.positions, 1, stride: 3) == row(moments.positions, 1, stride: 3))
    #expect(row(state.v.sh, 1, stride: shStride) == row(squared.sh, 1, stride: shStride))
    #expect(row(state.m.positions, 2, stride: 3) == [0, 0, 0])
    #expect(row(state.v.positions, 2, stride: 3) == [0, 0, 0])
    #expect(row(state.m.sh, 2, stride: shStride) == [Float](repeating: 0, count: shStride))
    #expect(row(state.m.positions, 3, stride: 3) == [0, 0, 0])
    #expect(row(state.v.positions, 4, stride: 3) == [0, 0, 0])
    #expect(row(state.m.sh, 4, stride: shStride) == [Float](repeating: 0, count: shStride))
    #expect(state.m.opacityLogits.count == 5)
    #expect(state.step == 7)
    // The clone child keeps the parent's geometry exactly.
    #expect(row(out.positions, 2, stride: 3) == row(cloud.positions, 1, stride: 3))
}

@Test func splitChildrenAreDeterministicAndShrunk() {
    let rotation = SIMD4<Float>(0.8, 0.3, 0.4, 0.2)   // not unit length on purpose
    let parentPosition = SIMD3<Float>(0.2, -0.1, 3)
    let parentLogScale = SIMD3<Float>(log(0.05), log(0.03), log(0.02))
    let cloud = makeCloud([Spec(position: parentPosition, logScale: parentLogScale,
                                logit: 0, rotation: rotation)])
    let plan = DensifyPlan(actions: [.split], outputCount: 2, pruned: 0, cloned: 0,
                           split: 1, demoted: 0, keptAll: false)
    let policy = DensificationPolicy()

    var stateA = AdamState(zeroLike: cloud)
    var stateB = AdamState(zeroLike: cloud)
    let a = Densification.apply(plan, to: cloud, state: &stateA, policy: policy, seed: 42, iteration: 500)
    let b = Densification.apply(plan, to: cloud, state: &stateB, policy: policy, seed: 42, iteration: 500)
    #expect(a.positions == b.positions)
    #expect(a.logScales == b.logScales)
    #expect(stateA == stateB)

    // Children: offset by R(q̂)·(exp(logScale) ⊙ ε_j) with the documented stream layout.
    let rotationMatrix = Matrix3.rotation(quaternion: rotation)
    let scale = SIMD3<Float>(exp(parentLogScale.x), exp(parentLogScale.y), exp(parentLogScale.z))
    let streamSeed: UInt32 = 42 &+ 500
    for j in 0..<2 {
        let jj = UInt32(3 * j)
        let eps = SIMD3<Float>(SplatRandom.normal(seed: streamSeed, index: 0, stream: jj),
                               SplatRandom.normal(seed: streamSeed, index: 0, stream: jj + 1),
                               SplatRandom.normal(seed: streamSeed, index: 0, stream: jj + 2))
        let expected = parentPosition + rotationMatrix * (scale * eps)
        let child = SIMD3<Float>(a.positions[3 * j], a.positions[3 * j + 1], a.positions[3 * j + 2])
        #expect(approxEqual(child, expected, tol: 1e-5))
        for c in 0..<3 {
            #expect(abs(a.logScales[3 * j + c] - (parentLogScale[c] - log(Float(1.6)))) <= 1e-6)
        }
        #expect(Array(a.rotations[(4 * j)..<(4 * j + 4)]) == Array(cloud.rotations[0..<4]))
        #expect(a.opacityLogits[j] == cloud.opacityLogits[0])
    }
    #expect(stateA.m.positions.allSatisfy { $0 == 0 })
    #expect(stateA.v.sh.allSatisfy { $0 == 0 })
}

@Test func applyPreservesSHAndRotationLayouts() throws {
    let cloud = makeCloud([
        Spec(position: SIMD3(0, 0, 3), logScale: SIMD3(repeating: log(0.05)), logit: 0,
             rotation: SIMD4(0.8, 0.3, 0.4, 0.2)),
        Spec(position: SIMD3(1, 0, 3), logScale: SIMD3(repeating: log(0.05)), logit: 0),
    ], shDegree: 2)
    let k3 = 3 * GaussianCloud.coefficientCount(degree: 2)
    var state = AdamState(zeroLike: cloud)
    let plan = DensifyPlan(actions: [.clone, .split], outputCount: 4, pruned: 0, cloned: 1,
                           split: 1, demoted: 0, keptAll: false)

    let out = Densification.apply(plan, to: cloud, state: &state, policy: DensificationPolicy(),
                                  seed: 1, iteration: 0)
    #expect(out.count == 4)
    #expect(out.shDegree == 2)
    #expect(out.sh.count == 4 * k3)
    #expect(out.rotations.count == 16)
    // Clone children of Gaussian 0 and split children of Gaussian 1 keep their parents' SH.
    #expect(Array(out.sh[0..<k3]) == Array(cloud.sh[0..<k3]))
    #expect(Array(out.sh[k3..<2 * k3]) == Array(cloud.sh[0..<k3]))
    #expect(Array(out.sh[2 * k3..<3 * k3]) == Array(cloud.sh[k3..<2 * k3]))
    #expect(Array(out.sh[3 * k3..<4 * k3]) == Array(cloud.sh[k3..<2 * k3]))
    // Rotations follow the same mapping.
    #expect(Array(out.rotations[0..<4]) == Array(cloud.rotations[0..<4]))
    #expect(Array(out.rotations[4..<8]) == Array(cloud.rotations[0..<4]))
    #expect(Array(out.rotations[8..<12]) == Array(cloud.rotations[4..<8]))
    #expect(Array(out.rotations[12..<16]) == Array(cloud.rotations[4..<8]))
    #expect(state.m.sh.count == out.sh.count)
    #expect(state.v.rotations.count == out.rotations.count)
}
