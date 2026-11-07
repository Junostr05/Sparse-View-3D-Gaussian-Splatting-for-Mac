import Foundation
import Testing
@testable import SplatCore

// Mip-Splatting anti-aliasing on the CPU reference: the 2D compensation, the 3D filter (as a bake), the sampling
// rate, and finite differences of the mip backward pass for every parameter group.

/// Projection of Gaussian 0 of a one-Gaussian cloud, with the given settings.
private func projectFirst(_ cloud: GaussianCloud, camera: Camera, settings: RasterSettings) -> ProjectedGaussian? {
    Projection.project(cloud, index: 0, camera: camera, settings: settings)
}

private func singleGaussian(logScale: SIMD3<Float>, rotation: SIMD4<Float>, logit: Float,
                            position: SIMD3<Float>) -> GaussianCloud {
    try! GaussianCloud(shDegree: 0,
                       positions: [position.x, position.y, position.z],
                       logScales: [logScale.x, logScale.y, logScale.z],
                       rotations: [rotation.x, rotation.y, rotation.z, rotation.w],
                       opacityLogits: [logit],
                       sh: [0.3, -0.2, 0.1])
}

private let testCamera = Camera(width: 48, height: 36, fx: 40, fy: 40, cx: 24, cy: 18,
                                rotation: .identity, translation: SIMD3(0.1, -0.05, 3))

// MARK: - 2D filter

@Test func compensationMatchesDeterminantRatio() {
    let cloud = singleGaussian(logScale: SIMD3(-1.4, -1.9, -1.6), rotation: SIMD4(0.9, 0.2, -0.3, 0.1),
                               logit: 1.2, position: SIMD3(0.2, -0.1, 0.4))
    let v2: Float = 0.1
    // Raw 2D covariance: project with no dilation (classic, eps2D = 0). Its conic is Σ⁻¹ = (A, B, C).
    var noDilation = RasterSettings(eps2D: 0, shDegreeLimit: 0)
    let raw = projectFirst(cloud, camera: testCamera, settings: noDilation)!
    noDilation.mipFilter = MipFilter(isEnabled: true, filter2DVariance: 0, filter3DVariance: 0,
                                     samplingRates: [1])
    let identity = projectFirst(cloud, camera: testCamera, settings: noDilation)!
    // With v2 = 0 and no 3D filter the mip path reduces to the classic path (up to rounding of sqrt(s²)).
    #expect(abs(identity.opacity - raw.opacity) <= 1e-6 * raw.opacity)
    #expect(abs(identity.conic.x - raw.conic.x) <= 1e-5 * abs(raw.conic.x))

    // Σ = K⁻¹ with K = [[A, B], [B, C]] (conic.y = B): det Σ = 1/det K, Σ + v2·I has det det Σ + v2·tr Σ + v2².
    let k = raw.conic
    let kxD: Double = Double(k.x)
    let kyD: Double = Double(k.y)
    let kzD: Double = Double(k.z)
    let detK: Double = kxD * kzD - kyD * kyD
    let trace: Double = (kzD + kxD) / detK
    let detRaw: Double = 1 / detK
    let v2D: Double = Double(v2)
    let withTrace: Double = detRaw + v2D * trace
    let detFiltered: Double = withTrace + v2D * v2D
    let expectedCompensation: Double = (detRaw / detFiltered).squareRoot()

    var settings = RasterSettings(shDegreeLimit: 0)
    settings.mipFilter = MipFilter(isEnabled: true, filter2DVariance: v2, filter3DVariance: 0,
                                   samplingRates: [1])
    let filtered = projectFirst(cloud, camera: testCamera, settings: settings)!
    let ratio = Double(filtered.opacity) / Double(raw.opacity)
    #expect(abs(ratio - expectedCompensation) < 1e-4 * expectedCompensation,
            "opacity ratio \(ratio) vs compensation \(expectedCompensation)")
    // The compensation never increases the opacity.
    #expect(ratio <= 1 + 1e-6)
    // Conic of Σ + v2·I: the inverse of the filtered matrix (its entries are larger in the diagonal).
    let kf = filtered.conic
    #expect(kf.x < k.x && kf.z < k.z, "mip conic must be the inverse of the dilated covariance")
}

@Test func mipTwoDFilterEqualsClassicDilationWithCompensation() {
    // Mip 2D filter with v2 = 0.1 covers the same covariance as classic eps2D = 0.1; the only difference is
    // the compensation, which can only reduce the opacity.
    let cloud = singleGaussian(logScale: SIMD3(-2.2, -2.2, -2.2), rotation: SIMD4(1, 0, 0, 0),
                               logit: 2, position: SIMD3(0.0, 0.0, 0.5))
    let classic = RasterSettings(eps2D: 0.1, shDegreeLimit: 0)
    let classicGaussian = projectFirst(cloud, camera: testCamera, settings: classic)!
    var mipSettings = RasterSettings(eps2D: 0.3, shDegreeLimit: 0)
    mipSettings.mipFilter = MipFilter(isEnabled: true, filter2DVariance: 0.1, filter3DVariance: 0,
                                      samplingRates: [1])
    let mip = projectFirst(cloud, camera: testCamera, settings: mipSettings)!
    #expect(mip.radius == classicGaussian.radius)
    #expect(abs(mip.conic.x - classicGaussian.conic.x) <= 1e-5 * abs(classicGaussian.conic.x))
    #expect(abs(mip.conic.z - classicGaussian.conic.z) <= 1e-5 * abs(classicGaussian.conic.z))
    #expect(mip.opacity <= classicGaussian.opacity)
    #expect(mip.opacity > 0.5 * classicGaussian.opacity, "a 0.1 px² compensation is mild")
}

@Test func threeDFilterGrowsFootprintOnlyAtLowSamplingRate() {
    // f² = v3/ν²: at high ν (fine sampling) the filter vanishes and the footprint matches the 2D-only result;
    // at low ν (the Gaussian is sub-pixel) the 3D filter widens the covariance, so the radius grows.
    let cloud = singleGaussian(logScale: SIMD3(-3.5, -3.5, -3.5), rotation: SIMD4(1, 0, 0, 0),
                               logit: 2, position: SIMD3(0.0, 0.0, 0.5))
    var twoDOnly = RasterSettings(eps2D: 0.3, shDegreeLimit: 0)
    twoDOnly.mipFilter = MipFilter(isEnabled: true, filter2DVariance: 0.1, filter3DVariance: 0,
                                   samplingRates: [1])
    let base = projectFirst(cloud, camera: testCamera, settings: twoDOnly)!

    var highRate = twoDOnly
    highRate.mipFilter.filter3DVariance = 0.2
    highRate.mipFilter.samplingRates = [1e4]
    let fine = projectFirst(cloud, camera: testCamera, settings: highRate)!
    #expect(abs(fine.conic.x - base.conic.x) <= 1e-4 * abs(base.conic.x))

    var lowRate = twoDOnly
    lowRate.mipFilter.filter3DVariance = 0.2
    lowRate.mipFilter.samplingRates = [0.5]
    let coarse = projectFirst(cloud, camera: testCamera, settings: lowRate)!
    #expect(coarse.radius >= base.radius)
    #expect(coarse.conic.x < base.conic.x, "wider covariance means a smaller inverse")
    #expect(coarse.opacity < base.opacity, "the 3D compensation reduces the opacity of sub-pixel Gaussians")
}

// MARK: - 3D filter

@Test func samplingRateIsMaximalFocalOverDepth() {
    // fx = 40, image 64×48. A centre at depth z contributes fx / z; the largest value over the cameras wins.
    let near = Camera(width: 64, height: 48, fx: 40, fy: 40, cx: 32, cy: 24,
                      rotation: .identity, translation: SIMD3(0, 0, 0))
    // Translated by +4 along z: the Gaussian at z = 2 is at depth 6 here, giving 40/6 < 20.
    let far = Camera(width: 64, height: 48, fx: 40, fy: 40, cx: 32, cy: 24,
                     rotation: .identity, translation: SIMD3(0, 0, 4))
    let rates = SamplingRate.compute(positions: [0, 0, 2], cameras: [near, far])
    #expect(abs(rates[0] - 20) < 1e-5)

    // Behind the camera in every view: invisible, so it takes the fallback (the mean over the visible ones, 20).
    let mixedRates = SamplingRate.compute(positions: [0, 0, 2, 0, 0, -3], cameras: [near])
    #expect(abs(mixedRates[0] - 20) < 1e-5)
    #expect(abs(mixedRates[1] - 20) < 1e-5)

    // Projected centre outside the image enlarged by 15 % per side: not visible. Here x_mean = 40·x/z + 32.
    let outside = SamplingRate.compute(positions: [2.2, 0, 2], cameras: [near])
    #expect(outside[0] == 1, "nothing visible: fallback is 1")

    // Inside the enlarged frustum: x_mean = 72 < 1.15·64 = 73.6.
    let edge = SamplingRate.compute(positions: [2, 0, 2], cameras: [near])
    #expect(abs(edge[0] - 20) < 1e-5)
}

@Test func bakingSmoothing3DMatchesFormulas() throws {
    let cloud = singleGaussian(logScale: SIMD3(-1.5, -2.0, -1.0), rotation: SIMD4(1, 0, 0, 0),
                               logit: 0.7, position: SIMD3(0, 0, 0))
    let f2: Float = 0.04
    let baked = cloud.bakingSmoothing3D(filterVariances: [f2])
    let scales = (0..<3).map { exp(Double(cloud.logScales[$0])) }
    var factor = 1.0
    for s in scales {
        let smoothed = (s * s + Double(f2)).squareRoot()
        factor *= s / smoothed
    }
    for j in 0..<3 {
        let s = scales[j]
        let expected = log((s * s + Double(f2)).squareRoot())
        #expect(abs(Double(baked.logScales[j]) - expected) < 1e-5)
    }
    let opacity = 1 / (1 + exp(-Double(cloud.opacityLogits[0])))
    let expectedOpacity = opacity * factor
    let bakedOpacity = 1 / (1 + exp(-Double(baked.opacityLogits[0])))
    #expect(abs(bakedOpacity - expectedOpacity) < 1e-5)
    #expect(bakedOpacity < opacity)
    // Zero variance leaves the cloud unchanged.
    let unchanged = cloud.bakingSmoothing3D(filterVariances: [0])
    for j in 0..<3 {
        #expect(abs(unchanged.logScales[j] - cloud.logScales[j]) < 1e-5)
    }
    #expect(abs(unchanged.opacityLogits[0] - cloud.opacityLogits[0]) < 1e-4)
}

@Test func mipProjectionWithoutDilationEqualsBakedClassicProjection() {
    // With v2 = 0 the mip path is only the 3D filter, which must equal classic rasterization of the baked cloud.
    let cloud = singleGaussian(logScale: SIMD3(-1.6, -1.3, -2.0), rotation: SIMD4(0.8, 0.3, 0.1, -0.2),
                               logit: 0.9, position: SIMD3(0.15, 0.1, 0.2))
    let rates: [Float] = [7.5]
    let f2 = Float(0.2) / (rates[0] * rates[0])
    let baked = cloud.bakingSmoothing3D(filterVariances: [f2])

    var mipSettings = RasterSettings(eps2D: 0, shDegreeLimit: 0)
    mipSettings.mipFilter = MipFilter(isEnabled: true, filter2DVariance: 0, filter3DVariance: 0.2,
                                      samplingRates: rates)
    let classicSettings = RasterSettings(eps2D: 0, shDegreeLimit: 0)
    guard let viaMip = projectFirst(cloud, camera: testCamera, settings: mipSettings),
          let viaBake = projectFirst(baked, camera: testCamera, settings: classicSettings) else {
        Issue.record("both projections must survive culling")
        return
    }
    #expect(abs(viaMip.opacity - viaBake.opacity) < 1e-5 * max(1, viaBake.opacity))
    #expect(abs(viaMip.conic.x - viaBake.conic.x) < 1e-4 * max(1, abs(viaBake.conic.x)))
    #expect(abs(viaMip.conic.z - viaBake.conic.z) < 1e-4 * max(1, abs(viaBake.conic.z)))
    #expect(abs(viaMip.conic.y - viaBake.conic.y) < 1e-4 * max(1, abs(viaBake.conic.y)))
    #expect(abs(viaMip.radius - viaBake.radius) <= 1)
}

// MARK: - Finite differences in mip mode

/// Mip settings with a large 3D variance, so that the 3D filter changes the covariance noticeably.
private func mipScene(_ base: Scene) -> Scene {
    var scene = base
    let rates = SamplingRate.compute(positions: base.cloud.positions, cameras: [base.camera])
    scene.settings.mipFilter = MipFilter(isEnabled: true, filter2DVariance: 0.1, filter3DVariance: 1.0,
                                         samplingRates: rates)
    return scene
}

private func mipLoss(_ scene: Scene, _ cloud: GaussianCloud) -> Double {
    let image = CPURasterizer.render(cloud, camera: scene.camera, settings: scene.settings)
    var total = 0.0
    for (p, w) in zip(image.pixels, scene.weights) {
        total += Double(p.x * w.x) + Double(p.y * w.y) + Double(p.z * w.z)
    }
    return total
}

/// Gating decisions (culling, tile rect, σ < 0, α < 1/255, termination) as a signature.
private func mipSignature(_ scene: Scene, _ cloud: GaussianCloud) -> [[Int]] {
    let sorted = CPURasterizer.projectedAndSorted(cloud, camera: scene.camera, settings: scene.settings)
    return CPURasterizer.blendedSamples(sorted, camera: scene.camera, settings: scene.settings)
        .map { $0.samples.map { sorted[$0.entry].index } }
}

private func mipAnalytic(_ scene: Scene) -> GaussianGradients {
    CPURasterizer.backward(scene.cloud, camera: scene.camera, settings: scene.settings,
                           imageGradient: scene.weights)
}

@Test func mipBackwardFiniteDifferencesMatchForEveryGroup() {
    for (name, base) in [("translated", translatedScene()), ("rotated", rotatedScene())] {
        let scene = mipScene(base)
        let grads = mipAnalytic(scene)
        let baseSignature = mipSignature(scene, scene.cloud)
        var rng = SplitMix64(state: 17)
        var checkedPerGroup: [Group: Int] = [:]
        var maxError = 0.0
        for g in Group.allCases {
            var candidates: [Int] = []
            for (i, v) in values(grads, g).enumerated() where abs(v) > 1e-2 { candidates.append(i) }
            candidates.shuffle(using: &rng)
            var checked = 0
            for index in candidates {
                if checked == 6 { break }
                let analytic = values(grads, g)[index]
                let h: Float = g == .sh ? 1e-2 : 1e-3
                let plus = perturbed(scene.cloud, g, index, h)
                let minus = perturbed(scene.cloud, g, index, -h)
                // A perturbation that flips a gating decision crosses a discontinuity: skip it.
                if mipSignature(scene, plus) != baseSignature || mipSignature(scene, minus) != baseSignature {
                    continue
                }
                let numeric = (mipLoss(scene, plus) - mipLoss(scene, minus)) / (2 * Double(h))
                let relative = abs(Double(analytic) - numeric) / max(abs(Double(analytic)), abs(numeric))
                maxError = max(maxError, relative)
                #expect(relative <= 5e-2, "\(name) \(g) [\(index)] analytic \(analytic) numeric \(numeric)")
                checked += 1
            }
            checkedPerGroup[g] = checked
        }
        print("mip FD \(name): checked per group \(checkedPerGroup), max rel err \(maxError)")
        for g in Group.allCases {
            #expect((checkedPerGroup[g] ?? 0) >= 2, "\(name): too few checked \(g) parameters")
        }
    }
}
