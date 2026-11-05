import Foundation
import Testing
import SplatCore

/// Spherical-harmonics DC coefficient that yields `rgb` for a degree-0 Gaussian
/// (color = C0 * dc + 0.5).
private func dcForColor(_ rgb: SIMD3<Float>) -> SIMD3<Float> {
    let c0: Float = 0.28209479177387814
    return (rgb - SIMD3(repeating: 0.5)) / c0
}

/// One Gaussian at the origin, scale 0.1, identity rotation, opacity logit 0,
/// color (1, 0.5, 0). Camera: 64×64, fx = fy = 100, cx = cy = 32, t = (0, 0, 5).
private func singleGaussianCloud() throws(GaussianCloudError) -> GaussianCloud {
    let logScale = Float(log(0.1))
    let dc = dcForColor(SIMD3(1, 0.5, 0))
    return try GaussianCloud(
        shDegree: 0,
        positions: [0, 0, 0],
        logScales: [logScale, logScale, logScale],
        rotations: [1, 0, 0, 0],
        opacityLogits: [0],
        sh: [dc.x, dc.y, dc.z]
    )
}

private func fixtureCamera(translation: SIMD3<Float> = SIMD3(0, 0, 5)) -> Camera {
    Camera(width: 64, height: 64, fx: 100, fy: 100, cx: 32, cy: 32,
           rotation: .identity, translation: translation)
}

@Test func projectsSingleGaussian() throws {
    let cloud = try singleGaussianCloud()
    let p = try #require(Projection.project(cloud, index: 0, camera: fixtureCamera(), settings: .init()))
    #expect(p.mean2D == SIMD2(32, 32))
    #expect(p.depth == 5)
    #expect(approxEqual(p.conic, SIMD3<Float>(1 / 4.3, 0, 1 / 4.3), tol: 1e-6))
    #expect(p.radius == 7)
    #expect(p.tileMin == SIMD2(1, 1))
    #expect(p.tileMax == SIMD2(3, 3))
    #expect(approxEqual(p.color, SIMD3(1, 0.5, 0), tol: 1e-6))
    #expect(p.opacity == 0.5)
}

@Test func rendersExpectedPixel() throws {
    let cloud = try singleGaussianCloud()
    let img = CPURasterizer.render(cloud, camera: fixtureCamera(), settings: .init())
    #expect(img.width == 64)
    #expect(img.height == 64)
    #expect(approxEqual(img.pixels[31 * 64 + 31], SIMD3(0.471759, 0.2358795, 0), tol: 1e-5))
    // Tile (0, 0) lies outside the Gaussian's tile rect [1, 3) × [1, 3).
    #expect(img.pixels[0] == .zero)
}

@Test func gaussianBehindCameraIsCulled() throws {
    let cloud = try singleGaussianCloud()
    let behind = fixtureCamera(translation: SIMD3(0, 0, -5))
    #expect(Projection.project(cloud, index: 0, camera: behind, settings: .init()) == nil)

    let background = SIMD3<Float>(0.2, 0.3, 0.4)
    let settings = RasterSettings(background: background)
    let img = CPURasterizer.render(cloud, camera: behind, settings: settings)
    #expect(img.pixels.count == 64 * 64)
    for pixel in img.pixels {
        #expect(approxEqual(pixel, background, tol: 1e-6))
    }
}

@Test func shDegreeLimitIgnoresHigherBands() throws {
    // Degree-3 Gaussian: DC gives color (1, 0.5, 0); all higher bands are non-zero.
    let k = GaussianCloud.coefficientCount(degree: 3)
    var sh = [Float](repeating: 0.7, count: k * 3)
    let dc = dcForColor(SIMD3(1, 0.5, 0))
    sh[0] = dc.x; sh[1] = dc.y; sh[2] = dc.z
    let logScale = Float(log(0.1))
    let cloud = try GaussianCloud(
        shDegree: 3,
        positions: [0, 0, 0],
        logScales: [logScale, logScale, logScale],
        rotations: [1, 0, 0, 0],
        opacityLogits: [0],
        sh: sh
    )
    let camera = fixtureCamera()

    let limited = try #require(Projection.project(
        cloud, index: 0, camera: camera, settings: RasterSettings(shDegreeLimit: 0)))
    #expect(approxEqual(limited.color, SIMD3(1, 0.5, 0), tol: 1e-6))

    let full = try #require(Projection.project(
        cloud, index: 0, camera: camera, settings: RasterSettings(shDegreeLimit: 3)))
    #expect(!approxEqual(full.color, SIMD3(1, 0.5, 0), tol: 1e-3))
}

@Test func shDegree1MatchesFormula() {
    // Coefficients [k][c]: sh0, sh1, sh2, sh3 with 3 channels each.
    let sh0 = SIMD3<Float>(1, 2, 3)
    let sh1 = SIMD3<Float>(4, 5, 6)
    let sh2 = SIMD3<Float>(7, 8, 9)
    let sh3 = SIMD3<Float>(10, 11, 12)
    let coefficients: [Float] = [sh0, sh1, sh2, sh3].flatMap { [$0.x, $0.y, $0.z] }

    let c0: Float = 0.28209479177387814
    let c1: Float = 0.4886025119029199
    let result = SphericalHarmonics.evaluate(
        degree: 1, coefficients: coefficients[...], direction: SIMD3(0, 0, 1))
    #expect(approxEqual(result, c0 * sh0 + c1 * sh2, tol: 1e-6))

    // Direction (1, 0, 0) selects sh3 with a negative sign.
    let alongX = SphericalHarmonics.evaluate(
        degree: 1, coefficients: coefficients[...], direction: SIMD3(1, 0, 0))
    #expect(approxEqual(alongX, c0 * sh0 - c1 * sh3, tol: 1e-6))
}

@Test func frontGaussianOccludes() throws {
    // Index 0: back Gaussian (blue) at z_world 2 → depth 7.
    // Index 1: front Gaussian (red) at z_world 0 → depth 5.
    // Scale 1 (logScale 0) so the front Gaussian covers the center pixel with alpha ≈ 0.999.
    // A scale of 0.1 would give alpha ≈ 0.94 at the pixel center (0.5 px off the mean),
    // which cannot meet a 1e-2 color tolerance with distinct colors.
    let front = SIMD3<Float>(1, 0, 0)
    let back = SIMD3<Float>(0, 0, 1)
    let dcBack = dcForColor(back)
    let dcFront = dcForColor(front)
    let cloud = try GaussianCloud(
        shDegree: 0,
        positions: [0, 0, 2, 0, 0, 0],
        logScales: [0, 0, 0, 0, 0, 0],
        rotations: [1, 0, 0, 0, 1, 0, 0, 0],
        opacityLogits: [10, 10],
        sh: [dcBack.x, dcBack.y, dcBack.z, dcFront.x, dcFront.y, dcFront.z]
    )
    let img = CPURasterizer.render(cloud, camera: fixtureCamera(), settings: .init())
    #expect(approxEqual(img.pixels[32 * 64 + 32], front, tol: 1e-2))
}
