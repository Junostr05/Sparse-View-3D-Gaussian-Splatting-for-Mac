import Foundation
import Metal
import SplatCore
import SplatMetal

// Helpers for the SplatTraining tests. Copied from SplatMetalTests where needed, because a test
// target cannot import another test target.

/// Deterministic SplitMix64 generator.
struct SplitMix64: RandomNumberGenerator {
    var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

enum TrainingTestEnvironment {
    static var hasMetal: Bool {
        MTLCreateSystemDefaultDevice() != nil
    }
}

/// Random scene in front of a camera with R = I, t = 0.
func makeRandomCloud(
    count: Int,
    seed: UInt64 = 42,
    shDegree: Int = 3,
    depth: ClosedRange<Float> = 2...8,
    scale: ClosedRange<Float> = 0.02...0.3
) throws -> GaussianCloud {
    var rng = SplitMix64(seed: seed)
    let k = GaussianCloud.coefficientCount(degree: shDegree)
    var positions: [Float] = []
    var logScales: [Float] = []
    var rotations: [Float] = []
    var opacity: [Float] = []
    var sh: [Float] = []
    positions.reserveCapacity(3 * count)
    logScales.reserveCapacity(3 * count)
    rotations.reserveCapacity(4 * count)
    opacity.reserveCapacity(count)
    sh.reserveCapacity(count * k * 3)

    let logLo = log(scale.lowerBound)
    let logHi = log(scale.upperBound)
    for _ in 0..<count {
        let z = Float.random(in: depth, using: &rng)
        let x = Float.random(in: -0.45...0.45, using: &rng) * abs(z)
        let y = Float.random(in: -0.45...0.45, using: &rng) * abs(z)
        positions += [x, y, z]
        for _ in 0..<3 {
            logScales.append(Float.random(in: logLo...logHi, using: &rng))
        }
        for _ in 0..<4 {
            rotations.append(Float.random(in: -1...1, using: &rng))
        }
        opacity.append(Float.random(in: -2...4, using: &rng))
        for coefficient in 0..<k {
            for _ in 0..<3 {
                if coefficient == 0 {
                    sh.append(Float.random(in: 0...1.5, using: &rng))
                } else {
                    sh.append(Float.random(in: -0.5...0.5, using: &rng))
                }
            }
        }
    }
    return try GaussianCloud(shDegree: shDegree, positions: positions, logScales: logScales,
                             rotations: rotations, opacityLogits: opacity, sh: sh)
}

/// Pinhole camera with R = I, fx = fy = height, principal point at the image centre, and the
/// camera translated by `shift` along x.
func makeCamera(width: Int, height: Int, shift: Float = 0) -> Camera {
    Camera(width: width, height: height,
           fx: Float(height), fy: Float(height),
           cx: Float(width) / 2, cy: Float(height) / 2,
           rotation: .identity, translation: SIMD3<Float>(shift, 0, 0))
}

/// Training views whose targets are the Metal renderer's images of `groundTruth`.
func makeViews(context: MetalContext, groundTruth: GaussianCloud, cameras: [Camera]) throws -> [TrainingView] {
    let renderer = try SplatRenderer(context: context, cloud: groundTruth)
    let settings = RasterSettings(shDegreeLimit: groundTruth.shDegree, background: .zero)
    var views: [TrainingView] = []
    for (index, camera) in cameras.enumerated() {
        let image = try renderer.renderImage(camera: camera, settings: settings)
        views.append(TrainingView(name: "view\(index)", camera: camera, image: image))
    }
    return views
}

/// Mean CPU L1 (SplatCore `ImageLoss`) of `cloud` over `views`, with the cloud rendered by Metal.
func meanL1(context: MetalContext, cloud: GaussianCloud, views: [TrainingView]) throws -> Double {
    let renderer = try SplatRenderer(context: context, cloud: cloud)
    let settings = RasterSettings(shDegreeLimit: cloud.shDegree, background: .zero)
    var sum: Double = 0
    for view in views {
        let rendered = try renderer.renderImage(camera: view.camera, settings: settings)
        sum += ImageLoss.evaluate(rendered: rendered, target: view.image, lambda: 0.2).l1
    }
    return sum / Double(views.count)
}

/// Largest absolute difference between two cloud snapshots, over every parameter group.
func maxAbsDifference(_ a: GaussianCloud, _ b: GaussianCloud) -> Float {
    var worst: Float = 0
    worst = max(worst, maxAbsDifference(a.positions, b.positions))
    worst = max(worst, maxAbsDifference(a.logScales, b.logScales))
    worst = max(worst, maxAbsDifference(a.rotations, b.rotations))
    worst = max(worst, maxAbsDifference(a.opacityLogits, b.opacityLogits))
    worst = max(worst, maxAbsDifference(a.sh, b.sh))
    return worst
}

private func maxAbsDifference(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count else { return .infinity }
    var worst: Float = 0
    for i in 0..<a.count {
        let delta: Float = abs(a[i] - b[i])
        worst = max(worst, delta)
    }
    return worst
}


extension TrainingConfiguration {
    /// The configuration most trainer tests were written against: the Phase 4 regularizer set and none of the
    /// v1.2 features (exposure compensation, MCMC, anti-aliasing, pose refinement), which are on in the app defaults
    /// since 2026-11-08 but are not implemented by the CPU reference trainer.
    static func testBaseline() -> TrainingConfiguration {
        var configuration = TrainingConfiguration()
        configuration.regularizers = [.depthPearson, .depthL1, .anisotropy, .isolationPrune]
        configuration.appearance.isEnabled = false
        configuration.mcmc.isEnabled = false
        configuration.antiAliasing.isEnabled = false
        configuration.poseRefinement.isEnabled = false
        return configuration
    }
}
