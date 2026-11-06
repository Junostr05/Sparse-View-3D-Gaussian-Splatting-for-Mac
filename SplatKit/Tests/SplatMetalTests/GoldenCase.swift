import Foundation
import SplatCore

/// One golden case written by `Tools/golden/generate.py`: a scene, camera, settings,
/// the PyTorch reference image and the parameter gradients for a fixed upstream gradient.
/// All arrays are flat and Float32. `image`, `imageGradient` are row-major W·H·3.
/// The Metal test target keeps an identical copy of this file.
struct GoldenCase: Decodable, Sendable {
    struct Gradients: Decodable, Sendable {
        let positions: [Float]
        let logScales: [Float]
        let rotations: [Float]
        let opacityLogits: [Float]
        let sh: [Float]
    }

    let name: String
    let width: Int
    let height: Int
    let fx: Float
    let fy: Float
    let cx: Float
    let cy: Float
    /// World → camera, 9 values, row-major.
    let rotation: [Float]
    /// 3 values.
    let translation: [Float]
    /// 3 values.
    let background: [Float]
    let shDegree: Int
    let shDegreeLimit: Int
    let positions: [Float]
    let logScales: [Float]
    /// (w, x, y, z) per Gaussian.
    let rotations: [Float]
    let opacityLogits: [Float]
    /// Layout `[gaussian][coefficient][channel]`.
    let sh: [Float]
    let image: [Float]
    let imageGradient: [Float]
    let gradients: Gradients
    /// Phase 4 depth cases only: row-major W·H, `Draw = Σ αᵢ Tᵢ zᵢ` (gsplat RGB+D, un-normalised).
    let renderedDepth: [Float]?
    /// Phase 4 depth cases only: row-major W·H, `A = 1 − T_final`.
    let renderedAlpha: [Float]?
    /// Phase 4 depth cases only: row-major W·H·2, `(∂L/∂Draw, ∂L/∂A)` per pixel.
    let depthGradient: [Float]?

    enum LoadError: Error, Equatable {
        case missingResource(String)
    }

    static func load(_ name: String) throws -> GoldenCase {
        guard let url = Bundle.module.url(forResource: "golden_\(name)",
                                          withExtension: "json",
                                          subdirectory: "Fixtures") else {
            throw LoadError.missingResource(name)
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(GoldenCase.self, from: data)
    }

    func cloud() throws -> GaussianCloud {
        try GaussianCloud(shDegree: shDegree,
                          positions: positions,
                          logScales: logScales,
                          rotations: rotations,
                          opacityLogits: opacityLogits,
                          sh: sh)
    }

    func camera() -> Camera {
        Camera(
            width: width,
            height: height,
            fx: fx,
            fy: fy,
            cx: cx,
            cy: cy,
            rotation: Matrix3(rows: [
                SIMD3(rotation[0], rotation[1], rotation[2]),
                SIMD3(rotation[3], rotation[4], rotation[5]),
                SIMD3(rotation[6], rotation[7], rotation[8]),
            ]),
            translation: SIMD3(translation[0], translation[1], translation[2])
        )
    }

    func settings() -> RasterSettings {
        RasterSettings(
            shDegreeLimit: shDegreeLimit,
            background: SIMD3(background[0], background[1], background[2])
        )
    }

    var imageGradientPixels: [SIMD3<Float>] {
        Self.pixels(from: imageGradient)
    }

    var imagePixels: [SIMD3<Float>] {
        Self.pixels(from: image)
    }

    private static func pixels(from flat: [Float]) -> [SIMD3<Float>] {
        stride(from: 0, to: flat.count, by: 3).map { i in
            SIMD3(flat[i], flat[i + 1], flat[i + 2])
        }
    }
}

/// Phase 4 Pearson/L1 depth loss references (plan Task 6), written by
/// `Tools/golden/generate.py --only phase4`. Shared inputs are row-major W·H. Each case holds
/// the prior in its own domain and the reference outputs; gradients are with respect to
/// `draw` and `alpha`. The Metal test target keeps an identical copy of this file.
struct PearsonGolden: Decodable, Sendable {
    struct Case: Decodable, Sendable {
        /// "depth" or "inverseDepth".
        let domain: String
        /// Prior in `domain`; values ≤ 0 are invalid.
        let target: [Float]
        let loss: Double
        /// ρ, 0 when degenerate.
        let pearson: Double
        let l1: Double
        let validPixels: Int
        let degenerate: Bool
        let gradDraw: [Float]
        let gradA: [Float]
    }

    let width: Int
    let height: Int
    let lambdaPearson: Float
    let lambdaL1: Float
    let alphaMask: Float
    let nearPlane: Float
    let minimumPixels: Int
    let isMeasuredMetric: Bool
    let l1Normalizer: Float
    let draw: [Float]
    let alpha: [Float]
    /// Measured camera-z depth for the L1 term; values ≤ 0 are invalid.
    let metricDepth: [Float]
    let cases: [Case]

    static func load(_ name: String) throws -> Self {
        guard let url = Bundle.module.url(forResource: "golden_\(name)",
                                          withExtension: "json",
                                          subdirectory: "Fixtures") else {
            throw GoldenCase.LoadError.missingResource(name)
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Self.self, from: data)
    }
}

/// Phase 4 anisotropy hinge references (plan Task 7): `λ/n · Σ max(0, s_max/s_min − maxRatio)`.
/// Written by `Tools/golden/generate.py --only phase4`. The Metal test target keeps an identical
/// copy of this file.
struct AnisotropyGolden: Decodable, Sendable {
    let count: Int
    let weight: Float
    let maxRatio: Float
    /// Row-major count × 3 log-scales.
    let logScales: [Float]
    let loss: Double
    /// Row-major count × 3.
    let gradLogScales: [Float]

    static func load(_ name: String) throws -> Self {
        guard let url = Bundle.module.url(forResource: "golden_\(name)",
                                          withExtension: "json",
                                          subdirectory: "Fixtures") else {
            throw GoldenCase.LoadError.missingResource(name)
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Self.self, from: data)
    }
}
