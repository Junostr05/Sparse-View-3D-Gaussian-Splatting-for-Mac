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
