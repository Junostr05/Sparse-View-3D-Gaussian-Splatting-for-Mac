import Foundation
import SplatCore
@testable import SplatDepth
import Testing

/// v1.5 normal prior: MoGe normal conversion (mask, normalisation, portrait rotation, resampling) and attaching
/// normals to depth priors.
@Suite struct NormalPreparationTests {

    @Test func landscapeNormalsAreNormalisedMaskedAndResampled() {
        // 2 × 1 model output: a valid normal of length 2 and a masked one.
        let raw: [Float] = [0, 0, -2, 1, 0, 0]
        let normals = MoGeDepthConversion.viewNormals(raw, mask: [0.9, 0.1], modelWidth: 2, modelHeight: 1,
                                                      isPortrait: false, viewWidth: 4, viewHeight: 2)
        #expect(normals.count == 8)
        #expect(normals[0] == SIMD3<Float>(0, 0, -1))
        #expect(normals[1] == SIMD3<Float>(0, 0, -1))
        #expect(normals[2] == .zero)
        #expect(normals[7] == .zero)
    }

    /// A portrait view runs through the model rotated clockwise. A normal pointing to the view's right (+x) is
    /// seen by the model pointing down (+y'), so a model normal (0, 1, 0) must come back as (1, 0, 0), and the
    /// pixel grid must rotate back as `DepthMap.rotatedCounterClockwise` does.
    @Test func portraitNormalsAreRotatedBackWithTheirPixels() {
        // Model frame 3 × 2 (landscape); only model pixel (2, 0) has a normal.
        var raw: [Float] = [Float](repeating: 0, count: 18)
        var mask: [Float] = [Float](repeating: 0, count: 6)
        raw[2 * 3 + 1] = 1
        mask[2] = 1
        let normals = MoGeDepthConversion.viewNormals(raw, mask: mask, modelWidth: 3, modelHeight: 2,
                                                      isPortrait: true, viewWidth: 2, viewHeight: 3)
        // Counter-clockwise: output (x, y) takes source (W − 1 − y, x); source (2, 0) lands at output (0, 0).
        #expect(normals[0] == SIMD3<Float>(1, 0, 0))
        for index in 1..<6 {
            #expect(normals[index] == .zero)
        }
        // The depth map rotation agrees on where the pixel goes.
        var values: [Float] = [Float](repeating: 1, count: 6)
        values[2] = 5
        let rotated = DepthMap(width: 3, height: 2, kind: .relativeDepth, values: values).rotatedCounterClockwise()
        #expect(rotated.values[0] == 5)
    }

    private struct FixedNormals: NormalProvider {
        let normals: [String: [SIMD3<Float>]]
        func normalMap(for view: TrainingView) throws -> [SIMD3<Float>]? {
            normals[view.name]
        }
    }

    @Test func attachSetsNormalsOnlyOnViewsWithAMatchingDepthPrior() {
        func view(_ name: String, withPrior: Bool) -> TrainingView {
            let camera = Camera(width: 2, height: 2, fx: 2, fy: 2, cx: 1, cy: 1,
                                rotation: Matrix3.identity, translation: SIMD3<Float>(0, 0, 0))
            let image = RGBImage(width: 2, height: 2, pixels: [SIMD3<Float>](repeating: .zero, count: 4))
            let prior: DepthPrior? = withPrior
                ? DepthPrior(width: 2, height: 2, correlationTarget: [1, 2, 3, 4], domain: .depth) : nil
            return TrainingView(name: name, camera: camera, image: image, depth: prior)
        }
        let up = [SIMD3<Float>](repeating: SIMD3<Float>(0, 0, -1), count: 4)
        let dataset = TrainingDataset(views: [view("a", withPrior: true), view("b", withPrior: false),
                                              view("c", withPrior: true), view("d", withPrior: true)])
        let provider = FixedNormals(normals: ["a": up, "b": up, "c": [SIMD3<Float>(0, 0, -1)]])
        let result = NormalPreparation.attach(dataset, provider: provider)
        #expect(result.attached == 1)
        #expect(result.failed == 2)   // "c" has the wrong size, "d" has no normals
        #expect(result.dataset.views[0].depth?.normals == up)
        #expect(result.dataset.views[1].depth == nil)
        #expect(result.dataset.views[2].depth?.normals == nil)
        #expect(result.dataset.views[3].depth?.normals == nil)
    }
}
