import Metal
import Testing
import SplatCore
@testable import SplatMetal

/// GPU backward pass against the PyTorch-autograd golden gradients (Tools/golden).
@Suite(.enabled(if: TestEnvironment.hasMetal))
struct GoldenBackwardTests {
    @Test(arguments: ["single", "multi", "rotated", "occluded"])
    func backwardMatchesGolden(name: String) throws {
        let golden = try GoldenCase.load(name)
        let cloud = try golden.cloud()
        let camera = golden.camera()
        let context = try makeTestContext()
        let renderer = try SplatRenderer(context: context, cloud: cloud)

        // The backward pass needs the state of a preceding render call.
        _ = try renderer.renderImage(camera: camera, settings: golden.settings())
        let gradients = try renderer.backward(imageGradient: golden.imageGradientPixels)

        func check(_ label: String, _ actual: [Float], _ expected: [Float]) {
            #expect(actual.count == expected.count, "\(name).\(label): length")
            guard actual.count == expected.count else { return }
            var worst: Float = 0
            var worstIndex = 0
            for i in 0..<actual.count {
                let allowed = 1e-4 + 1e-3 * abs(expected[i])
                let excess = abs(actual[i] - expected[i]) - allowed
                if !(excess <= worst) { worst = excess.isNaN ? .infinity : excess; worstIndex = i }
            }
            #expect(worst <= 0, "\(name).\(label)[\(worstIndex)]: gpu \(actual[worstIndex]) golden \(expected[worstIndex])")
        }
        check("positions", gradients.positions, golden.gradients.positions)
        check("logScales", gradients.logScales, golden.gradients.logScales)
        check("rotations", gradients.rotations, golden.gradients.rotations)
        check("opacityLogits", gradients.opacityLogits, golden.gradients.opacityLogits)
        check("sh", gradients.sh, golden.gradients.sh)
    }
}
