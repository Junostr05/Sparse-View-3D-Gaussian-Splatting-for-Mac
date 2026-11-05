import Foundation
import Testing
import SplatCore

@Suite struct GoldenLoaderTests {
    static let names = ["single", "multi", "rotated", "occluded"]

    @Test func loadsAllCases() throws {
        for name in Self.names {
            let golden = try GoldenCase.load(name)
            #expect(golden.name == name)

            let cloud = try golden.cloud()
            let n = cloud.count
            #expect(n > 0)
            #expect(golden.positions.count == 3 * n)
            #expect(golden.rotations.count == 4 * n)
            #expect(golden.gradients.positions.count == 3 * n)
            #expect(golden.gradients.logScales.count == 3 * n)
            #expect(golden.gradients.rotations.count == 4 * n)
            #expect(golden.gradients.opacityLogits.count == n)
            #expect(golden.gradients.sh.count == golden.sh.count)

            let pixelCount = golden.width * golden.height
            #expect(golden.image.count == 3 * pixelCount)
            #expect(golden.imageGradient.count == 3 * pixelCount)
            #expect(golden.imagePixels.count == pixelCount)
            #expect(golden.imageGradientPixels.count == pixelCount)

            #expect(golden.rotation.count == 9)
            #expect(golden.translation.count == 3)
            #expect(golden.background.count == 3)
        }
    }

    @Test(arguments: names) func forwardMatchesGolden(name: String) throws {
        let golden = try GoldenCase.load(name)
        let cloud = try golden.cloud()
        let image = CPURasterizer.render(cloud, camera: golden.camera(), settings: golden.settings())

        let expected = golden.imagePixels
        #expect(image.pixels.count == expected.count)

        var maxDiff: Float = 0
        for (actual, reference) in zip(image.pixels, expected) {
            let d = actual - reference
            maxDiff = max(maxDiff, Swift.abs(d.x), Swift.abs(d.y), Swift.abs(d.z))
        }
        #expect(maxDiff <= 1e-4, "max abs diff \(maxDiff) for \(name)")
    }
}
