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

    @Test func phase4GoldensDecode() throws {
        for name in ["depth_single", "depth_occluded"] {
            let golden = try GoldenCase.load(name)
            let n = try golden.cloud().count
            #expect(golden.gradients.positions.count == 3 * n)

            let pixelCount = golden.width * golden.height
            let depth = try #require(golden.renderedDepth)
            let alpha = try #require(golden.renderedAlpha)
            let depthGradient = try #require(golden.depthGradient)
            #expect(depth.count == pixelCount, "renderedDepth for \(name)")
            #expect(alpha.count == pixelCount, "renderedAlpha for \(name)")
            #expect(depthGradient.count == 2 * pixelCount, "depthGradient for \(name)")
        }

        let pearson = try PearsonGolden.load("pearson")
        let pixelCount = pearson.width * pearson.height
        #expect(pearson.draw.count == pixelCount)
        #expect(pearson.alpha.count == pixelCount)
        #expect(pearson.metricDepth.count == pixelCount)
        #expect(pearson.cases.map(\.domain) == ["depth", "inverseDepth"])
        for pearsonCase in pearson.cases {
            #expect(pearsonCase.target.count == pixelCount)
            #expect(pearsonCase.gradDraw.count == pixelCount)
            #expect(pearsonCase.gradA.count == pixelCount)
        }

        let anisotropy = try AnisotropyGolden.load("anisotropy")
        #expect(anisotropy.count == 16)
        #expect(anisotropy.logScales.count == 3 * anisotropy.count)
        #expect(anisotropy.gradLogScales.count == 3 * anisotropy.count)
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
