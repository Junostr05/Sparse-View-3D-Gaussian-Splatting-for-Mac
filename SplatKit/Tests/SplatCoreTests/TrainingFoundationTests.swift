import Foundation
import Testing
import SplatCore

@Suite struct TrainingFoundationTests {

    // MARK: - TrainingConfiguration

    @Test func defaultsMatchArchitecture() {
        let config = TrainingConfiguration()
        #expect(config.shDegree == 1)
        #expect(config.densification.opacityResetInterval == 0)
        #expect(config.lambdaDSSIM == 0.2)
        #expect(config.iterations == 10_000)
        #expect(config.regularizers == TrainingConfiguration.defaultRegularizers)
        #expect(config.heldOutViews.isEmpty)
    }

    @Test func decodesWithMissingKeysUsingDefaults() throws {
        let decoder = JSONDecoder()
        let empty = try decoder.decode(TrainingConfiguration.self, from: Data("{}".utf8))
        #expect(empty == TrainingConfiguration())

        let partial = try decoder.decode(
            TrainingConfiguration.self,
            from: Data("{\"iterations\": 50}".utf8)
        )
        var expected = TrainingConfiguration()
        expected.iterations = 50
        #expect(partial == expected)
    }

    @Test func configurationRoundTripsThroughJSON() throws {
        var config = TrainingConfiguration()
        config.iterations = 1_234
        config.shDegree = 3
        config.background = SIMD3<Float>(0.1, 0.2, 0.3)
        config.seed = 42
        config.densification.gradientThreshold = 0.5
        config.learningRates.opacity = 0.07

        let data = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(TrainingConfiguration.self, from: data)
        #expect(decoded == config)
    }

    @Test func activeSHDegreeSteps() {
        var config = TrainingConfiguration()
        config.shDegree = 3
        config.shDegreeInterval = 1_000
        #expect(config.activeSHDegree(iteration: 0) == 0)
        #expect(config.activeSHDegree(iteration: 999) == 0)
        #expect(config.activeSHDegree(iteration: 1_000) == 1)
        #expect(config.activeSHDegree(iteration: 5_000) == 3)
    }

    @Test func positionLearningRateDecaysExponentially() {
        var config = TrainingConfiguration()
        config.iterations = 1_000
        let radius: Float = 2

        let start = config.learningRates(iteration: 0, sceneRadius: radius)
        let end = config.learningRates(iteration: 1_000, sceneRadius: radius)
        let middle = config.learningRates(iteration: 500, sceneRadius: radius)

        #expect(abs(start.position - 3.2e-4) / 3.2e-4 < 1e-4)
        #expect(abs(end.position - 3.2e-6) / 3.2e-6 < 1e-4)
        #expect(abs(middle.position - 3.2e-5) / 3.2e-5 < 1e-4)

        // Non-position rates are not scaled by the scene radius.
        #expect(start.logScale == config.learningRates.logScale)
        #expect(start.rotation == config.learningRates.rotation)
        #expect(start.opacity == config.learningRates.opacity)
        #expect(start.shDC == config.learningRates.shDC)
        #expect(start.shRest == config.learningRates.shRest)
    }

    // MARK: - Seeded randomness

    @Test func seededGeneratorIsDeterministic() {
        var zeroSeed = SeededGenerator(seed: 0)
        #expect(zeroSeed.next() == 0xE220_A839_7B1D_CDAF)

        var first = SeededGenerator(seed: 7)
        var second = SeededGenerator(seed: 7)
        var other = SeededGenerator(seed: 8)
        var sameSeedA: [UInt64] = []
        var sameSeedB: [UInt64] = []
        var differentSeed: [UInt64] = []
        for _ in 0..<16 {
            sameSeedA.append(first.next())
            sameSeedB.append(second.next())
            differentSeed.append(other.next())
        }
        #expect(sameSeedA == sameSeedB)
        #expect(sameSeedA != differentSeed)
    }

    @Test func splatRandomNormalHasUnitMoments() {
        let count = 100_000
        var sum = 0.0
        var sumSquares = 0.0
        for index in 0..<count {
            let value = Double(SplatRandom.normal(seed: 1234, index: UInt32(index), stream: 0))
            sum += value
            sumSquares += value * value
        }
        let mean = sum / Double(count)
        let variance = sumSquares / Double(count) - mean * mean
        #expect(abs(mean) < 0.02)
        #expect(abs(variance - 1) < 0.03)
    }

    @Test func splatRandomUniformIsStrictlyInsideUnitInterval() {
        for index in 0..<100_000 {
            let value = SplatRandom.uniform(seed: 99, index: UInt32(index), stream: 3)
            #expect(value > 0 && value < 1)
        }
    }

    // MARK: - SceneExtent

    @Test func extentFromCameraCenters() {
        let extent = SceneExtent.estimate(
            cameraCenters: [SIMD3<Float>(1, 0, 0), SIMD3<Float>(-1, 0, 0)],
            points: []
        )
        #expect(abs(extent.center.x) < 1e-6)
        #expect(abs(extent.center.y) < 1e-6)
        #expect(abs(extent.center.z) < 1e-6)
        #expect(abs(extent.radius - 1.1) < 1e-5)
    }

    @Test func extentFallsBackToPointsForSingleCamera() {
        // 101 points on the diagonal x = y = z in [-1, 1] (step 0.02).
        var points: [SIMD3<Float>] = []
        for i in 0...100 {
            let value = -1 + Float(i) * 0.02
            points.append(SIMD3<Float>(value, value, value))
        }
        let extent = SceneExtent.estimate(
            cameraCenters: [SIMD3<Float>(5, 5, 5)],
            points: points
        )
        // Mean of symmetric points is the origin.
        #expect(abs(extent.center.x) < 1e-5)
        #expect(abs(extent.center.y) < 1e-5)
        #expect(abs(extent.center.z) < 1e-5)
        // 5-95 % box spans [-0.9, 0.9] per axis; half its diagonal is 0.9 * sqrt(3).
        let expectedRadius = Float(0.9 * 3.0.squareRoot())
        #expect(abs(extent.radius - expectedRadius) < 1e-4)
    }

    @Test func extentFallsBackToOneWithoutData() {
        let extent = SceneExtent.estimate(cameraCenters: [], points: [])
        #expect(extent.radius == 1)
    }

    // MARK: - TrainingMemoryBudget

    @Test func bytesPerGaussianValues() {
        #expect(TrainingMemoryBudget.bytesPerGaussian(shDegree: 0) == 576)
        #expect(TrainingMemoryBudget.bytesPerGaussian(shDegree: 1) == 756)
        #expect(TrainingMemoryBudget.bytesPerGaussian(shDegree: 3) == 1476)
    }

    @Test func reservedImageBytesUsesLargestView() {
        let sizes = [SIMD2<Int>(10, 20), SIMD2<Int>(4, 50)]
        // 16 * (200 + 200) + 144 * (10 * 50) with wmax = 10, hmax = 50.
        let expected = 16 * (200 + 200) + 144 * (10 * 50)
        #expect(TrainingMemoryBudget.reservedImageBytes(viewSizes: sizes) == expected)
        #expect(TrainingMemoryBudget.reservedImageBytes(viewSizes: []) == 0)
    }

    @Test func maxGaussiansRespectsConfiguredCapAndBudget() {
        // Tiny working set: budget-bound.
        let small = TrainingMemoryBudget.maxGaussians(
            workingSetBytes: 1_000_000,
            shDegree: 1,
            reservedBytes: 0,
            configuredMax: 1_500_000
        )
        #expect(small == Int(0.6 * 1_000_000 / 756.0))
        #expect(small < 1_500_000)

        // Huge working set: configured cap wins.
        let huge = TrainingMemoryBudget.maxGaussians(
            workingSetBytes: 16 * 1024 * 1024 * 1024,
            shDegree: 1,
            reservedBytes: 0,
            configuredMax: 1_500_000
        )
        #expect(huge == 1_500_000)

        // Reserved memory exceeds the budget: nothing fits.
        let none = TrainingMemoryBudget.maxGaussians(
            workingSetBytes: 1_000_000,
            shDegree: 1,
            reservedBytes: 2_000_000,
            configuredMax: 1_500_000
        )
        #expect(none == 0)
    }
}
