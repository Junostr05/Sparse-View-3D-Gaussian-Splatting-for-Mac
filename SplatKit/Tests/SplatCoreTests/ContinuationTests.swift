import Foundation
import Testing
import SplatCore

@Suite struct ContinuationTests {

    @Test func settingsDefaults() {
        let settings = ContinuationSettings()
        #expect(settings.roundIterations == 5_000)
        #expect(settings.isEndless == false)
        #expect(settings.addGaussians == false)
        #expect(settings.growthFactor == 1.5)
        #expect(settings.maxGaussians == 3_000_000)
    }

    @Test func settingsCodableRoundTrip() throws {
        var settings = ContinuationSettings()
        settings.roundIterations = 1_234
        settings.isEndless = true
        settings.addGaussians = true
        settings.growthFactor = 2.25
        settings.maxGaussians = 2_000_000

        let data = try JSONEncoder().encode(settings)
        let decoded = try JSONDecoder().decode(ContinuationSettings.self, from: data)
        #expect(decoded == settings)
    }

    @Test func shDegreeIsKeptForLoadedCloud() {
        var base = TrainingConfiguration()
        base.shDegree = 1
        base.shDegreeInterval = 1_000

        let config = base.continued(from: 100_000, shDegree: 3, round: 0,
                                    settings: ContinuationSettings())
        #expect(config.shDegree == 3)
        #expect(config.shDegreeInterval == 0)
        #expect(config.activeSHDegree(iteration: 0) == 3)
        #expect(config.activeSHDegree(iteration: 50_000) == 3)
    }

    @Test func positionLearningRateIsLowered() {
        var base = TrainingConfiguration()
        base.learningRates.positionInitial = 1.6e-4
        base.learningRates.positionFinal = 1.6e-6

        let config = base.continued(from: 100_000, shDegree: 3, round: 0,
                                    settings: ContinuationSettings())
        let tolerance: Float = 1e-11
        let expectedInitial: Float = 1.6e-6 * 10
        #expect(abs(config.learningRates.positionInitial - expectedInitial) <= tolerance)
        #expect(config.learningRates.positionFinal == base.learningRates.positionFinal)
        #expect(config.learningRates.logScale == base.learningRates.logScale)
        #expect(config.learningRates.rotation == base.learningRates.rotation)
        #expect(config.learningRates.opacity == base.learningRates.opacity)
        #expect(config.learningRates.shDC == base.learningRates.shDC)
        #expect(config.learningRates.shRest == base.learningRates.shRest)

        let resolved = config.learningRates(iteration: 0, sceneRadius: 1)
        let expectedPosition: Float = 1.6e-5
        #expect(abs(resolved.position - expectedPosition) <= 1e-11)
    }

    @Test func densificationDisabledWithoutAddGaussians() {
        let config = TrainingConfiguration().continued(from: 100_000, shDegree: 3, round: 0,
                                                       settings: ContinuationSettings())
        #expect(!config.densification.isEnabled)
        #expect(config.densification.startIteration == 0)
        #expect(config.densification.stopIteration == 0)
    }

    @Test func densificationEnabledWithAddGaussians() {
        var settings = ContinuationSettings()
        settings.addGaussians = true

        var base = TrainingConfiguration()
        base.iterations = 10_000
        let baseThreshold: Float = base.densification.gradientThreshold
        let config = base.continued(from: 100_000, shDegree: 3, round: 0, settings: settings)

        #expect(config.iterations == 5_000)
        #expect(config.densification.isEnabled)
        #expect(config.densification.startIteration == 0)
        #expect(config.densification.stopIteration == 4_000)
        #expect(config.densification.interval == base.densification.interval)
        let expectedThreshold: Float = baseThreshold * 0.5
        #expect(abs(config.densification.gradientThreshold - expectedThreshold) <= 1e-12)
    }

    @Test func densificationIntervalDefaultsWhenNonPositive() {
        var settings = ContinuationSettings()
        settings.addGaussians = true

        var base = TrainingConfiguration()
        base.densification.interval = 0
        let config = base.continued(from: 100_000, shDegree: 3, round: 0, settings: settings)
        #expect(config.densification.interval == 100)
        #expect(config.densification.isEnabled)
    }

    @Test func maxGaussiansGrowsByFactor() {
        var settings = ContinuationSettings()
        settings.addGaussians = true

        let config = TrainingConfiguration().continued(from: 1_000_000, shDegree: 3, round: 0,
                                                       settings: settings)
        #expect(config.maxGaussians == 1_500_000)
    }

    @Test func maxGaussiansRoundsUp() {
        var settings = ContinuationSettings()
        settings.addGaussians = true

        // 3 × 1.5 = 4.5, rounded up to 5.
        let config = TrainingConfiguration().continued(from: 3, shDegree: 3, round: 0,
                                                       settings: settings)
        #expect(config.maxGaussians == 5)
    }

    @Test func maxGaussiansClampsToCeiling() {
        var settings = ContinuationSettings()
        settings.addGaussians = true

        let capped = TrainingConfiguration().continued(from: 2_500_000, shDegree: 3, round: 0,
                                                       settings: settings)
        #expect(capped.maxGaussians == 3_000_000)

        // Already above the ceiling: never reduce the count.
        let above = TrainingConfiguration().continued(from: 4_000_000, shDegree: 3, round: 0,
                                                      settings: settings)
        #expect(above.maxGaussians == 4_000_000)
    }

    @Test func maxGaussiansWithoutAddGaussiansKeepsBaseOrCount() {
        var base = TrainingConfiguration()
        base.maxGaussians = 1_500_000

        let small = base.continued(from: 1_000_000, shDegree: 3, round: 0,
                                   settings: ContinuationSettings())
        #expect(small.maxGaussians == 1_500_000)

        let large = base.continued(from: 2_000_000, shDegree: 3, round: 0,
                                   settings: ContinuationSettings())
        #expect(large.maxGaussians == 2_000_000)
    }

    @Test func earlyStoppingAndHeldOutViewsAreCleared() {
        var base = TrainingConfiguration()
        base.earlyStopping.isEnabled = true
        base.heldOutViews = ["IMG_0001", "IMG_0002"]

        let config = base.continued(from: 100_000, shDegree: 3, round: 0,
                                    settings: ContinuationSettings())
        #expect(!config.earlyStopping.isEnabled)
        #expect(config.heldOutViews.isEmpty)
    }

    @Test func seedDiffersPerRound() {
        var base = TrainingConfiguration()
        base.seed = 42

        let settings = ContinuationSettings()
        let round0 = base.continued(from: 100_000, shDegree: 3, round: 0, settings: settings)
        let round1 = base.continued(from: 100_000, shDegree: 3, round: 1, settings: settings)
        let round2 = base.continued(from: 100_000, shDegree: 3, round: 2, settings: settings)

        #expect(round0.seed == 42)
        #expect(round1.seed == 43)
        #expect(round2.seed == 44)
        #expect(round0.seed != round1.seed)
        #expect(round1.seed != round2.seed)
    }

    @Test func baseConfigurationIsNotMutated() {
        var base = TrainingConfiguration()
        base.heldOutViews = ["IMG_0001"]
        base.earlyStopping.isEnabled = true
        let snapshot = base

        var settings = ContinuationSettings()
        settings.addGaussians = true
        _ = base.continued(from: 2_000_000, shDegree: 3, round: 4, settings: settings)

        #expect(base == snapshot)
        #expect(base.heldOutViews == ["IMG_0001"])
        #expect(base.earlyStopping.isEnabled)
    }

    @Test func otherSettingsAreUnchanged() {
        var base = TrainingConfiguration()
        base.regularizers = [.depthPearson, .anisotropy]
        base.logInterval = 7
        base.evalInterval = 123
        base.previewInterval = 456
        base.maxImageSide = 800
        base.lambdaDSSIM = 0.3
        base.configurationRevision = 2

        let config = base.continued(from: 100_000, shDegree: 3, round: 0,
                                    settings: ContinuationSettings())
        #expect(config.regularizers == base.regularizers)
        #expect(config.logInterval == 7)
        #expect(config.evalInterval == 123)
        #expect(config.previewInterval == 456)
        #expect(config.maxImageSide == 800)
        #expect(config.lambdaDSSIM == 0.3)
        #expect(config.configurationRevision == base.configurationRevision)
        #expect(config.depthPrior == base.depthPrior)
        #expect(config.artifacts == base.artifacts)
    }
}
