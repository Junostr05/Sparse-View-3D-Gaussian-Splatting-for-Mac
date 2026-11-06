import Testing
@testable import SplatCore

struct TrainingPresetTests {
    @Test func presetsAreOrderedByCost() {
        #expect(TrainingPreset.fast.maxImageSide < TrainingPreset.standard.maxImageSide)
        #expect(TrainingPreset.standard.maxImageSide < TrainingPreset.quality.maxImageSide)
        #expect(TrainingPreset.fast.iterations < TrainingPreset.standard.iterations)
        #expect(TrainingPreset.standard.iterations < TrainingPreset.quality.iterations)
    }

    @Test func applySetsSchedulesInsideTheRun() {
        for preset in TrainingPreset.allCases {
            let configuration = preset.configuration()
            #expect(configuration.iterations == preset.iterations)
            #expect(configuration.maxImageSide == preset.maxImageSide)
            #expect(configuration.densification.isEnabled)
            #expect(configuration.densification.stopIteration < configuration.iterations)
            #expect(configuration.densification.pruneScaleStartIteration < configuration.densification.stopIteration)
            #expect(configuration.shDegreeInterval * configuration.shDegree <= configuration.iterations)
        }
    }

    @Test func applyKeepsUnrelatedSettings() {
        var configuration = TrainingConfiguration()
        configuration.shDegree = 3
        configuration.maxGaussians = 123_456
        configuration.seed = 9
        TrainingPreset.fast.apply(to: &configuration)
        #expect(configuration.shDegree == 3)
        #expect(configuration.maxGaussians == 123_456)
        #expect(configuration.seed == 9)
    }

    @Test func standardPresetMatchesRecordedDefaultsForIterations() {
        #expect(TrainingPreset.default == .standard)
        #expect(TrainingPreset.standard.iterations == TrainingConfiguration().iterations)
    }
}
