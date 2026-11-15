import Foundation
import Testing
import SplatCore

@Suite struct DepthEstimatorConfigurationTests {

    @Test func defaultEstimatorIsAutomatic() {
        #expect(DepthPriorSettings().estimator == .automatic)
        #expect(TrainingConfiguration().depthPrior.estimator == .automatic)
    }

    @Test func estimatorKindsAreListedInPickerOrder() {
        #expect(DepthEstimatorKind.allCases == [.automatic, .depthAnythingV2Small, .moge2Small, .depthAnything3])
        #expect(DepthEstimatorKind.automatic.rawValue == "automatic")
        #expect(DepthEstimatorKind.moge2Small.rawValue == "moge2Small")
        #expect(DepthEstimatorKind.depthAnything3.rawValue == "depthAnything3")
    }

    @Test func confidenceWeightingIsOffByDefaultAndDecodes() throws {
        #expect(!DepthPriorSettings().confidenceWeighting)
        #expect(!TrainingConfiguration().depthPrior.confidenceWeighting)
        let missing = try JSONDecoder().decode(DepthPriorSettings.self, from: Data("{\"pearsonWeight\": 0.2}".utf8))
        #expect(!missing.confidenceWeighting)
        let stored = try JSONDecoder().decode(DepthPriorSettings.self,
                                              from: Data("{\"estimator\": \"depthAnything3\", \"confidenceWeighting\": true}".utf8))
        #expect(stored.confidenceWeighting)
        #expect(stored.estimator == .depthAnything3)
        var configuration = TrainingConfiguration()
        configuration.depthPrior.confidenceWeighting = true
        let decoded = try JSONDecoder().decode(TrainingConfiguration.self, from: try JSONEncoder().encode(configuration))
        #expect(decoded == configuration)
    }

    @Test func missingEstimatorKeyFallsBackToTheDefault() throws {
        let settings = try JSONDecoder().decode(DepthPriorSettings.self, from: Data("{\"pearsonWeight\": 0.2}".utf8))
        #expect(settings.estimator == .automatic)
        #expect(settings.pearsonWeight == 0.2)
    }

    @Test func storedExplicitEstimatorIsKeptWhenDecoding() throws {
        // A manifest that stored an explicit kind keeps it: automatic is only the default for a missing key.
        let settings = try JSONDecoder().decode(DepthPriorSettings.self,
                                                from: Data("{\"estimator\": \"depthAnythingV2Small\"}".utf8))
        #expect(settings.estimator == .depthAnythingV2Small)
        let automatic = try JSONDecoder().decode(DepthPriorSettings.self, from: Data("{\"estimator\": \"automatic\"}".utf8))
        #expect(automatic.estimator == .automatic)
    }

    @Test func automaticResolvesToDepthAnything3WhenAvailable() {
        #expect(DepthEstimatorKind.automatic.resolved(isDepthAnything3Available: true) == .depthAnything3)
    }

    @Test func automaticResolvesToDepthAnythingV2SmallWhenDepthAnything3IsMissing() {
        #expect(DepthEstimatorKind.automatic.resolved(isDepthAnything3Available: false) == .depthAnythingV2Small)
    }

    @Test func explicitEstimatorsResolveToThemselves() {
        for availability in [true, false] {
            #expect(DepthEstimatorKind.depthAnythingV2Small.resolved(isDepthAnything3Available: availability) == .depthAnythingV2Small)
            #expect(DepthEstimatorKind.moge2Small.resolved(isDepthAnything3Available: availability) == .moge2Small)
            #expect(DepthEstimatorKind.depthAnything3.resolved(isDepthAnything3Available: availability) == .depthAnything3)
        }
    }

    @Test func storedEstimatorDecodes() throws {
        let settings = try JSONDecoder().decode(DepthPriorSettings.self, from: Data("{\"estimator\": \"moge2Small\"}".utf8))
        #expect(settings.estimator == .moge2Small)
    }

    @Test func estimatorRoundTripsThroughConfiguration() throws {
        var configuration = TrainingConfiguration()
        configuration.depthPrior.estimator = .moge2Small
        let data = try JSONEncoder().encode(configuration)
        #expect(String(decoding: data, as: UTF8.self).contains("\"estimator\""))
        let decoded = try JSONDecoder().decode(TrainingConfiguration.self, from: data)
        #expect(decoded.depthPrior.estimator == .moge2Small)
        #expect(decoded == configuration)
    }

    @Test func manifestWithoutDepthPriorKeepsTheDefaultEstimator() throws {
        let configuration = try JSONDecoder().decode(TrainingConfiguration.self, from: Data("{\"iterations\": 10}".utf8))
        #expect(configuration.depthPrior.estimator == .automatic)
    }
}
