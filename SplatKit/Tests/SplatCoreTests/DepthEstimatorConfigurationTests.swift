import Foundation
import Testing
import SplatCore

@Suite struct DepthEstimatorConfigurationTests {

    @Test func defaultEstimatorIsDepthAnythingV2Small() {
        #expect(DepthPriorSettings().estimator == .depthAnythingV2Small)
        #expect(TrainingConfiguration().depthPrior.estimator == .depthAnythingV2Small)
    }

    @Test func estimatorKindsAreListedInPickerOrder() {
        #expect(DepthEstimatorKind.allCases == [.depthAnythingV2Small, .moge2Small])
        #expect(DepthEstimatorKind.moge2Small.rawValue == "moge2Small")
    }

    @Test func missingEstimatorKeyFallsBackToTheDefault() throws {
        let settings = try JSONDecoder().decode(DepthPriorSettings.self, from: Data("{\"pearsonWeight\": 0.2}".utf8))
        #expect(settings.estimator == .depthAnythingV2Small)
        #expect(settings.pearsonWeight == 0.2)
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
        #expect(configuration.depthPrior.estimator == .depthAnythingV2Small)
    }
}
