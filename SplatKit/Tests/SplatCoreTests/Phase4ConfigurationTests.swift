import Foundation
import Testing
import SplatCore

@Suite struct Phase4ConfigurationTests {

    @Test func defaultsMatchArchitecture() {
        let config = TrainingConfiguration()
        #expect(config.regularizers == [.depthPearson, .depthL1, .anisotropy, .isolationPrune])
        #expect(RegularizerKind.allCases.count == 6)
        #expect(config.artifacts.maxAxisRatio == 10)
        #expect(config.earlyStopping.minimumViews == 4)
        #expect(config.depthPrior.pearsonWeight == 0.05)
        #expect(config.shDegree == 1)
        #expect(config.densification.opacityResetInterval == 0)
        #expect(config.configurationRevision == TrainingConfiguration.currentRevision)
        #expect(TrainingConfiguration.currentRevision == 2)
        #expect(!config.isEnabled(.pseudoViews))
        #expect(!config.isEnabled(.scaleCap))
        #expect(!config.earlyStopping.isEnabled)
        #expect(config.denseInit.isEnabled)
    }

    @Test func phase3ManifestGetsNewDefaults() throws {
        let config = try JSONDecoder().decode(
            TrainingConfiguration.self,
            from: Data("{\"regularizers\": [], \"iterations\": 50}".utf8)
        )
        #expect(config.regularizers == TrainingConfiguration.defaultRegularizers)
        #expect(config.iterations == 50)
        #expect(config.configurationRevision == 2)
    }

    @Test func revision2KeepsStoredRegularizers() throws {
        let config = try JSONDecoder().decode(
            TrainingConfiguration.self,
            from: Data("{\"configurationRevision\": 2, \"regularizers\": [\"anisotropy\"]}".utf8)
        )
        #expect(config.regularizers == [.anisotropy])
        #expect(config.isEnabled(.anisotropy))
        #expect(!config.isEnabled(.depthL1))
    }

    @Test func phase4ConfigurationRoundTrips() throws {
        var config = TrainingConfiguration()
        config.regularizers = [.depthPearson, .pseudoViews]
        config.depthPrior.minimumPixels = 128
        config.pseudoViews.perPair = 2
        config.artifacts.isolationNeighbors = 6
        config.denseInit.maxPoints = 1_000
        config.earlyStopping.patience = 7
        config.earlyStopping.minDelta = 0.01

        let data = try JSONEncoder().encode(config)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains("configurationRevision"))

        let decoded = try JSONDecoder().decode(TrainingConfiguration.self, from: data)
        #expect(decoded == config)
        #expect(decoded.configurationRevision == 2)
    }

    @Test func depthMapValidity() {
        let invalid = DepthMap(width: 2, height: 2, kind: .metric,
                               values: [0, -1, Float.nan, Float.infinity])
        for index in 0..<4 {
            #expect(!invalid.isValid(index))
        }

        let valid = DepthMap(width: 2, height: 2, kind: .metric,
                             values: [2.5, 2.5, 2.5, 2.5],
                             confidence: [0, 1, 2, 1])
        #expect(!valid.isValid(0))
        #expect(valid.isValid(0, minimumConfidence: 0))
        #expect(valid.isValid(1))
        #expect(valid.isValid(2))
    }

    @Test func priorComputesL1Normalizer() {
        let target: [Float] = [1, 1, 1, 1]
        let depths: [Float] = [1, 2, 3, 0]

        let measured = DepthPrior(width: 2, height: 2, correlationTarget: target,
                                  domain: .depth, metricDepth: depths, isMeasuredMetric: true)
        #expect(measured.l1Normalizer == 2)

        let estimated = DepthPrior(width: 2, height: 2, correlationTarget: target,
                                   domain: .depth, metricDepth: depths, isMeasuredMetric: false)
        #expect(estimated.l1Normalizer == 0)
    }

    @Test func reservedDepthBytesFormula() {
        let views: [SIMD2<Int>] = [
            SIMD2<Int>(10, 10), SIMD2<Int>(10, 10), SIMD2<Int>(10, 10), SIMD2<Int>(10, 10),
        ]
        let pseudo: [SIMD2<Int>] = [SIMD2<Int>(5, 5), SIMD2<Int>(5, 5)]
        // 4 views: 8 * 100 each = 3200; largest view 16 * 100 = 1600; 2 pseudo: 4 * 25 each = 200.
        let expected: Int = 3_200 + 1_600 + 200
        #expect(TrainingMemoryBudget.reservedDepthBytes(viewSizes: views, pseudoViewSizes: pseudo) == expected)
        #expect(TrainingMemoryBudget.reservedDepthBytes(viewSizes: [], pseudoViewSizes: []) == 0)
    }

    // MARK: - Nested settings structs: missing keys fall back to defaults

    @Test func depthPriorSettingsEmptyObjectGivesDefaults() throws {
        let decoded = try JSONDecoder().decode(DepthPriorSettings.self, from: Data("{}".utf8))
        #expect(decoded == DepthPriorSettings())
    }

    @Test func depthPriorSettingsKeepsOneKnownKey() throws {
        let decoded = try JSONDecoder().decode(
            DepthPriorSettings.self, from: Data("{\"minimumPixels\": 128}".utf8))
        var expected = DepthPriorSettings()
        expected.minimumPixels = 128
        #expect(decoded == expected)
        #expect(decoded.pearsonWeight == 0.05)
    }

    @Test func pseudoViewSettingsEmptyObjectGivesDefaults() throws {
        let decoded = try JSONDecoder().decode(PseudoViewSettings.self, from: Data("{}".utf8))
        #expect(decoded == PseudoViewSettings())
    }

    @Test func pseudoViewSettingsKeepsOneKnownKey() throws {
        let decoded = try JSONDecoder().decode(
            PseudoViewSettings.self, from: Data("{\"perPair\": 5}".utf8))
        var expected = PseudoViewSettings()
        expected.perPair = 5
        #expect(decoded == expected)
        #expect(decoded.maxAngleDegrees == 60)
    }

    @Test func artifactRegularizerSettingsEmptyObjectGivesDefaults() throws {
        let decoded = try JSONDecoder().decode(ArtifactRegularizerSettings.self, from: Data("{}".utf8))
        #expect(decoded == ArtifactRegularizerSettings())
    }

    @Test func artifactRegularizerSettingsKeepsOneKnownKey() throws {
        let decoded = try JSONDecoder().decode(
            ArtifactRegularizerSettings.self, from: Data("{\"isolationNeighbors\": 6}".utf8))
        var expected = ArtifactRegularizerSettings()
        expected.isolationNeighbors = 6
        #expect(decoded == expected)
        #expect(decoded.maxAxisRatio == 10)
    }

    @Test func denseInitSettingsEmptyObjectGivesDefaults() throws {
        let decoded = try JSONDecoder().decode(DenseInitSettings.self, from: Data("{}".utf8))
        #expect(decoded == DenseInitSettings())
    }

    @Test func denseInitSettingsKeepsOneKnownKey() throws {
        let decoded = try JSONDecoder().decode(
            DenseInitSettings.self, from: Data("{\"isEnabled\": false}".utf8))
        var expected = DenseInitSettings()
        expected.isEnabled = false
        #expect(decoded == expected)
        #expect(decoded.maxPoints == 150_000)
    }

    @Test func earlyStoppingPolicyEmptyObjectGivesDefaults() throws {
        let decoded = try JSONDecoder().decode(EarlyStoppingPolicy.self, from: Data("{}".utf8))
        #expect(decoded == EarlyStoppingPolicy())
    }

    @Test func earlyStoppingPolicyKeepsOneKnownKey() throws {
        let decoded = try JSONDecoder().decode(
            EarlyStoppingPolicy.self, from: Data("{\"patience\": 9}".utf8))
        var expected = EarlyStoppingPolicy()
        expected.patience = 9
        #expect(decoded == expected)
        #expect(decoded.minimumViews == 4)
    }

    @Test func configurationWithEmptyNestedObjectsDecodes() throws {
        let json = """
        {
          "configurationRevision": 2,
          "iterations": 77,
          "regularizers": ["depthPearson", "pseudoViews"],
          "depthPrior": {},
          "pseudoViews": {},
          "artifacts": {},
          "denseInit": {},
          "earlyStopping": {}
        }
        """
        let decoded = try JSONDecoder().decode(TrainingConfiguration.self, from: Data(json.utf8))
        #expect(decoded.iterations == 77)
        #expect(decoded.regularizers == [.depthPearson, .pseudoViews])
        #expect(decoded.depthPrior == DepthPriorSettings())
        #expect(decoded.pseudoViews == PseudoViewSettings())
        #expect(decoded.artifacts == ArtifactRegularizerSettings())
        #expect(decoded.denseInit == DenseInitSettings())
        #expect(decoded.earlyStopping == EarlyStoppingPolicy())
    }
}

@Suite struct V12ConfigurationTests {
    @Test func oldManifestWithoutV12KeysDecodesWithFeaturesOff() throws {
        let json = Data("{\"iterations\": 123}".utf8)
        let configuration = try JSONDecoder().decode(TrainingConfiguration.self, from: json)
        #expect(configuration.iterations == 123)
        #expect(!configuration.appearance.isEnabled)
        #expect(!configuration.mcmc.isEnabled)
        #expect(!configuration.antiAliasing.isEnabled)
        #expect(!configuration.poseRefinement.isEnabled)
    }

    @Test func v12SettingsRoundTrip() throws {
        var configuration = TrainingConfiguration()
        configuration.appearance.isEnabled = true
        configuration.mcmc.isEnabled = true
        configuration.mcmc.growthFraction = 0.1
        configuration.antiAliasing.isEnabled = true
        configuration.poseRefinement.isEnabled = true
        let data = try JSONEncoder().encode(configuration)
        let decoded = try JSONDecoder().decode(TrainingConfiguration.self, from: data)
        #expect(decoded == configuration)
    }
}
