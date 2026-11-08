import Foundation
import Testing
import SplatCore

/// One Gaussian of a synthetic cloud.
private struct TestGaussian {
    var position: SIMD3<Float>
    var scale: SIMD3<Float>
    var opacity: Float
}

/// Builds a degree-0 cloud from explicit Gaussians.
private func makeCloud(_ gaussians: [TestGaussian]) throws(GaussianCloudError) -> GaussianCloud {
    var positions: [Float] = []
    var logScales: [Float] = []
    var rotations: [Float] = []
    var opacityLogits: [Float] = []
    var sh: [Float] = []
    for gaussian in gaussians {
        positions += [gaussian.position.x, gaussian.position.y, gaussian.position.z]
        logScales += [log(gaussian.scale.x), log(gaussian.scale.y), log(gaussian.scale.z)]
        rotations += [1, 0, 0, 0]
        let p: Float = gaussian.opacity
        opacityLogits.append(log(p / (1 - p)))
        sh += [0, 0, 0]
    }
    return try GaussianCloud(shDegree: 0,
                             positions: positions,
                             logScales: logScales,
                             rotations: rotations,
                             opacityLogits: opacityLogits,
                             sh: sh)
}

/// Deterministic pseudo-random numbers in 0..<1 (a 64-bit LCG), so that the synthetic clouds never change.
private struct Lcg {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> Float {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Float(state >> 40) / Float(1 << 24)
    }
}

/// A dense blob of `count` ordinary Gaussians inside the unit ball, with isotropic small scales.
private func blob(count: Int) -> [TestGaussian] {
    var random = Lcg(seed: 7)
    var result: [TestGaussian] = []
    result.reserveCapacity(count)
    while result.count < count {
        let x: Float = 2 * random.next() - 1
        let y: Float = 2 * random.next() - 1
        let z: Float = 2 * random.next() - 1
        guard x * x + y * y + z * z <= 1 else { continue }
        let scale: Float = 0.01 * (0.9 + 0.2 * random.next())
        result.append(TestGaussian(position: SIMD3<Float>(x, y, z),
                                   scale: SIMD3<Float>(scale, scale, scale),
                                   opacity: 0.8))
    }
    return result
}

/// The blob plus planted artifacts, and the indices of each planted group.
private struct PlantedScene {
    var gaussians: [TestGaussian]
    var blobIndices: [Int]
    var hugeIndices: [Int]
    var transparentIndices: [Int]
    var needleIndices: [Int]
    var floaterIndices: [Int]

    var plantedIndices: [Int] {
        hugeIndices + transparentIndices + needleIndices + floaterIndices
    }
}

private func plantedScene() -> PlantedScene {
    var gaussians: [TestGaussian] = blob(count: 2_000)
    let blobIndices: [Int] = Array(0..<gaussians.count)
    var hugeIndices: [Int] = []
    var transparentIndices: [Int] = []
    var needleIndices: [Int] = []
    var floaterIndices: [Int] = []

    for index in 0..<5 {
        hugeIndices.append(gaussians.count)
        let offset: Float = 0.1 * Float(index)
        gaussians.append(TestGaussian(position: SIMD3<Float>(offset, -offset, 0.2),
                                      scale: SIMD3<Float>(repeating: 0.5),
                                      opacity: 0.8))
    }
    for index in 0..<5 {
        transparentIndices.append(gaussians.count)
        let offset: Float = 0.15 * Float(index)
        gaussians.append(TestGaussian(position: SIMD3<Float>(-0.5 + offset, 0.4, -0.3),
                                      scale: SIMD3<Float>(repeating: 0.01),
                                      opacity: 0.01))
    }
    for index in 0..<5 {
        needleIndices.append(gaussians.count)
        let offset: Float = 0.1 * Float(index)
        gaussians.append(TestGaussian(position: SIMD3<Float>(0.3, offset - 0.2, -0.5),
                                      scale: SIMD3<Float>(0.05, 0.0005, 0.0005),
                                      opacity: 0.8))
    }
    for index in 0..<5 {
        floaterIndices.append(gaussians.count)
        gaussians.append(TestGaussian(position: SIMD3<Float>(8 + Float(index), -8, 8),
                                      scale: SIMD3<Float>(repeating: 0.01),
                                      opacity: 0.8))
    }
    return PlantedScene(gaussians: gaussians,
                        blobIndices: blobIndices,
                        hugeIndices: hugeIndices,
                        transparentIndices: transparentIndices,
                        needleIndices: needleIndices,
                        floaterIndices: floaterIndices)
}

/// Criteria with only `enabled` switched on, all at medium sensitivity.
private func criteria(only enabled: [CleanupCriterion]) -> CleanupCriteria {
    var rules: [CleanupCriterion: CleanupRule] = [:]
    for criterion in CleanupCriterion.allCases {
        rules[criterion] = CleanupRule(isEnabled: enabled.contains(criterion))
    }
    return CleanupCriteria(rules: rules)
}

@Suite struct CleanupAssistantTests {
    @Test func everyPlantedArtifactIsFoundAndTheBlobIsUntouched() throws {
        let scene: PlantedScene = plantedScene()
        let cloud: GaussianCloud = try makeCloud(scene.gaussians)
        let proposal: CleanupProposal = CleanupAssistant.propose(cloud: cloud, criteria: .standard)

        for index in scene.plantedIndices {
            #expect(proposal.selection[index], "planted Gaussian \(index) was not proposed")
        }
        for index in scene.blobIndices {
            #expect(!proposal.selection[index], "blob Gaussian \(index) was proposed")
        }
        #expect(proposal.count == scene.plantedIndices.count)
        #expect(!proposal.wasLimited)
    }

    @Test func eachCriterionFindsItsOwnPlantedGroup() throws {
        let scene: PlantedScene = plantedScene()
        let cloud: GaussianCloud = try makeCloud(scene.gaussians)
        let expected: [CleanupCriterion: [Int]] = [
            .huge: scene.hugeIndices,
            .transparent: scene.transparentIndices,
            .needle: scene.needleIndices,
            .isolated: scene.floaterIndices,
        ]
        for (criterion, indices) in expected {
            let proposal: CleanupProposal = CleanupAssistant.propose(cloud: cloud,
                                                                     criteria: criteria(only: [criterion]))
            let result: CleanupCriterionResult? = proposal.results.first { $0.criterion == criterion }
            #expect(result != nil, "no result for \(criterion)")
            for index in indices {
                #expect(proposal.selection[index], "\(criterion) missed Gaussian \(index)")
            }
            #expect(proposal.count == indices.count, "\(criterion) proposed \(proposal.count)")
        }
    }

    @Test func outsideCoreIsOffByDefaultAndFindsTheFloatersWhenOn() throws {
        let scene: PlantedScene = plantedScene()
        let cloud: GaussianCloud = try makeCloud(scene.gaussians)
        let off: CleanupProposal = CleanupAssistant.propose(cloud: cloud, criteria: .standard)
        #expect(!off.results.contains { $0.criterion == .outsideCore })

        let on: CleanupProposal = CleanupAssistant.propose(cloud: cloud, criteria: criteria(only: [.outsideCore]))
        for index in scene.floaterIndices {
            #expect(on.selection[index])
        }
        for index in scene.blobIndices {
            #expect(!on.selection[index], "blob Gaussian \(index) lies outside the core")
        }
    }

    @Test func disabledCriterionProposesNothing() throws {
        let scene: PlantedScene = plantedScene()
        let cloud: GaussianCloud = try makeCloud(scene.gaussians)
        let proposal: CleanupProposal = CleanupAssistant.propose(cloud: cloud,
                                                                 criteria: criteria(only: []))
        #expect(proposal.isEmpty)
        #expect(proposal.results.isEmpty)
    }

    @Test func sensitivityChangesTheTransparentThreshold() throws {
        // Opacity 0.07 is above the medium limit (0.05) and below the high one (0.10).
        let gaussians: [TestGaussian] = blob(count: 500) + [
            TestGaussian(position: SIMD3<Float>(0, 0, 0), scale: SIMD3<Float>(repeating: 0.01), opacity: 0.07),
        ]
        let cloud: GaussianCloud = try makeCloud(gaussians)
        var medium: CleanupCriteria = criteria(only: [.transparent])
        #expect(!CleanupAssistant.propose(cloud: cloud, criteria: medium).selection[500])
        medium[.transparent] = CleanupRule(isEnabled: true, sensitivity: .high)
        #expect(CleanupAssistant.propose(cloud: cloud, criteria: medium).selection[500])
    }

    @Test func safetyCapKeepsTheMostExtremeTransparentGaussians() throws {
        // 400 transparent Gaussians among 1 000: the criterion flags far more than 15 %.
        var gaussians: [TestGaussian] = blob(count: 600)
        for index in 0..<400 {
            let opacity: Float = 0.001 + 0.00001 * Float(index)
            gaussians.append(TestGaussian(position: SIMD3<Float>(0, 0, 0),
                                          scale: SIMD3<Float>(repeating: 0.01),
                                          opacity: opacity))
        }
        let cloud: GaussianCloud = try makeCloud(gaussians)
        let proposal: CleanupProposal = CleanupAssistant.propose(cloud: cloud,
                                                                 criteria: criteria(only: [.transparent]))
        let limit: Int = gaussians.count * CleanupAssistant.safetyPercent / 100
        #expect(proposal.safetyLimit == limit)
        #expect(proposal.count == limit)
        #expect(proposal.count <= gaussians.count * 15 / 100)
        #expect(proposal.wasLimited)
        #expect(proposal.results.first?.matchCount == 400)
        #expect(proposal.results.first?.wasLimited == true)

        // The kept Gaussians are the most transparent: the lowest opacities, which are the first ones added.
        for index in 600..<(600 + limit) {
            #expect(proposal.selection[index], "extreme Gaussian \(index) was dropped")
        }
        #expect(!proposal.selection[600 + limit])
    }

    @Test func unionIsLimitedWhenCriteriaTogetherExceedTheCap() throws {
        // 400 transparent Gaussians (capped to 120 on their own, the lowest opacities kept) and one huge Gaussian.
        // The union has 121 candidates against a limit of 120, so the least extreme of them is dropped: the 120th
        // transparent one (margin about 23), not the huge one (margin 30).
        var gaussians: [TestGaussian] = []
        for index in 0..<400 {
            gaussians.append(TestGaussian(position: SIMD3<Float>(0.5, 0.5, 0.5),
                                          scale: SIMD3<Float>(repeating: 0.01),
                                          opacity: 0.001 + 0.00001 * Float(index)))
        }
        for index in 0..<400 {
            gaussians.append(TestGaussian(position: SIMD3<Float>(0, 0, 0),
                                          scale: SIMD3<Float>(repeating: 0.01),
                                          opacity: 0.8 - 0.0001 * Float(index)))
        }
        gaussians.append(TestGaussian(position: SIMD3<Float>(1, 1, 1), scale: SIMD3<Float>(repeating: 3), opacity: 0.8))
        let cloud: GaussianCloud = try makeCloud(gaussians)
        let proposal: CleanupProposal = CleanupAssistant.propose(cloud: cloud, criteria: .standard)
        let limit: Int = gaussians.count * CleanupAssistant.safetyPercent / 100
        #expect(proposal.safetyLimit == limit)
        #expect(proposal.count == limit)
        #expect(proposal.fraction <= 0.15)
        #expect(proposal.unionWasLimited)
        #expect(proposal.wasLimited)
        #expect(proposal.selection[0], "the most transparent Gaussian must be kept")
        #expect(!proposal.selection[119], "the least extreme transparent candidate must be dropped")
        #expect(proposal.selection[gaussians.count - 1], "the huge Gaussian outranks the 120th transparent one")
        let shares: Int = proposal.results.reduce(0) { $0 + $1.count }
        #expect(shares >= proposal.count, "every proposed Gaussian belongs to at least one criterion share")
    }

    @Test func proposalIsDeterministic() throws {
        let scene: PlantedScene = plantedScene()
        let cloud: GaussianCloud = try makeCloud(scene.gaussians)
        let first: CleanupProposal = CleanupAssistant.propose(cloud: cloud, criteria: .standard)
        let second: CleanupProposal = CleanupAssistant.propose(cloud: cloud, criteria: .standard)
        #expect(first == second)
    }

    @Test func emptyAndTinyCloudsProposeNothingUnsafe() throws {
        let empty: GaussianCloud = try makeCloud([])
        let emptyProposal: CleanupProposal = CleanupAssistant.propose(cloud: empty, criteria: .standard)
        #expect(emptyProposal.isEmpty)
        #expect(emptyProposal.cloudCount == 0)
        #expect(emptyProposal.fraction == 0)

        // Five Gaussians: the limit is 0, so nothing may be proposed even though one is transparent.
        let tiny: GaussianCloud = try makeCloud(blob(count: 4) + [
            TestGaussian(position: SIMD3<Float>(0, 0, 0), scale: SIMD3<Float>(repeating: 0.01), opacity: 0.001),
        ])
        let tinyProposal: CleanupProposal = CleanupAssistant.propose(cloud: tiny, criteria: .standard)
        #expect(tinyProposal.safetyLimit == 0)
        #expect(tinyProposal.isEmpty)
    }
}
