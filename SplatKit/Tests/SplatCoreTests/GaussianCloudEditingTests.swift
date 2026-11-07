import Foundation
import Testing
import SplatCore

/// Cloud in which every stored value encodes its Gaussian index and array slot, so any misplaced
/// element is visible in the survivors.
private func indexedCloud(count: Int, shDegree: Int) throws(GaussianCloudError) -> GaussianCloud {
    let k: Int = GaussianCloud.coefficientCount(degree: shDegree)
    var positions: [Float] = []
    var logScales: [Float] = []
    var rotations: [Float] = []
    var opacity: [Float] = []
    var sh: [Float] = []
    for i in 0..<count {
        for c in 0..<3 {
            positions.append(Float(10 * i + c))
            logScales.append(Float(100 * i + c))
        }
        for c in 0..<4 {
            rotations.append(Float(1000 * i + c))
        }
        opacity.append(Float(i) + 0.5)
        for j in 0..<(k * 3) {
            sh.append(Float(10_000 * i + j))
        }
    }
    return try GaussianCloud(
        shDegree: shDegree,
        positions: positions,
        logScales: logScales,
        rotations: rotations,
        opacityLogits: opacity,
        sh: sh
    )
}

/// Expected arrays after keeping only `keptIndices`, built independently from the indexed encoding.
private func expectedArrays(keeping keptIndices: [Int], shDegree: Int)
    -> (positions: [Float], logScales: [Float], rotations: [Float], opacity: [Float], sh: [Float]) {
    let k: Int = GaussianCloud.coefficientCount(degree: shDegree)
    var positions: [Float] = []
    var logScales: [Float] = []
    var rotations: [Float] = []
    var opacity: [Float] = []
    var sh: [Float] = []
    for i in keptIndices {
        for c in 0..<3 {
            positions.append(Float(10 * i + c))
            logScales.append(Float(100 * i + c))
        }
        for c in 0..<4 {
            rotations.append(Float(1000 * i + c))
        }
        opacity.append(Float(i) + 0.5)
        for j in 0..<(k * 3) {
            sh.append(Float(10_000 * i + j))
        }
    }
    return (positions, logScales, rotations, opacity, sh)
}

@Suite struct GaussianCloudEditingTests {
    @Test(arguments: [0, 1, 2, 3])
    func removingKeepsSurvivorsInOrderInEveryArray(shDegree: Int) throws {
        let cloud = try indexedCloud(count: 7, shDegree: shDegree)
        let mask = SelectionMask(count: 7, indices: [1, 4])
        let edited = try cloud.removing(mask)
        let expected = expectedArrays(keeping: [0, 2, 3, 5, 6], shDegree: shDegree)

        #expect(edited.shDegree == shDegree)
        #expect(edited.count == 5)
        #expect(edited.positions == expected.positions)
        #expect(edited.logScales == expected.logScales)
        #expect(edited.rotations == expected.rotations)
        #expect(edited.opacityLogits == expected.opacity)
        #expect(edited.sh == expected.sh)
    }

    @Test func removingAcrossWordBoundaryIsCorrect() throws {
        let cloud = try indexedCloud(count: 130, shDegree: 0)
        let mask = SelectionMask(count: 130, indices: [0, 63, 64, 129])
        let edited = try cloud.removing(mask)
        let keep = Array(1..<129).filter { $0 != 63 && $0 != 64 }
        let expected = expectedArrays(keeping: keep, shDegree: 0)
        #expect(edited.count == keep.count)
        #expect(edited.positions == expected.positions)
        #expect(edited.sh == expected.sh)
    }

    @Test func removingEverythingYieldsEmptyCloud() throws {
        for shDegree in [0, 3] {
            let cloud = try indexedCloud(count: 5, shDegree: shDegree)
            let edited = try cloud.removing(SelectionMask.all(count: 5))
            #expect(edited.count == 0)
            #expect(edited.shDegree == shDegree)
            #expect(edited.positions.isEmpty)
            #expect(edited.logScales.isEmpty)
            #expect(edited.rotations.isEmpty)
            #expect(edited.opacityLogits.isEmpty)
            #expect(edited.sh.isEmpty)
        }
    }

    @Test func removingNothingReturnsIdenticalArrays() throws {
        let cloud = try indexedCloud(count: 9, shDegree: 2)
        let edited = try cloud.removing(SelectionMask(count: 9))
        #expect(edited.count == 9)
        #expect(edited.positions == cloud.positions)
        #expect(edited.logScales == cloud.logScales)
        #expect(edited.rotations == cloud.rotations)
        #expect(edited.opacityLogits == cloud.opacityLogits)
        #expect(edited.sh == cloud.sh)
    }

    @Test func wrongMaskCountThrowsInconsistentLengths() throws {
        let cloud = try indexedCloud(count: 4, shDegree: 1)
        do {
            _ = try cloud.removing(SelectionMask(count: 5))
            Issue.record("removing with a mask of the wrong count should throw")
        } catch {
            #expect(error == .inconsistentLengths)
        }
    }

    @Test func emptyCloudAcceptsEmptyMask() throws {
        let cloud = try indexedCloud(count: 0, shDegree: 0)
        let edited = try cloud.removing(SelectionMask(count: 0))
        #expect(edited.count == 0)
    }
}
