import Foundation
import Testing
import SplatCore
@testable import SplatIO

/// v1.5: the optional weight plane of `.sdpt` files and the view groups of the multi-view depth estimator.
@Suite struct DepthWeightsCodecTests {

    @Test func roundTripsWeightsWithAndWithoutConfidence() throws {
        for withConfidence in [false, true] {
            let values: [Float] = [0.5, 1, 1.5, 0, 2, 2.5]
            let weights: [Float] = [1, 6.5, 12.25, 0, 3, 1.125]
            let confidence: [UInt8]? = withConfidence ? [0, 1, 2, 2, 1, 0] : nil
            let map = DepthMap(width: 3, height: 2, kind: .relativeDepth, values: values,
                               confidence: confidence, isEstimated: true, weights: weights)
            let data: Data = DepthMapCodec.encode(map)
            let expectedLength: Int = DepthMapCodec.headerSize + 6 * 4 + (withConfidence ? 6 : 0) + 6 * 4
            #expect(data.count == expectedLength)
            #expect([UInt8](data)[7] & DepthMapCodec.weightsFlag != 0)
            let decoded = try DepthMapCodec.decode(data)
            #expect(decoded == map)
        }
    }

    @Test func mapsWithoutWeightsKeepTheOriginalLayout() throws {
        let map = DepthMap(width: 2, height: 2, kind: .relativeInverseDepth, values: [1, 2, 3, 4], isEstimated: true)
        let data: Data = DepthMapCodec.encode(map)
        #expect(data.count == DepthMapCodec.headerSize + 16)
        #expect([UInt8](data)[7] & DepthMapCodec.weightsFlag == 0)
        #expect(try DepthMapCodec.decode(data).weights == nil)
    }

    @Test func truncatedWeightPlaneIsRejected() {
        let map = DepthMap(width: 2, height: 2, kind: .relativeDepth, values: [1, 2, 3, 4], weights: [1, 1, 1, 1])
        let data: Data = DepthMapCodec.encode(map)
        let cut: Data = data.prefix(data.count - 4)
        #expect(throws: DepthMapError.truncated(expected: data.count, actual: cut.count)) {
            try DepthMapCodec.decode(cut)
        }
    }

    @Test func depthGroupsCoverEveryViewOnceInOrder() {
        #expect(DepthAnything3Math.depthGroups(count: 0).isEmpty)
        #expect(DepthAnything3Math.depthGroups(count: 1) == [[0]])
        #expect(DepthAnything3Math.depthGroups(count: 10) == [Array(0..<10)])
        #expect(DepthAnything3Math.depthGroups(count: 16) == [Array(0..<16)])
        // 17 views: two groups of 9 and 8, never a single leftover view.
        #expect(DepthAnything3Math.depthGroups(count: 17) == [Array(0..<9), Array(9..<17)])
        for count in 1...70 {
            let groups: [[Int]] = DepthAnything3Math.depthGroups(count: count)
            let flattened: [Int] = groups.flatMap { $0 }
            #expect(flattened == Array(0..<count))
            let sizes: [Int] = groups.map { $0.count }
            #expect(sizes.allSatisfy { $0 <= 16 })
            #expect((sizes.max() ?? 0) - (sizes.min() ?? 0) <= 1)
            if count >= 2 {
                #expect(sizes.allSatisfy { $0 >= 2 })
            }
        }
    }
}
