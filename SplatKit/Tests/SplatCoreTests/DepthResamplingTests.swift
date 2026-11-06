import Foundation
import Testing
@testable import SplatCore

private func plane(width: Int, height: Int, z: Float) -> DepthMap {
    DepthMap(width: width, height: height, kind: .metric,
             values: [Float](repeating: z, count: width * height))
}

@Suite struct DepthResamplingTests {

    @Test func identityResampleIsExact() throws {
        let map = DepthMap(width: 3, height: 2, kind: .relativeDepth,
                           values: [1, 2, 3, 4, 5, 6], confidence: [2, 1, 0, 2, 1, 2])
        let resampled = try map.resampled(width: 3, height: 2)
        #expect(resampled == map)
    }

    @Test func resamplesLiDARToImageSize() throws {
        let lidar = plane(width: 4, height: 3, z: 2)
        let resampled = try lidar.resampled(width: 16, height: 12)
        #expect(resampled.width == 16)
        #expect(resampled.height == 12)
        #expect(resampled.values.count == 16 * 12)
        for value in resampled.values {
            #expect(abs(value - 2) < 1e-6)
        }
    }

    @Test func rejectsAspectMismatch() {
        let lidar = plane(width: 4, height: 3, z: 2)
        #expect(throws: DepthResampleError.aspectMismatch(source: SIMD2(4, 3),
                                                          target: SIMD2(16, 9))) {
            try lidar.resampled(width: 16, height: 9)
        }
    }

    @Test func allowAspectChangeSkipsTheCheck() throws {
        let lidar = plane(width: 4, height: 3, z: 2)
        let resampled = try lidar.resampled(width: 16, height: 9, allowAspectChange: true)
        #expect(resampled.width == 16)
        #expect(resampled.height == 9)
    }

    @Test func edgeAwareBilinearKeepsDiscontinuity() throws {
        var values: [Float] = []
        for _ in 0..<3 {
            values.append(contentsOf: [1, 1, 10, 10])
        }
        let step = DepthMap(width: 4, height: 3, kind: .metric, values: values)
        let resampled = try step.resampled(width: 8, height: 6)
        for value in resampled.values {
            #expect(!(value > 1.2 && value < 9))
        }
    }

    @Test func invalidTapsAreNotInterpolated() throws {
        // An invalid source pixel must never be averaged into a neighbour: every output value is
        // either the valid depth 4 (within float tolerance) or 0.
        let values: [Float] = [4, 0, 4, 4, 4, 4, 4, 4, 4]
        let map = DepthMap(width: 3, height: 3, kind: .metric, values: values)
        let resampled = try map.resampled(width: 6, height: 6)
        for value in resampled.values {
            let isFour = abs(value - 4) < 1e-5
            #expect(isFour || value == 0)
        }
    }

    @Test func isolatedInvalidPixelFallsBackToNearestValid() throws {
        // Output (1, 1) of a 2x2 -> 4x4 map maps to source (0.25, 0.25). Its rounded tap is the
        // invalid (0, 0), and the valid taps (1, 0) and (0, 1) are equidistant, so raster order
        // picks (1, 0). The result is exactly 3, never a blend with the invalid 0.
        let map = DepthMap(width: 2, height: 2, kind: .metric, values: [0, 3, 3, 3])
        let resampled = try map.resampled(width: 4, height: 4)
        #expect(resampled.values[5] == 3)
        // Output (0, 0) clamps onto the invalid pixel only, so there is no valid tap: 0.
        #expect(resampled.values[0] == 0)
    }

    @Test func confidenceIsResampledNearest() throws {
        let map = DepthMap(width: 2, height: 1, kind: .metric, values: [1, 1],
                           confidence: [0, 2])
        // 2x1 to 4x1 changes the aspect ratio, so it needs the model-output escape hatch.
        let resampled = try map.resampled(width: 4, height: 1, allowAspectChange: true)
        #expect(resampled.confidence == [0, 0, 2, 2])
        #expect(resampled.values[0] == 0)
        #expect(resampled.values[3] == 1)
    }

    @Test func rotationsAreInverse() {
        let map = DepthMap(width: 3, height: 2, kind: .relativeInverseDepth,
                           values: [1, 2, 3, 4, 5, 6], confidence: [0, 1, 2, 0, 1, 2],
                           isEstimated: true)
        let clockwise = map.rotatedClockwise()
        #expect(clockwise.width == 2)
        #expect(clockwise.height == 3)
        #expect(clockwise.rotatedCounterClockwise() == map)
        #expect(clockwise.rotatedCounterClockwise().isEstimated)
        #expect(map.rotatedCounterClockwise().rotatedClockwise() == map)
    }
}
