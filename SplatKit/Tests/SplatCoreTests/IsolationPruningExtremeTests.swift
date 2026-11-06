import Testing
@testable import SplatCore

/// Regression for a crash seen in the app: a Gaussian that drifted far away during training made the
/// k-th neighbour search convert an infinite ratio to Int.
@Suite struct IsolationPruningExtremeTests {
    private static func cloud(count: Int) -> [SIMD3<Float>] {
        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(count)
        var state: UInt64 = 12345
        for _ in 0..<count {
            var values: [Float] = []
            for _ in 0..<3 {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                let unit = Float(state >> 40) / Float(1 << 24)
                values.append(unit * 2 - 1)
            }
            positions.append(SIMD3<Float>(values[0], values[1], values[2]))
        }
        return positions
    }

    @Test func hugeOutlierDoesNotTrapAndIsPruned() {
        var positions = Self.cloud(count: 400)
        positions.append(SIMD3<Float>(1e30, 0, 0))
        let mask = IsolationPruning.pruneMask(positions: positions, neighbors: 4, factor: 8,
                                              maxFraction: 0.05, sampleCount: 2_048, seed: 7)
        #expect(mask.count == positions.count)
        #expect(mask[positions.count - 1])
    }

    @Test func outliersAtTheFloatLimitDoNotTrap() {
        var positions = Self.cloud(count: 400)
        positions.append(SIMD3<Float>(3e38, -3e38, 1e37))
        positions.append(SIMD3<Float>(-1e25, 1e25, 0))
        let mask = IsolationPruning.pruneMask(positions: positions, neighbors: 4, factor: 8,
                                              maxFraction: 0.05, sampleCount: 2_048, seed: 7)
        #expect(mask[positions.count - 1])
        #expect(mask[positions.count - 2])
    }

    @Test func nonFinitePositionsDoNotTrapAndArePruned() {
        var positions = Self.cloud(count: 400)
        positions.append(SIMD3<Float>(.nan, 0, 0))
        positions.append(SIMD3<Float>(.infinity, 1, 1))
        let mask = IsolationPruning.pruneMask(positions: positions, neighbors: 4, factor: 8,
                                              maxFraction: 0.05, sampleCount: 2_048, seed: 7)
        #expect(mask[positions.count - 1])
        #expect(mask[positions.count - 2])
    }

    @Test func kthNeighborDistanceOfAFarPointIsInfiniteNotATrap() {
        var positions = Self.cloud(count: 50)
        positions.append(SIMD3<Float>(1e30, 0, 0))
        let grid = SpatialHashGrid(points: positions, cellSize: 0.01)
        let distance = grid.kthNeighborDistance(of: positions.count - 1, k: 4, maxRings: 64)
        #expect(distance == .infinity)
    }
}
