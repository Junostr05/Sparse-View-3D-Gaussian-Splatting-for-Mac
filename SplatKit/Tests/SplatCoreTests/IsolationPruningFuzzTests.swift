import Testing
@testable import SplatCore

/// Regression for an app crash ("Float value cannot be converted to Int because it is either infinite or
/// NaN" in `SpatialHashGrid.kthNeighborDistance`): diverged Gaussians with NaN, infinite or overflowing
/// coordinates must never trap the isolation pruning.
@Suite struct IsolationPruningFuzzTests {
    @Test func randomHostileCloudsNeverTrap() {
        var state: UInt64 = 99
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(state >> 40) / Float(1 << 24)
        }
        for round in 0..<60 {
            let n = 20 + Int(next() * 1500)
            let spread: Float = [1e-3, 1, 50, 1e4][round % 4]
            var positions: [SIMD3<Float>] = []
            for _ in 0..<n {
                positions.append(SIMD3<Float>((next() - 0.5) * spread, (next() - 0.5) * spread, (next() - 0.5) * spread))
            }
            // duplicates and tight clumps (as produced by cloning)
            for i in 0..<(n / 5) where i + 1 < positions.count {
                positions[i + 1] = positions[i]
            }
            let hostile = 1 + Int(next() * 6)
            for _ in 0..<hostile {
                let slot = Int(next() * Float(n))
                switch Int(next() * 6) {
                case 0: positions[slot] = SIMD3<Float>(.nan, 0, 0)
                case 1: positions[slot] = SIMD3<Float>(.infinity, 1, -.infinity)
                case 2: positions[slot] = SIMD3<Float>(1e30, -1e30, 1e30)
                case 3: positions[slot] = SIMD3<Float>(3e38, 0, 0)
                case 4: positions[slot] = SIMD3<Float>(1e-30, 1e-30, 1e-30)
                default: positions[slot] = SIMD3<Float>(0, 0, .nan)
                }
            }
            let mask = IsolationPruning.pruneMask(positions: positions, neighbors: 4, factor: 8,
                                                  maxFraction: 0.05, sampleCount: 2_048, seed: UInt64(round))
            #expect(mask.count == n)
        }
    }
}
