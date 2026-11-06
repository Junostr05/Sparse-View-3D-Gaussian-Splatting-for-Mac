import Foundation

/// Isolation pruning (Phase 4 P10): removes Gaussians whose k-th neighbour is far away compared
/// with the typical spacing of the model. Runs on CPU positions; the decision is discrete.
public enum IsolationPruning {
    /// Ring limit for the k-th neighbour searches, as in the Phase 4 plan.
    static let ringLimit = 64

    /// Returns one flag per position: `true` means prune. Never marks every point, and marks at most
    /// `floor(maxFraction · n)` points. Deterministic for a given seed; ties are broken by lowest index.
    ///
    /// Algorithm:
    /// 1. Cell size `c0 = 2·cbrt(V/n)`, where `V` is the product of the per-axis P1–P99 extents
    ///    (each floored at `1e-6` times the largest extent).
    /// 2. The median k-th neighbour distance `m` of `sampleCount` seeded samples (finite values only).
    /// 3. A Gaussian is isolated when fewer than `k` other points lie within `T = factor · m`.
    /// 4. If more than the cap are isolated, keep the ones with the largest k-th neighbour distance.
    public static func pruneMask(positions: [SIMD3<Float>],
                                 neighbors k: Int,
                                 factor: Float,
                                 maxFraction: Float,
                                 sampleCount: Int,
                                 seed: UInt64) -> [Bool] {
        let n = positions.count
        var mask = [Bool](repeating: false, count: n)
        guard k > 0, n > k, factor > 0, factor.isFinite else { return mask }

        guard let initialCellSize = initialCellSize(positions) else { return mask }
        let coarseGrid = SpatialHashGrid(points: positions, cellSize: initialCellSize)

        // Median k-th neighbour distance over a seeded sample.
        let sample = GaussianInitializer.subsetIndices(total: n, maxCount: sampleCount, seed: seed)
        var sampleDistances: [Double] = []
        sampleDistances.reserveCapacity(sample.count)
        for index in sample {
            let distance: Float = coarseGrid.kthNeighborDistance(of: index, k: k, maxRings: ringLimit)
            if distance.isFinite {
                sampleDistances.append(Double(distance))
            }
        }
        guard let median = median(of: sampleDistances), median > 0 else { return mask }
        let threshold: Float = Float(Double(factor) * median)
        guard threshold > 0, threshold.isFinite else { return mask }

        // Isolation test: fewer than k neighbours within the threshold.
        let thresholdGrid = SpatialHashGrid(points: positions, cellSize: threshold)
        var isolated: [Int] = []
        for index in 0..<n {
            let neighborCount = thresholdGrid.countNeighbors(of: index, within: threshold, limit: k)
            if neighborCount < k {
                isolated.append(index)
            }
        }

        // Cap: keep the most isolated points, ranked by k-th neighbour distance.
        let fraction = Double(min(max(maxFraction, 0), 1))
        let cap = Int((fraction * Double(n)).rounded(.down))
        if isolated.count > cap {
            var ranked: [(index: Int, distance: Float)] = []
            ranked.reserveCapacity(isolated.count)
            for index in isolated {
                let distance: Float = thresholdGrid.kthNeighborDistance(of: index, k: k, maxRings: ringLimit)
                ranked.append((index: index, distance: distance))
            }
            ranked.sort { (lhs: (index: Int, distance: Float), rhs: (index: Int, distance: Float)) -> Bool in
                if lhs.distance != rhs.distance {
                    return lhs.distance > rhs.distance
                }
                return lhs.index < rhs.index
            }
            var kept: [Int] = []
            kept.reserveCapacity(cap)
            for entry in ranked.prefix(cap) {
                kept.append(entry.index)
            }
            isolated = kept.sorted()
        }

        // Never empty the model.
        guard isolated.count < n else { return mask }
        for index in isolated {
            mask[index] = true
        }
        return mask
    }

    // MARK: - Helpers

    /// `c0 = 2·cbrt(V/n)` from the robust per-axis extents; nil when every coordinate is identical.
    private static func initialCellSize(_ positions: [SIMD3<Float>]) -> Float? {
        let n = positions.count
        var extents: [Double] = []
        var largest: Double = 0
        for axis in 0..<3 {
            var values: [Float] = []
            values.reserveCapacity(n)
            for position in positions {
                values.append(component(position, axis))
            }
            values.sort()
            let low = Double(percentile(values, 0.01))
            let high = Double(percentile(values, 0.99))
            let extent = high - low
            extents.append(extent)
            largest = max(largest, extent)
        }
        guard largest > 0 else { return nil }
        let floorExtent = 1e-6 * largest
        var volume: Double = 1
        for extent in extents {
            let axisExtent: Double = max(extent, floorExtent)
            volume *= axisExtent
        }
        let cellSize = 2 * cbrt(volume / Double(n))
        guard cellSize > 0, cellSize.isFinite else { return nil }
        return Float(cellSize)
    }

    /// Nearest-rank percentile of a sorted array.
    private static func percentile(_ sorted: [Float], _ fraction: Double) -> Float {
        let index = Int((Double(sorted.count - 1) * fraction).rounded())
        return sorted[index]
    }

    private static func median(of values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count % 2 == 1 {
            return sorted[middle]
        }
        return 0.5 * (sorted[middle - 1] + sorted[middle])
    }

    private static func component(_ position: SIMD3<Float>, _ axis: Int) -> Float {
        switch axis {
        case 0: return position.x
        case 1: return position.y
        default: return position.z
        }
    }
}
