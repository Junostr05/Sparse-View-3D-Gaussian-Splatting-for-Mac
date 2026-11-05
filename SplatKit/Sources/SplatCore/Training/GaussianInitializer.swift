import Foundation

/// Creates initial Gaussians from sparse points or a random cube (decision D9).
///
/// Scales are isotropic and derived from the three nearest neighbours; rotations are identity,
/// opacity is the requested initial value and the DC spherical-harmonic term carries the colour.
public enum GaussianInitializer {
    /// Linear scale used for a point without any neighbour (a lone point).
    static let isolatedScale: Float = 0.01
    /// Floor for the mean squared neighbour distance, so duplicate points get a finite scale.
    static let minimumMeanSquaredDistance: Float = 1e-7

    /// Linear isotropic scale per point: `sqrt(max(mean squared distance to up to k nearest
    /// neighbours, 1e-7))`. A point without neighbours gets `0.01`.
    public static func nearestNeighborScales(_ positions: [SIMD3<Float>], k: Int = 3) -> [Float] {
        let tree = KDTree(points: positions)
        var scales: [Float] = []
        scales.reserveCapacity(positions.count)
        for index in positions.indices {
            let neighbors = tree.nearestNeighbors(of: index, k: k)
            guard !neighbors.isEmpty else {
                scales.append(isolatedScale)
                continue
            }
            var sum: Double = 0
            for neighbor in neighbors {
                let delta = positions[neighbor] - positions[index]
                let squared = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z
                sum += Double(squared)
            }
            let meanSquared = Float(sum / Double(neighbors.count))
            scales.append(max(meanSquared, minimumMeanSquaredDistance).squareRoot())
        }
        return scales
    }

    /// Gaussians at the given points, coloured by the point colours. When there are more points
    /// than `maxCount`, a uniformly random subset of `maxCount` is kept in ascending index order.
    /// Scales are computed on the full point set before subsampling.
    ///
    /// - Precondition: `initialOpacity` in (0, 1), `shDegree` in 0...3, equal position and colour counts.
    public static func fromPoints(_ points: PointCloud,
                                  shDegree: Int,
                                  initialOpacity: Float,
                                  maxCount: Int,
                                  seed: UInt64) -> GaussianCloud {
        precondition(points.positions.count == points.colors.count,
                     "positions and colors must have the same count")
        let scales = nearestNeighborScales(points.positions, k: 3)
        let selected = subsetIndices(total: points.positions.count, maxCount: maxCount, seed: seed)

        var positions: [SIMD3<Float>] = []
        var pointScales: [Float] = []
        var colors: [SIMD3<Float>] = []
        positions.reserveCapacity(selected.count)
        pointScales.reserveCapacity(selected.count)
        colors.reserveCapacity(selected.count)
        for index in selected {
            positions.append(points.positions[index])
            pointScales.append(scales[index])
            colors.append(points.colors[index])
        }
        return makeCloud(shDegree: shDegree, initialOpacity: initialOpacity,
                         positions: positions, scales: pointScales, colors: colors)
    }

    /// Gaussians at uniform random positions in the cube `center ± radius`, with uniform random
    /// colours in [0, 1], initialized as in `fromPoints`.
    ///
    /// - Precondition: `extent.radius >= 0`.
    public static func random(count: Int,
                              extent: SceneExtent,
                              shDegree: Int,
                              initialOpacity: Float,
                              seed: UInt64) -> GaussianCloud {
        precondition(extent.radius >= 0, "scene radius must be non-negative")
        let total = max(0, count)
        var generator = SeededGenerator(seed: seed)
        let radius = extent.radius
        var positions: [SIMD3<Float>] = []
        var colors: [SIMD3<Float>] = []
        positions.reserveCapacity(total)
        colors.reserveCapacity(total)
        for _ in 0..<total {
            let dx = Float.random(in: -radius...radius, using: &generator)
            let dy = Float.random(in: -radius...radius, using: &generator)
            let dz = Float.random(in: -radius...radius, using: &generator)
            positions.append(extent.center + SIMD3<Float>(dx, dy, dz))
            let r = Float.random(in: 0...1, using: &generator)
            let g = Float.random(in: 0...1, using: &generator)
            let b = Float.random(in: 0...1, using: &generator)
            colors.append(SIMD3<Float>(r, g, b))
        }
        return fromPoints(PointCloud(positions: positions, colors: colors),
                          shDegree: shDegree, initialOpacity: initialOpacity,
                          maxCount: total, seed: seed)
    }

    /// Initialization for a training run: the dataset's points when present, otherwise a random
    /// cube sized from the camera centres, with `min(randomInitCount, maxCount)` Gaussians.
    public static func initialize(_ dataset: TrainingDataset,
                                  configuration: TrainingConfiguration,
                                  maxCount: Int) -> GaussianCloud {
        if let points = dataset.points, !points.positions.isEmpty {
            return fromPoints(points,
                              shDegree: configuration.shDegree,
                              initialOpacity: configuration.initialOpacity,
                              maxCount: maxCount,
                              seed: configuration.seed)
        }
        let count = min(configuration.randomInitCount, maxCount)
        let extent = SceneExtent.estimate(cameraCenters: dataset.cameraCenters, points: [])
        return random(count: count,
                      extent: extent,
                      shDegree: configuration.shDegree,
                      initialOpacity: configuration.initialOpacity,
                      seed: configuration.seed)
    }

    // MARK: - Helpers

    /// Indices of the kept points: all of them, or a uniform random subset of `maxCount` in
    /// ascending order (partial Fisher-Yates driven by `SeededGenerator`).
    static func subsetIndices(total: Int, maxCount: Int, seed: UInt64) -> [Int] {
        guard total > maxCount else { return Array(0..<total) }
        let keep = max(0, maxCount)
        var generator = SeededGenerator(seed: seed)
        var pool = Array(0..<total)
        for i in 0..<keep {
            let j = Int.random(in: i..<total, using: &generator)
            pool.swapAt(i, j)
        }
        return pool[0..<keep].sorted()
    }

    private static func makeCloud(shDegree: Int,
                                  initialOpacity: Float,
                                  positions: [SIMD3<Float>],
                                  scales: [Float],
                                  colors: [SIMD3<Float>]) -> GaussianCloud {
        precondition(initialOpacity > 0 && initialOpacity < 1, "initial opacity must be in (0, 1)")
        let n = positions.count
        let k = GaussianCloud.coefficientCount(degree: shDegree)
        let opacityLogit = log(initialOpacity / (1 - initialOpacity))

        var flatPositions: [Float] = []
        var logScales: [Float] = []
        var rotations: [Float] = []
        var opacityLogits: [Float] = []
        var sh = [Float](repeating: 0, count: n * k * 3)
        flatPositions.reserveCapacity(3 * n)
        logScales.reserveCapacity(3 * n)
        rotations.reserveCapacity(4 * n)
        opacityLogits.reserveCapacity(n)

        for i in 0..<n {
            let p = positions[i]
            flatPositions.append(contentsOf: [p.x, p.y, p.z])

            let logScale = log(scales[i])
            logScales.append(contentsOf: [logScale, logScale, logScale])

            // Identity quaternion, stored as (w, x, y, z).
            rotations.append(contentsOf: [1, 0, 0, 0])
            opacityLogits.append(opacityLogit)

            // DC term: colour = 0.5 + C0 · dc, so dc = (colour − 0.5) / C0. Other terms stay zero.
            let base = i * k * 3
            let color = colors[i]
            for channel in 0..<3 {
                sh[base + channel] = (color[channel] - 0.5) / SphericalHarmonics.c0
            }
        }

        do {
            return try GaussianCloud(shDegree: shDegree,
                                     positions: flatPositions,
                                     logScales: logScales,
                                     rotations: rotations,
                                     opacityLogits: opacityLogits,
                                     sh: sh)
        } catch {
            preconditionFailure("initializer produced an invalid cloud: \(error)")
        }
    }
}
