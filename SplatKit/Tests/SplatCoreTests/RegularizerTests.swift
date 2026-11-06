import Foundation
import Testing
@testable import SplatCore

// MARK: - Helpers

/// Builds a degree-0 cloud from flat arrays. Rotations are identity quaternions, opacity 0.
private func makeCloud(positions: [SIMD3<Float>], logScales: [SIMD3<Float>]) throws -> GaussianCloud {
    precondition(positions.count == logScales.count)
    let n = positions.count
    var flatPositions: [Float] = []
    var flatScales: [Float] = []
    var rotations: [Float] = []
    for i in 0..<n {
        flatPositions.append(contentsOf: [positions[i].x, positions[i].y, positions[i].z])
        flatScales.append(contentsOf: [logScales[i].x, logScales[i].y, logScales[i].z])
        rotations.append(contentsOf: [1, 0, 0, 0])
    }
    return try GaussianCloud(shDegree: 0,
                             positions: flatPositions,
                             logScales: flatScales,
                             rotations: rotations,
                             opacityLogits: [Float](repeating: 0, count: n),
                             sh: [Float](repeating: 0, count: n * 3))
}

private func zeroGradients(for cloud: GaussianCloud) -> GaussianGradients {
    GaussianGradients(zeroLike: cloud)
}

/// Groups a flat row-major n × 3 array into per-Gaussian log-scale triples.
private func rows(_ flat: [Float]) -> [SIMD3<Float>] {
    stride(from: 0, to: flat.count, by: 3).map { SIMD3<Float>(flat[$0], flat[$0 + 1], flat[$0 + 2]) }
}

/// Returns the anisotropy loss and the gradient vector for a cloud.
private func anisotropy(_ cloud: GaussianCloud, weight: Float, maxRatio: Float) -> (Double, [Float]) {
    var gradients = zeroGradients(for: cloud)
    let loss = Regularizers.addAnisotropyGradient(cloud, weight: weight, maxRatio: maxRatio, into: &gradients)
    return (loss, gradients.logScales)
}

/// Deterministic points on a unit sphere (Fibonacci lattice).
private func fibonacciSphere(count: Int) -> [SIMD3<Float>] {
    var points: [SIMD3<Float>] = []
    let goldenAngle = Double.pi * (3 - 5.0.squareRoot())
    for i in 0..<count {
        let y: Double = 1 - 2 * (Double(i) + 0.5) / Double(count)
        let radius: Double = (1 - y * y).squareRoot()
        let theta: Double = goldenAngle * Double(i)
        let x: Double = cos(theta) * radius
        let z: Double = sin(theta) * radius
        points.append(SIMD3<Float>(Float(x), Float(y), Float(z)))
    }
    return points
}

/// Uniform random points in a cube, reproducible from the seed.
private func randomPoints(count: Int, seed: UInt64, scale: Float) -> [SIMD3<Float>] {
    var rng = SeededGenerator(seed: seed)
    var points: [SIMD3<Float>] = []
    points.reserveCapacity(count)
    for _ in 0..<count {
        let x: Float = Float.random(in: -scale...scale, using: &rng)
        let y: Float = Float.random(in: -scale...scale, using: &rng)
        let z: Float = Float.random(in: -scale...scale, using: &rng)
        points.append(SIMD3<Float>(x, y, z))
    }
    return points
}

/// Same arithmetic as the grid, so brute-force comparisons agree bit for bit.
private func squaredDistance(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
    let d: SIMD3<Float> = a - b
    return d.x * d.x + d.y * d.y + d.z * d.z
}

private func bruteKthDistance(points: [SIMD3<Float>], index: Int, k: Int) -> Float {
    var distances: [Float] = []
    for j in points.indices where j != index {
        distances.append(squaredDistance(points[index], points[j]))
    }
    distances.sort()
    guard distances.count >= k else { return .infinity }
    return distances[k - 1].squareRoot()
}

/// Brute-force mirror of `IsolationPruning.pruneMask` with the same sampling and cap rules.
private func bruteForcePruneMask(positions: [SIMD3<Float>], k: Int, factor: Float,
                                 maxFraction: Float, sampleCount: Int, seed: UInt64) -> [Bool] {
    let n = positions.count
    var mask = [Bool](repeating: false, count: n)
    guard n > k, k > 0 else { return mask }
    let sample = GaussianInitializer.subsetIndices(total: n, maxCount: sampleCount, seed: seed)
    var finite: [Double] = []
    for s in sample {
        let d = bruteKthDistance(points: positions, index: s, k: k)
        if d.isFinite { finite.append(Double(d)) }
    }
    finite.sort()
    guard !finite.isEmpty else { return mask }
    let median: Double
    if finite.count % 2 == 1 {
        median = finite[finite.count / 2]
    } else {
        median = 0.5 * (finite[finite.count / 2 - 1] + finite[finite.count / 2])
    }
    guard median > 0 else { return mask }
    let threshold = Float(Double(factor) * median)
    let thresholdSquared = threshold * threshold
    var isolated: [Int] = []
    for i in 0..<n {
        var count = 0
        for j in 0..<n where j != i {
            if squaredDistance(positions[i], positions[j]) <= thresholdSquared { count += 1 }
        }
        if count < k { isolated.append(i) }
    }
    let cap = Int((Double(maxFraction) * Double(n)).rounded(.down))
    if isolated.count > cap {
        var ranked: [(index: Int, distance: Float)] = []
        for index in isolated {
            ranked.append((index, bruteKthDistance(points: positions, index: index, k: k)))
        }
        ranked.sort { (lhs: (index: Int, distance: Float), rhs: (index: Int, distance: Float)) -> Bool in
            lhs.distance > rhs.distance || (lhs.distance == rhs.distance && lhs.index < rhs.index)
        }
        var kept: [Int] = []
        for entry in ranked.prefix(cap) {
            kept.append(entry.index)
        }
        isolated = kept.sorted()
    }
    if isolated.count >= n { return mask }
    for i in isolated { mask[i] = true }
    return mask
}

// MARK: - Anisotropy

@Test func anisotropyZeroBelowRatio() throws {
    // max/min = exp(ln 9.9) = 9.9 < 10: no loss, no gradient.
    let spread = Float(log(9.9))
    let cloud = try makeCloud(positions: [.zero, .zero],
                              logScales: [SIMD3<Float>(0, spread * 0.5, spread),
                                          SIMD3<Float>(-1, -0.5, -1.2)])
    let (loss, gradient) = anisotropy(cloud, weight: 0.1, maxRatio: 10)
    #expect(loss == 0)
    #expect(gradient.allSatisfy { $0 == 0 })
}

@Test func anisotropyMatchesGolden() throws {
    let golden = try AnisotropyGolden.load("anisotropy")
    let positions = [SIMD3<Float>](repeating: .zero, count: golden.count)
    var logScales: [SIMD3<Float>] = []
    for i in 0..<golden.count {
        logScales.append(SIMD3<Float>(golden.logScales[3 * i], golden.logScales[3 * i + 1], golden.logScales[3 * i + 2]))
    }
    let cloud = try makeCloud(positions: positions, logScales: logScales)
    var gradients = zeroGradients(for: cloud)
    let loss = Regularizers.addAnisotropyGradient(cloud, weight: golden.weight,
                                                  maxRatio: golden.maxRatio, into: &gradients)

    #expect(abs(loss - golden.loss) <= 1e-6)
    #expect(gradients.logScales.count == golden.gradLogScales.count)
    for (a, b) in zip(gradients.logScales, golden.gradLogScales) {
        #expect(abs(a - b) <= 1e-5 * abs(b) + 1e-7)
    }
}

@Test func anisotropyMatchesFiniteDifferences() throws {
    // The golden cloud: above-ratio rows have a unique, well separated maximum axis and a
    // ratio of at least e^2.6, far from the hinge kink at ratio 10. Below-ratio rows stay below.
    let golden = try AnisotropyGolden.load("anisotropy")
    var logScales: [SIMD3<Float>] = []
    for i in 0..<golden.count {
        logScales.append(SIMD3<Float>(golden.logScales[3 * i], golden.logScales[3 * i + 1], golden.logScales[3 * i + 2]))
    }
    let positions = [SIMD3<Float>](repeating: .zero, count: golden.count)
    let cloud = try makeCloud(positions: positions, logScales: logScales)
    let (_, analytic) = anisotropy(cloud, weight: golden.weight, maxRatio: golden.maxRatio)

    let h: Double = 1e-4
    for i in 0..<golden.count {
        for axis in 0..<3 {
            let flatIndex = 3 * i + axis
            // Use the steps actually representable in Float, so the quotient divides by the true step.
            let x0 = Double(golden.logScales[flatIndex])
            let up = Float(x0 + h)
            let down = Float(x0 - h)
            let stepUp = Double(up) - x0
            let stepDown = x0 - Double(down)
            var plus = golden.logScales
            var minus = golden.logScales
            plus[flatIndex] = up
            minus[flatIndex] = down
            let lossPlus = try anisotropy(makeCloud(positions: positions, logScales: rows(plus)),
                                          weight: golden.weight, maxRatio: golden.maxRatio).0
            let lossMinus = try anisotropy(makeCloud(positions: positions, logScales: rows(minus)),
                                           weight: golden.weight, maxRatio: golden.maxRatio).0
            // Central difference with the actual steps: (L(x+a) - L(x-b)) / (a + b).
            let numeric = (lossPlus - lossMinus) / (stepUp + stepDown)
            let expected = Double(analytic[flatIndex])
            #expect(abs(numeric - expected) <= 1e-4 * abs(expected) + 1e-7,
                    "entry \(flatIndex): numeric \(numeric) analytic \(expected)")
        }
    }
}

@Test func scaleCapClampsOnlyLargeScales() throws {
    // Radius 2, centre and Gaussian at the origin: cap = 0.05 · 2 = 0.1.
    var cloud = try makeCloud(positions: [.zero, .zero],
                              logScales: [SIMD3<Float>(repeating: Float(log(0.5))),
                                          SIMD3<Float>(repeating: Float(log(0.05)))])
    Regularizers.applyScaleCap(&cloud, fraction: 0.05,
                               extent: SceneExtent(center: .zero, radius: 2))
    for axis in 0..<3 {
        #expect(abs(cloud.logScales[axis] - Float(log(0.1))) <= 1e-6)
        #expect(abs(cloud.logScales[3 + axis] - Float(log(0.05))) <= 1e-6)
    }
}

@Test func scaleCapGrowsWithDistance() throws {
    // Gaussian 10 units from the centre: cap = 0.05 · 10 = 0.5, so a scale of 1 is clamped to 0.5.
    var cloud = try makeCloud(positions: [SIMD3<Float>(10, 0, 0)],
                              logScales: [SIMD3<Float>(repeating: 0)])
    Regularizers.applyScaleCap(&cloud, fraction: 0.05,
                               extent: SceneExtent(center: .zero, radius: 2))
    for axis in 0..<3 {
        #expect(abs(cloud.logScales[axis] - Float(log(0.5))) <= 1e-6)
    }
}

// MARK: - Spatial hash grid

@Test func kthNeighborMatchesKDTree() {
    let points = randomPoints(count: 500, seed: 11, scale: 1)
    let tree = KDTree(points: points)
    let grid = SpatialHashGrid(points: points, cellSize: 0.2)
    for index in points.indices {
        let neighbors = tree.nearestNeighbors(of: index, k: 4)
        let treeDistance = squaredDistance(points[index], points[neighbors[3]]).squareRoot()
        let gridDistance = grid.kthNeighborDistance(of: index, k: 4, maxRings: 64)
        #expect(abs(gridDistance - treeDistance) <= 1e-6)
    }
}

@Test func countNeighborsMatchesBruteForce() {
    let points = randomPoints(count: 400, seed: 23, scale: 1)
    let radius: Float = 0.3
    let grid = SpatialHashGrid(points: points, cellSize: radius)
    for index in points.indices {
        var expected = 0
        for j in points.indices where j != index {
            if squaredDistance(points[index], points[j]) <= radius * radius { expected += 1 }
        }
        let unlimited = grid.countNeighbors(of: index, within: radius, limit: Int.max)
        #expect(unlimited == expected)
        let limited = grid.countNeighbors(of: index, within: radius, limit: 3)
        #expect(limited == min(expected, 3))
    }
}

// MARK: - Isolation pruning

@Test func prunesFarOutliers() {
    // 2 000 points on the unit sphere and 10 points at distance 20, spread over different
    // directions so that no two outliers are neighbours of each other.
    var positions = fibonacciSphere(count: 2_000)
    let outliers = fibonacciSphere(count: 10).map { $0 * 20 }
    positions.append(contentsOf: outliers)
    let mask = IsolationPruning.pruneMask(positions: positions, neighbors: 4, factor: 8,
                                          maxFraction: 0.05, sampleCount: 2_048, seed: 3)
    #expect(mask.count == positions.count)
    #expect(mask.prefix(2_000).allSatisfy { !$0 })
    #expect(mask.suffix(10).allSatisfy { $0 })
    #expect(mask.filter { $0 }.count == 10)
}

@Test func densePointsAreNeverPruned() {
    var positions: [SIMD3<Float>] = []
    for z in 0..<10 {
        for y in 0..<10 {
            for x in 0..<10 {
                positions.append(SIMD3<Float>(Float(x), Float(y), Float(z)))
            }
        }
    }
    let mask = IsolationPruning.pruneMask(positions: positions, neighbors: 4, factor: 8,
                                          maxFraction: 0.05, sampleCount: 2_048, seed: 5)
    #expect(mask.allSatisfy { !$0 })
}

@Test func pruningRespectsMaxFraction() {
    // 900 points in a unit cube and 100 outliers on a sphere of radius 50 (1 000 in total).
    // Only 50 may be pruned, and they must be the 50 farthest outliers by 4-NN distance.
    var positions = randomPoints(count: 900, seed: 41, scale: 0.5)
    let outliers = randomPoints(count: 100, seed: 43, scale: 1).map { p -> SIMD3<Float> in
        let length = (p.x * p.x + p.y * p.y + p.z * p.z).squareRoot()
        let direction = p / max(length, 1e-3)
        return direction * (50 + p.x * 2)
    }
    positions.append(contentsOf: outliers)
    let mask = IsolationPruning.pruneMask(positions: positions, neighbors: 4, factor: 8,
                                          maxFraction: 0.05, sampleCount: 2_048, seed: 7)
    let marked = mask.indices.filter { mask[$0] }
    #expect(marked.count == 50)
    #expect(marked.allSatisfy { $0 >= 900 })

    // The marked outliers are the 50 with the largest 4-NN distance among the isolated ones.
    var distances: [(index: Int, distance: Float)] = []
    for index in 900..<1_000 {
        distances.append((index, bruteKthDistance(points: positions, index: index, k: 4)))
    }
    distances.sort { lhs, rhs in
        lhs.distance > rhs.distance || (lhs.distance == rhs.distance && lhs.index < rhs.index)
    }
    let farthest = Set(distances.prefix(50).map { $0.index })
    #expect(Set(marked) == farthest)
}

@Test func pruneMaskIsDeterministic() {
    var positions = randomPoints(count: 600, seed: 61, scale: 1)
    positions.append(contentsOf: randomPoints(count: 20, seed: 67, scale: 30))
    let first = IsolationPruning.pruneMask(positions: positions, neighbors: 4, factor: 3,
                                           maxFraction: 0.05, sampleCount: 256, seed: 9)
    let second = IsolationPruning.pruneMask(positions: positions, neighbors: 4, factor: 3,
                                            maxFraction: 0.05, sampleCount: 256, seed: 9)
    #expect(first == second)
}

@Test func pruneMaskMatchesBruteForceOracle() {
    // Clustered points plus scattered outliers: exercises sampling, the threshold and the cap.
    var positions = randomPoints(count: 350, seed: 71, scale: 1)
    positions.append(contentsOf: randomPoints(count: 40, seed: 73, scale: 12))
    let mask = IsolationPruning.pruneMask(positions: positions, neighbors: 4, factor: 4,
                                          maxFraction: 0.05, sampleCount: 128, seed: 13)
    let oracle = bruteForcePruneMask(positions: positions, k: 4, factor: 4,
                                     maxFraction: 0.05, sampleCount: 128, seed: 13)
    #expect(mask == oracle)
    #expect(mask.contains(true))
    #expect(mask.filter { $0 }.count <= Int((0.05 * Double(positions.count)).rounded(.down)))
}

@Test func pruneMaskNeverEmptiesTheModel() {
    // Fewer points than neighbours: nothing may be pruned.
    let few = randomPoints(count: 4, seed: 81, scale: 1)
    #expect(IsolationPruning.pruneMask(positions: few, neighbors: 4, factor: 1,
                                       maxFraction: 1, sampleCount: 2_048, seed: 1).allSatisfy { !$0 })
    // A single point at a huge distance from a cluster is still kept when the sample median is 0
    // (every point identical): nothing may be pruned.
    let identical = [SIMD3<Float>](repeating: SIMD3<Float>(1, 2, 3), count: 50)
    #expect(IsolationPruning.pruneMask(positions: identical, neighbors: 4, factor: 2,
                                       maxFraction: 1, sampleCount: 2_048, seed: 1).allSatisfy { !$0 })
}
