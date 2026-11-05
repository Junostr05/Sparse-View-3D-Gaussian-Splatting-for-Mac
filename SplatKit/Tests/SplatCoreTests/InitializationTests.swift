import Foundation
import Testing
import SplatCore

@Suite struct InitializationTests {

    // MARK: - Helpers

    /// Brute-force k nearest neighbours with the same ordering as `KDTree`:
    /// ascending squared distance, ties broken by smaller index, excluding `index` itself.
    private func bruteForceNeighbors(_ points: [SIMD3<Float>], of index: Int, k: Int) -> [Int] {
        var best: [(distance: Float, index: Int)] = []
        let query = points[index]
        for other in points.indices where other != index {
            let delta = points[other] - query
            let distance = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z
            let candidate = (distance: distance, index: other)
            // Insert keeping `best` sorted by (distance, index) and bounded to k entries.
            var position = best.count
            while position > 0 {
                let previous = best[position - 1]
                let isBefore = candidate.distance < previous.distance
                    || (candidate.distance == previous.distance && candidate.index < previous.index)
                if !isBefore { break }
                position -= 1
            }
            if position < k {
                best.insert(candidate, at: position)
                if best.count > k { best.removeLast() }
            }
        }
        return best.map { $0.index }
    }

    private func randomPoints(count: Int, seed: UInt64) -> [SIMD3<Float>] {
        var generator = SeededGenerator(seed: seed)
        return (0..<count).map { _ in
            SIMD3<Float>(Float.random(in: -10...10, using: &generator),
                         Float.random(in: -10...10, using: &generator),
                         Float.random(in: -10...10, using: &generator))
        }
    }

    // MARK: - KDTree

    @Test func kdTreeMatchesBruteForce() {
        var points = randomPoints(count: 2_000, seed: 7)
        // 50 exact duplicates of existing points: distance-zero neighbours and index tie-breaks.
        for i in 0..<50 {
            points.append(points[i * 37 % 2_000])
        }
        let tree = KDTree(points: points)
        for index in points.indices {
            let expected = bruteForceNeighbors(points, of: index, k: 3)
            let actual = tree.nearestNeighbors(of: index, k: 3)
            #expect(actual == expected, "mismatch at index \(index)")
        }
    }

    @Test func kdTreeHandlesTinyInputs() {
        let empty = KDTree(points: [])
        _ = empty  // Building from no points must not trap.

        let single = KDTree(points: [SIMD3<Float>(1, 2, 3)])
        #expect(single.nearestNeighbors(of: 0, k: 3).isEmpty)

        let pair = KDTree(points: [SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 0, 0)])
        #expect(pair.nearestNeighbors(of: 0, k: 3) == [1])
        #expect(pair.nearestNeighbors(of: 1, k: 3) == [0])

        let triple = KDTree(points: [SIMD3<Float>(0, 0, 0), SIMD3<Float>(2, 0, 0), SIMD3<Float>(1, 0, 0)])
        #expect(triple.nearestNeighbors(of: 0, k: 10) == [2, 1])
        #expect(triple.nearestNeighbors(of: 0, k: 0).isEmpty)
    }

    // MARK: - Scales

    @Test func scalesOnRegularGrid() {
        var positions: [SIMD3<Float>] = []
        for z in 0..<5 {
            for y in 0..<5 {
                for x in 0..<5 {
                    positions.append(SIMD3<Float>(Float(x), Float(y), Float(z)))
                }
            }
        }
        let scales = GaussianInitializer.nearestNeighborScales(positions, k: 3)
        #expect(scales.count == positions.count)
        // Interior point (2, 2, 2) has six face neighbours at distance 1: scale 1.
        let interior = 2 * 25 + 2 * 5 + 2
        #expect(abs(scales[interior] - 1) <= 1e-6)
    }

    @Test func duplicatePointsGetMinimumScale() {
        let minimum = Float(1e-7).squareRoot()
        let allSame = [SIMD3<Float>](repeating: SIMD3<Float>(0.5, 0.5, 0.5), count: 4)
        for scale in GaussianInitializer.nearestNeighborScales(allSame, k: 3) {
            #expect(abs(scale - minimum) <= 1e-9)
        }
        let pair = [SIMD3<Float>(1, 1, 1), SIMD3<Float>(1, 1, 1)]
        for scale in GaussianInitializer.nearestNeighborScales(pair, k: 3) {
            #expect(abs(scale - minimum) <= 1e-9)
        }
    }

    @Test func singlePointUsesFallbackScale() {
        let scales = GaussianInitializer.nearestNeighborScales([SIMD3<Float>(4, 5, 6)], k: 3)
        #expect(scales.count == 1)
        #expect(abs(scales[0] - 0.01) <= 1e-7)
        #expect(GaussianInitializer.nearestNeighborScales([], k: 3).isEmpty)
    }

    // MARK: - Gaussian initialization

    @Test func fromPointsSetsActivationsAndColor() {
        let positions = randomPoints(count: 20, seed: 3)
        var generator = SeededGenerator(seed: 4)
        let colors = (0..<20).map { _ in
            SIMD3<Float>(Float.random(in: 0...1, using: &generator),
                         Float.random(in: 0...1, using: &generator),
                         Float.random(in: 0...1, using: &generator))
        }
        let cloud = GaussianInitializer.fromPoints(
            PointCloud(positions: positions, colors: colors),
            shDegree: 2, initialOpacity: 0.1, maxCount: 100, seed: 1)

        let k = GaussianCloud.coefficientCount(degree: 2)
        #expect(cloud.count == 20)
        #expect(cloud.shDegree == 2)
        let scales = GaussianInitializer.nearestNeighborScales(positions, k: 3)
        for i in 0..<cloud.count {
            // Opacity activation reproduces the requested initial opacity.
            let opacity = 1 / (1 + exp(-cloud.opacityLogits[i]))
            #expect(abs(opacity - 0.1) <= 1e-6)

            // Isotropic log-scale matches the nearest-neighbour scale.
            for axis in 0..<3 {
                #expect(abs(exp(cloud.logScales[3 * i + axis]) - scales[i]) <= 1e-5 * scales[i])
            }

            // Identity rotation (w, x, y, z).
            #expect(cloud.rotations[4 * i] == 1)
            #expect(cloud.rotations[4 * i + 1] == 0)
            #expect(cloud.rotations[4 * i + 2] == 0)
            #expect(cloud.rotations[4 * i + 3] == 0)

            // DC coefficient reproduces the colour through degree-0 SH evaluation.
            let base = i * k * 3
            let dc = SphericalHarmonics.evaluate(degree: 0,
                                                 coefficients: cloud.sh[base..<base + 3 * k],
                                                 direction: SIMD3<Float>(0, 0, 1))
            let reproduced = dc + SIMD3<Float>(repeating: 0.5)
            #expect(approxEqual(reproduced, colors[i], tol: 1e-6))

            // Higher-order coefficients start at zero.
            for j in 3..<(3 * k) {
                #expect(cloud.sh[base + j] == 0)
            }
        }
    }

    @Test func fromPointsSubsamplesDeterministically() {
        let positions = randomPoints(count: 1_000, seed: 21)
        let colors = [SIMD3<Float>](repeating: SIMD3<Float>(0.2, 0.4, 0.6), count: 1_000)
        let input = PointCloud(positions: positions, colors: colors)
        let first = GaussianInitializer.fromPoints(input, shDegree: 1, initialOpacity: 0.1,
                                                   maxCount: 100, seed: 99)
        let second = GaussianInitializer.fromPoints(input, shDegree: 1, initialOpacity: 0.1,
                                                    maxCount: 100, seed: 99)
        #expect(first.count == 100)
        #expect(first.positions == second.positions)
        #expect(first.sh == second.sh)
        #expect(first.logScales == second.logScales)

        // Every kept position is one of the input positions, in ascending input order.
        var searchStart = 0
        for i in 0..<first.count {
            let kept = SIMD3<Float>(first.positions[3 * i], first.positions[3 * i + 1],
                                    first.positions[3 * i + 2])
            var found = -1
            for j in searchStart..<positions.count where positions[j] == kept {
                found = j
                break
            }
            #expect(found >= 0)
            searchStart = found + 1
        }
    }

    @Test func randomFallbackStaysInCube() {
        let extent = SceneExtent(center: SIMD3<Float>(1, 2, 3), radius: 2)
        let cloud = GaussianInitializer.random(count: 500, extent: extent, shDegree: 1,
                                               initialOpacity: 0.1, seed: 5)
        #expect(cloud.count == 500)
        for i in 0..<cloud.count {
            let p = SIMD3<Float>(cloud.positions[3 * i], cloud.positions[3 * i + 1],
                                 cloud.positions[3 * i + 2])
            let offset = p - extent.center
            #expect(abs(offset.x) <= extent.radius + 1e-5)
            #expect(abs(offset.y) <= extent.radius + 1e-5)
            #expect(abs(offset.z) <= extent.radius + 1e-5)
        }
        // Same seed gives the same cloud.
        let again = GaussianInitializer.random(count: 500, extent: extent, shDegree: 1,
                                               initialOpacity: 0.1, seed: 5)
        #expect(again.positions == cloud.positions)
    }

    @Test func initializeFallsBackToRandomWithoutPoints() {
        let camera = Camera(width: 4, height: 4, fx: 4, fy: 4, cx: 2, cy: 2,
                            rotation: .identity, translation: SIMD3<Float>(0, 0, 3))
        let image = RGBImage(width: 4, height: 4,
                             pixels: [SIMD3<Float>](repeating: .zero, count: 16))
        let dataset = TrainingDataset(views: [TrainingView(name: "a", camera: camera, image: image)])
        var configuration = TrainingConfiguration()
        configuration.randomInitCount = 300
        let cloud = GaussianInitializer.initialize(dataset, configuration: configuration,
                                                   maxCount: 1_000)
        #expect(cloud.count == 300)
        #expect(cloud.shDegree == configuration.shDegree)

        let capped = GaussianInitializer.initialize(dataset, configuration: configuration,
                                                    maxCount: 50)
        #expect(capped.count == 50)
    }

    @Test func initializeUsesPointsWhenPresent() {
        let camera = Camera(width: 4, height: 4, fx: 4, fy: 4, cx: 2, cy: 2,
                            rotation: .identity, translation: SIMD3<Float>(0, 0, 3))
        let image = RGBImage(width: 4, height: 4,
                             pixels: [SIMD3<Float>](repeating: .zero, count: 16))
        let points = PointCloud(positions: randomPoints(count: 40, seed: 2),
                                colors: [SIMD3<Float>](repeating: SIMD3<Float>(0.5, 0.5, 0.5), count: 40))
        let dataset = TrainingDataset(views: [TrainingView(name: "a", camera: camera, image: image)],
                                      points: points)
        let cloud = GaussianInitializer.initialize(dataset, configuration: TrainingConfiguration(),
                                                   maxCount: 1_000)
        #expect(cloud.count == 40)
        // No subsampling needed: input order is kept.
        let first = SIMD3<Float>(cloud.positions[0], cloud.positions[1], cloud.positions[2])
        #expect(approxEqual(first, points.positions[0], tol: 0))
    }
}
