import Foundation
import Testing
import SplatCore

/// Pure colour and metric functions of the viewer display modes (no Metal).
@Suite struct ViewerDisplayTests {
    /// A cloud with degree-0 SH only, from per-Gaussian values. `logScales` and `opacities` are one value per Gaussian.
    private func makeCloud(positions: [SIMD3<Float>],
                           logScale: (Int) -> Float = { _ in log(0.05) },
                           opacityLogit: (Int) -> Float = { _ in 2 },
                           dc: (Int) -> SIMD3<Float> = { _ in .zero }) throws -> GaussianCloud {
        let n: Int = positions.count
        var flatPositions: [Float] = []
        var logScales: [Float] = []
        var rotations: [Float] = []
        var opacities: [Float] = []
        var sh: [Float] = []
        for i in 0..<n {
            flatPositions += [positions[i].x, positions[i].y, positions[i].z]
            logScales += [logScale(i), logScale(i), logScale(i)]
            rotations += [1, 0, 0, 0]
            opacities.append(opacityLogit(i))
            let c: SIMD3<Float> = dc(i)
            sh += [c.x, c.y, c.z]
        }
        return try GaussianCloud(shDegree: 0, positions: flatPositions, logScales: logScales,
                                 rotations: rotations, opacityLogits: opacities, sh: sh)
    }

    /// A line of `n` points spaced one unit apart on the x axis.
    private func line(_ n: Int) -> [SIMD3<Float>] {
        (0..<n).map { SIMD3<Float>(Float($0), 0, 0) }
    }

    @Test func modesCycleInOrderAndWrap() {
        #expect(ViewerDisplayMode.splat.next == .points)
        #expect(ViewerDisplayMode.points.next == .depth)
        #expect(ViewerDisplayMode.depth.next == .shapes)
        #expect(ViewerDisplayMode.shapes.next == .heatmap)
        #expect(ViewerDisplayMode.heatmap.next == .splat)
        #expect(ViewerDisplayMode.allCases.count == 5)
    }

    @Test func zKeyCyclesTheDisplayMode() {
        #expect(ViewCommand.command(character: "z", control: false, shift: false, keypad: false) == .cycleDisplayMode)
    }

    @Test func pointColorsUseTheDegreeZeroSH() throws {
        let cloud = try makeCloud(positions: line(3), dc: { i in
            [SIMD3<Float>(0, 1, -10), SIMD3<Float>(-2, 0.5, 0), SIMD3<Float>(0, 0, 0)][i]
        })
        let colors = ViewerColors.pointColors(cloud)
        #expect(colors.count == 3)
        // 0.5 + C0 · 0 is mid grey; 0.5 + C0 · -2 is below zero and clamps; 0.5 + C0 · -10 clamps too.
        let c0: Float = 0.28209479177387814
        #expect(abs(colors[0].x - 0.5) < 1e-6)
        #expect(abs(colors[0].y - (0.5 + c0)) < 1e-6)
        #expect(colors[0].z == 0)
        #expect(colors[1].x == 0)
        #expect(abs(colors[1].y - (0.5 + 0.5 * c0)) < 1e-6)
        for color in colors {
            #expect(color.w == 1)
            #expect(color.x >= 0 && color.x <= 1 && color.y >= 0 && color.y <= 1 && color.z >= 0 && color.z <= 1)
        }
    }

    @Test func shapeColorsAreDeterministicAndVaried() {
        let first = ViewerColors.shapeColors(count: 64)
        let second = ViewerColors.shapeColors(count: 64)
        #expect(first.count == 64)
        #expect(first == second)
        #expect(Set(first.map { "\($0.x),\($0.y),\($0.z)" }).count > 40)
        for color in first {
            #expect(color.x >= 0 && color.x <= 1 && color.y >= 0 && color.y <= 1 && color.z >= 0 && color.z <= 1)
            #expect(color.w == 1)
        }
        #expect(ViewerColors.shapeColors(count: 0).isEmpty)
    }

    @Test func turboRunsFromBlueToRedAndClamps() {
        let cold: SIMD3<Float> = ViewerColors.turbo(0)
        let warm: SIMD3<Float> = ViewerColors.turbo(1)
        #expect(cold.z > cold.x)
        #expect(warm.x > warm.z)
        #expect(ViewerColors.turbo(-3) == cold)
        #expect(ViewerColors.turbo(7) == warm)
        for step in 0...20 {
            let rgb: SIMD3<Float> = ViewerColors.turbo(Float(step) / 20)
            #expect(rgb.x >= 0 && rgb.x <= 1 && rgb.y >= 0 && rgb.y <= 1 && rgb.z >= 0 && rgb.z <= 1)
        }
    }

    @Test func heatmapSizeMakesHugeGaussiansRed() throws {
        // Log scales from small to large, twenty of them. The largest is clamped to the warm end.
        let cloud = try makeCloud(positions: line(20), logScale: { i in log(0.01) + Float(i) * 0.2 })
        let colors = ViewerColors.heatmap(cloud, metric: .size, sceneRadius: 1)
        #expect(colors.count == 20)
        #expect(colors[19].x > colors[19].z)
        #expect(colors[0].z > colors[0].x)
    }

    @Test func heatmapOpacityMakesTransparentGaussiansRed() throws {
        let cloud = try makeCloud(positions: line(20), opacityLogit: { i in -6 + Float(i) * 0.6 })
        let colors = ViewerColors.heatmap(cloud, metric: .opacity, sceneRadius: 1)
        #expect(colors[0].x > colors[0].z)
        #expect(colors[19].z > colors[19].x)
    }

    @Test func heatmapIsolationMakesDistantGaussiansRed() throws {
        // Thirty points in a unit grid and one far away: the far one is the isolated (suspicious) one.
        var positions: [SIMD3<Float>] = []
        for i in 0..<30 {
            positions.append(SIMD3<Float>(Float(i % 6), Float(i / 6), 0))
        }
        positions.append(SIMD3<Float>(100, 100, 100))
        let cloud = try makeCloud(positions: positions)
        let colors = ViewerColors.heatmap(cloud, metric: .isolation, sceneRadius: 100)
        #expect(colors[30].x > colors[30].z)
        // An interior grid point has four neighbours at distance 1, the least isolated value: blue.
        #expect(colors[14].z > colors[14].x)
    }

    @Test func isolationDistancesMatchBruteForce() {
        // Pseudo-random points from a fixed linear congruential sequence, so the test is deterministic.
        var positions: [SIMD3<Float>] = []
        var state: UInt32 = 12345
        for _ in 0..<80 {
            var coords: [Float] = []
            for _ in 0..<3 {
                state = state &* 1_664_525 &+ 1_013_904_223
                coords.append(Float(state >> 8) / Float(1 << 24) * 10)
            }
            positions.append(SIMD3<Float>(coords[0], coords[1], coords[2]))
        }
        let distances = ViewerColors.isolationDistances(positions: positions, k: 4)
        #expect(distances.count == positions.count)
        for i in 0..<positions.count {
            var others: [Float] = []
            for j in 0..<positions.count where j != i {
                let delta: SIMD3<Float> = positions[j] - positions[i]
                others.append((delta * delta).sum().squareRoot())
            }
            others.sort()
            #expect(abs(distances[i] - others[3]) < 1e-4)
        }
    }

    @Test func isolationDistancesHandleTinyInputs() {
        #expect(ViewerColors.isolationDistances(positions: []).isEmpty)
        #expect(ViewerColors.isolationDistances(positions: [SIMD3<Float>(1, 2, 3)]) == [0])
        let two = ViewerColors.isolationDistances(positions: [SIMD3<Float>(0, 0, 0), SIMD3<Float>(3, 4, 0)], k: 4)
        #expect(two.count == 2)
        #expect(abs(two[0] - 5) < 1e-5)
    }

    @Test func kdTreeQueryByLocationMatchesBruteForce() {
        var points: [SIMD3<Float>] = []
        for i in 0..<50 {
            let x: Float = Float((i * 37) % 17)
            let y: Float = Float((i * 11) % 13)
            let z: Float = Float((i * 7) % 5)
            points.append(SIMD3<Float>(x, y, z))
        }
        let tree = KDTree(points: points)
        let query = SIMD3<Float>(3.2, 4.1, 1.7)
        let found: [Int] = tree.nearestNeighbors(to: query, k: 6)
        var squared: [Float] = []
        for point in points {
            let delta: SIMD3<Float> = point - query
            squared.append((delta * delta).sum())
        }
        let brute: [Int] = points.indices.sorted { a, b in
            squared[a] < squared[b] || (squared[a] == squared[b] && a < b)
        }
        #expect(found == Array(brute.prefix(6)))
    }
}
