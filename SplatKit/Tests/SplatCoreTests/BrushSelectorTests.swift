import Foundation
import Testing
import SplatCore

/// Pinhole camera with focal length 1 and principal point at the origin, so that a camera-space
/// point (x, y, z) projects to pixel (x / z, y / z). Hand-computed expectations stay simple.
private func unitCamera() -> Camera {
    Camera(width: 640, height: 480, fx: 1, fy: 1, cx: 0, cy: 0,
           rotation: .identity, translation: SIMD3(0, 0, 0))
}

private func flatten(_ points: [SIMD3<Float>]) -> [Float] {
    var result: [Float] = []
    result.reserveCapacity(3 * points.count)
    for point in points {
        result.append(point.x)
        result.append(point.y)
        result.append(point.z)
    }
    return result
}

/// Brute-force reference: projects each point with the same function and compares against every
/// stroke point. Used to validate the grid implementation.
private func bruteForceSelect(positions: [Float], camera: Camera, stroke: [SIMD2<Float>],
                              radius: Float, near: Float) -> [Int] {
    var selected: [Int] = []
    let radiusSquared: Double = Double(radius) * Double(radius)
    for gaussian in 0..<(positions.count / 3) {
        let center = SIMD3<Float>(positions[3 * gaussian], positions[3 * gaussian + 1],
                                  positions[3 * gaussian + 2])
        guard center.x.isFinite, center.y.isFinite, center.z.isFinite else { continue }
        guard let projected = DepthGeometry.project(center, camera: camera, near: near) else {
            continue
        }
        let u: Double = Double(projected.u)
        let v: Double = Double(projected.v)
        var hit = false
        for point in stroke {
            let deltaX: Double = u - Double(point.x)
            let deltaY: Double = v - Double(point.y)
            if deltaX * deltaX + deltaY * deltaY <= radiusSquared {
                hit = true
                break
            }
        }
        if hit {
            selected.append(gaussian)
        }
    }
    return selected
}

@Suite struct BrushSelectorTests {
    @Test func singleStrokePointSelectsWithinRadiusInclusive() {
        // Projections with the unit camera: pixel = (x / z, y / z).
        let points: [SIMD3<Float>] = [
            SIMD3(3, 4, 1),         // pixel (3, 4): distance 5, exactly on the radius
            SIMD3(3.01, 4, 1),      // distance slightly above 5
            SIMD3(0, 0, 2),         // pixel (0, 0): centre of the brush
            SIMD3(-1, 1, 1),        // distance sqrt(2)
            SIMD3(0, 0, 0.02),      // depth 0.02 > near 0.01, pixel (0, 0)
            SIMD3(0, 0, 0.005),     // depth below near
            SIMD3(0, 0, -1),        // behind the camera, pixel (0, 0) before the depth test
            SIMD3(0, 0, 0),         // depth 0
        ]
        let mask = BrushSelector.select(positions: flatten(points), camera: unitCamera(),
                                        stroke: [SIMD2(0, 0)], radiusPixels: 5)
        #expect(mask.count == points.count)
        #expect(mask.selectedIndices() == [0, 2, 3, 4])
    }

    @Test func strokeWithSeveralPointsSelectsNearAnyPoint() {
        let stroke: [SIMD2<Float>] = [SIMD2(0, 0), SIMD2(10, 0), SIMD2(10, 10)]
        let points: [SIMD3<Float>] = [
            SIMD3(9.5, 9.5, 1),     // near (10, 10)
            SIMD3(5, 0.5, 1),       // about 5 from both (0, 0) and (10, 0)
            SIMD3(5, 0, 1),         // midway, 5 from each endpoint
            SIMD3(20, 0, 1),        // 10 from (10, 0)
            SIMD3(10.5, 0, 1),      // 0.5 from (10, 0)
            SIMD3(0, 1, 1),         // exactly 1 from (0, 0)
        ]
        let mask = BrushSelector.select(positions: flatten(points), camera: unitCamera(),
                                        stroke: stroke, radiusPixels: 1)
        #expect(mask.selectedIndices() == [0, 4, 5])
    }

    @Test func emptyStrokeOrNonPositiveRadiusSelectsNothing() {
        let points: [SIMD3<Float>] = [SIMD3(0, 0, 1), SIMD3(1, 1, 1)]
        let flat = flatten(points)
        let empty = BrushSelector.select(positions: flat, camera: unitCamera(),
                                         stroke: [], radiusPixels: 10)
        #expect(empty.count == 2)
        #expect(empty.isEmpty)

        let zeroRadius = BrushSelector.select(positions: flat, camera: unitCamera(),
                                              stroke: [SIMD2(0, 0)], radiusPixels: 0)
        #expect(zeroRadius.isEmpty)

        let negativeRadius = BrushSelector.select(positions: flat, camera: unitCamera(),
                                                  stroke: [SIMD2(0, 0)], radiusPixels: -3)
        #expect(negativeRadius.isEmpty)

        let nanRadius = BrushSelector.select(positions: flat, camera: unitCamera(),
                                             stroke: [SIMD2(0, 0)], radiusPixels: .nan)
        #expect(nanRadius.isEmpty)
    }

    @Test func nonFinitePositionsAreNeverSelected() {
        let points: [SIMD3<Float>] = [
            SIMD3(.nan, 0, 1),
            SIMD3(0, .infinity, 1),
            SIMD3(0, 0, .nan),
            SIMD3(0, 0, .infinity),
            SIMD3(0, 0, 1),         // the only valid point, at the brush centre
        ]
        let mask = BrushSelector.select(positions: flatten(points), camera: unitCamera(),
                                        stroke: [SIMD2(0, 0)], radiusPixels: 2)
        #expect(mask.selectedIndices() == [4])
    }

    @Test func nonFiniteStrokePointsAreIgnored() {
        let stroke: [SIMD2<Float>] = [SIMD2(.nan, 0), SIMD2(0, .infinity)]
        let mask = BrushSelector.select(positions: flatten([SIMD3(0, 0, 1)]), camera: unitCamera(),
                                        stroke: stroke, radiusPixels: 2)
        #expect(mask.isEmpty)
    }

    @Test func matchesBruteForceOnRandomScene() {
        var rng = SeededGenerator(seed: 5)
        let camera = Camera(
            width: 640, height: 480, fx: 300, fy: 310, cx: 320, cy: 240,
            rotation: Matrix3.rotation(quaternion: SIMD4<Float>(0.9, 0.1, -0.3, 0.2)),
            translation: SIMD3<Float>(0.2, -0.1, 0.5)
        )
        var points: [SIMD3<Float>] = []
        for _ in 0..<5_000 {
            let x: Float = Float.random(in: -3...3, using: &rng)
            let y: Float = Float.random(in: -3...3, using: &rng)
            let z: Float = Float.random(in: -1...6, using: &rng)
            points.append(SIMD3(x, y, z))
        }
        let flat = flatten(points)

        for radius in [Float(3), Float(25), Float(120)] {
            var stroke: [SIMD2<Float>] = []
            for _ in 0..<40 {
                let u: Float = Float.random(in: -50...700, using: &rng)
                let v: Float = Float.random(in: -50...530, using: &rng)
                stroke.append(SIMD2(u, v))
            }
            let mask = BrushSelector.select(positions: flat, camera: camera, stroke: stroke,
                                            radiusPixels: radius, near: 0.01)
            let expected = bruteForceSelect(positions: flat, camera: camera, stroke: stroke,
                                            radius: radius, near: 0.01)
            #expect(mask.selectedIndices() == expected, "radius \(radius)")
        }
    }

    @Test func longStrokeOnLargeCloudSelectsPlantedPoints() {
        var rng = SeededGenerator(seed: 9)
        let count: Int = 200_000
        var flat: [Float] = []
        flat.reserveCapacity(3 * count)
        for _ in 0..<count {
            flat.append(Float.random(in: -4...4, using: &rng))
            flat.append(Float.random(in: -4...4, using: &rng))
            flat.append(Float.random(in: 1...8, using: &rng))
        }
        // Gaussian 0 projects onto the stroke point below; Gaussian 1 projects far away at (500, 500).
        flat[0] = 0.5
        flat[1] = 0.25
        flat[2] = 2
        flat[3] = 1000
        flat[4] = 1000
        flat[5] = 2

        // Pixel of Gaussian 0 is (0.5 / 2, 0.25 / 2) = (0.25, 0.125) with the unit camera below.
        let mask = BrushSelector.select(positions: flat, camera: unitCamera(),
                                        stroke: [SIMD2(0.25, 0.125)], radiusPixels: 1)
        #expect(mask.count == count)
        #expect(mask[0])
        #expect(!mask[1])

        // A 300-point stroke over the same cloud must still run through the grid path and keep the
        // mask sized to the cloud.
        var stroke: [SIMD2<Float>] = []
        for step in 0..<300 {
            let t: Float = Float(step) / 299
            stroke.append(SIMD2(3 * t - 1.5, 3 * t - 1.5))
        }
        let swept = BrushSelector.select(positions: flat, camera: unitCamera(), stroke: stroke,
                                         radiusPixels: 2)
        #expect(swept.count == count)
        // Pixel (0.25, 0.125) is about 0.09 from the diagonal stroke y = x, well within radius 2.
        #expect(swept[0])
    }
}

@Suite struct BrushSelectorFrontOnlyTests {
    /// Single stroke point at pixel (0, 0), radius 1: a centre at (0, 0, z) is a candidate at depth z.
    private static let centreStroke: [SIMD2<Float>] = [SIMD2(0, 0)]

    /// Runs the front-only selector with the unit camera and returns the selected indices.
    private func select(_ points: [SIMD3<Float>], settings: FrontOnlySettings,
                        stroke: [SIMD2<Float>] = BrushSelectorFrontOnlyTests.centreStroke,
                        radius: Float = 1) -> [Int] {
        let mask = BrushSelector.select(positions: flatten(points), camera: unitCamera(),
                                        stroke: stroke, radiusPixels: radius, frontOnly: settings)
        return mask.selectedIndices()
    }

    @Test func floaterInFrontOfWallSelectsOnlyTheFloater() {
        // Index 0 is an opaque floater at depth 2, index 1 an opaque wall at depth 4, both at the brush centre.
        let points: [SIMD3<Float>] = [SIMD3(0, 0, 2), SIMD3(0, 0, 4)]
        let xray = BrushSelector.select(positions: flatten(points), camera: unitCamera(),
                                        stroke: BrushSelectorFrontOnlyTests.centreStroke, radiusPixels: 1)
        #expect(xray.selectedIndices() == [0, 1])
        #expect(select(points, settings: FrontOnlySettings(opacities: [0.9, 0.9])) == [0])
    }

    @Test func floaterBehindSurfaceIsSkipped() {
        // The opaque wall at depth 2 is in front, the floater at depth 4 lies behind it.
        let points: [SIMD3<Float>] = [SIMD3(0, 0, 4), SIMD3(0, 0, 2)]
        #expect(select(points, settings: FrontOnlySettings(opacities: [1, 1])) == [1])
    }

    @Test func semiTransparentFrontLayerDoesNotHideTheSurface() {
        // Index 0 is a layer at depth 2 with opacity 0.1, below minimumOpacity 0.3. The opaque wall at depth 4 is
        // the surface, so the layer (in front of the surface) and the wall are both selected.
        let points: [SIMD3<Float>] = [SIMD3(0, 0, 2), SIMD3(0, 0, 4)]
        #expect(select(points, settings: FrontOnlySettings(opacities: [0.1, 1])) == [0, 1])
        // An opaque candidate at depth 9 lies behind the surface at depth 4 and is rejected.
        let behind: [SIMD3<Float>] = [SIMD3(0, 0, 2), SIMD3(0, 0, 4), SIMD3(0, 0, 9)]
        #expect(select(behind, settings: FrontOnlySettings(opacities: [0.1, 1, 1])) == [0, 1])
    }

    @Test func opacityAtMinimumCountsAsOpaque() {
        // Opacity exactly at minimumOpacity (0.3) defines the surface, so the wall behind it is dropped.
        let points: [SIMD3<Float>] = [SIMD3(0, 0, 2), SIMD3(0, 0, 4)]
        #expect(select(points, settings: FrontOnlySettings(opacities: [0.3, 1])) == [0])
    }

    @Test func toleranceBoundaryIsInclusive() {
        // Surface depth 2 and tolerance 0.5 give a limit of exactly 3, which is exactly representable.
        let points: [SIMD3<Float>] = [SIMD3(0, 0, 2), SIMD3(0, 0, 3), SIMD3(0, 0, 3.001)]
        let settings = FrontOnlySettings(opacities: [1, 1, 1], depthTolerance: 0.5)
        #expect(select(points, settings: settings) == [0, 1])
    }

    @Test func zeroToleranceKeepsOnlyCandidatesAtTheSurfaceDepth() {
        let points: [SIMD3<Float>] = [SIMD3(0, 0, 2), SIMD3(0, 0, 2), SIMD3(0, 0, 2.5)]
        let settings = FrontOnlySettings(opacities: [1, 1, 1], depthTolerance: 0)
        #expect(select(points, settings: settings) == [0, 1])
    }

    @Test func negativeAndNonFiniteToleranceAreZero() {
        let points: [SIMD3<Float>] = [SIMD3(0, 0, 2), SIMD3(0, 0, 2.5)]
        for tolerance: Float in [-1, .nan] {
            let settings = FrontOnlySettings(opacities: [1, 1], depthTolerance: tolerance)
            #expect(select(points, settings: settings) == [0])
        }
    }

    @Test func nonFiniteOpacityCountsAsZero() {
        // +infinity must not count as opaque: the surface is the wall at depth 2, so the wall is kept and the
        // floater at depth 1 is in front of it. If the infinite value counted, the wall would be dropped.
        let points: [SIMD3<Float>] = [SIMD3(0, 0, 1), SIMD3(0, 0, 2)]
        #expect(select(points, settings: FrontOnlySettings(opacities: [.infinity, 1])) == [0, 1])
        #expect(select(points, settings: FrontOnlySettings(opacities: [.nan, 1])) == [0, 1])
    }

    @Test func surfaceIsLocalToEachScreenCell() {
        // The floater at pixel (0, 0) and the wall at pixel (30, 0) are in different cells, so the wall's depth
        // does not affect the floater's cell and both survive.
        let stroke: [SIMD2<Float>] = [SIMD2(0, 0), SIMD2(30, 0)]
        let points: [SIMD3<Float>] = [SIMD3(0, 0, 2), SIMD3(30 * 4, 0, 4), SIMD3(0, 0, 4)]
        // Index 2 is behind the floater in the same cell and is dropped.
        let settings = FrontOnlySettings(opacities: [0.9, 0.9, 0.9])
        #expect(select(points, settings: settings, stroke: stroke, radius: 1) == [0, 1])
    }

    @Test func identicalToXRayWhenAllCandidatesShareADepth() {
        var rng = SeededGenerator(seed: 21)
        var points: [SIMD3<Float>] = []
        var opacities: [Float] = []
        for _ in 0..<2_000 {
            points.append(SIMD3(Float.random(in: -40...40, using: &rng),
                                 Float.random(in: -40...40, using: &rng), 3))
            opacities.append(Float.random(in: 0...1, using: &rng))
        }
        let stroke: [SIMD2<Float>] = [SIMD2(0, 0), SIMD2(20, -10)]
        let xray = BrushSelector.select(positions: flatten(points), camera: unitCamera(),
                                        stroke: stroke, radiusPixels: 15)
        let front = BrushSelector.select(positions: flatten(points), camera: unitCamera(),
                                         stroke: stroke, radiusPixels: 15,
                                         frontOnly: FrontOnlySettings(opacities: opacities))
        #expect(!xray.isEmpty)
        #expect(front == xray)
    }

    @Test func matchesBruteForceReferenceOnRandomScene() {
        var rng = SeededGenerator(seed: 33)
        let camera = Camera(
            width: 640, height: 480, fx: 300, fy: 310, cx: 320, cy: 240,
            rotation: Matrix3.rotation(quaternion: SIMD4<Float>(0.9, 0.1, -0.3, 0.2)),
            translation: SIMD3<Float>(0.2, -0.1, 0.5)
        )
        var points: [SIMD3<Float>] = []
        var opacities: [Float] = []
        for _ in 0..<5_000 {
            points.append(SIMD3(Float.random(in: -3...3, using: &rng),
                                Float.random(in: -3...3, using: &rng),
                                Float.random(in: -1...6, using: &rng)))
            let roll: Float = Float.random(in: 0...1, using: &rng)
            opacities.append(roll < 0.02 ? .nan : roll)
        }
        let flat = flatten(points)
        var stroke: [SIMD2<Float>] = []
        for _ in 0..<30 {
            stroke.append(SIMD2(Float.random(in: 0...640, using: &rng),
                                Float.random(in: 0...480, using: &rng)))
        }
        let settings = FrontOnlySettings(opacities: opacities, minimumOpacity: 0.4,
                                         depthTolerance: 0.1, cellPixels: 9)
        for radius in [Float(4), Float(40)] {
            let mask = BrushSelector.select(positions: flat, camera: camera, stroke: stroke,
                                            radiusPixels: radius, near: 0.01, frontOnly: settings)
            let expected = bruteForceFrontOnly(positions: flat, camera: camera, stroke: stroke,
                                               radius: radius, near: 0.01, settings: settings)
            #expect(mask.selectedIndices() == expected, "radius \(radius)")
        }
    }

    /// Reference with the same rules as the selector but no grid: quadratic candidate search, and the
    /// surface per cell taken from a dictionary built in a separate pass.
    private func bruteForceFrontOnly(positions: [Float], camera: Camera, stroke: [SIMD2<Float>],
                                     radius: Float, near: Float, settings: FrontOnlySettings) -> [Int] {
        var minX: Double = .infinity
        var minY: Double = .infinity
        for point in stroke where point.x.isFinite && point.y.isFinite {
            minX = min(minX, Double(point.x))
            minY = min(minY, Double(point.y))
        }
        let cell: Double = Double(settings.cellPixels)
        struct Candidate { let index: Int; let depth: Double; let key: String }
        var candidates: [Candidate] = []
        let radiusSquared: Double = Double(radius) * Double(radius)
        for gaussian in 0..<(positions.count / 3) {
            let center = SIMD3<Float>(positions[3 * gaussian], positions[3 * gaussian + 1],
                                      positions[3 * gaussian + 2])
            guard center.x.isFinite, center.y.isFinite, center.z.isFinite else { continue }
            guard let projected = DepthGeometry.project(center, camera: camera, near: near) else { continue }
            let u = Double(projected.u)
            let v = Double(projected.v)
            let hit = stroke.contains { point in
                let dx = u - Double(point.x)
                let dy = v - Double(point.y)
                return point.x.isFinite && point.y.isFinite && dx * dx + dy * dy <= radiusSquared
            }
            guard hit else { continue }
            let cellX = Int(((u - minX) / cell).rounded(.down))
            let cellY = Int(((v - minY) / cell).rounded(.down))
            candidates.append(Candidate(index: gaussian, depth: Double(projected.z),
                                        key: "\(cellX),\(cellY)"))
        }
        var surface: [String: Double] = [:]
        for candidate in candidates {
            let opacity = settings.opacities[candidate.index]
            guard opacity.isFinite, opacity >= settings.minimumOpacity else { continue }
            surface[candidate.key] = min(surface[candidate.key] ?? .infinity, candidate.depth)
        }
        for candidate in candidates where surface[candidate.key] == nil {
            let nearest = candidates.filter { $0.key == candidate.key }.map(\.depth).min() ?? candidate.depth
            surface[candidate.key] = nearest
        }
        let tolerance = Double(max(settings.depthTolerance, 0))
        return candidates.filter { candidate in
            candidate.depth <= surface[candidate.key]! * (1 + tolerance)
        }.map(\.index).sorted()
    }
}
