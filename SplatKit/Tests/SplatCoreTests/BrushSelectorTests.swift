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
