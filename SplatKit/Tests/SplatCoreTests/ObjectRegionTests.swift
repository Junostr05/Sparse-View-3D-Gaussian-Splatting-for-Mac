import Foundation
import Testing
import simd
import SplatCore

@Suite struct ObjectRegionTests {
    /// A world-to-camera pose looking from `centre` at `target`, with world up +y and COLMAP axes (x right, y down, z forward).
    private func camera(from centre: SIMD3<Float>, looking target: SIMD3<Float>) -> Camera {
        let forward: SIMD3<Float> = simd_normalize(target - centre)
        let worldUp: SIMD3<Float> = SIMD3<Float>(0, 1, 0)
        let down: SIMD3<Float> = -worldUp
        let right: SIMD3<Float> = simd_normalize(simd_cross(down, forward))
        let down2: SIMD3<Float> = simd_cross(forward, right)
        // Rows of R are the camera axes, so the columns are (right.j, down2.j, forward.j).
        let rotation = Matrix3(
            c0: SIMD3<Float>(right.x, down2.x, forward.x),
            c1: SIMD3<Float>(right.y, down2.y, forward.y),
            c2: SIMD3<Float>(right.z, down2.z, forward.z))
        let translation: SIMD3<Float> = -(rotation * centre)
        return Camera(width: 100, height: 100, fx: 100, fy: 100, cx: 50, cy: 50,
                      rotation: rotation, translation: translation)
    }

    private func orbit(radius: Float, height: Float, count: Int, target: SIMD3<Float>) -> [Camera] {
        (0..<count).map { index in
            let angle: Float = Float(index) / Float(count) * 2 * .pi
            let centre = SIMD3<Float>(target.x + radius * cos(angle), target.y + height, target.z + radius * sin(angle))
            return camera(from: centre, looking: target)
        }
    }

    @Test func centreIsWhereTheCamerasLookOnAnOrbit() throws {
        let cameras: [Camera] = orbit(radius: 2, height: 0.5, count: 12, target: SIMD3<Float>(1, 0, -1))
        let region = try #require(ObjectRegion.estimate(cameras: cameras))
        #expect(simd_length(region.center - SIMD3<Float>(1, 0, -1)) < 1e-3)
    }

    @Test func cylinderIsTallerThanItIsWide() throws {
        let cameras: [Camera] = orbit(radius: 2, height: 0.5, count: 12, target: .zero)
        let region = try #require(ObjectRegion.estimate(cameras: cameras))
        #expect(region.verticalHalfHeight > region.horizontalRadius)
        // Points inside the cylinder: on the axis, near the rim, and high up.
        #expect(region.contains(region.center))
        #expect(region.contains(region.center + SIMD3<Float>(0.9 * region.horizontalRadius, 0, 0)))
        #expect(region.contains(region.center + region.up * (0.9 * region.verticalHalfHeight)))
        // Outside: just past the rim, and just above the top.
        #expect(!region.contains(region.center + SIMD3<Float>(1.1 * region.horizontalRadius, 0, 0)))
        #expect(!region.contains(region.center + region.up * (1.1 * region.verticalHalfHeight)))
    }

    @Test func userScaleGrowsTheRadiusLinearly() throws {
        let cameras: [Camera] = orbit(radius: 2, height: 0.5, count: 12, target: .zero)
        let base = try #require(ObjectRegion.estimate(cameras: cameras, radiusScale: 1))
        let double = try #require(ObjectRegion.estimate(cameras: cameras, radiusScale: 2))
        #expect(abs(double.horizontalRadius - 2 * base.horizontalRadius) < 1e-4)
        #expect(abs(double.verticalHalfHeight - 2 * base.verticalHalfHeight) < 1e-4)
    }

    @Test func aStraightLineOfCamerasStillGivesARegionAheadOfThem() throws {
        // Parallel rays do not meet: the region is placed along the view, in front of the cameras.
        let cameras: [Camera] = (0..<5).map { index in
            camera(from: SIMD3<Float>(Float(index) * 0.2, 0, 0), looking: SIMD3<Float>(0, 0, 3))
        }
        let region = try #require(ObjectRegion.estimate(cameras: cameras))
        #expect(region.center.z > 0)
        #expect(region.horizontalRadius > 0)
    }

    @Test func fewerThanTwoCamerasGiveNoRegion() {
        #expect(ObjectRegion.estimate(cameras: []) == nil)
        #expect(ObjectRegion.estimate(cameras: [camera(from: .zero, looking: SIMD3<Float>(0, 0, 1))]) == nil)
    }
}
