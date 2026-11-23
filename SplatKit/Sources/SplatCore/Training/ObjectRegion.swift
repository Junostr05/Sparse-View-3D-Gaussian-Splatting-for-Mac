import Foundation

/// The region a capture is about: a vertical cylinder around the object the cameras look at.
///
/// It is estimated from the camera poses alone, so it does not depend on the scale of the reconstruction. The centre
/// is the point nearest to all viewing rays (least squares). The radius is a fraction of the median distance from the
/// cameras to that centre, and the cylinder is taller than it is wide.
public struct ObjectRegion: Sendable, Equatable {
    /// Centre of the cylinder's axis, in world space.
    public var center: SIMD3<Float>
    /// Unit up direction of the cylinder's axis, in world space.
    public var up: SIMD3<Float>
    /// Radius of the cylinder in the horizontal plane, in world units.
    public var horizontalRadius: Float
    /// Half of the cylinder's height along `up`, in world units.
    public var verticalHalfHeight: Float

    public init(center: SIMD3<Float>, up: SIMD3<Float>, horizontalRadius: Float, verticalHalfHeight: Float) {
        self.center = center
        self.up = up
        self.horizontalRadius = horizontalRadius
        self.verticalHalfHeight = verticalHalfHeight
    }

    /// Horizontal radius as a fraction of the median camera distance, before the user's scale.
    public static let horizontalFraction: Float = 0.35
    /// The cylinder is this many times taller than its radius (half height over radius).
    public static let verticalFactor: Float = 1.6

    /// True when the point lies inside the cylinder.
    public func contains(_ point: SIMD3<Float>) -> Bool {
        let offset: SIMD3<Float> = point - center
        let height: Float = (offset * up).sum()
        let radial: SIMD3<Float> = offset - up * height
        let radialSquared: Float = (radial * radial).sum()
        return abs(height) <= verticalHalfHeight && radialSquared <= horizontalRadius * horizontalRadius
    }

    /// The region of a set of cameras, or `nil` when it cannot be placed (fewer than two cameras, or viewing rays that
    /// do not meet in front of the cameras).
    /// - Parameters:
    ///   - cameras: The cameras of the capture (world-to-camera poses in the usual COLMAP convention).
    ///   - radiusScale: The user's size: 1 is the default, larger takes more of the scene.
    public static func estimate(cameras: [Camera], radiusScale: Float = 1) -> ObjectRegion? {
        guard cameras.count >= 2, radiusScale > 0 else {
            return nil
        }
        var centers: [SIMD3<Double>] = []
        var forwards: [SIMD3<Double>] = []
        var ups: [SIMD3<Double>] = []
        centers.reserveCapacity(cameras.count)
        forwards.reserveCapacity(cameras.count)
        ups.reserveCapacity(cameras.count)
        for camera in cameras {
            let rotationT: Matrix3 = camera.rotation.transposed
            let center: SIMD3<Float> = camera.center
            let forward: SIMD3<Float> = rotationT * SIMD3<Float>(0, 0, 1)
            // COLMAP cameras have y pointing down, so up is -y.
            let up: SIMD3<Float> = -(rotationT * SIMD3<Float>(0, 1, 0))
            centers.append(SIMD3<Double>(Double(center.x), Double(center.y), Double(center.z)))
            forwards.append(normalized(SIMD3<Double>(Double(forward.x), Double(forward.y), Double(forward.z))))
            ups.append(normalized(SIMD3<Double>(Double(up.x), Double(up.y), Double(up.z))))
        }

        // Least squares point nearest to all rays: sum of (I - f fᵀ) p = sum of (I - f fᵀ) C.
        var normalMatrix: [Double] = [Double](repeating: 0, count: 9)
        var rightHand: SIMD3<Double> = .zero
        for index in 0..<centers.count {
            let f: SIMD3<Double> = forwards[index]
            let c: SIMD3<Double> = centers[index]
            for row in 0..<3 {
                for column in 0..<3 {
                    let identity: Double = row == column ? 1 : 0
                    normalMatrix[row * 3 + column] += identity - component(f, row) * component(f, column)
                }
            }
            rightHand += c - f * dot(f, c)
        }
        let meanCenter: SIMD3<Double> = centers.reduce(.zero, +) / Double(centers.count)
        let spread: Double = centers.map { length($0 - meanCenter) }.reduce(0, +) / Double(centers.count)

        var target: SIMD3<Double>
        if let solution = solve3x3(normalMatrix, rightHand), isFinite(solution) {
            target = solution
        } else {
            // Rays that do not meet, as in a straight line of cameras: the mean camera, pushed along its view.
            let meanForward: SIMD3<Double> = normalized(forwards.reduce(.zero, +))
            target = meanCenter + meanForward * spread
        }

        // The distance that sets the size is the median camera distance to the target.
        var cameraDistances: [Double] = centers.map { length($0 - target) }
        cameraDistances.sort()
        let median: Double = cameraDistances[cameraDistances.count / 2]
        guard median.isFinite, median > 0 else {
            return nil
        }
        var meanUp: SIMD3<Double> = normalized(ups.reduce(.zero, +))
        if !isFinite(meanUp) || length(meanUp) == 0 {
            meanUp = SIMD3<Double>(0, -1, 0)
        }
        let horizontal: Double = Double(radiusScale) * Double(horizontalFraction) * median
        return ObjectRegion(center: SIMD3<Float>(Float(target.x), Float(target.y), Float(target.z)),
                            up: SIMD3<Float>(Float(meanUp.x), Float(meanUp.y), Float(meanUp.z)),
                            horizontalRadius: Float(horizontal),
                            verticalHalfHeight: Float(horizontal * Double(verticalFactor)))
    }

    // MARK: - Small vector helpers (Double, for the least squares)

    private static func isFinite(_ v: SIMD3<Double>) -> Bool {
        v.x.isFinite && v.y.isFinite && v.z.isFinite
    }

    private static func component(_ v: SIMD3<Double>, _ index: Int) -> Double {
        index == 0 ? v.x : (index == 1 ? v.y : v.z)
    }

    private static func dot(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double {
        (a * b).sum()
    }

    private static func length(_ v: SIMD3<Double>) -> Double {
        dot(v, v).squareRoot()
    }

    private static func normalized(_ v: SIMD3<Double>) -> SIMD3<Double> {
        let n: Double = length(v)
        return n > 0 ? v / n : .zero
    }

    /// Solves a 3x3 system given row-major; `nil` when the matrix is singular (the rays are parallel).
    private static func solve3x3(_ m: [Double], _ b: SIMD3<Double>) -> SIMD3<Double>? {
        let a = m
        let det: Double = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
            + a[2] * (a[3] * a[7] - a[4] * a[6])
        let scale: Double = max(abs(a[0]) + abs(a[4]) + abs(a[8]), 1e-12)
        guard abs(det) > 1e-9 * scale * scale * scale else {
            return nil
        }
        let x: Double = (b.x * (a[4] * a[8] - a[5] * a[7]) - a[1] * (b.y * a[8] - a[5] * b.z)
            + a[2] * (b.y * a[7] - a[4] * b.z)) / det
        let y: Double = (a[0] * (b.y * a[8] - a[5] * b.z) - b.x * (a[3] * a[8] - a[5] * a[6])
            + a[2] * (a[3] * b.z - b.y * a[6])) / det
        let z: Double = (a[0] * (a[4] * b.z - b.y * a[7]) - a[1] * (a[3] * b.z - b.y * a[6])
            + b.x * (a[3] * a[7] - a[4] * a[6])) / det
        return SIMD3<Double>(x, y, z)
    }
}
