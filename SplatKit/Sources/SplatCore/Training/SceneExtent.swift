import Foundation

/// Scene center and radius used to scale learning rates and densification thresholds.
public struct SceneExtent: Sendable, Equatable {
    public var center: SIMD3<Float>
    public var radius: Float

    public init(center: SIMD3<Float>, radius: Float) {
        self.center = center
        self.radius = radius
    }

    /// Camera-centroid extent (`1.1 × max distance to centroid`). Falls back to the point cloud
    /// when the cameras do not span a volume, and to radius 1 when there is no usable data.
    public static func estimate(cameraCenters: [SIMD3<Float>], points: [SIMD3<Float>]) -> SceneExtent {
        let epsilon: Float = 1e-6
        let centroid = mean(cameraCenters) ?? .zero
        var maxDistance: Float = 0
        for center in cameraCenters {
            maxDistance = max(maxDistance, distance(center, centroid))
        }
        let radius = 1.1 * maxDistance
        if radius >= epsilon {
            return SceneExtent(center: centroid, radius: radius)
        }

        guard !points.isEmpty else {
            return SceneExtent(center: centroid, radius: 1)
        }
        let pointCenter = mean(points) ?? centroid
        let (low, high) = percentileBox(points, lower: 0.05, upper: 0.95)
        let halfDiagonal = distance(high, low) / 2
        if halfDiagonal < epsilon {
            return SceneExtent(center: pointCenter, radius: 1)
        }
        return SceneExtent(center: pointCenter, radius: halfDiagonal)
    }

    private static func mean(_ values: [SIMD3<Float>]) -> SIMD3<Float>? {
        guard !values.isEmpty else { return nil }
        var sum = SIMD3<Double>.zero
        for value in values {
            sum += SIMD3<Double>(Double(value.x), Double(value.y), Double(value.z))
        }
        let scale = 1 / Double(values.count)
        return SIMD3<Float>(Float(sum.x * scale), Float(sum.y * scale), Float(sum.z * scale))
    }

    private static func distance(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        let delta = a - b
        return (delta.x * delta.x + delta.y * delta.y + delta.z * delta.z).squareRoot()
    }

    /// Per-axis nearest-rank percentiles, returned as the low and high corners of the box.
    private static func percentileBox(_ points: [SIMD3<Float>],
                                      lower: Double,
                                      upper: Double) -> (SIMD3<Float>, SIMD3<Float>) {
        func rank(_ values: [Float], _ fraction: Double) -> Float {
            let sorted = values.sorted()
            let index = Int((Double(sorted.count - 1) * fraction).rounded())
            return sorted[index]
        }
        let xs = points.map { $0.x }
        let ys = points.map { $0.y }
        let zs = points.map { $0.z }
        let low = SIMD3<Float>(rank(xs, lower), rank(ys, lower), rank(zs, lower))
        let high = SIMD3<Float>(rank(xs, upper), rank(ys, upper), rank(zs, upper))
        return (low, high)
    }
}
