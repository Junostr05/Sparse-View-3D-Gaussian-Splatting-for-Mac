/// Axis-aligned bounding box of a point set.
public struct Bounds: Sendable, Equatable {
    public var min: SIMD3<Float>
    public var max: SIMD3<Float>

    public init(min: SIMD3<Float>, max: SIMD3<Float>) {
        self.min = min
        self.max = max
    }

    public var center: SIMD3<Float> { (min + max) / 2 }

    public var extent: SIMD3<Float> { max - min }
}

extension GaussianCloud {
    /// Per-axis percentile bounds over a deterministic strided sample of at most
    /// about `maxSamples` Gaussians, so that a few far-away outliers do not dominate
    /// the framing. An empty cloud yields `Bounds(min: .zero, max: .zero)`.
    public func robustBounds(lowerPercentile: Float = 0.05,
                             upperPercentile: Float = 0.95,
                             maxSamples: Int = 100_000) -> Bounds {
        precondition(maxSamples > 0, "maxSamples must be positive")
        let n = count
        guard n > 0 else { return Bounds(min: .zero, max: .zero) }

        let step = Swift.max(1, n / maxSamples)
        let sampled = Array(Swift.stride(from: 0, to: n, by: step))
        let last = sampled.count - 1

        func rank(_ percentile: Float) -> Int {
            Swift.min(Swift.max(Int(Float(last) * percentile), 0), last)
        }

        var lower = SIMD3<Float>.zero
        var upper = SIMD3<Float>.zero
        for axis in 0..<3 {
            var coords = sampled.map { positions[3 * $0 + axis] }
            coords.sort()
            lower[axis] = coords[rank(lowerPercentile)]
            upper[axis] = coords[rank(upperPercentile)]
        }
        return Bounds(min: lower, max: upper)
    }
}
