import Foundation

/// What the viewport draws. `splat` is the normal rendering; `depth` renders the splats and maps their depth to colour.
/// The other modes change how each Gaussian is drawn (`points`, `shapes`) or colour the Gaussians by a metric (`heatmap`).
public enum ViewerDisplayMode: String, CaseIterable, Sendable {
    case splat
    case points
    case depth
    case shapes
    case heatmap

    /// The next mode in `allCases` order, wrapping around after the last one (the Z key).
    public var next: ViewerDisplayMode {
        let modes: [ViewerDisplayMode] = Self.allCases
        let index: Int = modes.firstIndex(of: self) ?? 0
        return modes[(index + 1) % modes.count]
    }
}

/// The per-Gaussian value a heatmap colours. Red is always the suspicious end.
public enum HeatmapMetric: String, CaseIterable, Sendable {
    /// Largest axis of the Gaussian relative to the scene radius. Huge Gaussians are red.
    case size
    /// Opacity. Nearly transparent Gaussians are red.
    case opacity
    /// Distance to the 4th nearest neighbour relative to the scene radius. Isolated Gaussians are red.
    case isolation
}

/// Display parameters of one viewer frame, passed to `SplatRenderer.setDisplay`.
public struct ViewerDisplay: Sendable, Equatable {
    public var mode: ViewerDisplayMode
    /// Disc radius in rendered pixels for `points`.
    public var pointRadiusPixels: Float
    /// Opacity multiplier for `shapes`.
    public var opacityScale: Float
    /// Multiplier of the Gaussian scales in the non-splat modes.
    public var scaleScale: Float
    /// One RGB colour per Gaussian for `points`, `shapes` and `heatmap` (alpha is ignored). `nil` uses the SH colour.
    public var colorOverride: [SIMD4<Float>]?

    public init(mode: ViewerDisplayMode = .splat,
                pointRadiusPixels: Float = 3,
                opacityScale: Float = 0.35,
                scaleScale: Float = 1,
                colorOverride: [SIMD4<Float>]? = nil) {
        self.mode = mode
        self.pointRadiusPixels = pointRadiusPixels
        self.opacityScale = opacityScale
        self.scaleScale = scaleScale
        self.colorOverride = colorOverride
    }

    /// The normal rendering, which is what training uses.
    public static let splat = ViewerDisplay()
}

/// Colours of the viewer display modes. Everything here is pure; callers run it off the main actor for large clouds.
public enum ViewerColors {
    /// The degree-0 spherical harmonic constant C0 = 1 / (2√π).
    static let degreeZeroConstant: Float = 0.28209479177387814

    /// Colour stops of `turbo`, equally spaced from t = 0 to t = 1. `Shaders/Display.metal` repeats these values.
    static let colormapStops: [SIMD3<Float>] = [
        SIMD3<Float>(0.23, 0.30, 0.75),
        SIMD3<Float>(0.16, 0.56, 0.94),
        SIMD3<Float>(0.10, 0.80, 0.70),
        SIMD3<Float>(0.60, 0.90, 0.25),
        SIMD3<Float>(0.99, 0.72, 0.15),
        SIMD3<Float>(0.85, 0.12, 0.10),
    ]

    /// A turbo-like colormap: deep blue at t = 0 through green and yellow to red at t = 1. Values outside 0...1 are clamped.
    public static func turbo(_ t: Float) -> SIMD3<Float> {
        let clamped: Float = t.isNaN ? 0.5 : min(max(t, 0), 1)
        let last: Int = colormapStops.count - 1
        let x: Float = clamped * Float(last)
        let index: Int = min(Int(x), last - 1)
        let fraction: Float = x - Float(index)
        let low: SIMD3<Float> = colormapStops[index]
        let high: SIMD3<Float> = colormapStops[index + 1]
        let step: SIMD3<Float> = high - low
        return low + step * fraction
    }

    /// Colour of each Gaussian from its degree-0 SH coefficient: `0.5 + C0·f_dc`, clamped to 0...1.
    public static func pointColors(_ cloud: GaussianCloud) -> [SIMD4<Float>] {
        let n: Int = cloud.count
        let k: Int = GaussianCloud.coefficientCount(degree: cloud.shDegree)
        var colors: [SIMD4<Float>] = []
        colors.reserveCapacity(n)
        for i in 0..<n {
            let base: Int = i * k * 3
            let r: Float = min(max(0.5 + degreeZeroConstant * cloud.sh[base], 0), 1)
            let g: Float = min(max(0.5 + degreeZeroConstant * cloud.sh[base + 1], 0), 1)
            let b: Float = min(max(0.5 + degreeZeroConstant * cloud.sh[base + 2], 0), 1)
            colors.append(SIMD4<Float>(r, g, b, 1))
        }
        return colors
    }

    /// A deterministic, pleasant colour per Gaussian index: a hash of the index picks the hue.
    public static func shapeColors(count: Int) -> [SIMD4<Float>] {
        let n: Int = max(count, 0)
        var colors: [SIMD4<Float>] = []
        colors.reserveCapacity(n)
        for i in 0..<n {
            let hash: UInt32 = mixBits(UInt32(truncatingIfNeeded: i))
            let hue: Float = Float(hash) / 4_294_967_296
            let rgb: SIMD3<Float> = hsvToRGB(hue: hue, saturation: 0.55, value: 0.92)
            colors.append(SIMD4<Float>(rgb.x, rgb.y, rgb.z, 1))
        }
        return colors
    }

    /// Heatmap colours of the cloud for `metric`. `sceneRadius` normalises the size and the isolation (values ≤ 0 use 1).
    /// The metric is clamped at its 1st and 99th percentiles and mapped through `turbo`, red being the suspicious end.
    public static func heatmap(_ cloud: GaussianCloud, metric: HeatmapMetric, sceneRadius: Float) -> [SIMD4<Float>] {
        let radius: Float = (sceneRadius.isFinite && sceneRadius > 0) ? sceneRadius : 1
        let values: [Float] = heatmapValues(cloud, metric: metric, sceneRadius: radius)
        let range: (low: Float, high: Float) = percentileRange(values, lower: 0.01, upper: 0.99)
        // Opacity is suspicious when it is low; size and isolation when they are high.
        let highIsSuspicious: Bool = metric != .opacity
        var colors: [SIMD4<Float>] = []
        colors.reserveCapacity(values.count)
        for value in values {
            var t: Float = 0.5
            if value.isFinite, range.high - range.low > 1e-12 {
                t = min(max((value - range.low) / (range.high - range.low), 0), 1)
            }
            if !highIsSuspicious {
                t = 1 - t
            }
            let rgb: SIMD3<Float> = turbo(t)
            colors.append(SIMD4<Float>(rgb.x, rgb.y, rgb.z, 1))
        }
        return colors
    }

    /// Raw metric per Gaussian, before the percentile clamp: the log of the largest axis over the scene radius, the
    /// sigmoid opacity, or the isolation distance over the scene radius.
    static func heatmapValues(_ cloud: GaussianCloud, metric: HeatmapMetric, sceneRadius: Float) -> [Float] {
        let n: Int = cloud.count
        switch metric {
        case .size:
            let logRadius: Float = log(sceneRadius)
            var values: [Float] = []
            values.reserveCapacity(n)
            for i in 0..<n {
                let s0: Float = cloud.logScales[3 * i]
                let s1: Float = cloud.logScales[3 * i + 1]
                let s2: Float = cloud.logScales[3 * i + 2]
                let largest: Float = max(s0, max(s1, s2))
                values.append(largest - logRadius)
            }
            return values
        case .opacity:
            var values: [Float] = []
            values.reserveCapacity(n)
            for logit in cloud.opacityLogits {
                let opacity: Float = 1 / (1 + exp(-logit))
                values.append(opacity)
            }
            return values
        case .isolation:
            let distances: [Float] = isolationDistances(positions: positions(of: cloud), k: 4)
            return distances.map { $0 / sceneRadius }
        }
    }

    /// Distance from each point to its k-th nearest other point, exact with a k-d tree. Above `maxReference` points,
    /// only every s-th point is a neighbour candidate, which keeps the reference set bounded.
    public static func isolationDistances(positions: [SIMD3<Float>], k: Int = 4, maxReference: Int = 2_000_000) -> [Float] {
        let n: Int = positions.count
        guard n > 1, k > 0 else { return [Float](repeating: 0, count: n) }
        let step: Int = max(1, (n + maxReference - 1) / maxReference)
        var referenceIndices: [Int] = []
        var reference: [SIMD3<Float>] = []
        referenceIndices.reserveCapacity(n / step + 1)
        reference.reserveCapacity(n / step + 1)
        for index in Swift.stride(from: 0, to: n, by: step) {
            referenceIndices.append(index)
            reference.append(positions[index])
        }
        let tree = KDTree(points: reference)
        var distances: [Float] = [Float](repeating: 0, count: n)
        for i in 0..<n {
            // One extra candidate, so that the point itself can be dropped when it belongs to the reference set.
            let candidates: [Int] = tree.nearestNeighbors(to: positions[i], k: k + 1)
            var kept: Int = 0
            var last: Float = 0
            for local in candidates {
                let global: Int = referenceIndices[local]
                if global == i { continue }
                let delta: SIMD3<Float> = positions[global] - positions[i]
                last = (delta * delta).sum().squareRoot()
                kept += 1
                if kept == k { break }
            }
            distances[i] = last
        }
        return distances
    }

    /// Positions of a cloud as SIMD3 values.
    static func positions(of cloud: GaussianCloud) -> [SIMD3<Float>] {
        var result: [SIMD3<Float>] = []
        result.reserveCapacity(cloud.count)
        for i in 0..<cloud.count {
            result.append(SIMD3<Float>(cloud.positions[3 * i], cloud.positions[3 * i + 1], cloud.positions[3 * i + 2]))
        }
        return result
    }

    /// The values at the lower and upper fraction of the finite values (nearest rank). `(0, 0)` without finite values.
    static func percentileRange(_ values: [Float], lower: Float, upper: Float) -> (low: Float, high: Float) {
        var finite: [Float] = values.filter { $0.isFinite }
        guard !finite.isEmpty else { return (0, 0) }
        finite.sort()
        let last: Int = finite.count - 1
        let lowRank: Int = min(max(Int(Float(last) * lower), 0), last)
        let highRank: Int = min(max(Int(Float(last) * upper), 0), last)
        return (finite[lowRank], finite[highRank])
    }

    /// Murmur3 finaliser: a well-mixed 32-bit hash of an integer.
    static func mixBits(_ value: UInt32) -> UInt32 {
        var h: UInt32 = value
        h ^= h >> 16
        h = h &* 0x85EB_CA6B
        h ^= h >> 13
        h = h &* 0xC2B2_AE35
        h ^= h >> 16
        return h
    }

    /// HSV to RGB with the hue in 0...1 (wrapping), saturation and value in 0...1.
    static func hsvToRGB(hue: Float, saturation: Float, value: Float) -> SIMD3<Float> {
        let scaled: Float = (hue - hue.rounded(.down)) * 6
        let sector: Int = Int(scaled) % 6
        let fraction: Float = scaled - Float(Int(scaled))
        let p: Float = value * (1 - saturation)
        let q: Float = value * (1 - saturation * fraction)
        let t: Float = value * (1 - saturation * (1 - fraction))
        switch sector {
        case 0: return SIMD3<Float>(value, t, p)
        case 1: return SIMD3<Float>(q, value, p)
        case 2: return SIMD3<Float>(p, value, t)
        case 3: return SIMD3<Float>(p, q, value)
        case 4: return SIMD3<Float>(t, p, value)
        default: return SIMD3<Float>(value, p, q)
        }
    }
}
