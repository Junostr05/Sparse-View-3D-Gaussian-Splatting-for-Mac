import Foundation

/// What a depth map's values mean. Stored as a raw byte in the `.sdpt` codec (Phase 4 Task 2).
public enum DepthKind: UInt8, Codable, Sendable {
    /// Metres along the camera z axis (LiDAR, ARKit, COLMAP exports).
    case metric = 0
    /// Affine-ambiguous depth from a model; scale and shift must be fitted.
    case relativeDepth = 1
    /// Affine-ambiguous inverse depth from a model (Depth Anything output convention).
    case relativeInverseDepth = 2
}

/// A depth map as read from disk or produced by a model, before alignment to a view.
public struct DepthMap: Sendable, Equatable {
    public var width: Int
    public var height: Int
    public var kind: DepthKind
    /// Row-major. Non-finite or ≤ 0 values are invalid.
    public var values: [Float]
    /// ARKit confidence 0/1/2, same count as `values`, when available.
    public var confidence: [UInt8]?
    /// Produced by a model (flags bit 1) rather than measured.
    public var isEstimated: Bool

    public init(width: Int, height: Int, kind: DepthKind, values: [Float],
                confidence: [UInt8]? = nil, isEstimated: Bool = false) {
        self.width = width
        self.height = height
        self.kind = kind
        self.values = values
        self.confidence = confidence
        self.isEstimated = isEstimated
    }

    /// True when the pixel at `index` has a finite, positive value and, if confidence is
    /// present, a confidence of at least `minimumConfidence`.
    public func isValid(_ index: Int, minimumConfidence: UInt8 = 1) -> Bool {
        guard index >= 0, index < values.count else { return false }
        let value = values[index]
        guard value.isFinite, value > 0 else { return false }
        if let confidence {
            guard index < confidence.count else { return false }
            return confidence[index] >= minimumConfidence
        }
        return true
    }
}

/// The domain in which the Pearson correlation of a depth prior is evaluated.
public enum DepthDomain: Sendable, Equatable {
    case depth
    case inverseDepth
}

/// Per-view depth prior at view resolution, ready for the loss.
public struct DepthPrior: Sendable, Equatable {
    public var width: Int
    public var height: Int
    /// Values in `domain`. ≤ 0 or non-finite means invalid.
    public var correlationTarget: [Float]
    public var domain: DepthDomain
    /// Camera-z depth in scene units (aligned or LiDAR). 0 means invalid.
    public var metricDepth: [Float]?
    /// LiDAR only: enables the L1 term.
    public var isMeasuredMetric: Bool
    /// Median of valid `metricDepth` when `isMeasuredMetric`, else 0.
    public var l1Normalizer: Float

    public init(width: Int, height: Int, correlationTarget: [Float], domain: DepthDomain,
                metricDepth: [Float]? = nil, isMeasuredMetric: Bool = false) {
        self.width = width
        self.height = height
        self.correlationTarget = correlationTarget
        self.domain = domain
        self.metricDepth = metricDepth
        self.isMeasuredMetric = isMeasuredMetric
        self.l1Normalizer = isMeasuredMetric ? DepthPrior.median(of: metricDepth) : 0
    }

    /// Median of the finite, positive entries; 0 when there are none. For an even count the
    /// two middle values are averaged.
    static func median(of values: [Float]?) -> Float {
        guard let values else { return 0 }
        var valid: [Float] = []
        valid.reserveCapacity(values.count)
        for value in values where value.isFinite && value > 0 {
            valid.append(value)
        }
        guard !valid.isEmpty else { return 0 }
        valid.sort()
        let middle = valid.count / 2
        if valid.count % 2 == 1 {
            return valid[middle]
        }
        let lower: Double = Double(valid[middle - 1])
        let upper: Double = Double(valid[middle])
        return Float((lower + upper) * 0.5)
    }
}
