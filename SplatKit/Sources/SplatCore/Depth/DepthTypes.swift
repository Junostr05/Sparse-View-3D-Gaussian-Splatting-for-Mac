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
    /// Optional per-pixel loss weights, same count as `values` (for example a depth model's own
    /// confidence, v1.5). Non-negative; larger means more reliable. `nil` means uniform. The scale is
    /// arbitrary: `DepthPrior` normalises the weights to mean 1.
    public var weights: [Float]?

    public init(width: Int, height: Int, kind: DepthKind, values: [Float],
                confidence: [UInt8]? = nil, isEstimated: Bool = false, weights: [Float]? = nil) {
        self.width = width
        self.height = height
        self.kind = kind
        self.values = values
        self.confidence = confidence
        self.isEstimated = isEstimated
        self.weights = weights
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
    /// Optional per-pixel weights of the depth loss terms (v1.5 confidence weighting), normalised to
    /// mean 1 over the pixels with a valid `correlationTarget`. `nil` means every pixel weighs 1, which
    /// is exactly the unweighted loss.
    public var weights: [Float]?
    /// Optional surface-normal prior (v1.5): unit camera-frame normals (OpenCV frame, pointing towards the camera),
    /// one per pixel; a zero vector means no prior at that pixel. Used by `NormalLoss` when the normal prior is on.
    public var normals: [SIMD3<Float>]?

    /// `weights` are sanitised (non-finite or negative values become 0) and normalised to mean 1 over
    /// the pixels with a valid target. Weights that sum to 0 over those pixels are dropped (`nil`).
    public init(width: Int, height: Int, correlationTarget: [Float], domain: DepthDomain,
                metricDepth: [Float]? = nil, isMeasuredMetric: Bool = false, weights: [Float]? = nil,
                normals: [SIMD3<Float>]? = nil) {
        self.width = width
        self.height = height
        self.correlationTarget = correlationTarget
        self.domain = domain
        self.metricDepth = metricDepth
        self.isMeasuredMetric = isMeasuredMetric
        self.l1Normalizer = isMeasuredMetric ? DepthPrior.median(of: metricDepth) : 0
        self.weights = DepthPrior.normalizedWeights(weights, target: correlationTarget)
        if let normals {
            precondition(normals.count == correlationTarget.count, "normals must match the pixel count")
        }
        self.normals = normals
    }

    /// Sanitised weights with mean 1 over the valid-target pixels, or nil (see `init`).
    static func normalizedWeights(_ weights: [Float]?, target: [Float]) -> [Float]? {
        guard let weights else { return nil }
        precondition(weights.count == target.count, "depth weights must match the pixel count")
        var cleaned: [Float] = [Float](repeating: 0, count: weights.count)
        var sum: Double = 0
        var validCount: Int = 0
        for index in 0..<weights.count {
            let weight: Float = weights[index]
            let clean: Float = (weight.isFinite && weight > 0) ? weight : 0
            cleaned[index] = clean
            let value: Float = target[index]
            if value.isFinite && value > 0 {
                sum += Double(clean)
                validCount += 1
            }
        }
        guard validCount > 0, sum > 0 else { return nil }
        let scale: Double = Double(validCount) / sum
        for index in 0..<cleaned.count {
            let scaled: Double = Double(cleaned[index]) * scale
            cleaned[index] = Float(scaled)
        }
        return cleaned
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
