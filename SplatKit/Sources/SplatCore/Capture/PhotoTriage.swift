import Foundation

/// One photo entering triage: its name as stored in the project and its measurements.
public struct PhotoInput: Sendable, Equatable {
    public var name: String
    public var metrics: PhotoMetrics

    public init(name: String, metrics: PhotoMetrics) {
        self.name = name
        self.metrics = metrics
    }
}

/// What triage thinks of one photo. `duplicate` names the earlier photo it repeats.
public enum PhotoVerdictKind: Sendable, Equatable {
    case ok
    case blurry
    case tooDark
    case tooBright
    case duplicate(of: String)
}

/// The verdict for one photo, with a plain-language reason for the UI.
public struct PhotoVerdict: Sendable, Equatable {
    public var name: String
    public var kind: PhotoVerdictKind
    public var reason: String

    public init(name: String, kind: PhotoVerdictKind, reason: String) {
        self.name = name
        self.kind = kind
        self.reason = reason
    }

    public var isOK: Bool {
        kind == .ok
    }
}

/// Flags photos that are blurry, badly exposed or near-duplicates of another photo. Pure logic, Linux-testable.
///
/// - Blur is relative to the set: a photo counts as blurry only when its sharpness is well below the median
///   of the set (robust z-score against the median absolute deviation, with a floor so a uniform set flags nothing).
/// - Exposure uses absolute thresholds on mean luma and clipped fractions.
/// - Redundancy uses camera poses when both photos have one: centre distance below `centreSpacingFraction` of the
///   median nearest-neighbour spacing and viewing angle below `viewingAngleDegrees`. Otherwise the 8 x 8
///   signatures must correlate above `signatureCorrelation`.
/// Photos are judged in input order. A photo flagged for exposure or blur never serves as the original of a duplicate.
public enum PhotoTriage {
    public struct Thresholds: Sendable, Equatable {
        /// Blur needs at least this many photos; smaller sets have no meaningful median.
        public var minimumSetSizeForBlur: Int
        /// How many robust standard deviations below the median a photo must be to count as blurry.
        public var blurDeviations: Double
        /// A blurry photo also has to fall below this fraction of the median sharpness.
        public var blurRelativeFloor: Double
        /// Lower bound of the robust scale, as a fraction of the median, so a uniform set does not flag noise.
        public var blurScaleFloor: Double
        public var darkMeanLuminance: Double
        public var darkShadowFraction: Double
        public var brightMeanLuminance: Double
        public var brightHighlightFraction: Double
        /// Near-duplicate: centre distance below this fraction of the median camera spacing.
        public var centreSpacingFraction: Double
        /// Near-duplicate: viewing directions within this angle, in degrees.
        public var viewingAngleDegrees: Double
        /// Near-duplicate without poses: signature correlation above this value.
        public var signatureCorrelation: Double

        public init() {
            minimumSetSizeForBlur = 4
            blurDeviations = 2.5
            blurRelativeFloor = 0.7
            blurScaleFloor = 0.1
            darkMeanLuminance = 0.12
            darkShadowFraction = 0.5
            brightMeanLuminance = 0.88
            brightHighlightFraction = 0.5
            centreSpacingFraction = 0.03
            viewingAngleDegrees = 3
            signatureCorrelation = 0.98
        }
    }

    /// One verdict per input photo, in input order. `cameras` is keyed by photo name; photos without an entry
    /// are compared by their signatures.
    public static func assess(images: [PhotoInput],
                              cameras: [String: Camera] = [:],
                              thresholds: Thresholds = Thresholds()) -> [PhotoVerdict] {
        var kinds: [PhotoVerdictKind?] = Array(repeating: nil, count: images.count)

        // Exposure: absolute thresholds.
        for index in images.indices {
            let metrics: PhotoMetrics = images[index].metrics
            if metrics.meanLuminance < thresholds.darkMeanLuminance
                || metrics.shadowFraction > thresholds.darkShadowFraction {
                kinds[index] = .tooDark
            } else if metrics.meanLuminance > thresholds.brightMeanLuminance
                        || metrics.highlightFraction > thresholds.brightHighlightFraction {
                kinds[index] = .tooBright
            }
        }

        // Blur: relative to the median of the whole set.
        if images.count >= thresholds.minimumSetSizeForBlur {
            let sharpness: [Double] = images.map { $0.metrics.sharpness }
            let median: Double = medianOf(sharpness)
            if median > 0 {
                let deviations: [Double] = sharpness.map { abs($0 - median) }
                let spread: Double = 1.4826 * medianOf(deviations)
                let scale: Double = max(spread, thresholds.blurScaleFloor * median)
                for index in images.indices where kinds[index] == nil {
                    let value: Double = sharpness[index]
                    let wellBelow: Bool = value < median - thresholds.blurDeviations * scale
                    let belowFloor: Bool = value < thresholds.blurRelativeFloor * median
                    if wellBelow && belowFloor {
                        kinds[index] = .blurry
                    }
                }
            }
        }

        // Redundancy: judged in input order against the photos kept so far.
        let spacing: Double? = medianSpacing(images: images, cameras: cameras)
        var kept: [Int] = []
        for index in images.indices where kinds[index] == nil {
            var original: Int?
            for keptIndex in kept where isDuplicate(images[index], images[keptIndex],
                                                    cameras: cameras, spacing: spacing,
                                                    thresholds: thresholds) {
                original = keptIndex
                break
            }
            if let original {
                kinds[index] = .duplicate(of: images[original].name)
            } else {
                kept.append(index)
            }
        }

        var verdicts: [PhotoVerdict] = []
        verdicts.reserveCapacity(images.count)
        for index in images.indices {
            let kind: PhotoVerdictKind = kinds[index] ?? .ok
            verdicts.append(PhotoVerdict(name: images[index].name,
                                         kind: kind,
                                         reason: reason(for: kind, metrics: images[index].metrics)))
        }
        return verdicts
    }

    // MARK: - Redundancy

    /// Median distance from each posed camera to its nearest posed neighbour. `nil` with fewer than two poses.
    private static func medianSpacing(images: [PhotoInput], cameras: [String: Camera]) -> Double? {
        var centres: [SIMD3<Float>] = []
        for input in images {
            if let camera = cameras[input.name] {
                centres.append(camera.center)
            }
        }
        guard centres.count >= 2 else {
            return nil
        }
        var nearest: [Double] = []
        nearest.reserveCapacity(centres.count)
        for index in centres.indices {
            var best: Double = .infinity
            for other in centres.indices where other != index {
                let distance: Double = Double(length(centres[index] - centres[other]))
                best = min(best, distance)
            }
            nearest.append(best)
        }
        let spacing: Double = medianOf(nearest)
        return spacing > 0 ? spacing : nil
    }

    private static func isDuplicate(_ photo: PhotoInput,
                                    _ other: PhotoInput,
                                    cameras: [String: Camera],
                                    spacing: Double?,
                                    thresholds: Thresholds) -> Bool {
        if let spacing,
           let camera = cameras[photo.name],
           let otherCamera = cameras[other.name] {
            let distance: Double = Double(length(camera.center - otherCamera.center))
            let angle: Double = viewingAngleDegrees(forward(camera), forward(otherCamera))
            return distance < thresholds.centreSpacingFraction * spacing && angle < thresholds.viewingAngleDegrees
        }
        return correlation(photo.metrics.signature, other.metrics.signature) > thresholds.signatureCorrelation
    }

    /// The viewing direction of a camera in world space: the camera's z axis, `Rᵀ e_z`.
    private static func forward(_ camera: Camera) -> SIMD3<Float> {
        camera.rotation.transposed * SIMD3<Float>(0, 0, 1)
    }

    private static func viewingAngleDegrees(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Double {
        let dotProduct: Double = Double(a.x * b.x + a.y * b.y + a.z * b.z)
        let clamped: Double = min(1.0, max(-1.0, dotProduct))
        return acos(clamped) * 180.0 / Double.pi
    }

    private static func length(_ v: SIMD3<Float>) -> Float {
        (v.x * v.x + v.y * v.y + v.z * v.z).squareRoot()
    }

    /// Pearson correlation of two signatures. Two flat signatures correlate when their means agree.
    static func correlation(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count, !a.isEmpty else {
            return 0
        }
        let n: Double = Double(a.count)
        var meanA: Double = 0
        var meanB: Double = 0
        for index in a.indices {
            meanA += a[index]
            meanB += b[index]
        }
        meanA /= n
        meanB /= n

        var covariance: Double = 0
        var varianceA: Double = 0
        var varianceB: Double = 0
        for index in a.indices {
            let deltaA: Double = a[index] - meanA
            let deltaB: Double = b[index] - meanB
            covariance += deltaA * deltaB
            varianceA += deltaA * deltaA
            varianceB += deltaB * deltaB
        }
        let flatLimit: Double = 1e-12
        if varianceA < flatLimit || varianceB < flatLimit {
            return (varianceA < flatLimit && varianceB < flatLimit && abs(meanA - meanB) < 0.01) ? 1 : 0
        }
        return covariance / (varianceA * varianceB).squareRoot()
    }

    // MARK: - Helpers

    private static func medianOf(_ values: [Double]) -> Double {
        guard !values.isEmpty else {
            return 0
        }
        let sorted: [Double] = values.sorted()
        let middle: Int = sorted.count / 2
        if sorted.count % 2 == 1 {
            return sorted[middle]
        }
        return 0.5 * (sorted[middle - 1] + sorted[middle])
    }

    private static func percent(_ fraction: Double) -> String {
        "\(Int((fraction * 100).rounded()))%"
    }

    private static func reason(for kind: PhotoVerdictKind, metrics: PhotoMetrics) -> String {
        switch kind {
        case .ok:
            return "Sharp, well exposed and a distinct view."
        case .tooDark:
            return "Too dark: mean brightness \(percent(metrics.meanLuminance)), \(percent(metrics.shadowFraction)) of pixels in deep shadow."
        case .tooBright:
            return "Too bright: mean brightness \(percent(metrics.meanLuminance)), \(percent(metrics.highlightFraction)) of pixels clipped."
        case .blurry:
            return "Much blurrier than the other photos in this set."
        case .duplicate(let original):
            return "Nearly the same view as \(original)."
        }
    }
}
