import Foundation
import SplatCore

/// What depth preparation decided for one view. Problems are reported here, never thrown.
public enum ViewDepthStatus: Sendable, Equatable {
    /// The provider had no depth for this view.
    case none
    /// Measured metric depth (ARKit / LiDAR) used directly, with the L1 term enabled.
    case measuredMetric
    /// A relative map, or a COLMAP metric map, aligned to the SfM points of this view.
    case aligned(AlignmentResult)
    /// Depth exists but could not be aligned. Pearson still applies (affine-invariant), but no
    /// metric depth is available for dense initialization or pseudo-views.
    case unaligned(AlignmentFailure)
    /// The map was unusable (provider error, mismatched size, no valid pixels). The reason is
    /// for display.
    case rejected(String)
}

/// Per-view outcome of a depth preparation run.
public struct DepthPreparationReport: Sendable, Equatable {
    public var views: [ViewDepthStatus]

    public init(views: [ViewDepthStatus]) {
        self.views = views
    }

    /// Views with measured or aligned metric depth (the views dense init and pseudo-views use).
    public var alignedOrMetricCount: Int {
        var count: Int = 0
        for status in views {
            switch status {
            case .measuredMetric, .aligned:
                count += 1
            case .none, .unaligned, .rejected:
                break
            }
        }
        return count
    }

    /// True when at least one view received a depth prior of any kind.
    public var hasAnyPrior: Bool {
        for status in views {
            switch status {
            case .measuredMetric, .aligned, .unaligned:
                return true
            case .none, .rejected:
                continue
            }
        }
        return false
    }
}

/// Turns provider maps into per-view `DepthPrior`s at view resolution (Phase 4 P1, P3, P4, P5).
public enum DepthPreparation {
    /// A relative map keeps its declared kind when at least this fraction of samples are inliers.
    static let declaredKindMinimumInlierRatio: Double = 0.5
    /// The other kind must explain at least this fraction and this multiple of the declared
    /// kind's inliers to replace it.
    static let correctionInlierMultiple: Double = 1.5
    /// zMax (the far limit for inverse depth) as a multiple of the median SfM depth.
    static let zMaxMultiple: Double = 100

    private struct ViewOutcome {
        var status: ViewDepthStatus
        var prior: DepthPrior?
    }

    /// Resamples each provided map to its view, aligns it, and returns the dataset with
    /// `TrainingView.depth` set for every view (`nil` where there is no usable depth).
    ///
    /// `trustMetric`: true for ARKit sources, whose metric depth is taken as-is (P4). Metric depth
    /// from other sources is fitted with scale only. Without a provider the dataset is returned
    /// unchanged and every view is reported as `.none`. Each view gets its own seed derived from
    /// `seed` and its index, so the result does not depend on the order of work.
    public static func prepare(_ dataset: TrainingDataset, provider: (any DepthProvider)?, trustMetric: Bool,
                               settings: DepthPriorSettings, seed: UInt64) -> (dataset: TrainingDataset, report: DepthPreparationReport) {
        guard let provider else {
            let statuses: [ViewDepthStatus] = [ViewDepthStatus](repeating: .none, count: dataset.views.count)
            return (dataset, DepthPreparationReport(views: statuses))
        }

        let points: [SIMD3<Float>] = dataset.points?.positions ?? []
        var prepared: TrainingDataset = dataset
        var statuses: [ViewDepthStatus] = []
        statuses.reserveCapacity(dataset.views.count)
        for index in 0..<dataset.views.count {
            let outcome: ViewOutcome = prepareView(dataset.views[index],
                                                   provider: provider,
                                                   points: points,
                                                   trustMetric: trustMetric,
                                                   settings: settings,
                                                   seed: seed &+ UInt64(index))
            prepared.views[index].depth = outcome.prior
            statuses.append(outcome.status)
        }
        return (prepared, DepthPreparationReport(views: statuses))
    }

    // MARK: - Per view

    private static func prepareView(_ view: TrainingView, provider: any DepthProvider,
                                    points: [SIMD3<Float>], trustMetric: Bool,
                                    settings: DepthPriorSettings, seed: UInt64) -> ViewOutcome {
        let width: Int = view.camera.width
        let height: Int = view.camera.height

        let found: DepthMap?
        do {
            found = try provider.depthMap(for: view)
        } catch {
            return ViewOutcome(status: .rejected("depth provider failed: \(error)"), prior: nil)
        }
        guard let map = found else {
            return ViewOutcome(status: .none, prior: nil)
        }

        let resampled: DepthMap
        do {
            resampled = try map.resampled(width: width, height: height)
        } catch {
            return ViewOutcome(status: .rejected("depth map \(map.width)x\(map.height) does not match view \(width)x\(height)"),
                               prior: nil)
        }

        // v1.5: the map's own per-pixel weights (model confidence) only when confidence weighting is on.
        let pixelWeights: [Float]? = settings.confidenceWeighting ? resampled.weights : nil

        if map.kind == .metric && trustMetric {
            return measuredOutcome(resampled, weights: pixelWeights)
        }

        let samples: [(prior: Double, z: Double)] = DepthAlignment.samples(points: points,
                                                                           camera: view.camera,
                                                                           map: resampled)
        if map.kind == .metric {
            let fit = DepthAlignment.fit(samples, kind: .metric, scaleOnly: true,
                                         minimumPoints: settings.minimumAlignmentPoints,
                                         minimumInlierFraction: Double(settings.minimumInlierFraction),
                                         seed: seed)
            return alignedOutcome(resampled, samples: samples, fit: fit, weights: pixelWeights)
        }

        let fit: Result<AlignmentResult, AlignmentFailure> = fitRelative(samples,
                                                                         declaredKind: map.kind,
                                                                         settings: settings,
                                                                         seed: seed)
        return alignedOutcome(resampled, samples: samples, fit: fit, weights: pixelWeights)
    }

    /// Measured metric depth: invalid pixels become 0 and the L1 term is enabled.
    private static func measuredOutcome(_ map: DepthMap, weights: [Float]?) -> ViewOutcome {
        let count: Int = map.values.count
        var metric: [Float] = [Float](repeating: 0, count: count)
        var validCount: Int = 0
        for index in 0..<count where map.isValid(index) {
            metric[index] = map.values[index]
            validCount += 1
        }
        guard validCount > 0 else {
            return ViewOutcome(status: .rejected("measured depth has no valid pixels"), prior: nil)
        }
        let prior = DepthPrior(width: map.width, height: map.height, correlationTarget: metric,
                               domain: .depth, metricDepth: metric, isMeasuredMetric: true, weights: weights)
        return ViewOutcome(status: .measuredMetric, prior: prior)
    }

    /// Fits a relative map with its declared kind. The other relative kind replaces it only when
    /// the declared fit fails or explains less than half of the points, and the other kind
    /// explains at least half and more than 1.5 times as many points.
    private static func fitRelative(_ samples: [(prior: Double, z: Double)], declaredKind: DepthKind,
                                    settings: DepthPriorSettings, seed: UInt64) -> Result<AlignmentResult, AlignmentFailure> {
        let fraction: Double = Double(settings.minimumInlierFraction)
        let declared = DepthAlignment.fit(samples, kind: declaredKind, scaleOnly: false,
                                          minimumPoints: settings.minimumAlignmentPoints,
                                          minimumInlierFraction: fraction, seed: seed)
        let total: Int = samples.count
        var declaredSucceeded: Bool = false
        var declaredInliers: Int = 0
        if case .success(let result) = declared {
            declaredSucceeded = true
            declaredInliers = result.inliers
        }
        let declaredRatio: Double = total > 0 ? Double(declaredInliers) / Double(total) : 0
        if declaredSucceeded && declaredRatio >= declaredKindMinimumInlierRatio {
            return declared
        }
        guard total > 0 else {
            return declared
        }

        let otherKind: DepthKind = (declaredKind == .relativeInverseDepth) ? .relativeDepth : .relativeInverseDepth
        let alternative = DepthAlignment.fit(samples, kind: otherKind, scaleOnly: false,
                                             minimumPoints: settings.minimumAlignmentPoints,
                                             minimumInlierFraction: fraction, seed: seed)
        if case .success(var result) = alternative {
            let enoughInliers: Bool = result.inliers * 2 >= total
            let clearlyBetter: Bool = Double(result.inliers) > correctionInlierMultiple * Double(declaredInliers)
            if enoughInliers && clearlyBetter {
                result.kindCorrected = true
                return .success(result)
            }
        }
        return declared
    }

    /// Builds the status and prior from a fit. Metric depth is set only when the fit succeeded.
    /// The correlation target is always the raw map in the domain of the kind it was fitted as.
    private static func alignedOutcome(_ map: DepthMap, samples: [(prior: Double, z: Double)],
                                       fit: Result<AlignmentResult, AlignmentFailure>,
                                       weights: [Float]?) -> ViewOutcome {
        let declaredDomain: DepthDomain = (map.kind == .relativeInverseDepth) ? .inverseDepth : .depth
        switch fit {
        case .success(let alignment):
            let zMax: Double = zMaxFor(samples)
            let metric: [Float] = DepthAlignment.metricDepth(map, alignment: alignment, zMax: zMax)
            let domain: DepthDomain = (alignment.fittedKind == .relativeInverseDepth) ? .inverseDepth : .depth
            let prior = DepthPrior(width: map.width, height: map.height, correlationTarget: map.values,
                                   domain: domain, metricDepth: metric, isMeasuredMetric: false,
                                   weights: weights)
            return ViewOutcome(status: .aligned(alignment), prior: prior)
        case .failure(let failure):
            let prior = DepthPrior(width: map.width, height: map.height, correlationTarget: map.values,
                                   domain: declaredDomain, weights: weights)
            return ViewOutcome(status: .unaligned(failure), prior: prior)
        }
    }

    private static func zMaxFor(_ samples: [(prior: Double, z: Double)]) -> Double {
        var depths: [Double] = []
        depths.reserveCapacity(samples.count)
        for sample in samples {
            depths.append(sample.z)
        }
        return zMaxMultiple * DepthAlignment.median(depths)
    }
}
