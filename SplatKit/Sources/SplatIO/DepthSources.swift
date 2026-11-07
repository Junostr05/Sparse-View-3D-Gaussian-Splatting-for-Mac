import Foundation
import SplatCore

/// The depth maps a training run may use, split by origin.
public struct DepthSources: Sendable {
    /// Maps imported with the project: unflagged maps directly under `depth/`. They serve every estimator.
    public var supplied: [String: DepthMap]
    public var suppliedFailures: [String: DepthMapError]
    /// Maps the chosen estimator produced in an earlier run. Only that estimator's cache is read.
    public var cached: [String: DepthMap]
    public var cachedFailures: [String: DepthMapError]

    public init(supplied: [String: DepthMap], suppliedFailures: [String: DepthMapError],
                cached: [String: DepthMap], cachedFailures: [String: DepthMapError]) {
        self.supplied = supplied
        self.suppliedFailures = suppliedFailures
        self.cached = cached
        self.cachedFailures = cachedFailures
    }
}

extension ProjectBundle {
    /// Splits the depth files of a project by origin for one estimator.
    ///
    /// Maps flagged as estimated directly under `depth/` were written before the per-estimator cache existed. They
    /// came from Depth Anything, so they count as its cache, and only when it is the chosen estimator. Unflagged maps
    /// are supplied. The cache of `estimator` wins over a legacy map of the same image.
    public static func depthSources(_ contents: ProjectContents, estimator: DepthEstimatorKind) -> DepthSources {
        let direct = depthMaps(contents)
        let own = estimatedDepthMaps(contents, estimator: estimator)
        let supplied: [String: DepthMap] = direct.maps.filter { !$0.value.isEstimated }
        var cached: [String: DepthMap] = own.maps
        if estimator == .depthAnythingV2Small {
            let legacy: [String: DepthMap] = direct.maps.filter { $0.value.isEstimated }
            cached = legacy.merging(own.maps) { _, ownMap in ownMap }
        }
        return DepthSources(supplied: supplied,
                            suppliedFailures: direct.failures,
                            cached: cached,
                            cachedFailures: own.failures)
    }
}
