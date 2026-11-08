import SplatCore
import SplatIO

extension DepthEstimatorKind {
    /// The catalogue entry of the model that backs this estimator.
    public var descriptor: ModelDescriptor {
        switch self {
        case .depthAnythingV2Small:
            return ModelCatalog.depthAnythingV2Small
        case .moge2Small:
            return ModelCatalog.moge2SmallInt8
        case .depthAnything3:
            return ModelDescriptor.depthAnything3Base
        }
    }

    /// True when the model is imported by the user (the pose-free Depth Anything 3 package) rather than
    /// downloaded from the Depth Model section.
    public var needsImportedPackage: Bool {
        self == .depthAnything3
    }
}

/// Builds the estimator a kind names, from the installed model.
public enum DepthEstimatorFactory {
    /// The provider for `kind`, or nil when its model is not installed. Throws only when an installed model fails to load.
    ///
    /// `views` are the views a multi-view estimator (Depth Anything 3) runs on together; the single-image
    /// estimators ignore them. Depth Anything 3 loads its model lazily, on the first map requested.
    public static func installed(_ kind: DepthEstimatorKind,
                                 in store: ModelStore,
                                 views: [TrainingView] = []) throws(MonocularDepthError) -> (any DepthProvider)? {
        switch kind {
        case .depthAnythingV2Small:
            guard let estimator: MonocularDepthEstimator = try MonocularDepthEstimator.installed(in: store,
                                                                                                descriptor: kind.descriptor) else {
                return nil
            }
            return estimator
        case .moge2Small:
            guard let estimator: MoGeDepthEstimator = try MoGeDepthEstimator.installed(in: store,
                                                                                      descriptor: kind.descriptor) else {
                return nil
            }
            return estimator
        case .depthAnything3:
            return DepthAnything3DepthEstimator.installed(in: store, views: views)
        }
    }
}
