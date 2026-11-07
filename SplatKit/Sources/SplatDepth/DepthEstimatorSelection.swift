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
        }
    }
}

/// Builds the estimator a kind names, from the installed model.
public enum DepthEstimatorFactory {
    /// The provider for `kind`, or nil when its model is not installed. Throws only when an installed model fails to load.
    public static func installed(_ kind: DepthEstimatorKind,
                                 in store: ModelStore) throws(MonocularDepthError) -> (any DepthProvider)? {
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
        }
    }
}
