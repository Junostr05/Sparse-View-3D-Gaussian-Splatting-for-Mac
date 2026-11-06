import Foundation

/// Whether depth-based features can run, given the depth model's store status and connectivity.
/// The UI shows the `unavailable` reason as the "depth features off" notice.
public enum DepthFeatureAvailability: Equatable, Sendable {
    case ready
    /// The model is missing and can be downloaded now.
    case needsDownload
    case unavailable(reason: String)

    public static func evaluate(status: ModelStatus, networkAllowed: Bool) -> DepthFeatureAvailability {
        switch status {
        case .ready:
            return .ready
        case .notConfigured:
            return .unavailable(reason: "Depth features are off: the depth model is not configured in this build yet.")
        case .missing:
            if networkAllowed {
                return .needsDownload
            }
            return .unavailable(reason: "Depth features are off: the depth model is not downloaded and no network connection is available.")
        case .corrupt:
            // A corrupt file is replaced by downloading it again, which needs a connection.
            if networkAllowed {
                return .needsDownload
            }
            return .unavailable(reason: "Depth features are off: the stored depth model failed its checksum and no network connection is available to download it again.")
        }
    }
}
