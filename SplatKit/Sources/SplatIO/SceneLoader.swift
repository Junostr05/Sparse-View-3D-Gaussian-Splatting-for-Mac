import Foundation
import SplatCore

/// A parsed scene with every non-finite Gaussian removed.
public struct LoadedScene: Sendable {
    public let cloud: GaussianCloud
    /// Number of Gaussians dropped because they contained NaN or infinite values.
    public let removedNonFinite: Int

    public init(cloud: GaussianCloud, removedNonFinite: Int) {
        self.cloud = cloud
        self.removedNonFinite = removedNonFinite
    }
}

/// Dispatches a file's bytes to the matching reader by extension.
public enum SceneLoader {
    /// `fileExtension` is matched case-insensitively against `ply` and `splat`.
    public static func load(_ data: Data, fileExtension: String) throws(SplatIOError) -> LoadedScene {
        let lowercasedExtension = fileExtension.lowercased()
        let parsed: GaussianCloud
        switch lowercasedExtension {
        case "ply":
            parsed = try PLYReader.read(data)
        case "splat":
            parsed = try AntimatterSplatReader.read(data)
        default:
            throw .unsupportedFormat(lowercasedExtension)
        }

        let (cloud, removed) = parsed.removingNonFinite()
        guard cloud.count > 0 else { throw .empty }
        return LoadedScene(cloud: cloud, removedNonFinite: removed)
    }
}
