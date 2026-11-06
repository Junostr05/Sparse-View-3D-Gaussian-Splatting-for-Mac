import Foundation

/// Describes a downloadable model: where it comes from, how to verify it and where it is stored.
public struct ModelDescriptor: Codable, Sendable, Equatable {
    /// Stable identifier, also the first directory component in the model store.
    public let id: String
    public let displayName: String
    /// Version string, the second directory component in the model store.
    public let version: String
    /// Download location. `nil` means the model is not configured yet.
    public let sourceURL: URL?
    /// Lower-case hex SHA-256 of the downloaded file. `nil` means not pinned yet.
    public let sha256: String?
    public let sizeBytes: Int64?
    public let license: String
    /// Name of the downloaded file inside the version directory.
    public let fileName: String
    /// `true` when the download is a zip archive that must be unpacked before use.
    public let isArchive: Bool

    public init(
        id: String,
        displayName: String,
        version: String,
        sourceURL: URL?,
        sha256: String?,
        sizeBytes: Int64?,
        license: String,
        fileName: String,
        isArchive: Bool
    ) {
        self.id = id
        self.displayName = displayName
        self.version = version
        self.sourceURL = sourceURL
        self.sha256 = sha256
        self.sizeBytes = sizeBytes
        self.license = license
        self.fileName = fileName
        self.isArchive = isArchive
    }

    /// `true` when a download URL is set and the checksum is a 64-character lower-case hex string.
    /// Models that are not configured are never downloaded or installed.
    public var isConfigured: Bool {
        guard sourceURL != nil, let sha256 else { return false }
        guard sha256.utf8.count == 64 else { return false }
        for byte in sha256.utf8 {
            let isDigit = byte >= 0x30 && byte <= 0x39
            let isLowerHex = byte >= 0x61 && byte <= 0x66
            if !isDigit && !isLowerHex {
                return false
            }
        }
        return true
    }
}

/// Built-in model descriptors.
public enum ModelCatalog {
    /// Depth Anything V2 Small as a Core ML package, downloaded on first use.
    /// TODO: pin `sourceURL`, `sha256` and `sizeBytes` after verifying Apple's Core ML conversion
    /// on Hugging Face on a Mac. Until then the model reports `isConfigured == false`.
    public static let depthAnythingV2Small = ModelDescriptor(
        id: "depth-anything-v2-small",
        displayName: "Depth Anything V2 Small",
        version: "1",
        sourceURL: nil,
        sha256: nil,
        sizeBytes: nil,
        license: "Apache-2.0",
        fileName: "DepthAnythingV2Small.mlpackage.zip",
        isArchive: true
    )

    public static let all: [ModelDescriptor] = [depthAnythingV2Small]
}
