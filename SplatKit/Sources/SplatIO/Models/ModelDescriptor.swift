import Foundation

/// One file of a multi-file model package.
public struct ModelFile: Codable, Sendable, Equatable {
    /// Path inside the package, with forward slashes, e.g. "Data/com.apple.CoreML/model.mlmodel".
    public let relativePath: String
    /// Lower-case hex SHA-256 of the file.
    public let sha256: String
    public let sizeBytes: Int64

    public init(relativePath: String, sha256: String, sizeBytes: Int64) {
        self.relativePath = relativePath
        self.sha256 = sha256
        self.sizeBytes = sizeBytes
    }
}

/// Describes a downloadable model package: where its files come from, how to verify them and where they are stored.
public struct ModelDescriptor: Codable, Sendable, Equatable {
    /// Stable identifier, also the first directory component in the model store.
    public let id: String
    public let displayName: String
    /// Version string, the second directory component in the model store.
    public let version: String
    public let license: String
    /// Files are fetched from `baseURL/<packageName>/<relativePath>`. `nil` means the model is not configured.
    public let baseURL: URL?
    /// Directory name of the package inside the version directory, e.g. "X.mlpackage".
    public let packageName: String
    public let files: [ModelFile]

    public init(
        id: String,
        displayName: String,
        version: String,
        license: String,
        baseURL: URL?,
        packageName: String,
        files: [ModelFile]
    ) {
        self.id = id
        self.displayName = displayName
        self.version = version
        self.license = license
        self.baseURL = baseURL
        self.packageName = packageName
        self.files = files
    }

    /// Sum of the declared file sizes in bytes.
    public var totalSizeBytes: Int64 {
        var total: Int64 = 0
        for file in files {
            total += file.sizeBytes
        }
        return total
    }

    /// `true` when a base URL is set, there is at least one file, every checksum is a 64-character
    /// lower-case hex string and every path is safe. Models that are not configured are never downloaded or installed.
    public var isConfigured: Bool {
        guard baseURL != nil, !files.isEmpty, isSafeRelativePath(packageName) else {
            return false
        }
        for file in files {
            guard isSafeRelativePath(file.relativePath), isLowerHexSHA256(file.sha256) else {
                return false
            }
        }
        return true
    }
}

/// `true` when `path` is a non-empty, relative, forward-slash path whose components are all real names:
/// no empty, `.` or `..` component and no backslash. Keeps downloads and installs inside the package directory.
func isSafeRelativePath(_ path: String) -> Bool {
    guard !path.isEmpty, !path.hasPrefix("/"), !path.contains("\\") else {
        return false
    }
    for component in path.split(separator: "/", omittingEmptySubsequences: false) {
        if component.isEmpty || component == "." || component == ".." {
            return false
        }
    }
    return true
}

/// `true` when `value` is exactly 64 lower-case hexadecimal characters.
func isLowerHexSHA256(_ value: String) -> Bool {
    guard value.utf8.count == 64 else {
        return false
    }
    for byte in value.utf8 {
        let isDigit = byte >= 0x30 && byte <= 0x39
        let isLowerHex = byte >= 0x61 && byte <= 0x66
        if !isDigit && !isLowerHex {
            return false
        }
    }
    return true
}

extension URL {
    /// Appends each component of a relative path that has passed `isSafeRelativePath`, one at a time.
    func appendingComponents(of relativePath: String) -> URL {
        var url = self
        for component in relativePath.split(separator: "/") {
            url = url.appendingPathComponent(String(component))
        }
        return url
    }
}

/// Built-in model descriptors.
public enum ModelCatalog {
    /// Apple's Core ML conversion of Depth Anything V2 Small (Hugging Face `apple/coreml-depth-anything-v2-small`,
    /// Apache-2.0), Float16 variant. Fixed input: RGB image 518x392, input name `image`.
    /// Fixed output: `Grayscale16Half` image 518x392, named `depth`.
    /// Pinned to the commit, so the URL and the checksums cannot change underneath us.
    public static let depthAnythingV2Small = ModelDescriptor(
        id: "depth-anything-v2-small",
        displayName: "Depth Anything V2 Small",
        version: "cfef6f6",
        license: "Apache-2.0",
        baseURL: URL(string: "https://huggingface.co/apple/coreml-depth-anything-v2-small/resolve/cfef6f6f2a70783dedc0bfae40cecbc2052285d3"),
        packageName: "DepthAnythingV2SmallF16.mlpackage",
        files: [
            ModelFile(
                relativePath: "Manifest.json",
                sha256: "2883ae290c48fe916dc5ececac03a7d847fa277165a49ef5652fa1d2b9cb55f7",
                sizeBytes: 617
            ),
            ModelFile(
                relativePath: "Data/com.apple.CoreML/model.mlmodel",
                sha256: "44ac97a3efcfd52113183fb2862ff59cd0368e9ec2e30a90a54980dd11407042",
                sizeBytes: 399_433
            ),
            ModelFile(
                relativePath: "Data/com.apple.CoreML/weights/weight.bin",
                sha256: "fa60d9b6a155734f59029ebb882fd54e549bfaee3539c1a9cbd2cbbab64a0fed",
                sizeBytes: 49_419_072
            ),
        ]
    )

    public static let all: [ModelDescriptor] = [depthAnythingV2Small]
}
