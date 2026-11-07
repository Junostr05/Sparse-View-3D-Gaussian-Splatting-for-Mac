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

    /// MoGe-2 ViT-S (Microsoft, MIT) as an unofficial Core ML conversion, `ebowwa/moge-2-vits-normal-coreml`,
    /// int8 variant. Pinned to the commit below. Fixed input: multi-array `image` of shape [1, 3, 512, 768]
    /// (NCHW, RGB 0...1). Outputs used: `points` [1, 512, 768, 3] (camera-space points; z is the depth) and
    /// `mask` [1, 512, 768] (valid where > 0.5). The model is not bundled: it is downloaded on request.
    public static let moge2SmallInt8 = ModelDescriptor(
        id: "moge-2-vits-int8",
        displayName: "MoGe-2 Small (depth)",
        version: "e01dfc7",
        license: "MIT",
        baseURL: URL(string: "https://huggingface.co/ebowwa/moge-2-vits-normal-coreml/resolve/e01dfc7e5148e57fbad3b72fee832f8e9d6fdbda"),
        packageName: "moge_vits_normal_int8.mlpackage",
        files: [
            ModelFile(
                relativePath: "Manifest.json",
                sha256: "a370d9b38cca877bb62ba44ab37d94f751358acd146fbf7a8f7ec23f72ff8e16",
                sizeBytes: 617
            ),
            ModelFile(
                relativePath: "Data/com.apple.CoreML/model.mlmodel",
                sha256: "9a117c8978535ea9d9d02d1aa8031cc85b1d5f0e3acb75ee59eb416023bfd937",
                sizeBytes: 301_577
            ),
            ModelFile(
                relativePath: "Data/com.apple.CoreML/weights/weight.bin",
                sha256: "b3a01f24323632b5a4e6d41dfefadb69cad505efeb399c736ca0e8c0817cde2b",
                sizeBytes: 36_128_704
            ),
        ]
    )

    /// Every downloadable depth model, in the order the app lists them.
    public static let all: [ModelDescriptor] = [depthAnythingV2Small, moge2SmallInt8]
}

extension ModelDescriptor {
    /// Depth Anything 3 Base as a multi-function Core ML package for pose-free estimation (Apache-2.0).
    /// Not published yet: `baseURL` is `nil`, so the model is never downloaded. The user imports a converted
    /// package with `ModelStore.importPackage(from:for:)`, and it is found with `ModelStore.importedPackageURL(for:)`.
    /// Kept out of `ModelCatalog.all` until a download source exists.
    public static let depthAnything3Base = ModelDescriptor(
        id: "depth-anything-3-base",
        displayName: "Depth Anything 3 Base (pose-free)",
        version: "1",
        license: "Apache-2.0",
        baseURL: nil,
        packageName: "DepthAnything3BaseQ8.mlpackage",
        // Checksums of the package built by Tools/da3/convert_da3.py. They become active once `baseURL`
        // points at the published copy (Tools/da3/prepare_hf_repo.py prepares it; see docs/models/index.html).
        files: [
            ModelFile(relativePath: "Manifest.json",
                      sha256: "083e08c6080ac35ddc7b91810e9d7cf970fd46d3a23561d467a4cc92595e60fd",
                      sizeBytes: 617),
            ModelFile(relativePath: "Data/com.apple.CoreML/model.mlmodel",
                      sha256: "50faeb905fac1b6fd260872ba1db0f865170d9b53d56d0a6c3ddf1658a34ccab",
                      sizeBytes: 3_454_932),
            ModelFile(relativePath: "Data/com.apple.CoreML/weights/weight.bin",
                      sha256: "326259a56ddbebdd7de59dc3324a99e495cffe5e3c15f11e2431f59f7a39127a",
                      sizeBytes: 607_552_064),
        ]
    )
}
