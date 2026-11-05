import Foundation
import SplatCore

/// Where the camera poses of a project came from.
public enum PoseSource: String, Codable, Sendable {
    case colmap
    case arkit
}

/// One saved checkpoint of a training run.
public struct CheckpointRecord: Codable, Sendable, Equatable {
    public var iteration: Int
    public var gaussianCount: Int
    public var psnr: Double?
    public var date: Date

    public init(iteration: Int, gaussianCount: Int, psnr: Double?, date: Date) {
        self.iteration = iteration
        self.gaussianCount = gaussianCount
        self.psnr = psnr
        self.date = date
    }
}

/// Contents of `manifest.json` in a `.gsproj` package.
public struct ProjectManifest: Codable, Sendable, Equatable {
    public static let currentFormatVersion = 1

    public var formatVersion: Int
    public var source: PoseSource?
    /// Image names as given by the pose source, sorted. Stored under `images/<name>`.
    public var imageFiles: [String]
    public var training: TrainingConfiguration
    public var checkpoints: [CheckpointRecord]

    /// An empty project at the current format version.
    public init() {
        formatVersion = Self.currentFormatVersion
        source = nil
        imageFiles = []
        training = TrainingConfiguration()
        checkpoints = []
    }
}

public enum ProjectError: Error, Equatable {
    case missingManifest
    case unsupportedFormatVersion(Int)
    case corruptManifest(String)
    case corruptCheckpoint(String)
}

/// Relative paths inside a `.gsproj` package.
public enum ProjectLayout {
    public static let manifest = "manifest.json"
    public static let imagesDirectory = "images"
    public static let colmapDirectory = "poses/colmap"
    public static let arkitCapture = "poses/arkit.json"
    public static let latestCheckpoint = "checkpoints/latest.ply"
}

/// The package contents as plain data: the manifest plus every other file by relative path.
/// `files` never contains `manifest.json`; `ProjectBundle.encode` adds it.
public struct ProjectContents: Sendable, Equatable {
    public var manifest: ProjectManifest
    public var files: [String: Data]

    public init(manifest: ProjectManifest, files: [String: Data]) {
        self.manifest = manifest
        self.files = files
    }
}

/// Encoding, decoding and import of `.gsproj` contents. Operates on `Data` only.
public enum ProjectBundle {
    // MARK: - Manifest

    public static func encodeManifest(_ manifest: ProjectManifest) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(manifest)
    }

    /// Reads `manifest.json` from `files`. Newer format versions are rejected before the rest of
    /// the manifest is interpreted. Every other path must be a safe relative path.
    public static func decode(files: [String: Data]) throws(ProjectError) -> ProjectContents {
        guard let manifestData = files[ProjectLayout.manifest] else {
            throw .missingManifest
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        let version: Int
        do {
            version = try decoder.decode(VersionProbe.self, from: manifestData).formatVersion
        } catch {
            throw .corruptManifest(String(describing: error))
        }
        guard version <= ProjectManifest.currentFormatVersion else {
            throw .unsupportedFormatVersion(version)
        }

        let manifest: ProjectManifest
        do {
            manifest = try decoder.decode(ProjectManifest.self, from: manifestData)
        } catch {
            throw .corruptManifest(String(describing: error))
        }

        var contentFiles: [String: Data] = [:]
        for (path, data) in files where path != ProjectLayout.manifest {
            guard isSafeRelativePath(path) else {
                throw .corruptManifest("unsafe file path \(path)")
            }
            contentFiles[path] = data
        }
        for name in manifest.imageFiles where !isSafeRelativePath(name) {
            throw .corruptManifest("unsafe image name \(name)")
        }
        return ProjectContents(manifest: manifest, files: contentFiles)
    }

    /// Returns every package file, including the freshly encoded `manifest.json`.
    public static func encode(_ contents: ProjectContents) throws -> [String: Data] {
        var result: [String: Data] = [:]
        for (path, data) in contents.files {
            guard path != ProjectLayout.manifest, isSafeRelativePath(path) else {
                throw ProjectError.corruptManifest("unsafe file path \(path)")
            }
            result[path] = data
        }
        for name in contents.manifest.imageFiles where !isSafeRelativePath(name) {
            throw ProjectError.corruptManifest("unsafe image name \(name)")
        }
        result[ProjectLayout.manifest] = try encodeManifest(contents.manifest)
        return result
    }

    // MARK: - Import

    /// Replaces the images, poses and checkpoints of `contents` with a COLMAP reconstruction.
    /// `imageFiles` maps COLMAP image names to their bytes. Training settings are kept.
    public static func importCOLMAP(cameras: Data,
                                    images: Data,
                                    points3D: Data?,
                                    imageFiles: [String: Data],
                                    into contents: ProjectContents) -> ProjectContents {
        var result = clearedForImport(contents, source: .colmap, imageFiles: imageFiles)
        let directory = ProjectLayout.colmapDirectory
        result.files[directory + "/cameras.bin"] = cameras
        result.files[directory + "/images.bin"] = images
        if let points3D {
            result.files[directory + "/points3D.bin"] = points3D
        }
        return result
    }

    /// Replaces the images, poses and checkpoints of `contents` with an ARKit JSON capture.
    public static func importARKit(capture: Data,
                                   imageFiles: [String: Data],
                                   into contents: ProjectContents) -> ProjectContents {
        var result = clearedForImport(contents, source: .arkit, imageFiles: imageFiles)
        result.files[ProjectLayout.arkitCapture] = capture
        return result
    }

    // MARK: - Dataset and checkpoints

    /// Builds the training dataset from the stored poses and images. A project without a pose
    /// source has no views. Missing pose files surface as the reader's own errors.
    public static func loadDataset(_ contents: ProjectContents,
                                   decode: TrainingDatasetLoader.ImageDecoder) throws(DatasetError) -> TrainingDataset {
        let files = contents.files
        let imageData: (String) -> Data? = { name in
            files[ProjectLayout.imagesDirectory + "/" + name]
        }
        let maxImageSide = contents.manifest.training.maxImageSide

        switch contents.manifest.source {
        case .colmap?:
            let directory = ProjectLayout.colmapDirectory
            return try TrainingDatasetLoader.loadCOLMAP(
                cameras: files[directory + "/cameras.bin"] ?? Data(),
                images: files[directory + "/images.bin"] ?? Data(),
                points3D: files[directory + "/points3D.bin"],
                imageData: imageData,
                maxImageSide: maxImageSide,
                decode: decode
            )
        case .arkit?:
            return try TrainingDatasetLoader.loadARKit(
                capture: files[ProjectLayout.arkitCapture] ?? Data(),
                imageData: imageData,
                maxImageSide: maxImageSide,
                decode: decode
            )
        case nil:
            throw .noViews
        }
    }

    /// The stored checkpoint, or `nil` when the project has none.
    public static func latestCheckpoint(_ contents: ProjectContents) throws(ProjectError) -> GaussianCloud? {
        guard let data = contents.files[ProjectLayout.latestCheckpoint] else {
            return nil
        }
        do {
            return try PLYReader.read(data)
        } catch {
            throw .corruptCheckpoint(String(describing: error))
        }
    }

    /// Stores `cloud` as the latest checkpoint and appends `record` to the manifest.
    public static func withCheckpoint(_ contents: ProjectContents,
                                      cloud: GaussianCloud,
                                      record: CheckpointRecord) -> ProjectContents {
        var result = contents
        result.files[ProjectLayout.latestCheckpoint] = PLYWriter.write(cloud)
        result.manifest.checkpoints.append(record)
        return result
    }

    // MARK: - Helpers

    /// Only the format version is read first, so a newer manifest is reported as such.
    private struct VersionProbe: Decodable {
        let formatVersion: Int
    }

    /// A safe package path is relative, non-empty, uses `/` and has no empty, `.` or `..`
    /// components and no backslashes.
    static func isSafeRelativePath(_ path: String) -> Bool {
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

    /// Drops everything an import replaces, then records the new source and image names.
    private static func clearedForImport(_ contents: ProjectContents,
                                         source: PoseSource,
                                         imageFiles: [String: Data]) -> ProjectContents {
        let managedPrefixes = [
            ProjectLayout.imagesDirectory + "/",
            "poses/",
            "checkpoints/",
        ]
        var result = contents
        result.files = contents.files.filter { path, _ in
            !managedPrefixes.contains { path.hasPrefix($0) }
        }
        result.manifest.source = source
        result.manifest.imageFiles = imageFiles.keys.sorted()
        result.manifest.checkpoints = []
        for (name, data) in imageFiles {
            result.files[ProjectLayout.imagesDirectory + "/" + name] = data
        }
        return result
    }
}
