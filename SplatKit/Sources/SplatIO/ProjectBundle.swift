import Foundation
import SplatCore

/// Where the camera poses of a project came from.
public enum PoseSource: String, Codable, Sendable {
    case colmap
    case arkit
}

/// One saved checkpoint of a training run.
///
/// The held-out fields are Phase 4 additions. They are optional, so checkpoints written before
/// them still decode.
public struct CheckpointRecord: Codable, Sendable, Equatable {
    public var iteration: Int
    public var gaussianCount: Int
    public var psnr: Double?
    public var date: Date
    /// Name of the view held out from training for evaluation, if any.
    public var heldOutView: String?
    public var heldOutPSNR: Double?
    public var heldOutSSIM: Double?
    /// Iteration with the best held-out score, when early stopping was used.
    public var bestIteration: Int?

    public init(iteration: Int, gaussianCount: Int, psnr: Double?, date: Date,
                heldOutView: String? = nil, heldOutPSNR: Double? = nil,
                heldOutSSIM: Double? = nil, bestIteration: Int? = nil) {
        self.iteration = iteration
        self.gaussianCount = gaussianCount
        self.psnr = psnr
        self.date = date
        self.heldOutView = heldOutView
        self.heldOutPSNR = heldOutPSNR
        self.heldOutSSIM = heldOutSSIM
        self.bestIteration = bestIteration
    }
}

/// What the project was captured from. Optional: Phase 3 projects do not have one.
public enum InputKind: String, Codable, Sendable {
    case video
    case photos
    case colmap
    case arkit
}

/// Where one stored image came from. Only written for projects built from frames.
public struct FrameProvenance: Codable, Sendable, Equatable {
    /// Name of the image under `images/`.
    public var imageName: String
    /// Original file name the image came from.
    public var sourceName: String
    /// Position in the video in seconds, or `nil` for a photo.
    public var timestamp: Double?
    /// 35 mm-equivalent focal length from the photo's EXIF, or `nil` when unknown. Optional, so
    /// manifests written before this field existed still decode.
    public var focalLength35mm: Double?

    public init(imageName: String, sourceName: String, timestamp: Double?, focalLength35mm: Double? = nil) {
        self.imageName = imageName
        self.sourceName = sourceName
        self.timestamp = timestamp
        self.focalLength35mm = focalLength35mm
    }
}

/// Contents of `manifest.json` in a `.gsproj` package.
///
/// `inputKind` and `frameProvenance` are optional additions within format version 1. A manifest
/// without them (every Phase 3 project) decodes with `nil` and `[]`.
public struct ProjectManifest: Codable, Sendable, Equatable {
    public static let currentFormatVersion = 1

    public var formatVersion: Int
    public var source: PoseSource?
    /// Image names as given by the pose source, sorted. Stored under `images/<name>`.
    public var imageFiles: [String]
    public var training: TrainingConfiguration
    public var checkpoints: [CheckpointRecord]
    /// The kind of capture the images came from, if recorded.
    public var inputKind: InputKind?
    /// Per-image origin, in the order the images were stored. Empty when not recorded.
    public var frameProvenance: [FrameProvenance]

    /// An empty project at the current format version.
    public init() {
        formatVersion = Self.currentFormatVersion
        source = nil
        imageFiles = []
        training = TrainingConfiguration()
        checkpoints = []
        inputKind = nil
        frameProvenance = []
    }

    /// Decodes a manifest. Keys added after format version 1 shipped are optional, so older
    /// manifests still load.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        formatVersion = try container.decode(Int.self, forKey: .formatVersion)
        source = try container.decodeIfPresent(PoseSource.self, forKey: .source)
        imageFiles = try container.decode([String].self, forKey: .imageFiles)
        training = try container.decode(TrainingConfiguration.self, forKey: .training)
        checkpoints = try container.decode([CheckpointRecord].self, forKey: .checkpoints)
        inputKind = try container.decodeIfPresent(InputKind.self, forKey: .inputKind)
        frameProvenance = try container.decodeIfPresent([FrameProvenance].self,
                                                        forKey: .frameProvenance) ?? []
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
    public static let depthDirectory = "depth"

    /// `depth/<name>.sdpt` for the image stored as `name`.
    public static func depthPath(forImage name: String) -> String {
        depthDirectory + "/" + name + "." + DepthMapCodec.fileExtension
    }
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
        var result = clearedForImport(contents,
                                      source: .colmap,
                                      inputKind: .colmap,
                                      imageFiles: imageFiles,
                                      frameProvenance: [])
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
        importARKit(capture: capture, imageFiles: imageFiles, depthFiles: [:], into: contents)
    }

    /// Replaces the images, poses, depth and checkpoints of `contents` with an ARKit JSON capture.
    /// `depthFiles` is keyed by image name and holds the stored `.sdpt` bytes of each image's depth.
    public static func importARKit(capture: Data,
                                   imageFiles: [String: Data],
                                   depthFiles: [String: Data],
                                   into contents: ProjectContents) -> ProjectContents {
        var result = clearedForImport(contents,
                                      source: .arkit,
                                      inputKind: .arkit,
                                      imageFiles: imageFiles,
                                      frameProvenance: [])
        result.files[ProjectLayout.arkitCapture] = capture
        for (name, data) in depthFiles {
            result.files[ProjectLayout.depthPath(forImage: name)] = data
        }
        return result
    }

    /// Imports the frames a pose provider registered, with the poses and sparse points it produced.
    ///
    /// Only frames whose name is in `result.registeredImageNames` are kept. They are stored under
    /// `images/`, and their poses are written as COLMAP binaries under `poses/colmap/`. Provenance
    /// is recorded in sorted name order. A point cloud is written only when it is not empty.
    public static func importEstimatedPoses(frames: [CapturedFrame],
                                            result: PoseEstimationResult,
                                            inputKind: InputKind,
                                            into contents: ProjectContents) -> ProjectContents {
        let registered = Set(result.registeredImageNames)
        let kept = frames
            .filter { registered.contains($0.name) }
            .sorted { $0.name < $1.name }
        let keptNames = Set(kept.map(\.name))
        let imageFiles = Dictionary(kept.map { ($0.name, $0.data) },
                                    uniquingKeysWith: { first, _ in first })
        let provenance = kept.map {
            FrameProvenance(imageName: $0.name,
                            sourceName: $0.sourceName,
                            timestamp: $0.timestamp,
                            focalLength35mm: $0.focalLength35mm)
        }

        var imported = clearedForImport(contents,
                                       source: .colmap,
                                       inputKind: inputKind,
                                       imageFiles: imageFiles,
                                       frameProvenance: provenance)
        let directory = ProjectLayout.colmapDirectory
        let images = result.images.filter { keptNames.contains($0.name) }
        imported.files[directory + "/cameras.bin"] = COLMAPWriter.cameras(result.cameras)
        imported.files[directory + "/images.bin"] = COLMAPWriter.images(images)
        if !result.points.isEmpty {
            imported.files[directory + "/points3D.bin"] = COLMAPWriter.points3D(result.points)
        }
        return imported
    }

    // MARK: - Depth maps

    /// Stores each map as `depth/<name>.sdpt`, keyed by image name. Entries for other images are
    /// left as they are.
    public static func withDepthMaps(_ contents: ProjectContents,
                                     _ maps: [String: DepthMap]) -> ProjectContents {
        var result = contents
        for (name, map) in maps {
            result.files[ProjectLayout.depthPath(forImage: name)] = DepthMapCodec.encode(map)
        }
        return result
    }

    /// Decodes every `depth/<name>.sdpt` file, keyed by image name. A file that does not decode is
    /// reported in `failures` under its name, and the other maps still load.
    public static func depthMaps(_ contents: ProjectContents)
        -> (maps: [String: DepthMap], failures: [String: DepthMapError]) {
        var maps: [String: DepthMap] = [:]
        var failures: [String: DepthMapError] = [:]
        let prefix = ProjectLayout.depthDirectory + "/"
        let suffix = "." + DepthMapCodec.fileExtension
        for (path, data) in contents.files where path.hasPrefix(prefix) && path.hasSuffix(suffix) {
            let name = String(path.dropFirst(prefix.count).dropLast(suffix.count))
            guard !name.isEmpty else {
                continue
            }
            do {
                maps[name] = try DepthMapCodec.decode(data)
            } catch {
                failures[name] = error
            }
        }
        return (maps, failures)
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

    /// Drops everything an import replaces, then records the new source, input kind, provenance
    /// and image names.
    private static func clearedForImport(_ contents: ProjectContents,
                                         source: PoseSource,
                                         inputKind: InputKind,
                                         imageFiles: [String: Data],
                                         frameProvenance: [FrameProvenance]) -> ProjectContents {
        let managedPrefixes = [
            ProjectLayout.imagesDirectory + "/",
            ProjectLayout.depthDirectory + "/",
            "poses/",
            "checkpoints/",
        ]
        var result = contents
        result.files = contents.files.filter { path, _ in
            !managedPrefixes.contains { path.hasPrefix($0) }
        }
        result.manifest.source = source
        result.manifest.inputKind = inputKind
        result.manifest.frameProvenance = frameProvenance
        result.manifest.imageFiles = imageFiles.keys.sorted()
        result.manifest.checkpoints = []
        for (name, data) in imageFiles {
            result.files[ProjectLayout.imagesDirectory + "/" + name] = data
        }
        return result
    }
}
