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

/// The pose provider that registered the frames of a project. Optional: projects without estimated poses
/// (COLMAP and ARKit imports, Phase 3 projects) keep `nil`.
public enum PoseMethod: String, Codable, Sendable {
    case apple
    case depthAnything3
}

extension UpAxis {
    /// The up axis a viewer should start with for a project. COLMAP world frames follow the OpenCV convention
    /// (-Y up). Apple Photogrammetry (photos, video) and ARKit keep their gravity-aligned +Y-up world; only the
    /// camera axes are converted on import. Measured on the 13 table photos: the mean camera up vector in the
    /// world frame is +Y and the cameras sit above the point cloud.
    public static func forInputKind(_ kind: InputKind?) -> UpAxis {
        switch kind {
        case .photos, .video, .arkit: .positiveY
        case .colmap, nil: .negativeY
        }
    }

    /// The start up axis of a project. Depth Anything 3 uses the OpenCV frame of its first camera as the world
    /// frame (y down), so its projects start at -Y up. Other projects follow their input kind.
    public static func forProject(_ manifest: ProjectManifest) -> UpAxis {
        if manifest.poseMethod == .depthAnything3 {
            return .negativeY
        }
        return forInputKind(manifest.inputKind)
    }
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

/// Summary of the deletions stored in `edits/deleted.bin`. The indices themselves live in that file.
public struct EditRecord: Codable, Sendable, Equatable {
    /// Gaussian count of the checkpoint the deletions refer to.
    public var baseGaussianCount: Int
    /// Number of deleted indices in the file.
    public var deletedCount: Int

    public init(baseGaussianCount: Int, deletedCount: Int) {
        self.baseGaussianCount = baseGaussianCount
        self.deletedCount = deletedCount
    }
}

/// Contents of `manifest.json` in a `.gsproj` package.
///
/// `inputKind`, `frameProvenance`, `edits` and `poseMethod` are optional additions within format version 1.
/// A manifest without them (every Phase 3 project) decodes with `nil`, `[]`, `nil` and `nil`.
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
    /// Non-destructive deletions of the latest checkpoint, if any.
    public var edits: EditRecord?
    /// The pose provider behind the stored poses, when they were estimated. `nil` for COLMAP and ARKit imports.
    public var poseMethod: PoseMethod?

    /// An empty project at the current format version.
    public init() {
        formatVersion = Self.currentFormatVersion
        source = nil
        imageFiles = []
        training = TrainingConfiguration()
        checkpoints = []
        inputKind = nil
        frameProvenance = []
        edits = nil
        poseMethod = nil
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
        edits = try container.decodeIfPresent(EditRecord.self, forKey: .edits)
        poseMethod = try container.decodeIfPresent(PoseMethod.self, forKey: .poseMethod)
    }
}

public enum ProjectError: Error, Equatable {
    case missingManifest
    case unsupportedFormatVersion(Int)
    case corruptManifest(String)
    case corruptCheckpoint(String)
    case corruptRefinedPoses(String)
}

/// A camera pose after training-time refinement: the world→camera rotation and translation of one view.
public struct RefinedPose: Sendable, Equatable {
    public var rotation: Matrix3
    public var translation: SIMD3<Float>

    public init(rotation: Matrix3, translation: SIMD3<Float>) {
        self.rotation = rotation
        self.translation = translation
    }
}

/// Relative paths inside a `.gsproj` package.
public enum ProjectLayout {
    public static let manifest = "manifest.json"
    public static let imagesDirectory = "images"
    public static let colmapDirectory = "poses/colmap"
    public static let arkitCapture = "poses/arkit.json"
    public static let latestCheckpoint = "checkpoints/latest.ply"
    public static let depthDirectory = "depth"
    /// Deleted Gaussian indices of the latest checkpoint, as `DeletionListCodec` data.
    public static let deletedIndices = "edits/deleted.bin"
    /// Camera poses refined by training, as `RefinedPoseFile` JSON. Lives under `poses/`, so every import
    /// (which replaces that directory) removes it.
    public static let refinedPoses = "poses/refined.json"
    /// Directory of files the app derives from the project, such as the library cover.
    public static let previewDirectory = "preview"
    /// JPEG cover of the project, rendered from the latest checkpoint. Optional: older projects have none.
    public static let coverImage = "preview/cover.jpg"

    /// `depth/<name>.sdpt` for the image stored as `name`.
    public static func depthPath(forImage name: String) -> String {
        depthDirectory + "/" + name + "." + DepthMapCodec.fileExtension
    }

    /// Directory of the maps one monocular estimator produced: `depth/estimated/<estimator>/`. Kept apart from the
    /// supplied maps directly under `depth/`, and apart per estimator, so a map of one model is never used for another.
    public static func estimatedDepthDirectory(for estimator: DepthEstimatorKind) -> String {
        depthDirectory + "/estimated/" + estimator.rawValue
    }

    /// `depth/estimated/<estimator>/<name>.sdpt` for the image stored as `name`.
    public static func estimatedDepthPath(forImage name: String, estimator: DepthEstimatorKind) -> String {
        estimatedDepthDirectory(for: estimator) + "/" + name + "." + DepthMapCodec.fileExtension
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
    /// `method` records which provider produced the poses.
    public static func importEstimatedPoses(frames: [CapturedFrame],
                                            result: PoseEstimationResult,
                                            inputKind: InputKind,
                                            method: PoseMethod = .apple,
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
        imported.manifest.poseMethod = method
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

    /// Stores the maps of one monocular estimator under `depth/estimated/<estimator>/`. Maps of other estimators
    /// and the supplied maps are left as they are.
    public static func withEstimatedDepthMaps(_ contents: ProjectContents,
                                              _ maps: [String: DepthMap],
                                              estimator: DepthEstimatorKind) -> ProjectContents {
        var result = contents
        for (name, map) in maps {
            result.files[ProjectLayout.estimatedDepthPath(forImage: name, estimator: estimator)] = DepthMapCodec.encode(map)
        }
        return result
    }

    /// Decodes every `depth/<name>.sdpt` file, keyed by image name. A file that does not decode is
    /// reported in `failures` under its name, and the other maps still load. Only direct children of
    /// `depth/` count: the per-estimator cache under `depth/estimated/` is not a supplied map.
    public static func depthMaps(_ contents: ProjectContents)
        -> (maps: [String: DepthMap], failures: [String: DepthMapError]) {
        let prefix = ProjectLayout.depthDirectory + "/"
        return decodeMaps(contents, prefix: prefix)
    }

    /// Decodes the maps one monocular estimator stored under `depth/estimated/<estimator>/`, keyed by image name.
    /// Failures are reported as in `depthMaps(_:)`.
    public static func estimatedDepthMaps(_ contents: ProjectContents, estimator: DepthEstimatorKind)
        -> (maps: [String: DepthMap], failures: [String: DepthMapError]) {
        let prefix = ProjectLayout.estimatedDepthDirectory(for: estimator) + "/"
        return decodeMaps(contents, prefix: prefix)
    }

    /// Shared decoding for `depthMaps` and `estimatedDepthMaps`: every `<prefix><name>.sdpt` whose name has no
    /// slash, so a file in a deeper directory is never taken for a map of this one.
    private static func decodeMaps(_ contents: ProjectContents, prefix: String)
        -> (maps: [String: DepthMap], failures: [String: DepthMapError]) {
        var maps: [String: DepthMap] = [:]
        var failures: [String: DepthMapError] = [:]
        let suffix = "." + DepthMapCodec.fileExtension
        for (path, data) in contents.files where path.hasPrefix(prefix) && path.hasSuffix(suffix) {
            let name = String(path.dropFirst(prefix.count).dropLast(suffix.count))
            guard !name.isEmpty, !name.contains("/") else {
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
    /// With `useRefinedPoses`, each view's rotation and translation come from the stored refined poses
    /// (see `withRefinedPoses`); views without an entry keep their original pose.
    public static func loadDataset(_ contents: ProjectContents,
                                   decode: TrainingDatasetLoader.ImageDecoder,
                                   useRefinedPoses: Bool = false) throws(DatasetError) -> TrainingDataset {
        let files = contents.files
        let imageData: (String) -> Data? = { name in
            files[ProjectLayout.imagesDirectory + "/" + name]
        }
        let maxImageSide = contents.manifest.training.maxImageSide
        let overrides = try refinedOverrides(contents, enabled: useRefinedPoses)

        var dataset: TrainingDataset
        switch contents.manifest.source {
        case .colmap?:
            let directory = ProjectLayout.colmapDirectory
            dataset = try TrainingDatasetLoader.loadCOLMAP(
                cameras: files[directory + "/cameras.bin"] ?? Data(),
                images: files[directory + "/images.bin"] ?? Data(),
                points3D: files[directory + "/points3D.bin"],
                imageData: imageData,
                maxImageSide: maxImageSide,
                decode: decode
            )
        case .arkit?:
            dataset = try TrainingDatasetLoader.loadARKit(
                capture: files[ProjectLayout.arkitCapture] ?? Data(),
                imageData: imageData,
                maxImageSide: maxImageSide,
                decode: decode
            )
        case nil:
            throw .noViews
        }
        if !overrides.isEmpty {
            for index in dataset.views.indices {
                if let pose = overrides[dataset.views[index].name] {
                    dataset.views[index].camera = applying(pose, to: dataset.views[index].camera)
                }
            }
        }
        return dataset
    }

    /// Camera poses of the project without decoding any image. Uses the same pose files as
    /// `loadDataset`. A project without a pose source has no cameras, so the result is `[]`.
    /// With `useRefinedPoses`, the stored refined poses replace the rotation and translation of their cameras.
    public static func cameraPoses(_ contents: ProjectContents,
                                   useRefinedPoses: Bool = false) throws(DatasetError) -> [NamedCamera] {
        let files = contents.files
        let overrides = try refinedOverrides(contents, enabled: useRefinedPoses)

        var cameras: [NamedCamera]
        switch contents.manifest.source {
        case .colmap?:
            let directory = ProjectLayout.colmapDirectory
            cameras = try TrainingDatasetLoader.cameras(
                colmapCameras: files[directory + "/cameras.bin"] ?? Data(),
                colmapImages: files[directory + "/images.bin"] ?? Data()
            )
        case .arkit?:
            cameras = try TrainingDatasetLoader.cameras(
                arkitCapture: files[ProjectLayout.arkitCapture] ?? Data()
            )
        case nil:
            return []
        }
        if !overrides.isEmpty {
            for index in cameras.indices {
                if let pose = overrides[cameras[index].name] {
                    cameras[index].camera = applying(pose, to: cameras[index].camera)
                }
            }
        }
        return cameras
    }

    // MARK: - Refined poses

    /// Stores the poses training refined, keyed by image name, in place of any earlier refined poses.
    /// `cameras` are the training cameras of the run, each named after its image. Non-finite cameras are
    /// skipped. When nothing is left, the file is removed.
    public static func withRefinedPoses(_ contents: ProjectContents, cameras: [NamedCamera]) -> ProjectContents {
        var poses: [String: RefinedPoseRecord] = [:]
        for named in cameras where isFinite(named.camera) {
            poses[named.name] = RefinedPoseRecord(pose: RefinedPose(rotation: named.camera.rotation,
                                                                     translation: named.camera.translation))
        }
        var result = contents
        if poses.isEmpty {
            result.files[ProjectLayout.refinedPoses] = nil
        } else {
            // Finite values always encode. A failure would drop the stale file, never keep it.
            result.files[ProjectLayout.refinedPoses] = try? encodeRefinedPoses(RefinedPoseFile(poses: poses))
        }
        return result
    }

    /// Removes the stored refined poses, so the project's cameras are the original ones again.
    public static func withoutRefinedPoses(_ contents: ProjectContents) -> ProjectContents {
        var result = contents
        result.files[ProjectLayout.refinedPoses] = nil
        return result
    }

    /// The stored refined poses by image name. Empty when the project has none.
    /// Throws `ProjectError.corruptRefinedPoses` when the file does not decode or holds an invalid pose.
    public static func refinedPoses(_ contents: ProjectContents) throws(ProjectError) -> [String: RefinedPose] {
        guard let data = contents.files[ProjectLayout.refinedPoses] else {
            return [:]
        }
        let file: RefinedPoseFile
        do {
            file = try JSONDecoder().decode(RefinedPoseFile.self, from: data)
        } catch {
            throw .corruptRefinedPoses(String(describing: error))
        }
        guard file.version == RefinedPoseFile.currentVersion else {
            throw .corruptRefinedPoses("unsupported version \(file.version)")
        }
        var poses: [String: RefinedPose] = [:]
        for (name, record) in file.poses {
            guard let pose = record.pose else {
                throw .corruptRefinedPoses("invalid pose for \(name)")
            }
            poses[name] = pose
        }
        return poses
    }

    /// Stored refined poses to apply, or none when `enabled` is false. Maps a decoding failure to `DatasetError`.
    private static func refinedOverrides(_ contents: ProjectContents,
                                         enabled: Bool) throws(DatasetError) -> [String: RefinedPose] {
        guard enabled else {
            return [:]
        }
        do throws(ProjectError) {
            return try refinedPoses(contents)
        } catch {
            throw .invalidRefinedPoses(String(describing: error))
        }
    }

    /// The camera with the refined rotation and translation. Intrinsics and size are unchanged.
    private static func applying(_ pose: RefinedPose, to camera: Camera) -> Camera {
        var result = camera
        result.rotation = pose.rotation
        result.translation = pose.translation
        return result
    }

    private static func isFinite(_ camera: Camera) -> Bool {
        let r = camera.rotation
        let values: [SIMD3<Float>] = [r.c0, r.c1, r.c2, camera.translation]
        return values.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }
    }

    private static func encodeRefinedPoses(_ file: RefinedPoseFile) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(file)
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
    /// Deletions of the previous checkpoint are dropped, because they refer to its Gaussians.
    public static func withCheckpoint(_ contents: ProjectContents,
                                      cloud: GaussianCloud,
                                      record: CheckpointRecord) -> ProjectContents {
        var result = contents
        result.files[ProjectLayout.latestCheckpoint] = PLYWriter.write(cloud)
        result.files[ProjectLayout.deletedIndices] = nil
        result.manifest.edits = nil
        result.manifest.checkpoints.append(record)
        return result
    }

    // MARK: - Cover

    /// The stored cover, or `nil` when the package has none.
    public static func cover(_ contents: ProjectContents) -> Data? {
        contents.files[ProjectLayout.coverImage]
    }

    /// Stores `jpeg` as the cover. Covers are written after a checkpoint, so earlier covers are replaced.
    public static func withCover(_ contents: ProjectContents, jpeg: Data) -> ProjectContents {
        var result = contents
        result.files[ProjectLayout.coverImage] = jpeg
        return result
    }

    // MARK: - Deletions

    /// Stores the deleted Gaussian indices of the latest checkpoint. Duplicates are removed and the
    /// indices are sorted. An empty list removes the file and the `edits` record.
    /// `baseGaussianCount` is the Gaussian count of that checkpoint.
    public static func withDeletions(_ contents: ProjectContents,
                                     baseGaussianCount: Int,
                                     deleted: [UInt32]) -> ProjectContents {
        var result = contents
        let sorted = Array(Set(deleted)).sorted()
        if sorted.isEmpty {
            result.files[ProjectLayout.deletedIndices] = nil
            result.manifest.edits = nil
        } else {
            result.files[ProjectLayout.deletedIndices] = DeletionListCodec.encode(sorted)
            result.manifest.edits = EditRecord(baseGaussianCount: baseGaussianCount,
                                               deletedCount: sorted.count)
        }
        return result
    }

    /// The stored deletions, or `nil` when there are none or they do not belong to the latest checkpoint.
    ///
    /// - `nil` when there is no `edits` record, or when the latest checkpoint's Gaussian count (from the
    ///   manifest, no PLY is decoded) differs from the record's `baseGaussianCount`. Stale deletions are
    ///   ignored, not reported.
    /// - Throws `ProjectError.corruptManifest` when the record exists but `edits/deleted.bin` is missing
    ///   or holds a different number of indices than the record says.
    /// - Throws `ProjectError.corruptCheckpoint` when the file does not decode, including an index at or
    ///   beyond the base Gaussian count.
    public static func deletions(_ contents: ProjectContents)
        throws(ProjectError) -> (baseGaussianCount: Int, deleted: [UInt32])? {
        guard let record = contents.manifest.edits else {
            return nil
        }
        guard let latest = contents.manifest.checkpoints.last,
              latest.gaussianCount == record.baseGaussianCount else {
            return nil
        }
        guard let data = contents.files[ProjectLayout.deletedIndices] else {
            throw .corruptManifest("edits record without \(ProjectLayout.deletedIndices)")
        }
        let deleted: [UInt32]
        do throws(DeletionListError) {
            deleted = try DeletionListCodec.decode(data, maxIndex: record.baseGaussianCount)
        } catch {
            throw .corruptCheckpoint("deleted indices: \(error)")
        }
        guard deleted.count == record.deletedCount else {
            throw .corruptManifest("edits record says \(record.deletedCount) deletions, file holds \(deleted.count)")
        }
        return (record.baseGaussianCount, deleted)
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
    /// and image names. The pose method is reset; `importEstimatedPoses` sets it again afterwards.
    /// Everything under `poses/` goes, which includes `ProjectLayout.refinedPoses`: refined poses
    /// belong to the poses they refined and are stale after any import.
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
            "edits/",
            // The cover shows the checkpoint that is replaced here, so it goes too.
            ProjectLayout.previewDirectory + "/",
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
        result.manifest.edits = nil
        result.manifest.poseMethod = nil
        for (name, data) in imageFiles {
            result.files[ProjectLayout.imagesDirectory + "/" + name] = data
        }
        return result
    }
}

/// `poses/refined.json`: the refined pose of each image, by image name.
private struct RefinedPoseFile: Codable {
    static let currentVersion = 1

    var version: Int
    var poses: [String: RefinedPoseRecord]

    init(poses: [String: RefinedPoseRecord]) {
        self.version = Self.currentVersion
        self.poses = poses
    }
}

/// One refined pose as stored. The rotation is column-major (c0, c1, c2 as x, y, z triples) and the
/// translation is (x, y, z). Both map world to camera, as in `Camera`.
private struct RefinedPoseRecord: Codable {
    var rotation: [Float]
    var translation: [Float]

    init(pose: RefinedPose) {
        let r = pose.rotation
        rotation = [r.c0.x, r.c0.y, r.c0.z, r.c1.x, r.c1.y, r.c1.z, r.c2.x, r.c2.y, r.c2.z]
        translation = [pose.translation.x, pose.translation.y, pose.translation.z]
    }

    /// The pose, or `nil` when an array has the wrong size or a value is not finite.
    var pose: RefinedPose? {
        guard rotation.count == 9, translation.count == 3, (rotation + translation).allSatisfy(\.isFinite) else {
            return nil
        }
        let matrix = Matrix3(c0: SIMD3(rotation[0], rotation[1], rotation[2]),
                             c1: SIMD3(rotation[3], rotation[4], rotation[5]),
                             c2: SIMD3(rotation[6], rotation[7], rotation[8]))
        return RefinedPose(rotation: matrix, translation: SIMD3(translation[0], translation[1], translation[2]))
    }
}
