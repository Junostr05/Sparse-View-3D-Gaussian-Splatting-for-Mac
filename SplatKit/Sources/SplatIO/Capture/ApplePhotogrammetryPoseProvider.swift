#if canImport(RealityKit) && canImport(ImageIO)
import Foundation
import ImageIO
import RealityKit
import SplatCore

/// Pose provider backed by Apple's `PhotogrammetrySession` (RealityKit).
///
/// Apple returns camera poses and a sparse point cloud. Each pose also carries intrinsics
/// (`Pose.intrinsics`, fx, fy, cx, cy in the stored image's pixel coordinates), which are used as the
/// primary source. Such a camera is PINHOLE. A pose without valid intrinsics falls back to the
/// 35 mm-equivalent EXIF focal length, or to `fallbackFocalLength35mm`, and its camera is
/// SIMPLE_PINHOLE. Cameras are shared between frames with the same model, pixel size and parameters.
///
/// `Pose.lensDistortionData` is deliberately ignored for now. The distortion model is not mapped to a
/// COLMAP model yet, so the cameras assume no lens distortion.
public struct ApplePhotogrammetryPoseProvider: PoseProvider {
    public let name = "Apple Photogrammetry"

    /// 35 mm-equivalent focal length used for frames without EXIF focal length. `nil` makes such
    /// frames fail the estimation.
    public let fallbackFocalLength35mm: Double?

    public init(fallbackFocalLength35mm: Double? = 26) {
        self.fallbackFocalLength35mm = fallbackFocalLength35mm
    }

    public func estimatePoses(for frames: [CapturedFrame]) async throws(PoseEstimationFailure) -> PoseEstimationResult {
        // Checked first, so that too few frames never touch RealityKit.
        if frames.count < PoseEstimationPolicy.minimumFrames {
            throw .tooFewFrames(have: frames.count, need: PoseEstimationPolicy.minimumFrames)
        }
        guard PhotogrammetrySession.isSupported else {
            throw .unavailable(reason: "PhotogrammetrySession is not supported on this Mac")
        }

        let directory: URL = try writeInput(frames)
        defer { try? FileManager.default.removeItem(at: directory) }

        let output: SessionOutput = try await runSession(input: directory)
        if Task.isCancelled {
            throw .cancelled
        }

        var poses: [ProviderPose] = []
        for id in output.poses.keys.sorted() {
            guard let url = output.urls[id], let pose = output.poses[id] else {
                continue
            }
            let translation: SIMD3<Float> = pose.translation
            let quaternion: SIMD4<Float> = pose.rotation.vector
            // Column-major simd_float3x3: fx = [0][0], fy = [1][1], cx = [2][0], cy = [2][1].
            // Validity is checked in buildResult, where the frame's pixel size is known.
            let intrinsics: SIMD4<Double>? = pose.intrinsics.map { (matrix) -> SIMD4<Double> in
                SIMD4<Double>(Double(matrix[0][0]), Double(matrix[1][1]),
                              Double(matrix[2][0]), Double(matrix[2][1]))
            }
            poses.append((
                name: url.lastPathComponent,
                centre: SIMD3<Double>(Double(translation.x), Double(translation.y), Double(translation.z)),
                quaternionXYZW: SIMD4<Double>(Double(quaternion.x), Double(quaternion.y),
                                              Double(quaternion.z), Double(quaternion.w)),
                intrinsics: intrinsics
            ))
        }

        var points: [(position: SIMD3<Double>, color: SIMD3<UInt8>)] = []
        points.reserveCapacity(output.points.count)
        for point in output.points {
            let position: SIMD3<Float> = point.position
            let color: SIMD4<UInt8> = point.color
            points.append((
                position: SIMD3<Double>(Double(position.x), Double(position.y), Double(position.z)),
                color: SIMD3<UInt8>(color.x, color.y, color.z)
            ))
        }

        return try Self.buildResult(frames: frames,
                                    poses: poses,
                                    points: points,
                                    fallbackFocalLength35mm: fallbackFocalLength35mm)
    }

    /// Builds the COLMAP-convention result from Apple's poses. Testable without running a session.
    ///
    /// - Parameters:
    ///   - frames: The input frames. Their `name` matches `poses[].name`.
    ///   - poses: Apple poses by image name, with the camera centre, the `(x, y, z, w)` quaternion
    ///     (camera-to-world, OpenGL convention) and optional intrinsics `(fx, fy, cx, cy)`. Intrinsics
    ///     that are not valid for the frame (see `intrinsicsAreValid`) are ignored as if missing. Poses
    ///     whose name matches no frame are dropped.
    ///   - points: Sparse points in the same world frame as the camera centres.
    ///   - fallbackFocalLength35mm: Used for frames without intrinsics whose `focalLength35mm` is `nil`.
    /// - Throws: `.failed` when a frame has neither valid intrinsics nor a focal length, or its pixel
    ///   size cannot be read.
    public static func buildResult(frames: [CapturedFrame],
                                   poses: [ProviderPose],
                                   points: [(position: SIMD3<Double>, color: SIMD3<UInt8>)],
                                   fallbackFocalLength35mm: Double?) throws(PoseEstimationFailure) -> PoseEstimationResult {
        var framesByName: [String: CapturedFrame] = [:]
        for frame in frames {
            framesByName[frame.name] = frame
        }

        var cameras: [EstimatedCamera] = []
        var cameraIndexByKey: [CameraKey: Int] = [:]
        var images: [EstimatedImagePose] = []

        for pose in poses {
            guard let frame = framesByName[pose.name] else {
                continue
            }
            let size: (width: Int, height: Int) = try pixelSize(of: frame)
            let camera: (modelID: Int32, params: [Double]) = try cameraParameters(
                for: frame,
                intrinsics: pose.intrinsics,
                width: size.width,
                height: size.height,
                fallbackFocalLength35mm: fallbackFocalLength35mm)
            // Frames share a camera when model, size and parameters agree to 0.01 px.
            let key = CameraKey(width: size.width,
                                height: size.height,
                                modelID: camera.modelID,
                                params: camera.params.map { (value: Double) -> Double in (value * 100).rounded() / 100 })

            let cameraIndex: Int
            if let existing = cameraIndexByKey[key] {
                cameraIndex = existing
            } else {
                cameras.append(EstimatedCamera(modelID: camera.modelID,
                                               width: size.width,
                                               height: size.height,
                                               params: camera.params))
                cameraIndex = cameras.count - 1
                cameraIndexByKey[key] = cameraIndex
            }

            let colmap = ApplePoseConversion.colmapPose(centre: pose.centre,
                                                        quaternionXYZW: pose.quaternionXYZW)
            images.append(EstimatedImagePose(name: frame.name,
                                             cameraIndex: cameraIndex,
                                             rotation: colmap.rotation,
                                             translation: colmap.translation))
        }

        images.sort { (a: EstimatedImagePose, b: EstimatedImagePose) -> Bool in a.name < b.name }

        var estimatedPoints: [EstimatedPoint] = []
        estimatedPoints.reserveCapacity(points.count)
        for point in points {
            estimatedPoints.append(EstimatedPoint(position: point.position, color: point.color, error: 0))
        }

        return PoseEstimationResult(cameras: cameras, images: images, points: estimatedPoints)
    }

    // MARK: - Private

    /// Pose as collected from the session: name, camera centre, `(x, y, z, w)` quaternion and optional
    /// intrinsics `(fx, fy, cx, cy)`.
    public typealias ProviderPose = (name: String, centre: SIMD3<Double>, quaternionXYZW: SIMD4<Double>, intrinsics: SIMD4<Double>?)

    /// Key for sharing one camera: model, pixel size and parameters rounded to 0.01 px.
    private struct CameraKey: Hashable {
        var width: Int
        var height: Int
        var modelID: Int32
        var params: [Double]
    }

    /// Camera model and parameters for one frame. Valid Apple intrinsics give PINHOLE
    /// `[fx, fy, cx, cy]`. Otherwise the focal length comes from EXIF or the fallback and gives
    /// SIMPLE_PINHOLE `[f, cx, cy]`. Throws when neither source exists.
    private static func cameraParameters(for frame: CapturedFrame,
                                         intrinsics: SIMD4<Double>?,
                                         width: Int,
                                         height: Int,
                                         fallbackFocalLength35mm: Double?) throws(PoseEstimationFailure) -> (modelID: Int32, params: [Double]) {
        if let values = intrinsics, intrinsicsAreValid(values, width: width, height: height) {
            return (modelID: 1, params: [values.x, values.y, values.z, values.w])
        }
        guard let focalLength35mm = frame.focalLength35mm ?? fallbackFocalLength35mm else {
            throw .failed(reason: "no focal length known for \(frame.name)")
        }
        let focal: Double = ApplePoseConversion.focalLengthPixels(focalLength35mm: focalLength35mm,
                                                                  width: width,
                                                                  height: height)
        let halfWidth: Double = Double(width) / 2
        let halfHeight: Double = Double(height) / 2
        return (modelID: 0, params: [focal, halfWidth, halfHeight])
    }

    /// Apple's intrinsics `(fx, fy, cx, cy)` are usable when all four are finite, `fx` and `fy` are
    /// positive, and the principal point lies inside the image: `0 < cx < width`, `0 < cy < height`.
    static func intrinsicsAreValid(_ values: SIMD4<Double>, width: Int, height: Int) -> Bool {
        guard values.x.isFinite, values.y.isFinite, values.z.isFinite, values.w.isFinite else {
            return false
        }
        let widthValue: Double = Double(width)
        let heightValue: Double = Double(height)
        return values.x > 0 && values.y > 0
            && values.z > 0 && values.z < widthValue
            && values.w > 0 && values.w < heightValue
    }

    /// Everything read from a finished session, kept as the raw Apple values.
    private struct SessionOutput {
        var poses: [Int: PhotogrammetrySession.Pose] = [:]
        var urls: [Int: URL] = [:]
        var points: [PhotogrammetrySession.PointCloud.Point] = []
    }

    /// Wraps the session so the cancellation handler can capture it. The SDK does not declare
    /// `PhotogrammetrySession` as `Sendable`. The handler only calls `cancel()`, which is safe from any thread.
    private struct SessionHandle: @unchecked Sendable {
        let session: PhotogrammetrySession
    }

    /// Writes every frame to a fresh temporary directory, named exactly as the frame.
    /// The caller removes the directory.
    private func writeInput(_ frames: [CapturedFrame]) throws(PoseEstimationFailure) -> URL {
        let directory: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("apple-pg-\(UUID().uuidString)", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            throw .failed(reason: "cannot create \(directory.lastPathComponent): \(error.localizedDescription)")
        }
        do {
            for frame in frames {
                try frame.data.write(to: directory.appendingPathComponent(frame.name))
            }
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw .failed(reason: "cannot write the input frames: \(error.localizedDescription)")
        }
        return directory
    }

    /// Runs the session over `input` and waits until it finishes. Cancelling the task cancels the session.
    private func runSession(input: URL) async throws(PoseEstimationFailure) -> SessionOutput {
        var configuration = PhotogrammetrySession.Configuration()
        configuration.sampleOrdering = .unordered
        configuration.featureSensitivity = .normal

        let session: PhotogrammetrySession
        do {
            session = try PhotogrammetrySession(input: input, configuration: configuration)
        } catch {
            throw .failed(reason: "cannot start PhotogrammetrySession: \(error.localizedDescription)")
        }
        let handle = SessionHandle(session: session)

        do {
            return try await withTaskCancellationHandler {
                try await Self.collectOutputs(of: handle.session)
            } onCancel: {
                handle.session.cancel()
            }
        } catch let failure as PoseEstimationFailure {
            throw failure
        } catch {
            throw .failed(reason: String(describing: error))
        }
    }

    /// Requests poses and the point cloud, then consumes the outputs until processing ends.
    /// The outputs stream does not end by itself after `.processingComplete`, so the loop breaks there.
    private static func collectOutputs(of session: PhotogrammetrySession) async throws(PoseEstimationFailure) -> SessionOutput {
        do {
            try session.process(requests: [.poses, .pointCloud])
        } catch {
            throw .failed(reason: "cannot process the frames: \(error.localizedDescription)")
        }

        var output = SessionOutput()
        do {
            outputLoop: for try await item in session.outputs {
                switch item {
                case .requestComplete(_, let result):
                    switch result {
                    case .poses(let poses):
                        output.poses = poses.posesBySample
                        output.urls = poses.urlsBySample
                    case .pointCloud(let cloud):
                        output.points = cloud.points
                    default:
                        break
                    }
                case .requestError(_, let error):
                    session.cancel()
                    throw PoseEstimationFailure.failed(reason: "Apple Photogrammetry could not reconstruct the scene (\(String(describing: error))). Nearly identical frames, too little overlap between them or too few distinct views are the usual causes")
                case .processingComplete, .processingCancelled:
                    break outputLoop
                default:
                    break
                }
            }
        } catch let failure as PoseEstimationFailure {
            throw failure
        } catch {
            throw .failed(reason: String(describing: error))
        }
        return output
    }

    /// Pixel size of the first image in `frame.data`, read from its header only.
    private static func pixelSize(of frame: CapturedFrame) throws(PoseEstimationFailure) -> (width: Int, height: Int) {
        guard let source = CGImageSourceCreateWithData(frame.data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
              let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
              width > 0, height > 0 else {
            throw .failed(reason: "cannot read the pixel size of \(frame.name)")
        }
        return (width: width, height: height)
    }
}
#endif
