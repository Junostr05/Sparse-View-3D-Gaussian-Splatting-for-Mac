import Foundation
import SplatCore

/// A camera model with its parameters, in the layout of a COLMAP `cameras.bin` record without the
/// id (ids are assigned when the cameras are written). `modelID` is the COLMAP model id; the
/// parameter order follows `COLMAPReader.parameterCount(modelID:)`.
public struct EstimatedCamera: Sendable, Equatable {
    /// COLMAP camera model id (0 = SIMPLE_PINHOLE, 1 = PINHOLE, ...).
    public var modelID: Int32
    public var width: Int
    public var height: Int
    /// Model parameters in COLMAP order.
    public var params: [Double]

    public init(modelID: Int32, width: Int, height: Int, params: [Double]) {
        self.modelID = modelID
        self.width = width
        self.height = height
        self.params = params
    }
}

/// The world-to-camera pose of one registered image.
public struct EstimatedImagePose: Sendable, Equatable {
    /// Image file name. Must match a `CapturedFrame.name`.
    public var name: String
    /// Index into `PoseEstimationResult.cameras`.
    public var cameraIndex: Int
    /// Unit quaternion `(w, x, y, z)` of the world→camera rotation, COLMAP convention.
    public var rotation: SIMD4<Double>
    /// World→camera translation.
    public var translation: SIMD3<Double>

    public init(name: String, cameraIndex: Int, rotation: SIMD4<Double>, translation: SIMD3<Double>) {
        self.name = name
        self.cameraIndex = cameraIndex
        self.rotation = rotation
        self.translation = translation
    }
}

/// One sparse 3D point from a reconstruction.
public struct EstimatedPoint: Sendable, Equatable {
    public var position: SIMD3<Double>
    public var color: SIMD3<UInt8>
    /// Reprojection error in pixels.
    public var error: Double

    public init(position: SIMD3<Double>, color: SIMD3<UInt8>, error: Double) {
        self.position = position
        self.color = color
        self.error = error
    }
}

/// The output of a successful pose estimation: cameras, the registered images and sparse points.
public struct PoseEstimationResult: Sendable, Equatable {
    public var cameras: [EstimatedCamera]
    public var images: [EstimatedImagePose]
    public var points: [EstimatedPoint]

    public init(cameras: [EstimatedCamera], images: [EstimatedImagePose], points: [EstimatedPoint]) {
        self.cameras = cameras
        self.images = images
        self.points = points
    }

    /// Names of the registered images, sorted.
    public var registeredImageNames: [String] {
        images.map(\.name).sorted()
    }

    /// Checks that the result is usable for `totalFrames` input frames and returns it unchanged.
    ///
    /// Throws `.tooFewFrames` for fewer than `PoseEstimationPolicy.minimumFrames` input frames,
    /// `.registrationFailed` when fewer than `PoseEstimationPolicy.minimumRegistered` images are
    /// registered, and `.failed` for duplicate image names or dangling camera references.
    public func validated(totalFrames: Int) throws(PoseEstimationFailure) -> PoseEstimationResult {
        if totalFrames < PoseEstimationPolicy.minimumFrames {
            throw .tooFewFrames(have: totalFrames, need: PoseEstimationPolicy.minimumFrames)
        }
        let uniqueNames = Set(images.map(\.name))
        if uniqueNames.count != images.count {
            throw .failed(reason: "more than one pose for the same image name")
        }
        let registered = uniqueNames.count
        let needed = PoseEstimationPolicy.minimumRegistered(forTotal: totalFrames)
        if registered < needed {
            throw .registrationFailed(registered: registered, total: totalFrames)
        }
        for image in images where !cameras.indices.contains(image.cameraIndex) {
            throw .failed(reason: "image \(image.name) refers to a camera that does not exist")
        }
        for camera in cameras {
            guard let count = COLMAPReader.parameterCount(modelID: camera.modelID),
                  count == camera.params.count else {
                throw .failed(reason: "camera model \(camera.modelID) has the wrong number of parameters")
            }
        }
        return self
    }
}

/// Thresholds shared by every pose provider.
public enum PoseEstimationPolicy {
    /// Fewest input frames worth running pose estimation on.
    public static let minimumFrames = 2

    /// Fewest registered images that count as a successful estimation for `total` input frames:
    /// at least half of them (rounded up), and never fewer than two.
    public static func minimumRegistered(forTotal total: Int) -> Int {
        max(2, (total + 1) / 2)
    }
}

/// Why a pose provider did not produce a usable result.
public enum PoseEstimationFailure: Error, Sendable, Equatable {
    /// The provider cannot run here, for example its tool is not installed or the sandbox denies it.
    case unavailable(reason: String)
    /// Fewer input frames than the provider needs.
    case tooFewFrames(have: Int, need: Int)
    /// Too few images were registered. See `PoseEstimationPolicy.minimumRegistered(forTotal:)`.
    case registrationFailed(registered: Int, total: Int)
    /// The caller cancelled the estimation. The chain does not try further providers.
    case cancelled
    /// Any other failure, with a human-readable reason.
    case failed(reason: String)
}

/// A source of camera poses for a set of frames, such as COLMAP run on the frames or a
/// device-provided pose estimator.
public protocol PoseProvider: Sendable {
    /// Short name shown in progress and error messages.
    var name: String { get }

    /// Estimates poses for `frames`. Frames are unique by `name`.
    func estimatePoses(for frames: [CapturedFrame]) async throws(PoseEstimationFailure) -> PoseEstimationResult
}

/// One provider attempt inside a `PoseProviderChain` run.
public struct PoseAttempt: Sendable, Equatable {
    public var providerName: String
    public var outcome: PoseAttemptOutcome

    public init(providerName: String, outcome: PoseAttemptOutcome) {
        self.providerName = providerName
        self.outcome = outcome
    }
}

/// Result of one `PoseAttempt`.
public enum PoseAttemptOutcome: Sendable, Equatable {
    case success
    case failure(PoseEstimationFailure)
}

/// Final result of a `PoseProviderChain` run.
public enum PoseChainOutcome: Sendable, Equatable {
    /// The first provider whose result validated, its name, and every attempt made (the last one
    /// is the success).
    case success(PoseEstimationResult, String, [PoseAttempt])
    /// No provider produced a usable result, or the run was cancelled. Lists every attempt made.
    case failure([PoseAttempt])
}

/// Tries pose providers in order and returns the first validated result.
///
/// A failing provider falls through to the next one. `.cancelled` stops the chain at once.
public struct PoseProviderChain: Sendable {
    public var providers: [any PoseProvider]

    public init(providers: [any PoseProvider]) {
        self.providers = providers
    }

    /// Runs the providers in order. `progress` is called once per attempt, as it finishes.
    public func estimate(for frames: [CapturedFrame],
                         progress: (@Sendable (PoseAttempt) -> Void)? = nil) async -> PoseChainOutcome {
        var attempts: [PoseAttempt] = []
        for provider in providers {
            if Task.isCancelled {
                return .failure(attempts)
            }
            do throws(PoseEstimationFailure) {
                let result = try await provider.estimatePoses(for: frames)
                let validated = try result.validated(totalFrames: frames.count)
                let attempt = PoseAttempt(providerName: provider.name, outcome: .success)
                attempts.append(attempt)
                progress?(attempt)
                return .success(validated, provider.name, attempts)
            } catch {
                let attempt = PoseAttempt(providerName: provider.name, outcome: .failure(error))
                attempts.append(attempt)
                progress?(attempt)
                if error == .cancelled {
                    return .failure(attempts)
                }
            }
        }
        return .failure(attempts)
    }
}
