/// Errors thrown while importing a training dataset (COLMAP or ARKit, with its images).
public enum DatasetError: Error, Equatable {
    /// A dataset file ends before its declared content, or a count or size in it is impossible.
    /// The payload names the file (`cameras.bin`, `images.bin`, `points3D.bin`).
    case truncated(file: String)
    /// A COLMAP camera model id this reader does not know, so its record cannot be skipped.
    case unknownCameraModel(Int32)
    /// A known COLMAP camera model that the trainer does not support. The payload is its name.
    case unsupportedCameraModel(String)
    /// An image refers to a camera id that `cameras.bin` does not define.
    case missingCamera(image: String, cameraID: UInt32)
    /// Images listed in the dataset are absent from the image folder. Lists every missing name.
    case missingImages([String])
    /// An image file could not be decoded. The payload names the image and the reason.
    case imageDecodingFailed(name: String, reason: String)
    /// An image's pixel size does not match its camera and cannot be rescaled to it.
    case imageSizeMismatch(name: String, expected: SIMD2<Int>, actual: SIMD2<Int>)
    /// A pose is not a finite rigid transform (for example a zero quaternion).
    case invalidPose(name: String)
    /// An ARKit capture file is malformed. The payload describes the problem.
    case invalidARKitCapture(String)
    /// The dataset contains no views.
    case noViews
}
