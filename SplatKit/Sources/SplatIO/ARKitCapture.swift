import Foundation
import SplatCore

/// JSON capture written by the iOS companion for ARKit sessions.
///
/// Each frame names its image (relative to the image folder), the image size in pixels, the
/// pinhole intrinsics `[fx, fy, cx, cy]` for that size, and the ARKit camera-to-world transform
/// as 16 values in column-major order (the `simd_float4x4` columns). ARKit camera axes are
/// x right, y up, z backward.
public struct ARKitCapture: Codable, Sendable, Equatable {
    public struct Frame: Codable, Sendable, Equatable {
        public var image: String
        public var width: Int
        public var height: Int
        /// `fx, fy, cx, cy` in pixels of `width × height`.
        public var intrinsics: [Double]
        /// 16 values, column-major, ARKit camera convention.
        public var cameraToWorld: [Double]
        /// Phase 4 extension point. Ignored in Phase 3.
        public var depth: String?

        public init(image: String,
                    width: Int,
                    height: Int,
                    intrinsics: [Double],
                    cameraToWorld: [Double],
                    depth: String? = nil) {
            self.image = image
            self.width = width
            self.height = height
            self.intrinsics = intrinsics
            self.cameraToWorld = cameraToWorld
            self.depth = depth
        }
    }

    public var version: Int
    public var frames: [Frame]

    public init(version: Int, frames: [Frame]) {
        self.version = version
        self.frames = frames
    }

    /// OpenCV-convention camera for one frame. Its image size is the frame's `width × height`.
    public static func camera(for frame: Frame) throws(DatasetError) -> Camera {
        let resolved = try resolve(frame)
        return Camera(model: resolved.model,
                      width: frame.width,
                      height: frame.height,
                      rotation: resolved.rotation,
                      translation: resolved.translation)
    }

    /// Pinhole model, world→camera rotation and translation of one frame.
    static func resolve(_ frame: Frame) throws(DatasetError) -> (model: CameraModel, rotation: Matrix3, translation: SIMD3<Float>) {
        guard frame.width > 0, frame.height > 0 else {
            throw .invalidARKitCapture("frame \(frame.image) has a non-positive size")
        }
        guard frame.intrinsics.count == 4, frame.intrinsics.allSatisfy(\.isFinite) else {
            throw .invalidARKitCapture("frame \(frame.image) must have four finite intrinsics")
        }
        let fx = frame.intrinsics[0]
        let fy = frame.intrinsics[1]
        guard fx > 0, fy > 0 else {
            throw .invalidARKitCapture("frame \(frame.image) has non-positive focal lengths")
        }
        guard frame.cameraToWorld.count == 16 else {
            throw .invalidARKitCapture("frame \(frame.image) must have 16 pose values")
        }
        let m = frame.cameraToWorld
        guard m.allSatisfy(\.isFinite), isRigidARKitPose(m) else {
            throw .invalidPose(name: frame.image)
        }

        // Columns of the camera-to-world rotation (ARKit: right, up, backward).
        let right = SIMD3<Double>(m[0], m[1], m[2])
        let up = SIMD3<Double>(m[4], m[5], m[6])
        let backward = SIMD3<Double>(m[8], m[9], m[10])
        let position = SIMD3<Double>(m[12], m[13], m[14])

        // OpenCV camera axes in world space: x right, y down, z forward.
        let xAxis = right
        let yAxis = -up
        let zAxis = -backward
        let rotation = Matrix3(rows: [float3(xAxis), float3(yAxis), float3(zAxis)])
        let translation = float3(SIMD3<Double>(-dot(xAxis, position),
                                               -dot(yAxis, position),
                                               -dot(zAxis, position)))

        let model = CameraModel.pinhole(fx: fx,
                                        fy: fy,
                                        cx: frame.intrinsics[2],
                                        cy: frame.intrinsics[3])
        return (model, rotation, translation)
    }

    /// Column-major `cameraToWorld` is a rigid transform: orthonormal rotation columns and the
    /// affine row `(0, 0, 0, 1)`.
    private static func isRigidARKitPose(_ m: [Double]) -> Bool {
        let columns = [SIMD3<Double>(m[0], m[1], m[2]),
                       SIMD3<Double>(m[4], m[5], m[6]),
                       SIMD3<Double>(m[8], m[9], m[10])]
        for i in 0..<3 {
            for j in 0..<3 {
                let expected: Double = i == j ? 1 : 0
                if abs(dot(columns[i], columns[j]) - expected) > 1e-3 {
                    return false
                }
            }
        }
        let lastRowIsAffine = abs(m[3]) <= 1e-6 && abs(m[7]) <= 1e-6 && abs(m[11]) <= 1e-6
            && abs(m[15] - 1) <= 1e-6
        return lastRowIsAffine
    }

    private static func dot(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    private static func float3(_ v: SIMD3<Double>) -> SIMD3<Float> {
        SIMD3(Float(v.x), Float(v.y), Float(v.z))
    }
}
