import Foundation

/// Which world axis points up on screen.
public enum UpAxis: String, CaseIterable, Sendable {
    /// COLMAP/OpenCV convention (default).
    case negativeY
    case positiveY
    case positiveZ

    /// Unit vector of the up direction in world space.
    public var vector: SIMD3<Float> {
        switch self {
        case .negativeY: SIMD3(0, -1, 0)
        case .positiveY: SIMD3(0, 1, 0)
        case .positiveZ: SIMD3(0, 0, 1)
        }
    }
}

/// Camera that orbits a target point at a given distance.
///
/// The eye is `target − forward · distance`, where `forward` is set by `yaw`
/// (rotation around the up axis) and `pitch` (elevation toward the up axis;
/// positive pitch looks down from above). Produces an OpenCV-convention `Camera`.
public struct OrbitCamera: Sendable, Equatable {
    /// Vertical field of view of 50 degrees, in radians.
    public static let defaultFovY: Float = 50 * .pi / 180
    /// Pitch limit, keeping the view direction off the up axis.
    public static let maxPitch: Float = .pi / 2 - 0.01

    public var target: SIMD3<Float>
    public var distance: Float
    public var yaw: Float
    public var pitch: Float
    public var fovY: Float
    public var upAxis: UpAxis
    public var minDistance: Float
    public var maxDistance: Float

    public init(target: SIMD3<Float>,
                distance: Float,
                yaw: Float,
                pitch: Float,
                fovY: Float = OrbitCamera.defaultFovY,
                upAxis: UpAxis = .negativeY,
                minDistance: Float,
                maxDistance: Float) {
        self.target = target
        self.distance = distance
        self.yaw = yaw
        self.pitch = Self.clampPitch(pitch)
        self.fovY = fovY
        self.upAxis = upAxis
        self.minDistance = minDistance
        self.maxDistance = maxDistance
    }

    /// Rotates the eye around the target. Pitch is clamped to `±maxPitch`.
    public mutating func orbit(deltaYaw: Float, deltaPitch: Float) {
        yaw += deltaYaw
        pitch = Self.clampPitch(pitch + deltaPitch)
    }

    /// Multiplies the distance by `factor`, clamped to `minDistance...maxDistance`.
    public mutating func zoom(factor: Float) {
        distance = Swift.min(Swift.max(distance * factor, minDistance), maxDistance)
    }

    /// Moves the target in the image plane. Dragging right by `deltaPixels.x` moves
    /// the target along −x_cam and dragging down by `deltaPixels.y` moves it along
    /// −y_cam, so the scene follows the cursor. The distance is unchanged.
    public mutating func pan(deltaPixels: SIMD2<Float>, viewportHeight: Float) {
        let basis = self.basis()
        let worldPerPixel = 2 * distance * tan(fovY / 2) / viewportHeight
        target += basis.right * (-deltaPixels.x * worldPerPixel)
        target += basis.down * (-deltaPixels.y * worldPerPixel)
    }

    /// World-to-camera transform for an image of `width` × `height` pixels.
    public func camera(width: Int, height: Int) -> Camera {
        let basis = self.basis()
        let fy = (Float(height) / 2) / tan(fovY / 2)
        let rotation = Matrix3(rows: [basis.right, basis.down, basis.forward])
        return Camera(
            width: width,
            height: height,
            fx: fy,
            fy: fy,
            cx: Float(width) / 2,
            cy: Float(height) / 2,
            rotation: rotation,
            translation: -(rotation * basis.eye)
        )
    }

    /// Camera that frames the whole bounding sphere of `bounds` at the default
    /// vertical field of view, looking at the center from a 0.3 rad pitch.
    public static func framing(_ bounds: Bounds, upAxis: UpAxis) -> OrbitCamera {
        let radius = 0.5 * (bounds.extent * bounds.extent).sum().squareRoot()
        let safeRadius: Float = radius > 0 ? radius : 1
        let distance = safeRadius / sin(defaultFovY / 2) * 1.1
        return OrbitCamera(
            target: bounds.center,
            distance: distance,
            yaw: 0,
            pitch: 0.3,
            fovY: defaultFovY,
            upAxis: upAxis,
            minDistance: distance * 0.01,
            maxDistance: distance * 100
        )
    }

    // MARK: - Private

    private static func clampPitch(_ pitch: Float) -> Float {
        Swift.min(Swift.max(pitch, -maxPitch), maxPitch)
    }

    /// Camera axes in world space (x right, y down, z forward) and the eye position.
    private func basis() -> (right: SIMD3<Float>, down: SIMD3<Float>, forward: SIMD3<Float>, eye: SIMD3<Float>) {
        let up = upAxis.vector
        let h1 = SIMD3<Float>(1, 0, 0)
        let h2 = normalized(cross(up, h1))
        let horizontal = cos(yaw) * h1 + sin(yaw) * h2
        let forward = cos(pitch) * horizontal - sin(pitch) * up
        let eye = target - forward * distance
        let right = normalized(cross(forward, up))
        let down = cross(forward, right)
        return (right, down, forward, eye)
    }

    private func cross(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
    }

    private func normalized(_ v: SIMD3<Float>) -> SIMD3<Float> {
        v / (v * v).sum().squareRoot()
    }
}
