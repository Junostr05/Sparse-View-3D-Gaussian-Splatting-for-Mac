/// Pinhole camera in OpenCV convention (x right, y down, z forward).
/// `rotation` and `translation` map world to camera space: `p_cam = R · p_world + t`.
public struct Camera: Sendable, Equatable {
    public var width: Int
    public var height: Int
    public var fx: Float
    public var fy: Float
    public var cx: Float
    public var cy: Float
    public var rotation: Matrix3
    public var translation: SIMD3<Float>

    public init(width: Int, height: Int, fx: Float, fy: Float, cx: Float, cy: Float,
                rotation: Matrix3, translation: SIMD3<Float>) {
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.rotation = rotation
        self.translation = translation
    }

    /// Camera center in world space: `C = -Rᵀ t`.
    public var center: SIMD3<Float> {
        -(rotation.transposed * translation)
    }
}
