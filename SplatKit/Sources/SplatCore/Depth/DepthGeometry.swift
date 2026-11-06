import Foundation

/// Pixel and world geometry shared by dense initialization and the depth-consistency checks.
///
/// Pixel coordinates are continuous: the centre of pixel `(x, y)` is `(x + 0.5, y + 0.5)`, and
/// pixel index `(floor(u), floor(v))` contains the point `(u, v)`.
public enum DepthGeometry {
    /// World point of pixel centre (x + 0.5, y + 0.5) at camera depth z: R^T·(z·K⁻¹·[u, v, 1] − t).
    public static func backProject(x: Int, y: Int, depth z: Float, camera: Camera) -> SIMD3<Float> {
        let u: Float = Float(x) + 0.5
        let v: Float = Float(y) + 0.5
        let normalizedX: Float = (u - camera.cx) / camera.fx
        let normalizedY: Float = (v - camera.cy) / camera.fy
        let cameraPoint = SIMD3<Float>(z * normalizedX, z * normalizedY, z)
        let shifted: SIMD3<Float> = cameraPoint - camera.translation
        return camera.rotation.transposed * shifted
    }

    /// Pixel coordinates and camera z of a world point; nil if z ≤ near.
    public static func project(_ p: SIMD3<Float>, camera: Camera, near: Float) -> (u: Float, v: Float, z: Float)? {
        let rotated: SIMD3<Float> = camera.rotation * p
        let cameraX: Float = rotated.x + camera.translation.x
        let cameraY: Float = rotated.y + camera.translation.y
        let cameraZ: Float = rotated.z + camera.translation.z
        guard cameraZ > near else { return nil }
        let u: Float = camera.fx * (cameraX / cameraZ) + camera.cx
        let v: Float = camera.fy * (cameraY / cameraZ) + camera.cy
        return (u, v, cameraZ)
    }

    /// True if the 3×3 neighbourhood of valid depths has max/min > ratio (or fewer than 2 valid
    /// neighbours, the centre pixel included). Valid means finite and > 0.
    public static func isDepthEdge(_ depth: [Float], width: Int, height: Int, x: Int, y: Int, ratio: Float) -> Bool {
        var minimum: Float = .infinity
        var maximum: Float = 0
        var validCount = 0
        for offsetY in -1...1 {
            let row = y + offsetY
            guard row >= 0, row < height else { continue }
            for offsetX in -1...1 {
                let column = x + offsetX
                guard column >= 0, column < width else { continue }
                let value: Float = depth[row * width + column]
                guard value.isFinite, value > 0 else { continue }
                validCount += 1
                minimum = min(minimum, value)
                maximum = max(maximum, value)
            }
        }
        guard validCount >= 2 else { return true }
        let spread: Float = maximum / minimum
        return spread > ratio
    }
}
