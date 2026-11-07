import Foundation

/// The surface under a cursor: its camera-space depth and the world point on the cursor ray at that depth.
public struct CursorPick: Sendable, Equatable {
    /// Camera-space depth along the forward axis.
    public var depth: Float
    /// World-space point on the cursor ray at `depth`.
    public var worldPoint: SIMD3<Float>

    public init(depth: Float, worldPoint: SIMD3<Float>) {
        self.depth = depth
        self.worldPoint = worldPoint
    }
}

/// Picks the depth of the surface under a cursor, for pivoting gestures at what the user points at.
public enum CursorDepth {
    /// Minimum number of candidates for a pick. Fewer candidates give `nil`.
    static let minimumCandidates: Int = 3

    /// Projects the Gaussian centres with `camera` and keeps those in front of it whose projected centre lies
    /// within `radiusPixels` of `pixel` and whose sigmoid opacity is at least `minimumOpacity`. The depth is the
    /// value at about the 25th percentile of the candidate depths, which favours the near surface and ignores
    /// a few floaters in front of it. The world point lies on the cursor ray at that depth.
    ///
    /// - Parameters:
    ///   - pixel: cursor position in the pixel space of `camera`, top-left origin.
    /// - Returns: `nil` when fewer than 3 candidates qualify.
    public static func pick(cloud: GaussianCloud,
                            camera: Camera,
                            pixel: SIMD2<Float>,
                            radiusPixels: Float = 12,
                            minimumOpacity: Float = 0.3) -> CursorPick? {
        guard pixel.x.isFinite, pixel.y.isFinite, radiusPixels.isFinite, radiusPixels > 0 else { return nil }
        let count: Int = cloud.count
        guard count > 0 else { return nil }

        // sigmoid(logit) >= m  <=>  logit >= log(m / (1 - m)). This avoids an exp per Gaussian.
        let logitThreshold: Float
        if minimumOpacity <= 0 {
            logitThreshold = -Float.infinity
        } else if minimumOpacity >= 1 {
            logitThreshold = Float.infinity
        } else {
            logitThreshold = Float(log(Double(minimumOpacity) / Double(1 - minimumOpacity)))
        }
        let radiusSquared: Float = radiusPixels * radiusPixels
        let near: Float = 0.01
        let rotation: Matrix3 = camera.rotation
        let translation: SIMD3<Float> = camera.translation
        let fx: Float = camera.fx
        let fy: Float = camera.fy
        let cx: Float = camera.cx
        let cy: Float = camera.cy

        var depths: [Float] = []
        cloud.positions.withUnsafeBufferPointer { positions in
            cloud.opacityLogits.withUnsafeBufferPointer { logits in
                for index in 0..<count {
                    // Cheap opacity test first. A NaN logit fails the comparison and is skipped.
                    guard logits[index] >= logitThreshold else { continue }
                    let base: Int = 3 * index
                    let p0: Float = positions[base]
                    let p1: Float = positions[base + 1]
                    let p2: Float = positions[base + 2]
                    let x: Float = rotation.c0.x * p0 + rotation.c1.x * p1 + rotation.c2.x * p2 + translation.x
                    let y: Float = rotation.c0.y * p0 + rotation.c1.y * p1 + rotation.c2.y * p2 + translation.y
                    let z: Float = rotation.c0.z * p0 + rotation.c1.z * p1 + rotation.c2.z * p2 + translation.z
                    guard z > near else { continue }
                    let u: Float = fx * x / z + cx
                    let v: Float = fy * y / z + cy
                    let du: Float = u - pixel.x
                    let dv: Float = v - pixel.y
                    guard du * du + dv * dv <= radiusSquared else { continue }
                    depths.append(z)
                }
            }
        }

        guard depths.count >= minimumCandidates else { return nil }
        depths.sort()
        let lastIndex: Int = depths.count - 1
        let quarter: Float = Float(lastIndex) * 0.25
        let index: Int = Int(quarter)
        let depth: Float = depths[index]

        // The point on the cursor ray at `depth`: camera-space ray direction scaled to the depth, then to world.
        let direction: SIMD3<Float> = SIMD3<Float>((pixel.x - cx) / fx, (pixel.y - cy) / fy, 1)
        let cameraPoint: SIMD3<Float> = direction * depth
        let worldOffset: SIMD3<Float> = rotation.transposed * cameraPoint
        let worldPoint: SIMD3<Float> = camera.center + worldOffset
        return CursorPick(depth: depth, worldPoint: worldPoint)
    }
}
