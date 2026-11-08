import Foundation

/// Pure maths for the Depth Anything 3 pose provider: function selection, frame sub-sampling, pose
/// conversion, intrinsics and back-projection. Kept in SplatIO so it runs without Core ML.
///
/// Conventions: the model's pose is camera-to-world `[R | T]` with `R` from a quaternion `(x, y, z, w)`.
/// Everything returned here is world-to-camera in the OpenCV/COLMAP convention (x right, y down,
/// z forward): `R_w2c = Rᵀ`, `t_w2c = −Rᵀ·T`.
public enum DepthAnything3Math {
    /// View counts the multi-function Core ML package provides, ascending.
    public static let supportedViewCounts: [Int] = [4, 6, 8, 12, 16]

    /// Model height and width for portrait views (height > width).
    public static let portraitHeight: Int = 504
    public static let portraitWidth: Int = 378

    /// The function of the multi-function package that fits `n` frames.
    ///
    /// Returns the smallest supported view count `N >= n`, the function name `v<N>_<H>x<W>` and the
    /// image size. Returns `nil` for `n < 2` (one frame is not supported) and for `n > 16`, where the
    /// caller must sub-sample with `evenlySpacedIndices(count:keep:)` first.
    public static func function(forFrameCount n: Int, portrait: Bool) -> (name: String, viewCount: Int, height: Int, width: Int)? {
        guard n >= 2, let viewCount = supportedViewCounts.first(where: { (candidate: Int) -> Bool in candidate >= n }) else {
            return nil
        }
        let height: Int = portrait ? portraitHeight : portraitWidth
        let width: Int = portrait ? portraitWidth : portraitHeight
        return (name: "v\(viewCount)_\(height)x\(width)", viewCount: viewCount, height: height, width: width)
    }

    /// Splits `count` views into contiguous groups for multi-view depth (v1.5): one group when
    /// `count <= maximum`, else `ceil(count / maximum)` groups whose sizes differ by at most one, so no
    /// group is left with a single view. Every index appears exactly once, in order. Empty for `count <= 0`.
    ///
    /// Groups share no anchor views: each view's depth is aligned to the SfM points on its own later, so
    /// the scale shared within one call is not needed across calls.
    public static func depthGroups(count: Int, maximum: Int = 16) -> [[Int]] {
        guard count > 0 else {
            return []
        }
        let limit: Int = max(1, maximum)
        let groupCount: Int = (count + limit - 1) / limit
        let base: Int = count / groupCount
        let remainder: Int = count % groupCount
        var groups: [[Int]] = []
        groups.reserveCapacity(groupCount)
        var start: Int = 0
        for group in 0..<groupCount {
            let size: Int = base + (group < remainder ? 1 : 0)
            groups.append(Array(start..<(start + size)))
            start += size
        }
        return groups
    }

    /// `keep` indices spread evenly over `0..<count`, strictly increasing, including the first and the
    /// last index when `keep >= 2`. Returns all indices when `keep >= count`, and `[]` for empty input.
    public static func evenlySpacedIndices(count: Int, keep: Int) -> [Int] {
        guard count > 0, keep > 0 else {
            return []
        }
        if keep >= count {
            return Array(0..<count)
        }
        if keep == 1 {
            return [0]
        }
        // keep < count, so the step is above 1 and the rounded indices never collide.
        let step: Double = Double(count - 1) / Double(keep - 1)
        var indices: [Int] = []
        indices.reserveCapacity(keep)
        for position in 0..<keep {
            let scaled: Double = (Double(position) * step).rounded()
            indices.append(Int(scaled))
        }
        return indices
    }

    /// Splits a 9-value model pose `[tx, ty, tz, qx, qy, qz, qw, fovH, fovW]` into the world-to-camera
    /// rotation (row-major 3×3), the world-to-camera translation and the field of view.
    ///
    /// The quaternion is normalised first. A zero quaternion gives the identity rotation.
    /// Precondition: `pose.count == 9`.
    public static func worldToCamera(pose: [Double]) -> (rotation: [Double], translation: SIMD3<Double>, fovH: Double, fovW: Double) {
        precondition(pose.count == 9, "a Depth Anything 3 pose has 9 values")
        let cameraTranslation: SIMD3<Double> = SIMD3<Double>(pose[0], pose[1], pose[2])
        let quaternionX: Double = pose[3]
        let quaternionY: Double = pose[4]
        let quaternionZ: Double = pose[5]
        let quaternionW: Double = pose[6]

        let norm: Double = (quaternionX * quaternionX + quaternionY * quaternionY
                            + quaternionZ * quaternionZ + quaternionW * quaternionW).squareRoot()
        var cameraToWorld: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        if norm > 0 && norm.isFinite {
            let x: Double = quaternionX / norm
            let y: Double = quaternionY / norm
            let z: Double = quaternionZ / norm
            let w: Double = quaternionW / norm
            let xx: Double = x * x
            let yy: Double = y * y
            let zz: Double = z * z
            let xy: Double = x * y
            let xz: Double = x * z
            let yz: Double = y * z
            let wx: Double = w * x
            let wy: Double = w * y
            let wz: Double = w * z
            cameraToWorld = [
                1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
                2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
                2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
            ]
        }

        // R_w2c = R_c2wᵀ, row-major.
        let rotation: [Double] = [
            cameraToWorld[0], cameraToWorld[3], cameraToWorld[6],
            cameraToWorld[1], cameraToWorld[4], cameraToWorld[7],
            cameraToWorld[2], cameraToWorld[5], cameraToWorld[8],
        ]
        // t_w2c = −R_w2c · T, so that x_cam = R_w2c · x_world + t_w2c.
        let rotatedX: Double = cameraToWorld[0] * cameraTranslation.x + cameraToWorld[3] * cameraTranslation.y + cameraToWorld[6] * cameraTranslation.z
        let rotatedY: Double = cameraToWorld[1] * cameraTranslation.x + cameraToWorld[4] * cameraTranslation.y + cameraToWorld[7] * cameraTranslation.z
        let rotatedZ: Double = cameraToWorld[2] * cameraTranslation.x + cameraToWorld[5] * cameraTranslation.y + cameraToWorld[8] * cameraTranslation.z
        let translation: SIMD3<Double> = SIMD3<Double>(-rotatedX, -rotatedY, -rotatedZ)
        return (rotation: rotation, translation: translation, fovH: pose[7], fovW: pose[8])
    }

    /// Pinhole intrinsics for an image of `width` × `height` pixels with the given fields of view.
    ///
    /// `fy = (H / 2) / tan(fovH / 2)` and `fx = (W / 2) / tan(fovW / 2)`, with the principal point at the
    /// image centre. The tangents are clamped to at least `1e-6`.
    public static func intrinsics(fovH: Double, fovW: Double, width: Int, height: Int) -> (fx: Double, fy: Double, cx: Double, cy: Double) {
        let tanHalfH: Double = max(tan(fovH / 2), 1e-6)
        let tanHalfW: Double = max(tan(fovW / 2), 1e-6)
        let halfWidth: Double = Double(width) / 2
        let halfHeight: Double = Double(height) / 2
        let fx: Double = halfWidth / tanHalfW
        let fy: Double = halfHeight / tanHalfH
        return (fx: fx, fy: fy, cx: halfWidth, cy: halfHeight)
    }

    /// Unit quaternion `(w, x, y, z)` of a row-major 3×3 rotation matrix, with `w >= 0`.
    ///
    /// Uses Shepperd's method, choosing the largest of the four components for numerical stability.
    /// Precondition: `r.count == 9`.
    public static func quaternionWXYZ(fromRowMajor r: [Double]) -> SIMD4<Double> {
        precondition(r.count == 9, "a rotation matrix has 9 values")
        let trace: Double = r[0] + r[4] + r[8]
        var w: Double
        var x: Double
        var y: Double
        var z: Double
        if trace > 0 {
            let s: Double = (trace + 1).squareRoot() * 2
            w = 0.25 * s
            x = (r[7] - r[5]) / s
            y = (r[2] - r[6]) / s
            z = (r[3] - r[1]) / s
        } else if r[0] > r[4] && r[0] > r[8] {
            let s: Double = (1 + r[0] - r[4] - r[8]).squareRoot() * 2
            w = (r[7] - r[5]) / s
            x = 0.25 * s
            y = (r[1] + r[3]) / s
            z = (r[2] + r[6]) / s
        } else if r[4] > r[8] {
            let s: Double = (1 + r[4] - r[0] - r[8]).squareRoot() * 2
            w = (r[2] - r[6]) / s
            x = (r[1] + r[3]) / s
            y = 0.25 * s
            z = (r[5] + r[7]) / s
        } else {
            let s: Double = (1 + r[8] - r[0] - r[4]).squareRoot() * 2
            w = (r[3] - r[1]) / s
            x = (r[2] + r[6]) / s
            y = (r[5] + r[7]) / s
            z = 0.25 * s
        }
        if w < 0 {
            w = -w
            x = -x
            y = -y
            z = -z
        }
        let norm: Double = (w * w + x * x + y * y + z * z).squareRoot()
        guard norm > 0, norm.isFinite else {
            return SIMD4<Double>(1, 0, 0, 0)
        }
        return SIMD4<Double>(w / norm, x / norm, y / norm, z / norm)
    }

    /// Back-projects a depth map into world space.
    ///
    /// Pixels are sampled every `stride` rows and columns, starting at the top-left. A pixel is skipped
    /// when its depth is not finite or ≤ 0, or when its confidence is not finite or below `minConfidence`.
    /// The world point is `R_w2cᵀ · (d · K⁻¹ · [u, v, 1] − t_w2c)`.
    ///
    /// - Parameters:
    ///   - depth: Row-major depth, `width × height` values.
    ///   - confidence: Row-major confidence with the same count as `depth`.
    ///   - rotationW2C: Row-major 3×3 world-to-camera rotation.
    ///   - translationW2C: World-to-camera translation.
    /// - Returns: The world points and the row-major pixel indices they came from. Empty when the input
    ///   counts do not match.
    public static func unproject(depth: [Float],
                                 confidence: [Float],
                                 width: Int,
                                 height: Int,
                                 fx: Double,
                                 fy: Double,
                                 cx: Double,
                                 cy: Double,
                                 rotationW2C: [Double],
                                 translationW2C: SIMD3<Double>,
                                 stride: Int,
                                 minConfidence: Float) -> (points: [SIMD3<Double>], pixelIndices: [Int]) {
        guard width > 0, height > 0, depth.count == width * height, confidence.count == depth.count,
              rotationW2C.count == 9 else {
            return (points: [], pixelIndices: [])
        }
        let step: Int = max(1, stride)
        var points: [SIMD3<Double>] = []
        var pixelIndices: [Int] = []
        for row in Swift.stride(from: 0, to: height, by: step) {
            for column in Swift.stride(from: 0, to: width, by: step) {
                let index: Int = row * width + column
                let depthValue: Float = depth[index]
                guard depthValue.isFinite, depthValue > 0 else {
                    continue
                }
                let confidenceValue: Float = confidence[index]
                guard confidenceValue.isFinite, confidenceValue >= minConfidence else {
                    continue
                }
                let d: Double = Double(depthValue)
                let cameraX: Double = (Double(column) - cx) / fx * d
                let cameraY: Double = (Double(row) - cy) / fy * d
                let cameraZ: Double = d
                // q = x_cam − t_w2c, then x_world = R_w2cᵀ · q.
                let qx: Double = cameraX - translationW2C.x
                let qy: Double = cameraY - translationW2C.y
                let qz: Double = cameraZ - translationW2C.z
                let worldX: Double = rotationW2C[0] * qx + rotationW2C[3] * qy + rotationW2C[6] * qz
                let worldY: Double = rotationW2C[1] * qx + rotationW2C[4] * qy + rotationW2C[7] * qz
                let worldZ: Double = rotationW2C[2] * qx + rotationW2C[5] * qy + rotationW2C[8] * qz
                points.append(SIMD3<Double>(worldX, worldY, worldZ))
                pixelIndices.append(index)
            }
        }
        return (points: points, pixelIndices: pixelIndices)
    }
}
