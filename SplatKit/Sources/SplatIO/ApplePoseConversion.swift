/// Converts poses from Apple's photogrammetry (RealityKit, OpenGL camera convention, camera-to-world)
/// into COLMAP's convention (world-to-camera, OpenCV camera axes). Pure math, no RealityKit.
///
/// Apple's camera looks along -Z with +Y up. COLMAP's camera looks along +Z with +Y down. Measured on
/// real data: `R_c2w_cv = R_apple_c2w * diag(1, -1, -1)` matches COLMAP's rotations within 0.27 degrees
/// on average. Do not change this without re-measuring.
public enum ApplePoseConversion {
    /// Converts one Apple pose to COLMAP world-to-camera.
    ///
    /// - Parameters:
    ///   - centre: Camera centre in the world frame (`PhotogrammetrySession.Pose.translation`).
    ///   - quaternionXYZW: Camera-to-world rotation as `(x, y, z, w)` (`simd_quatf.vector`). It is
    ///     normalized first.
    /// - Returns: `rotation` is the unit quaternion `(w, x, y, z)` of the world-to-camera rotation with
    ///   `w >= 0`. `translation` is `-R * centre`.
    public static func colmapPose(centre: SIMD3<Double>,
                                  quaternionXYZW: SIMD4<Double>) -> (rotation: SIMD4<Double>, translation: SIMD3<Double>) {
        let unit: SIMD4<Double> = normalizedQuaternion(quaternionXYZW)
        let appleCameraToWorld: ConversionMatrix3 = matrix(fromXYZW: unit)
        // Flip the camera's Y and Z axes: the second and third columns change sign.
        let cvCameraToWorld: ConversionMatrix3 = appleCameraToWorld.negatingColumns(1, 2)
        let worldToCamera: ConversionMatrix3 = cvCameraToWorld.transposed()
        let rotation: SIMD4<Double> = quaternionWXYZ(from: worldToCamera)
        let rotatedCentre: SIMD3<Double> = worldToCamera.times(centre)
        let translation: SIMD3<Double> = SIMD3<Double>(-rotatedCentre.x, -rotatedCentre.y, -rotatedCentre.z)
        return (rotation, translation)
    }

    /// Focal length in pixels from a 35 mm-equivalent focal length, using the diagonal rule:
    /// `f = f35 * sqrt(w^2 + h^2) / 43.267`, where 43.267 mm is the diagonal of a 36 x 24 mm frame.
    public static func focalLengthPixels(focalLength35mm: Double, width: Int, height: Int) -> Double {
        let widthValue: Double = Double(width)
        let heightValue: Double = Double(height)
        let diagonalPixels: Double = (widthValue * widthValue + heightValue * heightValue).squareRoot()
        let focal: Double = focalLength35mm * diagonalPixels / 43.267
        return focal
    }

    // MARK: - Private

    private static func normalizedQuaternion(_ q: SIMD4<Double>) -> SIMD4<Double> {
        let squared: Double = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
        let length: Double = squared.squareRoot()
        return SIMD4<Double>(q.x / length, q.y / length, q.z / length, q.w / length)
    }

    /// Rotation matrix of a unit quaternion given as `(x, y, z, w)`.
    private static func matrix(fromXYZW q: SIMD4<Double>) -> ConversionMatrix3 {
        let x: Double = q.x
        let y: Double = q.y
        let z: Double = q.z
        let w: Double = q.w
        let xx: Double = x * x
        let yy: Double = y * y
        let zz: Double = z * z
        let xy: Double = x * y
        let xz: Double = x * z
        let yz: Double = y * z
        let wx: Double = w * x
        let wy: Double = w * y
        let wz: Double = w * z

        let r00: Double = 1 - 2 * (yy + zz)
        let r01: Double = 2 * (xy - wz)
        let r02: Double = 2 * (xz + wy)
        let r10: Double = 2 * (xy + wz)
        let r11: Double = 1 - 2 * (xx + zz)
        let r12: Double = 2 * (yz - wx)
        let r20: Double = 2 * (xz - wy)
        let r21: Double = 2 * (yz + wx)
        let r22: Double = 1 - 2 * (xx + yy)

        return ConversionMatrix3(row0: SIMD3<Double>(r00, r01, r02),
                                 row1: SIMD3<Double>(r10, r11, r12),
                                 row2: SIMD3<Double>(r20, r21, r22))
    }

    /// Unit quaternion `(w, x, y, z)` of a rotation matrix, with `w >= 0`.
    /// Shepperd's method: the branch is chosen by the largest of the trace and the diagonal entries,
    /// so the square root is never taken of a value near zero.
    private static func quaternionWXYZ(from r: ConversionMatrix3) -> SIMD4<Double> {
        let trace: Double = r[0, 0] + r[1, 1] + r[2, 2]
        var w: Double = 0
        var x: Double = 0
        var y: Double = 0
        var z: Double = 0

        if trace > 0 {
            let s: Double = (trace + 1).squareRoot() * 2
            w = 0.25 * s
            x = (r[2, 1] - r[1, 2]) / s
            y = (r[0, 2] - r[2, 0]) / s
            z = (r[1, 0] - r[0, 1]) / s
        } else if r[0, 0] > r[1, 1] && r[0, 0] > r[2, 2] {
            let s: Double = (1 + r[0, 0] - r[1, 1] - r[2, 2]).squareRoot() * 2
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        } else if r[1, 1] > r[2, 2] {
            let s: Double = (1 + r[1, 1] - r[0, 0] - r[2, 2]).squareRoot() * 2
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        } else {
            let s: Double = (1 + r[2, 2] - r[0, 0] - r[1, 1]).squareRoot() * 2
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
        }

        let norm: Double = (w * w + x * x + y * y + z * z).squareRoot()
        let sign: Double = w < 0 ? -1 : 1
        return SIMD4<Double>(sign * w / norm, sign * x / norm, sign * y / norm, sign * z / norm)
    }
}

/// Row-major 3 x 3 matrix used by the conversion. Private to this file.
private struct ConversionMatrix3 {
    var row0: SIMD3<Double>
    var row1: SIMD3<Double>
    var row2: SIMD3<Double>

    subscript(row: Int, column: Int) -> Double {
        get {
            switch row {
            case 0: return row0[column]
            case 1: return row1[column]
            default: return row2[column]
            }
        }
    }

    func transposed() -> ConversionMatrix3 {
        return ConversionMatrix3(
            row0: SIMD3<Double>(row0.x, row1.x, row2.x),
            row1: SIMD3<Double>(row0.y, row1.y, row2.y),
            row2: SIMD3<Double>(row0.z, row1.z, row2.z)
        )
    }

    /// Multiplies every entry of columns `first` and `second` by -1, i.e. right-multiplies by a
    /// diagonal matrix with -1 at those positions.
    func negatingColumns(_ first: Int, _ second: Int) -> ConversionMatrix3 {
        var result: ConversionMatrix3 = self
        for row in 0..<3 {
            var values: SIMD3<Double> = result.rowValues(row)
            values[first] = -values[first]
            values[second] = -values[second]
            result.setRow(row, values)
        }
        return result
    }

    func times(_ v: SIMD3<Double>) -> SIMD3<Double> {
        let x: Double = row0.x * v.x + row0.y * v.y + row0.z * v.z
        let y: Double = row1.x * v.x + row1.y * v.y + row1.z * v.z
        let z: Double = row2.x * v.x + row2.y * v.y + row2.z * v.z
        return SIMD3<Double>(x, y, z)
    }

    private func rowValues(_ row: Int) -> SIMD3<Double> {
        switch row {
        case 0: return row0
        case 1: return row1
        default: return row2
        }
    }

    private mutating func setRow(_ row: Int, _ values: SIMD3<Double>) {
        switch row {
        case 0: row0 = values
        case 1: row1 = values
        default: row2 = values
        }
    }
}
