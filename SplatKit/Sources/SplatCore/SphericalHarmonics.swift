/// Real spherical-harmonics evaluation in the gsplat convention (degrees 0…3).
public enum SphericalHarmonics {
    static let c0: Float = 0.28209479177387814
    static let c1: Float = 0.4886025119029199
    static let c2: [Float] = [
        1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
        -1.0925484305920792, 0.5462742152960396,
    ]
    static let c3: [Float] = [
        -0.5900435899266435, 2.890611442640554, -0.4570457994644658,
        0.3731763325901154, -0.4570457994644658, 1.445305721320277,
        -0.5900435899266435,
    ]

    /// Raw SH sum for one Gaussian and one color channel set, without the `+ 0.5` offset.
    ///
    /// - Parameters:
    ///   - degree: SH degree to evaluate, 0…3. Bands above `degree` are ignored.
    ///   - coefficients: One Gaussian's SH slice, layout `[k][c]` (`k * 3 + c` relative to
    ///     `startIndex`). Must hold at least `(degree + 1)² * 3` values.
    ///   - direction: Unit view direction from the camera center to the Gaussian.
    public static func evaluate(degree: Int, coefficients: ArraySlice<Float>,
                                direction: SIMD3<Float>) -> SIMD3<Float> {
        precondition((0...3).contains(degree), "SH degree must be 0...3")
        let k = (degree + 1) * (degree + 1)
        precondition(coefficients.count >= k * 3, "not enough SH coefficients for degree")
        let base = coefficients.startIndex

        func sh(_ i: Int) -> SIMD3<Float> {
            SIMD3(coefficients[base + 3 * i],
                  coefficients[base + 3 * i + 1],
                  coefficients[base + 3 * i + 2])
        }

        let x = direction.x, y = direction.y, z = direction.z
        var result = c0 * sh(0)

        guard degree >= 1 else { return result }
        result += c1 * (-y * sh(1) + z * sh(2) - x * sh(3))

        guard degree >= 2 else { return result }
        let xx = x * x, yy = y * y, zz = z * z
        let xy = x * y, yz = y * z, xz = x * z
        result += c2[0] * xy * sh(4)
        result += c2[1] * yz * sh(5)
        result += c2[2] * (2 * zz - xx - yy) * sh(6)
        result += c2[3] * xz * sh(7)
        result += c2[4] * (xx - yy) * sh(8)

        guard degree >= 3 else { return result }
        result += c3[0] * y * (3 * xx - yy) * sh(9)
        result += c3[1] * xy * z * sh(10)
        result += c3[2] * y * (4 * zz - xx - yy) * sh(11)
        result += c3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh(12)
        result += c3[4] * x * (4 * zz - xx - yy) * sh(13)
        result += c3[5] * z * (xx - yy) * sh(14)
        result += c3[6] * x * (xx - 3 * yy) * sh(15)
        return result
    }
}
