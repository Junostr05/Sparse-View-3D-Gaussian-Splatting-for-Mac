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
        // Every band term is `scalar * coefficient`; the scalars are computed as plain
        // Floats first so the type checker never sees long mixed SIMD/Float expressions.
        let b1: [Float] = [-c1 * y, c1 * z, -c1 * x]
        for i in 0..<3 { result += b1[i] * sh(1 + i) }

        guard degree >= 2 else { return result }
        let xx: Float = x * x, yy: Float = y * y, zz: Float = z * z
        let xy: Float = x * y, yz: Float = y * z, xz: Float = x * z
        let b2: [Float] = [
            c2[0] * xy,
            c2[1] * yz,
            c2[2] * (2 * zz - xx - yy),
            c2[3] * xz,
            c2[4] * (xx - yy),
        ]
        for i in 0..<5 { result += b2[i] * sh(4 + i) }

        guard degree >= 3 else { return result }
        let p9: Float = 3 * xx - yy
        let p11: Float = 4 * zz - xx - yy
        let p12: Float = 2 * zz - 3 * xx - 3 * yy
        let p14: Float = xx - yy
        let p15: Float = xx - 3 * yy
        let b3: [Float] = [
            c3[0] * y * p9,
            c3[1] * xy * z,
            c3[2] * y * p11,
            c3[3] * z * p12,
            c3[4] * x * p11,
            c3[5] * z * p14,
            c3[6] * x * p15,
        ]
        for i in 0..<7 { result += b3[i] * sh(9 + i) }
        return result
    }
}
