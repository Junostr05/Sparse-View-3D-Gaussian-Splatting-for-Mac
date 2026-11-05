/// Column-major 3×3 matrix of `Float`.
public struct Matrix3: Sendable, Equatable {
    public var c0: SIMD3<Float>
    public var c1: SIMD3<Float>
    public var c2: SIMD3<Float>

    public static let identity = Matrix3(
        c0: SIMD3(1, 0, 0),
        c1: SIMD3(0, 1, 0),
        c2: SIMD3(0, 0, 1)
    )

    public init(c0: SIMD3<Float>, c1: SIMD3<Float>, c2: SIMD3<Float>) {
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
    }

    /// Builds the matrix from its three rows.
    public init(rows: [SIMD3<Float>]) {
        precondition(rows.count == 3, "Matrix3 requires exactly three rows")
        self.init(
            c0: SIMD3(rows[0].x, rows[1].x, rows[2].x),
            c1: SIMD3(rows[0].y, rows[1].y, rows[2].y),
            c2: SIMD3(rows[0].z, rows[1].z, rows[2].z)
        )
    }

    public var transposed: Matrix3 {
        Matrix3(
            c0: SIMD3(c0.x, c1.x, c2.x),
            c1: SIMD3(c0.y, c1.y, c2.y),
            c2: SIMD3(c0.z, c1.z, c2.z)
        )
    }

    /// Rotation matrix for quaternion `q = (w, x, y, z)`. The quaternion is normalized;
    /// a zero quaternion yields the identity.
    public static func rotation(quaternion q: SIMD4<Float>) -> Matrix3 {
        let lengthSquared = (q * q).sum()
        guard lengthSquared > 0 else { return .identity }
        let n = q / lengthSquared.squareRoot()
        let w = n.x, x = n.y, y = n.z, z = n.w
        return Matrix3(
            c0: SIMD3(1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)),
            c1: SIMD3(2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)),
            c2: SIMD3(2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
        )
    }

    public static func * (lhs: Matrix3, rhs: Matrix3) -> Matrix3 {
        Matrix3(c0: lhs * rhs.c0, c1: lhs * rhs.c1, c2: lhs * rhs.c2)
    }

    public static func * (lhs: Matrix3, rhs: SIMD3<Float>) -> SIMD3<Float> {
        lhs.c0 * rhs.x + lhs.c1 * rhs.y + lhs.c2 * rhs.z
    }
}
