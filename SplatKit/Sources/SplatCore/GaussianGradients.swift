/// Gradients of a scalar loss with respect to every parameter of a `GaussianCloud`.
/// Same layouts as `GaussianCloud`; `rotations` holds the gradient with respect to the
/// RAW stored quaternion (before normalization).
public struct GaussianGradients: Sendable, Equatable {
    /// 3n
    public var positions: [Float]
    /// 3n
    public var logScales: [Float]
    /// 4n, (w, x, y, z)
    public var rotations: [Float]
    /// n
    public var opacityLogits: [Float]
    /// n·K·3 with K = (cloud.shDegree + 1)², layout [i][k][c].
    public var sh: [Float]

    /// All-zero gradients sized for `cloud`.
    public init(zeroLike cloud: GaussianCloud) {
        let n = cloud.count
        positions = [Float](repeating: 0, count: 3 * n)
        logScales = [Float](repeating: 0, count: 3 * n)
        rotations = [Float](repeating: 0, count: 4 * n)
        opacityLogits = [Float](repeating: 0, count: n)
        sh = [Float](repeating: 0, count: cloud.sh.count)
    }
}
