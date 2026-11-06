import Foundation

/// Rendered depth of one view (Phase 4 P6): `raw = Σ αᵢTᵢzᵢ` and `alpha = 1 − T_final`,
/// row-major W·H. Expected depth at a pixel is `raw / alpha`.
public struct DepthRender: Sendable, Equatable {
    public var raw: [Float]
    public var alpha: [Float]

    public init(raw: [Float], alpha: [Float]) {
        self.raw = raw
        self.alpha = alpha
    }

    /// `raw / alpha` where `alpha > minimumAlpha`, else 0.
    public func normalized(minimumAlpha: Float) -> [Float] {
        precondition(raw.count == alpha.count, "raw and alpha must have the same count")
        var out = [Float](repeating: 0, count: raw.count)
        for i in 0..<raw.count where alpha[i] > minimumAlpha {
            out[i] = raw[i] / alpha[i]
        }
        return out
    }
}
