import Foundation

/// COLMAP camera models (double precision, pixel-corner convention: pixel centers at `x + 0.5`).
/// Distortion is applied to normalized image coordinates `(u, v) = ((x − cx)/fx, (y − cy)/fy)`.
public enum CameraModel: Sendable, Equatable {
    case simplePinhole(f: Double, cx: Double, cy: Double)
    case pinhole(fx: Double, fy: Double, cx: Double, cy: Double)
    case simpleRadial(f: Double, cx: Double, cy: Double, k: Double)
    case radial(f: Double, cx: Double, cy: Double, k1: Double, k2: Double)
    case opencv(fx: Double, fy: Double, cx: Double, cy: Double,
                k1: Double, k2: Double, p1: Double, p2: Double)

    /// Focal lengths `(fx, fy)` in pixels.
    public var focal: SIMD2<Double> {
        switch self {
        case let .simplePinhole(f, _, _), let .simpleRadial(f, _, _, _), let .radial(f, _, _, _, _):
            return SIMD2(f, f)
        case let .pinhole(fx, fy, _, _), let .opencv(fx, fy, _, _, _, _, _, _):
            return SIMD2(fx, fy)
        }
    }

    /// Principal point `(cx, cy)` in pixels.
    public var principalPoint: SIMD2<Double> {
        switch self {
        case let .simplePinhole(_, cx, cy), let .pinhole(_, _, cx, cy),
             let .simpleRadial(_, cx, cy, _), let .radial(_, cx, cy, _, _),
             let .opencv(_, _, cx, cy, _, _, _, _):
            return SIMD2(cx, cy)
        }
    }

    /// `false` for the pinhole kinds and whenever every distortion coefficient is zero.
    public var hasDistortion: Bool {
        switch self {
        case .simplePinhole, .pinhole:
            return false
        case let .simpleRadial(_, _, _, k):
            return k != 0
        case let .radial(_, _, _, k1, k2):
            return k1 != 0 || k2 != 0
        case let .opencv(_, _, _, _, k1, k2, p1, p2):
            return k1 != 0 || k2 != 0 || p1 != 0 || p2 != 0
        }
    }

    /// Maps an undistorted normalized point to its distorted normalized position.
    public func distort(_ normalized: SIMD2<Double>) -> SIMD2<Double> {
        let u = normalized.x
        let v = normalized.y
        let r2 = u * u + v * v
        switch self {
        case .simplePinhole, .pinhole:
            return normalized
        case let .simpleRadial(_, _, _, k):
            let s = 1 + k * r2
            return SIMD2(u * s, v * s)
        case let .radial(_, _, _, k1, k2):
            let s = 1 + k1 * r2 + k2 * r2 * r2
            return SIMD2(u * s, v * s)
        case let .opencv(_, _, _, _, k1, k2, p1, p2):
            let radial = k1 * r2 + k2 * r2 * r2
            let du = u * radial + 2 * p1 * u * v + p2 * (r2 + 2 * u * u)
            let dv = v * radial + 2 * p2 * u * v + p1 * (r2 + 2 * v * v)
            return SIMD2(u + du, v + dv)
        }
    }
}

extension Camera {
    /// Pinhole part of `model` (focal lengths and principal point); distortion is not stored.
    public init(model: CameraModel, width: Int, height: Int,
                rotation: Matrix3, translation: SIMD3<Float>) {
        let focal = model.focal
        let center = model.principalPoint
        self.init(width: width,
                  height: height,
                  fx: Float(focal.x),
                  fy: Float(focal.y),
                  cx: Float(center.x),
                  cy: Float(center.y),
                  rotation: rotation,
                  translation: translation)
    }

    /// Camera for an image downscaled by an integer factor (floor size). Intrinsics are divided
    /// by the same factor, so continuous pixel coordinates scale by `1 / factor`.
    public func downscaled(by factor: Int) -> Camera {
        precondition(factor >= 1, "downscale factor must be at least 1")
        let s = Float(factor)
        return Camera(width: width / factor,
                      height: height / factor,
                      fx: fx / s,
                      fy: fy / s,
                      cx: cx / s,
                      cy: cy / s,
                      rotation: rotation,
                      translation: translation)
    }

    /// Camera for an image resized to `toWidth × height` (same field of view).
    public func rescaled(toWidth newWidth: Int, height newHeight: Int) -> Camera {
        let sx = Float(newWidth) / Float(width)
        let sy = Float(newHeight) / Float(height)
        return Camera(width: newWidth,
                      height: newHeight,
                      fx: fx * sx,
                      fy: fy * sy,
                      cx: cx * sx,
                      cy: cy * sy,
                      rotation: rotation,
                      translation: translation)
    }
}
