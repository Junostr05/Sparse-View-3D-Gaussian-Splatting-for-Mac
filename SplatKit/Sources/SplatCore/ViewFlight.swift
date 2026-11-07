import Foundation

/// A smooth camera path through a list of keyframe views.
///
/// Positions run from `0` (first keyframe) to `count − 1` (last keyframe). The eye follows a
/// centripetal Catmull–Rom spline through the keyframe eyes, with the end points duplicated.
/// Orientation is spherically interpolated between the two neighbouring keyframes. Field of
/// view and distance are linear within a segment. Keyframes are passed through exactly at
/// integer positions.
public struct ViewFlight: Sendable {
    public let keyframes: [ViewCamera]

    /// `keyframes` must not be empty. An empty flight is a precondition failure.
    public init(keyframes: [ViewCamera]) {
        precondition(!keyframes.isEmpty, "ViewFlight requires at least one keyframe")
        self.keyframes = keyframes
    }

    /// Number of keyframes.
    public var count: Int { keyframes.count }

    /// View at a fractional keyframe position. The position is clamped to `0...(count − 1)`.
    public func camera(at position: Float) -> ViewCamera {
        let last = keyframes.count - 1
        if last == 0 {
            return keyframes[0]
        }
        let clamped = Swift.min(Swift.max(position, 0), Float(last))
        let whole = clamped.rounded(.down)
        if clamped == whole {
            return keyframes[Int(whole)]
        }

        let segment = Swift.min(Int(whole), last - 1)
        let u = Double(clamped) - Double(segment)
        let a = keyframes[segment]
        let b = keyframes[segment + 1]

        let p0 = ViewMath.toDouble(keyframes[Swift.max(segment - 1, 0)].eye)
        let p1 = ViewMath.toDouble(a.eye)
        let p2 = ViewMath.toDouble(b.eye)
        let p3 = ViewMath.toDouble(keyframes[Swift.min(segment + 2, last)].eye)
        let eye = ViewFlight.centripetalCatmullRom(p0, p1, p2, p3, u: u)

        let axes = ViewMath.slerp(a.axes, b.axes, t: u)
        let distance = Double(a.distance) + (Double(b.distance) - Double(a.distance)) * u
        let fovY = Double(a.fovY) + (Double(b.fovY) - Double(a.fovY)) * u
        let forward = axes.forward
        let distanceF = Float(distance)
        return ViewCamera(
            pivot: ViewMath.toFloat(eye) + forward * distanceF,
            distance: distanceF,
            right: axes.right,
            down: axes.down,
            forward: forward,
            fovY: Float(fovY),
            upAxis: a.upAxis,
            minDistance: a.minDistance,
            maxDistance: a.maxDistance
        )
    }

    /// Eased transition from `a` to `b`. `progress` is clamped to `0...1` and smoothstep-eased
    /// (`3s² − 2s³`). Used for flying to a camera, a preset or the reset view.
    public static func transition(from a: ViewCamera, to b: ViewCamera, progress: Float) -> ViewCamera {
        let x = Swift.min(Swift.max(progress, 0), 1)
        let eased = x * x * (3 - 2 * x)
        return ViewCamera.interpolated(a, b, t: eased)
    }

    /// Playback length in seconds for a flight with `count − 1` segments.
    public func duration(secondsPerSegment: Double) -> Double {
        Double(count - 1) * secondsPerSegment
    }

    // MARK: - Private

    /// Point on the centripetal (α = ½) Catmull–Rom segment from `p1` to `p2`, evaluated with
    /// the Barry–Goldman recursion. Knot gaps are floored at 1e-6, so repeated keyframes are safe.
    static func centripetalCatmullRom(_ p0: SIMD3<Double>, _ p1: SIMD3<Double>,
                                      _ p2: SIMD3<Double>, _ p3: SIMD3<Double>,
                                      u: Double) -> SIMD3<Double> {
        // Centripetal knot gap: |P_{k+1} − P_k|^{1/2}, floored to keep the recursion finite.
        func knotGap(_ d: SIMD3<Double>) -> Double {
            let length: Double = (d * d).sum().squareRoot()
            return Swift.max(length.squareRoot(), 1e-6)
        }
        let t0: Double = 0
        let t1: Double = t0 + knotGap(p1 - p0)
        let t2: Double = t1 + knotGap(p2 - p1)
        let t3: Double = t2 + knotGap(p3 - p2)
        let t: Double = t1 + u * (t2 - t1)

        let w01: Double = (t1 - t) / (t1 - t0)
        let w11: Double = (t - t0) / (t1 - t0)
        let a1: SIMD3<Double> = w01 * p0 + w11 * p1

        let w12: Double = (t2 - t) / (t2 - t1)
        let w22: Double = (t - t1) / (t2 - t1)
        let a2: SIMD3<Double> = w12 * p1 + w22 * p2

        let w23: Double = (t3 - t) / (t3 - t2)
        let w33: Double = (t - t2) / (t3 - t2)
        let a3: SIMD3<Double> = w23 * p2 + w33 * p3

        let wb10: Double = (t2 - t) / (t2 - t0)
        let wb11: Double = (t - t0) / (t2 - t0)
        let b1: SIMD3<Double> = wb10 * a1 + wb11 * a2

        let wb20: Double = (t3 - t) / (t3 - t1)
        let wb21: Double = (t - t1) / (t3 - t1)
        let b2: SIMD3<Double> = wb20 * a2 + wb21 * a3

        let wc0: Double = (t2 - t) / (t2 - t1)
        let wc1: Double = (t - t1) / (t2 - t1)
        let c: SIMD3<Double> = wc0 * b1 + wc1 * b2
        return c
    }
}
