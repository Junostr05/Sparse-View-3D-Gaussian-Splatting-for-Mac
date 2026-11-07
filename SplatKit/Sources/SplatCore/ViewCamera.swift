import Foundation

/// Numpad-style viewing directions (Blender convention).
public enum ViewPreset: CaseIterable, Sendable, Equatable {
    case front, back, right, left, top, bottom
}

/// Turntable camera that can hold roll.
///
/// The camera is a pivot point, a distance along the view direction and an orthonormal frame
/// `right`, `down`, `forward` in world space, with OpenCV axes (x right, y down, z forward).
/// The eye is `pivot − forward · distance`. Orientation is stored as axes rather than yaw and
/// pitch, so a rolled camera (as estimated for handheld phone captures) keeps its roll while
/// it is orbited. `OrbitCamera` stays the reference for on-screen direction of the gestures.
///
/// The axes are expected to be orthonormal and right-handed (`right × down = forward`).
/// The mutating operations restore that, the memberwise initializers do not check it.
public struct ViewCamera: Sendable, Equatable {
    /// Upper bound on `|forward · up|`: the view direction stays at least 0.01 rad from the up axis.
    public static let maxUpAlignment: Float = Float(cos(0.01 as Double))

    public var pivot: SIMD3<Float>
    public var distance: Float
    public var right: SIMD3<Float>
    public var down: SIMD3<Float>
    public var forward: SIMD3<Float>
    /// Vertical field of view in radians.
    public var fovY: Float
    public var upAxis: UpAxis
    public var minDistance: Float
    public var maxDistance: Float

    public init(pivot: SIMD3<Float>,
                distance: Float,
                right: SIMD3<Float>,
                down: SIMD3<Float>,
                forward: SIMD3<Float>,
                fovY: Float = OrbitCamera.defaultFovY,
                upAxis: UpAxis = .negativeY,
                minDistance: Float,
                maxDistance: Float) {
        self.pivot = pivot
        self.distance = distance
        self.right = right
        self.down = down
        self.forward = forward
        self.fovY = fovY
        self.upAxis = upAxis
        self.minDistance = minDistance
        self.maxDistance = maxDistance
    }

    /// Camera position in world space.
    public var eye: SIMD3<Float> {
        pivot - forward * distance
    }

    /// Same distance, limits, field of view and pitch as `OrbitCamera.framing`, with yaw 0,
    /// so the result renders identically to that camera for the same bounds.
    public static func framing(_ bounds: Bounds, upAxis: UpAxis) -> ViewCamera {
        let reference = OrbitCamera.framing(bounds, upAxis: upAxis)
        let axes = ViewMath.orbitAxes(yaw: 0, pitch: 0.3, upAxis: upAxis)
        return ViewCamera(
            pivot: bounds.center,
            distance: reference.distance,
            right: axes.right,
            down: axes.down,
            forward: axes.forward,
            fovY: reference.fovY,
            upAxis: upAxis,
            minDistance: reference.minDistance,
            maxDistance: reference.maxDistance
        )
    }

    /// Builds the view from a world-to-camera `Camera`. The pivot is placed `pivotDistance`
    /// in front of the eye. The field of view is taken from the focal length `fy`.
    public init(camera: Camera, pivotDistance: Float, upAxis: UpAxis, minDistance: Float, maxDistance: Float) {
        let rows = camera.rotation.transposed
        let eye = camera.center
        let forward = rows.c2
        self.init(
            pivot: eye + forward * pivotDistance,
            distance: pivotDistance,
            right: rows.c0,
            down: rows.c1,
            forward: forward,
            fovY: 2 * atan(Float(camera.height) / (2 * camera.fy)),
            upAxis: upAxis,
            minDistance: minDistance,
            maxDistance: maxDistance
        )
    }

    /// World-to-camera transform for an image of `width` × `height` pixels.
    public func camera(width: Int, height: Int) -> Camera {
        let fy: Float = (Float(height) / 2) / tan(fovY / 2)
        let rotation = Matrix3(rows: [right, down, forward])
        return Camera(
            width: width,
            height: height,
            fx: fy,
            fy: fy,
            cx: Float(width) / 2,
            cy: Float(height) / 2,
            rotation: rotation,
            translation: -(rotation * eye)
        )
    }

    /// Turntable rotation. A positive `deltaYaw` turns the view the way `OrbitCamera.orbit` does.
    /// Yaw turns all axes about the world up vector. Pitch turns them about the camera's own
    /// right axis (a positive pitch looks down). The pitch is reduced if it would bring the view
    /// direction within 0.01 rad of the up axis. Roll is kept.
    public mutating func orbit(deltaYaw: Float, deltaPitch: Float) {
        let up = upAxis.vector
        right = ViewMath.rotate(right, around: up, by: deltaYaw)
        down = ViewMath.rotate(down, around: up, by: deltaYaw)
        forward = ViewMath.rotate(forward, around: up, by: deltaYaw)

        // Pitch about the current right axis. Rotating by -deltaPitch matches OrbitCamera.
        let axis = right
        // A view that already lies beyond the limit (after a top or bottom preset) may leave it
        // but not move further out. The small tolerance absorbs rounding at exactly 1.
        let limit = Swift.max(ViewCamera.maxUpAlignment, Swift.abs(ViewMath.dot(forward, up))) + 1e-6
        let angle = ViewMath.allowedPitch(forward: forward, axis: axis, up: up,
                                          angle: -deltaPitch, limit: limit)
        forward = ViewMath.rotate(forward, around: axis, by: angle)
        down = ViewMath.rotate(down, around: axis, by: angle)
        orthonormalize()
    }

    /// Non-mutating form of `orbit(deltaYaw:deltaPitch:)`.
    public func orbited(deltaYaw: Float, deltaPitch: Float) -> ViewCamera {
        var copy = self
        copy.orbit(deltaYaw: deltaYaw, deltaPitch: deltaPitch)
        return copy
    }

    /// Multiplies the distance by `factor`, clamped to `minDistance...maxDistance`.
    public mutating func zoom(factor: Float) {
        distance = Swift.min(Swift.max(distance * factor, minDistance), maxDistance)
    }

    /// Moves the pivot in the image plane with the same sign convention as `OrbitCamera.pan`.
    public mutating func pan(deltaPixels: SIMD2<Float>, viewportHeight: Float) {
        let worldPerPixel: Float = 2 * distance * tan(fovY / 2) / viewportHeight
        pivot += right * (-deltaPixels.x * worldPerPixel)
        pivot += down * (-deltaPixels.y * worldPerPixel)
    }

    /// Makes `right` horizontal again (keeps `forward`). If `forward` is parallel to the up axis,
    /// `right` has no horizontal direction and is left unchanged.
    public mutating func removeRoll() {
        let horizontal = ViewMath.cross(forward, upAxis.vector)
        if ViewMath.length(horizontal) > 1e-6 {
            right = ViewMath.normalized(horizontal)
        }
        orthonormalize()
    }

    /// Keeps the orientation and moves the pivot to the center of `bounds`. The distance is set
    /// to the framing distance. The limits are not changed.
    public mutating func frame(_ bounds: Bounds) {
        pivot = bounds.center
        distance = OrbitCamera.framing(bounds, upAxis: upAxis).distance
    }

    /// Camera looking along a numpad direction. Pivot, distance, field of view, up axis and
    /// limits come from `current`. Blender's numpad views are the reference for the directions.
    public static func aligned(_ preset: ViewPreset, like current: ViewCamera) -> ViewCamera {
        let up = current.upAxis.vector
        let h1 = SIMD3<Float>(1, 0, 0)
        let h2 = ViewMath.normalized(ViewMath.cross(up, h1))
        let forward: SIMD3<Float>
        switch preset {
        case .front: forward = h2
        case .back: forward = -h2
        case .right: forward = -h1
        case .left: forward = h1
        case .top: forward = -up
        case .bottom: forward = up
        }
        let right: SIMD3<Float>
        if preset == .top || preset == .bottom {
            // h1 is perpendicular to the up axis, so it is already perpendicular to forward.
            right = ViewMath.normalized(h1 - forward * ViewMath.dot(forward, h1))
        } else {
            right = ViewMath.normalized(ViewMath.cross(forward, up))
        }
        let down = ViewMath.cross(forward, right)
        return ViewCamera(
            pivot: current.pivot,
            distance: current.distance,
            right: right,
            down: down,
            forward: forward,
            fovY: current.fovY,
            upAxis: current.upAxis,
            minDistance: current.minDistance,
            maxDistance: current.maxDistance
        )
    }

    /// Interpolates between two views. The eye is interpolated linearly and the orientation by
    /// spherical interpolation (shortest path). The field of view and distance are linear. The
    /// up axis and limits are taken from `a`. `t = 0` gives the orientation of `a` and `t = 1`
    /// the orientation of `b`.
    public static func interpolated(_ a: ViewCamera, _ b: ViewCamera, t: Float) -> ViewCamera {
        let s = Double(t)
        let eyeA = ViewMath.toDouble(a.eye)
        let eyeB = ViewMath.toDouble(b.eye)
        let eye = eyeA + (eyeB - eyeA) * s
        let distance = Double(a.distance) + (Double(b.distance) - Double(a.distance)) * s
        let fovY = Double(a.fovY) + (Double(b.fovY) - Double(a.fovY)) * s
        let axes = ViewMath.slerp(a.axes, b.axes, t: s)
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

    /// The camera axes as one value.
    var axes: ViewAxes {
        ViewAxes(right: right, down: down, forward: forward)
    }

    // MARK: - Private

    /// Gram–Schmidt on forward, then right, then down = forward × right.
    private mutating func orthonormalize() {
        let f = ViewMath.normalized(forward)
        var r = right - f * ViewMath.dot(f, right)
        if ViewMath.length(r) < 1e-12 {
            // right was parallel to forward: take the horizontal direction instead.
            r = ViewMath.cross(f, upAxis.vector)
            if ViewMath.length(r) < 1e-12 {
                r = SIMD3<Float>(1, 0, 0)
            }
        }
        r = ViewMath.normalized(r)
        forward = f
        right = r
        down = ViewMath.cross(f, r)
    }
}

/// Camera axes as rows of the world-to-camera rotation: `right`, `down`, `forward`.
struct ViewAxes: Equatable {
    var right: SIMD3<Float>
    var down: SIMD3<Float>
    var forward: SIMD3<Float>
}

/// Vector and rotation helpers shared by `ViewCamera`, `ViewFlight` and `ViewCommand`.
/// Kept internal: the public surface is the types themselves.
enum ViewMath {
    static func dot(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    static func cross(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
    }

    static func length(_ v: SIMD3<Float>) -> Float {
        dot(v, v).squareRoot()
    }

    static func normalized(_ v: SIMD3<Float>) -> SIMD3<Float> {
        v / length(v)
    }

    static func toDouble(_ v: SIMD3<Float>) -> SIMD3<Double> {
        SIMD3<Double>(Double(v.x), Double(v.y), Double(v.z))
    }

    static func toFloat(_ v: SIMD3<Double>) -> SIMD3<Float> {
        SIMD3<Float>(Float(v.x), Float(v.y), Float(v.z))
    }

    /// Camera axes for a turntable with the given yaw and pitch, using the same formulas as
    /// `OrbitCamera` (yaw 0 looks along h1 = x, positive pitch looks down).
    static func orbitAxes(yaw: Float, pitch: Float, upAxis: UpAxis) -> ViewAxes {
        let up = upAxis.vector
        let h1 = SIMD3<Float>(1, 0, 0)
        let h2 = normalized(cross(up, h1))
        let horizontal: SIMD3<Float> = cos(yaw) * h1 + sin(yaw) * h2
        let forward: SIMD3<Float> = cos(pitch) * horizontal - sin(pitch) * up
        let right = normalized(cross(forward, up))
        let down = cross(forward, right)
        return ViewAxes(right: right, down: down, forward: forward)
    }

    /// Rodrigues rotation of `v` about `axis` (normalized here) by `angle` radians.
    static func rotate(_ v: SIMD3<Float>, around axis: SIMD3<Float>, by angle: Float) -> SIMD3<Float> {
        let k = normalized(axis)
        let c: Float = cos(angle)
        let s: Float = sin(angle)
        let kv = cross(k, v)
        let kd = dot(k, v)
        return v * c + kv * s + k * (kd * (1 - c))
    }

    /// The part of a pitch rotation of `angle` about `axis` that keeps `|forward · up| <= limit`
    /// along the whole path, measured from the start. Rotations wrap around the sphere, so a
    /// large angle can pass over a pole and return inside the band. The path is therefore
    /// scanned for the first exit (one full turn at most), and the exit is bisected.
    /// The start must satisfy the limit; then the result is never beyond it.
    static func allowedPitch(forward: SIMD3<Float>, axis: SIMD3<Float>, up: SIMD3<Float>,
                             angle: Float, limit: Float) -> Float {
        func alignment(_ a: Float) -> Float {
            Swift.abs(dot(rotate(forward, around: axis, by: a), up))
        }
        if angle == 0 {
            return 0
        }
        let sign: Float = angle < 0 ? -1 : 1
        let span = Swift.min(Swift.abs(angle), 2 * Float.pi)
        // About one sample per 0.003 rad, at least 16 and at most 2048 (a full turn).
        let steps: Int = Swift.min(2048, Swift.max(16, Int(span / 0.003) + 1))
        var inside: Float = 0
        for i in 1...steps {
            let outside = span * Float(i) / Float(steps)
            if alignment(sign * outside) > limit {
                var lo = inside
                var hi = outside
                for _ in 0..<30 {
                    let mid = 0.5 * (lo + hi)
                    if alignment(sign * mid) <= limit {
                        lo = mid
                    } else {
                        hi = mid
                    }
                }
                return sign * lo
            }
            inside = outside
        }
        return angle
    }

    // MARK: Orientation interpolation (Double precision)

    /// Unit quaternion (w, x, y, z).
    struct Quaternion {
        var w: Double
        var x: Double
        var y: Double
        var z: Double
    }

    /// Rotation matrix `m[row][column]` with rows `right`, `down`, `forward`.
    static func rows(_ axes: ViewAxes) -> [[Double]] {
        let r = toDouble(axes.right)
        let d = toDouble(axes.down)
        let f = toDouble(axes.forward)
        return [[r.x, r.y, r.z], [d.x, d.y, d.z], [f.x, f.y, f.z]]
    }

    /// Shepperd's method for the quaternion of a proper rotation matrix.
    static func quaternion(_ m: [[Double]]) -> Quaternion {
        let trace = m[0][0] + m[1][1] + m[2][2]
        if trace > 0 {
            let s = (trace + 1).squareRoot() * 2
            return Quaternion(w: 0.25 * s,
                              x: (m[2][1] - m[1][2]) / s,
                              y: (m[0][2] - m[2][0]) / s,
                              z: (m[1][0] - m[0][1]) / s)
        } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
            let s = (1 + m[0][0] - m[1][1] - m[2][2]).squareRoot() * 2
            return Quaternion(w: (m[2][1] - m[1][2]) / s,
                              x: 0.25 * s,
                              y: (m[0][1] + m[1][0]) / s,
                              z: (m[0][2] + m[2][0]) / s)
        } else if m[1][1] > m[2][2] {
            let s = (1 + m[1][1] - m[0][0] - m[2][2]).squareRoot() * 2
            return Quaternion(w: (m[0][2] - m[2][0]) / s,
                              x: (m[0][1] + m[1][0]) / s,
                              y: 0.25 * s,
                              z: (m[1][2] + m[2][1]) / s)
        } else {
            let s = (1 + m[2][2] - m[0][0] - m[1][1]).squareRoot() * 2
            return Quaternion(w: (m[1][0] - m[0][1]) / s,
                              x: (m[0][2] + m[2][0]) / s,
                              y: (m[1][2] + m[2][1]) / s,
                              z: 0.25 * s)
        }
    }

    /// Rows of the rotation matrix of a unit quaternion.
    static func axes(_ q: Quaternion) -> ViewAxes {
        let w = q.w, x = q.x, y = q.y, z = q.z
        let row0 = SIMD3<Double>(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y))
        let row1 = SIMD3<Double>(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x))
        let row2 = SIMD3<Double>(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))
        return ViewAxes(right: toFloat(row0), down: toFloat(row1), forward: toFloat(row2))
    }

    /// Spherical interpolation of two camera orientations along the shortest path.
    /// `t <= 0` returns `a` and `t >= 1` returns `b` exactly.
    static func slerp(_ a: ViewAxes, _ b: ViewAxes, t: Double) -> ViewAxes {
        if t <= 0 { return a }
        if t >= 1 { return b }
        let qa = quaternion(rows(a))
        var qb = quaternion(rows(b))
        var cosTheta = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z
        if cosTheta < 0 {
            qb = Quaternion(w: -qb.w, x: -qb.x, y: -qb.y, z: -qb.z)
            cosTheta = -cosTheta
        }
        let wa: Double
        let wb: Double
        if cosTheta > 0.9995 {
            wa = 1 - t
            wb = t
        } else {
            let theta = acos(Swift.min(cosTheta, 1))
            let sinTheta = sin(theta)
            wa = sin((1 - t) * theta) / sinTheta
            wb = sin(t * theta) / sinTheta
        }
        let w = wa * qa.w + wb * qb.w
        let x = wa * qa.x + wb * qb.x
        let y = wa * qa.y + wb * qb.y
        let z = wa * qa.z + wb * qb.z
        let norm = (w * w + x * x + y * y + z * z).squareRoot()
        return axes(Quaternion(w: w / norm, x: x / norm, y: y / norm, z: z / norm))
    }
}
