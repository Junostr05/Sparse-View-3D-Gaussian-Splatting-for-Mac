import Foundation

/// What an overlay line is. The role decides its colour, its weight and whether it fades with distance.
public enum OverlayRole: Sendable, Equatable {
    /// A ground grid line that is neither an axis nor a major line.
    case minorGrid
    /// Every tenth ground grid line, away from the origin.
    case majorGrid
    /// The ground lines through the origin along the world axes.
    case axisX
    case axisY
    case axisZ
    /// An edge of the frustum of the camera with this index in the list given to `cameraFrusta`.
    case camera(index: Int)

    /// `true` for the ground grid lines, which fade with distance.
    public var isGrid: Bool {
        switch self {
        case .minorGrid, .majorGrid: true
        case .axisX, .axisY, .axisZ, .camera: false
        }
    }
}

/// A line in world space.
public struct OverlaySegment: Sendable, Equatable {
    public var start: SIMD3<Float>
    public var end: SIMD3<Float>
    public var role: OverlayRole

    public init(start: SIMD3<Float>, end: SIMD3<Float>, role: OverlayRole) {
        self.start = start
        self.end = end
        self.role = role
    }
}

/// A line in view points, origin at the top left, y downward.
public struct ProjectedOverlaySegment: Sendable, Equatable {
    public var start: SIMD2<Float>
    public var end: SIMD2<Float>
    public var role: OverlayRole
    /// Opacity in `0...1`. Below 1 only for grid lines that are far from the eye.
    public var alpha: Float
}

/// One arm of the axes gizmo.
public struct OverlayGizmoAxis: Sendable, Equatable {
    /// 0 for X, 1 for Y, 2 for Z.
    public var axis: Int
    /// Offset from the gizmo anchor in view points, y downward.
    public var offset: SIMD2<Float>
    /// Camera-space depth of the arm's tip. Positive values point away from the viewer.
    public var depth: Float
}

/// Pure geometry for the viewport overlays: the ground grid, the axes gizmo and the camera frusta.
/// Nothing here depends on Metal or SwiftUI. The app draws the results with a `Canvas`.
public enum OverlayGeometry {
    /// Points closer to the camera than this (camera units) are clipped.
    public static let nearPlane: Float = 0.01
    /// Number of grid lines across the scene that the spacing aims for.
    public static let targetGridLines: Float = 15
    /// Frustum size as a fraction of the scene radius.
    public static let frustumFraction: Float = 0.04

    /// A power of ten, chosen so that the scene spans about `targetGridLines` lines. Only powers of ten are
    /// allowed, so the actual line count can be between about 5 and 47 for a given scene size.
    public static func gridSpacing(sceneSize: Float) -> Float {
        guard sceneSize.isFinite, sceneSize > 0 else { return 1 }
        let ratio: Double = Double(sceneSize / targetGridLines)
        let exponent: Double = log10(ratio).rounded()
        let spacing: Double = pow(10.0, exponent)
        return Float(spacing)
    }

    /// Ground grid on the plane perpendicular to the up axis, at the lowest point of `bounds` along the up
    /// axis. With `nil` bounds (an empty scene) the plane goes through `fallbackPivot`. The two lines through
    /// the origin are the axes, coloured X red, Y green and Z blue. Every tenth line is a major line.
    public static func groundGrid(bounds: Bounds?,
                                  upAxis: UpAxis,
                                  fallbackPivot: SIMD3<Float> = .zero) -> [OverlaySegment] {
        let up: SIMD3<Float> = upAxis.vector
        let upIndex: Int = upAxis == .positiveZ ? 2 : 1
        // The two horizontal world axes are X (index 0) and the other one.
        let secondIndex: Int = upIndex == 1 ? 2 : 1

        var height: Float = (up * fallbackPivot).sum()
        var reach: Float = 0
        var extent: Float = 0
        if let bounds {
            var lowest: Float = Float.greatestFiniteMagnitude
            for corner in 0..<8 {
                let x: Float = (corner & 1) == 0 ? bounds.min.x : bounds.max.x
                let y: Float = (corner & 2) == 0 ? bounds.min.y : bounds.max.y
                let z: Float = (corner & 4) == 0 ? bounds.min.z : bounds.max.z
                let p: SIMD3<Float> = SIMD3<Float>(x, y, z)
                lowest = Swift.min(lowest, (up * p).sum())
                reach = Swift.max(reach, Swift.abs(p[0]), Swift.abs(p[secondIndex]))
            }
            height = lowest
            extent = Swift.max(bounds.extent[0], bounds.extent[secondIndex])
        }

        let sceneSize: Float = Swift.max(extent, reach) > 0 ? Swift.max(extent, reach) : 1
        let spacing: Float = gridSpacing(sceneSize: sceneSize)
        let reachLines: Float = (reach / spacing).rounded(.up)
        let halfCount: Int = Swift.min(100, Swift.max(5, Int(reachLines)))
        let limit: Float = Float(halfCount) * spacing

        // A point on the plane, with coordinates on the varying and the fixed horizontal axis.
        func point(varying: Int, value: Float, fixed: Int, position: Float) -> SIMD3<Float> {
            var p: SIMD3<Float> = .zero
            p[upIndex] = height * up[upIndex]
            p[varying] = value
            p[fixed] = position
            return p
        }

        var segments: [OverlaySegment] = []
        segments.reserveCapacity(4 * halfCount + 4)
        for i in -halfCount...halfCount {
            let position: Float = Float(i) * spacing
            let isMajor: Bool = i % 10 == 0
            let firstRole: OverlayRole = i == 0 ? .axisX : (isMajor ? .majorGrid : .minorGrid)
            segments.append(OverlaySegment(
                start: point(varying: 0, value: -limit, fixed: secondIndex, position: position),
                end: point(varying: 0, value: limit, fixed: secondIndex, position: position),
                role: firstRole))
            let secondRole: OverlayRole = i == 0 ? axisRole(secondIndex) : (isMajor ? .majorGrid : .minorGrid)
            segments.append(OverlaySegment(
                start: point(varying: secondIndex, value: -limit, fixed: 0, position: position),
                end: point(varying: secondIndex, value: limit, fixed: 0, position: position),
                role: secondRole))
        }
        return segments
    }

    /// Radius of the scene (half the diagonal of `bounds`), or 1 when it is empty.
    public static func sceneRadius(bounds: Bounds?) -> Float {
        guard let bounds else { return 1 }
        let extent: SIMD3<Float> = bounds.extent
        let halfDiagonal: Float = 0.5 * (extent * extent).sum().squareRoot()
        return halfDiagonal > 0 ? halfDiagonal : 1
    }

    /// Frustum depth used for the camera pyramids: `frustumFraction` of the scene radius.
    public static func frustumSize(bounds: Bounds?) -> Float {
        frustumFraction * sceneRadius(bounds: bounds)
    }

    /// The axes gizmo: one arm per world axis, from a common anchor, in the directions the camera
    /// rotation gives. Only the rotation is used, so the gizmo does not depend on the camera position.
    /// `length` is in view points.
    public static func axesGizmo(rotation: Matrix3, length: Float) -> [OverlayGizmoAxis] {
        // Column i of the world-to-camera rotation is the camera-space image of world axis i.
        let columns: [SIMD3<Float>] = [rotation.c0, rotation.c1, rotation.c2]
        var arms: [OverlayGizmoAxis] = []
        arms.reserveCapacity(3)
        for index in 0..<3 {
            let local: SIMD3<Float> = columns[index]
            let offset: SIMD2<Float> = SIMD2<Float>(length * local.x, length * local.y)
            arms.append(OverlayGizmoAxis(axis: index, offset: offset, depth: local.z))
        }
        return arms
    }

    /// One camera pyramid per camera: rays from the camera centre to the four corners of a rectangle at
    /// `size` in front of it, the rectangle itself and a short tick on its top edge marking "up".
    /// The rectangle follows the camera's field of view and aspect ratio.
    public static func cameraFrusta(cameras: [Camera], size: Float) -> [OverlaySegment] {
        var segments: [OverlaySegment] = []
        segments.reserveCapacity(cameras.count * 10)
        for (index, camera) in cameras.enumerated() {
            let role: OverlayRole = .camera(index: index)
            let apex: SIMD3<Float> = camera.center
            let rotationT: Matrix3 = camera.rotation.transposed
            let halfWidth: Float = size * Float(camera.width) / (2 * camera.fx)
            let halfHeight: Float = size * Float(camera.height) / (2 * camera.fy)

            // Camera-space corners in the order top left, top right, bottom right, bottom left (y points down).
            var corners: [SIMD3<Float>] = []
            corners.reserveCapacity(4)
            let local: [SIMD3<Float>] = [
                SIMD3<Float>(-halfWidth, -halfHeight, size),
                SIMD3<Float>(halfWidth, -halfHeight, size),
                SIMD3<Float>(halfWidth, halfHeight, size),
                SIMD3<Float>(-halfWidth, halfHeight, size),
            ]
            for p in local {
                // World point of a camera-space point: p_world = Rᵀ p_camera + center.
                corners.append(rotationT * p + apex)
            }
            for corner in corners {
                segments.append(OverlaySegment(start: apex, end: corner, role: role))
            }
            for edge in 0..<4 {
                segments.append(OverlaySegment(start: corners[edge], end: corners[(edge + 1) % 4], role: role))
            }
            let tickStart: SIMD3<Float> = SIMD3<Float>(0, -halfHeight, size)
            let tickEnd: SIMD3<Float> = SIMD3<Float>(0, -1.5 * halfHeight, size)
            segments.append(OverlaySegment(start: rotationT * tickStart + apex,
                                           end: rotationT * tickEnd + apex,
                                           role: role))
        }
        return segments
    }

    /// Projects world lines with `camera`, clipping each one against the near plane first. Segments that lie
    /// entirely outside the view (`viewSize` in points) on one side are dropped. Grid lines fade linearly from
    /// full opacity at the eye to 10 % at `fadeDistance`. Other roles are opaque.
    public static func project(segments: [OverlaySegment],
                               camera: Camera,
                               viewSize: SIMD2<Float>,
                               fadeDistance: Float = .infinity) -> [ProjectedOverlaySegment] {
        let eye: SIMD3<Float> = camera.center
        let fades: Bool = fadeDistance.isFinite && fadeDistance > 0
        var result: [ProjectedOverlaySegment] = []
        result.reserveCapacity(segments.count)
        for segment in segments {
            let a: SIMD3<Float> = cameraSpace(segment.start, camera: camera)
            let b: SIMD3<Float> = cameraSpace(segment.end, camera: camera)
            guard let clipped = clipToNearPlane(a, b) else { continue }
            let sa: SIMD2<Float> = screen(clipped.0, camera: camera)
            let sb: SIMD2<Float> = screen(clipped.1, camera: camera)
            let leftOut: Bool = sa.x < 0 && sb.x < 0
            let rightOut: Bool = sa.x > viewSize.x && sb.x > viewSize.x
            let aboveOut: Bool = sa.y < 0 && sb.y < 0
            let belowOut: Bool = sa.y > viewSize.y && sb.y > viewSize.y
            if leftOut || rightOut || aboveOut || belowOut {
                continue
            }
            var alpha: Float = 1
            if fades, segment.role.isGrid {
                let middle: SIMD3<Float> = (segment.start + segment.end) * 0.5
                let offset: SIMD3<Float> = middle - eye
                let distance: Float = (offset * offset).sum().squareRoot()
                alpha = Swift.max(0.1, 1 - distance / fadeDistance)
            }
            result.append(ProjectedOverlaySegment(start: sa, end: sb, role: segment.role, alpha: alpha))
        }
        return result
    }

    /// Projects one world point. `nil` when it lies at or behind the near plane.
    public static func projectPoint(_ point: SIMD3<Float>, camera: Camera) -> SIMD2<Float>? {
        let local: SIMD3<Float> = cameraSpace(point, camera: camera)
        guard local.z > nearPlane else { return nil }
        return screen(local, camera: camera)
    }

    // MARK: - Private

    private static func axisRole(_ index: Int) -> OverlayRole {
        switch index {
        case 0: .axisX
        case 1: .axisY
        default: .axisZ
        }
    }

    private static func cameraSpace(_ point: SIMD3<Float>, camera: Camera) -> SIMD3<Float> {
        camera.rotation * point + camera.translation
    }

    private static func screen(_ local: SIMD3<Float>, camera: Camera) -> SIMD2<Float> {
        let u: Float = camera.fx * local.x / local.z + camera.cx
        let v: Float = camera.fy * local.y / local.z + camera.cy
        return SIMD2<Float>(u, v)
    }

    /// The part of a camera-space segment in front of the near plane, or `nil` if none of it is.
    private static func clipToNearPlane(_ a: SIMD3<Float>,
                                        _ b: SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>)? {
        let aFront: Bool = a.z > nearPlane
        let bFront: Bool = b.z > nearPlane
        if !aFront && !bFront { return nil }
        if aFront && bFront { return (a, b) }
        let t: Float = (nearPlane - a.z) / (b.z - a.z)
        let crossing: SIMD3<Float> = a + (b - a) * t
        return aFront ? (a, crossing) : (crossing, b)
    }
}
