import SwiftUI
import SplatCore

/// Draws the viewport tools into the overlay Canvas: the crop box with its handles, the floor picks and the fitted
/// plane, and the measured segment with its length. Projected with the same camera as the grid, in points. The
/// tools are read-only drawings here; the input goes through `SplatMTKView` and `ViewerModel+SceneTools`.
enum ViewportToolDrawing {
    static func draw(_ state: ToolOverlayState, into context: inout GraphicsContext, camera: Camera, ink: Color) {
        if let box = state.crop {
            drawCrop(box, into: &context, camera: camera, ink: ink)
        }
        if let preview = state.floorPreview {
            drawFloorPlane(preview, picks: state.floorPoints, into: &context, camera: camera, ink: ink)
        }
        for point in state.floorPoints {
            drawDot(point, radius: 4, fill: ink, into: &context, camera: camera)
        }
        if state.measurePoints.count == 2 {
            drawSegment(state.measurePoints, label: state.measureLabel, into: &context, camera: camera, ink: ink)
        } else {
            for point in state.measurePoints {
                drawDot(point, radius: 4, fill: ink, into: &context, camera: camera)
            }
        }
    }

    // MARK: - Crop

    /// The twelve edges of the box, and a handle at each face centre and at the centre.
    private static func drawCrop(_ box: CropBox, into context: inout GraphicsContext, camera: Camera, ink: Color) {
        let rotation: Matrix3 = Matrix3.rotation(quaternion: box.rotation)
        let axes: [SIMD3<Float>] = [rotation.c0, rotation.c1, rotation.c2]
        // Corner k has the sign of axis i given by bit i of k.
        var corners: [CGPoint?] = []
        for index in 0..<8 {
            var world: SIMD3<Float> = box.center
            for axis in 0..<3 {
                let sign: Float = (index >> axis) & 1 == 1 ? 1 : -1
                world += axes[axis] * (box.halfExtents[axis] * sign)
            }
            corners.append(project(world, camera: camera))
        }
        var edges: Path = Path()
        for index in 0..<8 {
            for axis in 0..<3 where (index >> axis) & 1 == 0 {
                let other: Int = index | (1 << axis)
                if let a = corners[index], let b = corners[other] {
                    edges.move(to: a)
                    edges.addLine(to: b)
                }
            }
        }
        context.stroke(edges, with: .color(ink.opacity(0.9)), style: StrokeStyle(lineWidth: 1.5))

        for (handle, position) in ViewerModel.cropHandlePositions(box, camera: camera) {
            let point: CGPoint = CGPoint(x: CGFloat(position.x), y: CGFloat(position.y))
            switch handle {
            case .face:
                let rect: CGRect = CGRect(x: point.x - 4.5, y: point.y - 4.5, width: 9, height: 9)
                context.fill(Path(rect), with: .color(ink))
                context.stroke(Path(rect), with: .color(ink.opacity(0.5)), style: StrokeStyle(lineWidth: 1))
            case .move:
                let rect: CGRect = CGRect(x: point.x - 6, y: point.y - 6, width: 12, height: 12)
                context.fill(Path(ellipseIn: rect), with: .color(.accentColor))
                context.stroke(Path(ellipseIn: rect), with: .color(ink), style: StrokeStyle(lineWidth: 1))
            }
        }
    }

    // MARK: - Floor

    /// A square of the fitted plane around the picks, with its diagonals.
    private static func drawFloorPlane(_ preview: FloorPreview,
                                       picks: [SIMD3<Float>],
                                       into context: inout GraphicsContext,
                                       camera: Camera,
                                       ink: Color) {
        let normal: SIMD3<Float> = preview.normal
        // Any vector not parallel to the normal gives an in-plane basis.
        let helper: SIMD3<Float> = abs(normal.y) < 0.9 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0)
        let uRaw: SIMD3<Float> = cross(normal, helper)
        let u: SIMD3<Float> = uRaw / length(uRaw)
        let v: SIMD3<Float> = cross(normal, u)
        var spread: Float = 0
        for point in picks {
            spread = max(spread, length(point - preview.centre))
        }
        let half: Float = max(0.5, 1.5 * spread)
        let corners: [SIMD3<Float>] = [
            preview.centre + u * half + v * half,
            preview.centre - u * half + v * half,
            preview.centre - u * half - v * half,
            preview.centre + u * half - v * half,
        ]
        var outline: Path = Path()
        var diagonals: Path = Path()
        let projected: [CGPoint?] = corners.map { project($0, camera: camera) }
        for index in 0..<4 {
            if let a = projected[index], let b = projected[(index + 1) % 4] {
                outline.move(to: a)
                outline.addLine(to: b)
            }
        }
        if let a = projected[0], let b = projected[2] {
            diagonals.move(to: a)
            diagonals.addLine(to: b)
        }
        if let a = projected[1], let b = projected[3] {
            diagonals.move(to: a)
            diagonals.addLine(to: b)
        }
        context.stroke(outline, with: .color(.accentColor), style: StrokeStyle(lineWidth: 1.5))
        context.stroke(diagonals, with: .color(.accentColor.opacity(0.5)), style: StrokeStyle(lineWidth: 1))
    }

    // MARK: - Measure

    private static func drawSegment(_ points: [SIMD3<Float>],
                                    label: String?,
                                    into context: inout GraphicsContext,
                                    camera: Camera,
                                    ink: Color) {
        guard let a = project(points[0], camera: camera), let b = project(points[1], camera: camera) else { return }
        var line: Path = Path()
        line.move(to: a)
        line.addLine(to: b)
        context.stroke(line, with: .color(.accentColor), style: StrokeStyle(lineWidth: 2))
        drawDot(points[0], radius: 4, fill: ink, into: &context, camera: camera)
        drawDot(points[1], radius: 4, fill: ink, into: &context, camera: camera)
        if let label {
            let middle: CGPoint = CGPoint(x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 - 12)
            context.draw(Text(label).font(.callout.monospacedDigit()).foregroundStyle(ink), at: middle, anchor: .center)
        }
    }

    // MARK: - Helpers

    private static func drawDot(_ world: SIMD3<Float>,
                                radius: CGFloat,
                                fill: Color,
                                into context: inout GraphicsContext,
                                camera: Camera) {
        guard let point = project(world, camera: camera) else { return }
        let rect: CGRect = CGRect(x: point.x - radius, y: point.y - radius, width: radius * 2, height: radius * 2)
        context.fill(Path(ellipseIn: rect), with: .color(fill))
    }

    private static func project(_ world: SIMD3<Float>, camera: Camera) -> CGPoint? {
        guard let projected = OverlayGeometry.projectPoint(world, camera: camera) else { return nil }
        return CGPoint(x: CGFloat(projected.x), y: CGFloat(projected.y))
    }

    private static func cross(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3<Float>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
    }

    private static func length(_ v: SIMD3<Float>) -> Float {
        (v * v).sum().squareRoot()
    }
}
