import SwiftUI
import SplatCore
import SplatIO

/// The overlay geometry of one scene: the ground grid, the camera frusta and the camera centres. It is built by
/// `ViewerModel.viewportOverlayScene` only when the bounds, the up axis or the camera list change. Every frame
/// projects it again with the current camera.
struct ViewportOverlayScene {
    /// Ground grid, including the axes through the origin.
    let grid: [OverlaySegment]
    /// Frustum edges of all cameras. The role carries the index of the camera.
    let frusta: [OverlaySegment]
    /// Camera centres in world space, in the order of the camera list.
    let cameraCentres: [SIMD3<Float>]
    /// Camera names, in the same order as `cameraCentres`.
    let cameraNames: [String]
    /// Radius of the scene. It sets the range of the camera labels and the fading of the grid.
    let sceneRadius: Float

    init(bounds: Bounds?, upAxis: UpAxis, cameras: [NamedCamera]) {
        let radius: Float = OverlayGeometry.sceneRadius(bounds: bounds)
        let plainCameras: [Camera] = cameras.map { $0.camera }
        let frustumSize: Float = OverlayGeometry.frustumSize(bounds: bounds)
        self.sceneRadius = radius
        self.grid = OverlayGeometry.groundGrid(bounds: bounds, upAxis: upAxis)
        self.frusta = OverlayGeometry.cameraFrusta(cameras: plainCameras, size: frustumSize)
        self.cameraCentres = plainCameras.map { $0.center }
        self.cameraNames = cameras.map { $0.name }
    }

    /// Grid lines at this distance from the eye have the lowest opacity.
    var fadeDistance: Float { 6 * sceneRadius }
}

extension ViewerModel {
    /// Bounds of the scene, or `nil` when it has no Gaussians.
    var overlayBounds: Bounds? {
        scene.cloud.count > 0 ? bounds : nil
    }

    /// The overlay geometry. It is built on first use after the bounds, the up axis or the cameras changed.
    var viewportOverlayScene: ViewportOverlayScene {
        if let cached = overlayCache {
            return cached
        }
        let built: ViewportOverlayScene = ViewportOverlayScene(bounds: overlayBounds, upAxis: upAxis, cameras: cameras)
        overlayCache = built
        return built
    }
}

/// Ground grid, axes gizmo and estimated cameras drawn above the 3D view. The overlay is plain line work on the
/// content layer, so it carries no glass, and it never takes input: the viewport underneath keeps every event.
struct ViewportOverlay: View {
    let model: ViewerModel

    /// Most camera names drawn at once. The cameras nearest to the eye win.
    private static let maximumLabels: Int = 40

    var body: some View {
        // Read the observed state here, so that SwiftUI draws the overlay again when any of it changes.
        let camera: ViewCamera = model.view
        let showsGrid: Bool = model.showsGrid
        let showsAxes: Bool = model.showsAxes
        let showsCameras: Bool = model.showsCameras
        let showsNames: Bool = model.showsCameraNames
        let activeCamera: Int? = model.activeCameraIndex
        let background: ViewerModel.Background = model.background
        let scene: ViewportOverlayScene = model.viewportOverlayScene

        Canvas { context, size in
            let width: Int = Int(size.width.rounded())
            let height: Int = Int(size.height.rounded())
            guard width > 0, height > 0 else { return }
            // Projected in points, like the edit overlay, so the lines match the rendered view.
            let projection: Camera = camera.camera(width: width, height: height)
            let viewSize: SIMD2<Float> = SIMD2<Float>(Float(size.width), Float(size.height))
            let ink: Color = background == .black ? .white : .black

            if showsGrid || showsAxes {
                let projected: [ProjectedOverlaySegment] = OverlayGeometry.project(
                    segments: scene.grid, camera: projection, viewSize: viewSize,
                    fadeDistance: scene.fadeDistance)
                for line in projected {
                    let visible: Bool = line.role.isGrid ? showsGrid : showsAxes
                    if visible {
                        Self.stroke(line, into: &context, ink: ink)
                    }
                }
            }
            if showsAxes {
                Self.drawGizmo(into: &context, size: size, rotation: projection.rotation)
            }
            if showsCameras {
                Self.drawCameras(into: &context, segments: scene.frusta, projection: projection,
                                 viewSize: viewSize, active: activeCamera)
                if showsNames {
                    Self.drawNames(into: &context, scene: scene, projection: projection,
                                   eye: camera.eye, viewSize: viewSize, ink: ink)
                }
            }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    // MARK: - Drawing

    /// One grid or axis line. Grid lines use their alpha; the axes are opaque.
    private static func stroke(_ line: ProjectedOverlaySegment,
                               into context: inout GraphicsContext,
                               ink: Color) {
        var path: Path = Path()
        path.move(to: point(line.start))
        path.addLine(to: point(line.end))
        let style: (color: Color, width: CGFloat) = gridStyle(line.role, ink: ink)
        context.stroke(path, with: .color(style.color.opacity(Double(line.alpha))), lineWidth: style.width)
    }

    /// The Blender-style axes gizmo in the bottom left corner, from the camera rotation only.
    private static func drawGizmo(into context: inout GraphicsContext, size: CGSize, rotation: Matrix3) {
        let anchor: CGPoint = CGPoint(x: 48, y: size.height - 48)
        let arms: [OverlayGizmoAxis] = OverlayGeometry.axesGizmo(rotation: rotation, length: 24)
        // Arms that point away from the viewer are drawn first and dimmed, so the nearer arms stay on top.
        for arm in arms.sorted(by: { $0.depth > $1.depth }) {
            let dx: CGFloat = CGFloat(arm.offset.x)
            let dy: CGFloat = CGFloat(arm.offset.y)
            let tip: CGPoint = CGPoint(x: anchor.x + dx, y: anchor.y + dy)
            var path: Path = Path()
            path.move(to: anchor)
            path.addLine(to: tip)
            let color: Color = axisColor(arm.axis).opacity(arm.depth > 0 ? 0.45 : 1)
            context.stroke(path, with: .color(color), lineWidth: 1.5)
            let labelAt: CGPoint = CGPoint(x: anchor.x + 1.4 * dx, y: anchor.y + 1.4 * dy)
            let label: String = ["X", "Y", "Z"][arm.axis]
            context.draw(Text(label).font(.caption.bold()).foregroundStyle(color), at: labelAt, anchor: .center)
        }
    }

    /// Camera frusta: all in one colour, the active camera in the accent colour and thicker.
    private static func drawCameras(into context: inout GraphicsContext,
                                    segments: [OverlaySegment],
                                    projection: Camera,
                                    viewSize: SIMD2<Float>,
                                    active: Int?) {
        let projected: [ProjectedOverlaySegment] = OverlayGeometry.project(
            segments: segments, camera: projection, viewSize: viewSize)
        var others: Path = Path()
        var selected: Path = Path()
        for line in projected {
            guard case .camera(let index) = line.role else { continue }
            if index == active {
                selected.move(to: point(line.start))
                selected.addLine(to: point(line.end))
            } else {
                others.move(to: point(line.start))
                others.addLine(to: point(line.end))
            }
        }
        context.stroke(others, with: .color(Color.orange.opacity(0.85)), lineWidth: 1)
        context.stroke(selected, with: .color(Color.accentColor), lineWidth: 2)
    }

    /// Camera names at the camera centres. Only cameras within three scene radii of the eye are considered,
    /// nearest first, and at most `maximumLabels` are drawn.
    private static func drawNames(into context: inout GraphicsContext,
                                  scene: ViewportOverlayScene,
                                  projection: Camera,
                                  eye: SIMD3<Float>,
                                  viewSize: SIMD2<Float>,
                                  ink: Color) {
        let range: Float = 3 * scene.sceneRadius
        var candidates: [(index: Int, distance: Float)] = []
        for (index, centre) in scene.cameraCentres.enumerated() {
            let offset: SIMD3<Float> = centre - eye
            let distance: Float = (offset * offset).sum().squareRoot()
            if distance <= range {
                candidates.append((index: index, distance: distance))
            }
        }
        candidates.sort { $0.distance < $1.distance }

        var drawn: Int = 0
        for candidate in candidates {
            if drawn >= maximumLabels {
                break
            }
            guard let at = OverlayGeometry.projectPoint(scene.cameraCentres[candidate.index], camera: projection) else {
                continue
            }
            // Labels that would start off the view are skipped and do not use up the budget.
            guard at.x >= 0, at.x <= viewSize.x, at.y >= 0, at.y <= viewSize.y else {
                continue
            }
            let text: Text = Text(scene.cameraNames[candidate.index])
                .font(.caption2)
                .foregroundStyle(ink.opacity(0.9))
            context.draw(text, at: point(at), anchor: .bottomLeading)
            drawn += 1
        }
    }

    // MARK: - Styles

    /// Colour and width of a grid or axis line. The base colour is white on the black background and black otherwise.
    private static func gridStyle(_ role: OverlayRole, ink: Color) -> (color: Color, width: CGFloat) {
        switch role {
        case .minorGrid:
            return (ink.opacity(0.16), 0.5)
        case .majorGrid:
            return (ink.opacity(0.32), 1)
        case .axisX, .axisY, .axisZ:
            return (axisColor(axisIndex(role)), 1)
        case .camera:
            return (Color.orange, 1)
        }
    }

    /// Blender's axis colours: X red, Y green, Z blue.
    private static func axisColor(_ index: Int) -> Color {
        switch index {
        case 0: Color(red: 0.92, green: 0.27, blue: 0.27)
        case 1: Color(red: 0.35, green: 0.80, blue: 0.30)
        default: Color(red: 0.30, green: 0.50, blue: 0.95)
        }
    }

    private static func axisIndex(_ role: OverlayRole) -> Int {
        switch role {
        case .axisX: 0
        case .axisY: 1
        default: 2
        }
    }

    private static func point(_ p: SIMD2<Float>) -> CGPoint {
        CGPoint(x: CGFloat(p.x), y: CGFloat(p.y))
    }
}
