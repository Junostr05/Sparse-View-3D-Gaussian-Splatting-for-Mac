import CoreGraphics
import Foundation
import SplatCore
import SplatIO

/// The viewport tools. While one is active its clicks and handle drags go to the tool; orbit, pan and zoom keep
/// working everywhere else. Escape returns to `.none`.
enum ViewerTool: String, CaseIterable, Identifiable, Sendable {
    case none
    case crop
    case floor
    case measure

    var id: String { rawValue }

    var title: String {
        switch self {
        case .none: "None"
        case .crop: "Crop"
        case .floor: "Floor"
        case .measure: "Measure"
        }
    }
}

/// Units of the scale calibration and of the metric measurements.
enum LengthUnit: String, CaseIterable, Identifiable, Sendable {
    case metre
    case centimetre
    case inch

    var id: String { rawValue }

    var title: String {
        switch self {
        case .metre: "m"
        case .centimetre: "cm"
        case .inch: "in"
        }
    }

    /// Metres in one unit.
    var metresPerUnit: Double {
        switch self {
        case .metre: 1
        case .centimetre: 0.01
        case .inch: 0.0254
        }
    }

    /// A length given in metres, shown in this unit with two decimals.
    func format(metres: Float) -> String {
        let value: Double = Double(metres) / metresPerUnit
        return String(format: "%.2f", value) + " " + title
    }
}

/// A handle of the crop box: one of its six faces, or the move handle at the centre.
enum CropHandle: Equatable, Sendable {
    /// Face `axis` (0, 1, 2 for the box's own x, y, z axes), on the positive or the negative side.
    case face(axis: Int, positive: Bool)
    case move
}

/// The handle a press grabbed and the crop box it started from. The drag is computed from the start, so it does
/// not accumulate rounding.
struct CropDragState {
    let handle: CropHandle
    let startBox: CropBox
    let startPoint: CGPoint
}

/// What the viewport does with a press while a tool is active.
enum ToolPressOutcome: Equatable {
    /// The tool does not take the press: orbit, pan and select as usual.
    case passThrough
    /// The press is a tool click when the mouse is released without moving; a drag past a few points orbits.
    case pendingClick
    /// A crop handle is grabbed. The drag moves it until the release.
    case dragHandle
}

/// The plane fitted to the floor picks, for the overlay.
struct FloorPreview: Equatable {
    let centre: SIMD3<Float>
    let normal: SIMD3<Float>
}

/// Everything the viewport overlay draws for the active tool.
struct ToolOverlayState {
    let tool: ViewerTool
    /// The crop box, drawn while the crop tool is active.
    let crop: CropBox?
    let floorPoints: [SIMD3<Float>]
    let floorPreview: FloorPreview?
    let measurePoints: [SIMD3<Float>]
    let measureLabel: String?
}

/// Dot products written out, so the type checker never sees the SIMD operators in long expressions.
private func sceneDot(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
    let products: SIMD3<Float> = a * b
    return products.x + products.y + products.z
}

private func sceneDot(_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
    let products: SIMD2<Float> = a * b
    return products.x + products.y
}

/// Scene transform, crop box and the measuring, floor and crop tools. The state lives on `ViewerModel`; the
/// stored parts are in ViewerModel.swift, the drawing in App/Viewport/ViewportToolDrawing.swift and the panel in
/// App/Viewport/ViewportToolsControls.swift.
///
/// Frames: the transform maps the stored scene to the displayed one. The crop box, the floor picks and the
/// measured points are all in the displayed frame. Changing the transform moves the crop box with it.
extension ViewerModel {
    /// The crop box the renderer clips to: only while the crop tool is active and the clip is on.
    var cropClip: CropBox? {
        activeTool == .crop && clipsToCropBox ? crop : nil
    }

    /// True when the scene has a transform or a crop, which the export sheet offers to apply.
    var hasSceneEdits: Bool {
        !sceneTransform.isIdentity || crop != nil
    }

    /// Length of the measured segment in displayed units, when two points are set.
    var measuredDistance: Float? {
        guard measurePoints.count == 2 else { return nil }
        let difference: SIMD3<Float> = measurePoints[0] - measurePoints[1]
        return (difference * difference).sum().squareRoot()
    }

    /// The measured length as text: in the chosen unit once the scene is metric, otherwise in scene units.
    var measureLabel: String? {
        guard let distance = measuredDistance else { return nil }
        if sceneTransform.isMetric {
            return measureUnit.format(metres: distance)
        }
        return String(format: "%.3f", distance) + " scene units"
    }

    /// The overlay's view of the active tool.
    var toolOverlayState: ToolOverlayState {
        ToolOverlayState(tool: activeTool,
                         crop: activeTool == .crop ? crop : nil,
                         floorPoints: floorPoints,
                         floorPreview: floorPreview,
                         measurePoints: measurePoints,
                         measureLabel: measureLabel)
    }

    // MARK: - Scene transform

    /// Sets the scene transform and moves everything stored in its frame: the displayed cloud, the cameras, the
    /// bounds and the crop box. The view is framed again and the change is reported for storage.
    func setSceneTransform(_ next: SceneTransform) {
        guard next != sceneTransform else { return }
        // Maps points of the old displayed frame into the new one; the crop box is stored in the displayed frame.
        let delta: SceneTransform = next.composing(sceneTransform.inverse)
        if let box = crop {
            crop = box.transformed(by: delta)
        }
        stopMotion()
        sceneTransform = next
        rebuildDisplayCloud()
        bounds = scene.cloud.robustBounds()
        replaceCameras(baseCameras)
        resetView()
        onSceneEditsChanged?(sceneTransform, crop)
    }

    /// Sets the crop box in the displayed frame (`nil` removes it) and reports it for storage.
    func setCrop(_ box: CropBox?) {
        crop = box
        onSceneEditsChanged?(sceneTransform, crop)
    }

    /// Removes the Gaussians outside the crop box through the deletion history, so the step can be undone.
    func deleteOutsideCrop() {
        guard canEdit, let box = crop else { return }
        let outside: SelectionMask = box.mask(for: scene.cloud)
        guard !outside.isEmpty else { return }
        selection = outside
        deleteSelection()
    }

    // MARK: - Floor

    /// The plane through the floor picks, once there are three. Its normal is the up axis seen through the fit.
    var floorPreview: FloorPreview? {
        guard floorPoints.count >= 3,
              let fit = SceneTransform.fromFloor(points: floorPoints, upAxis: upAxis) else {
            return nil
        }
        let normal: SIMD3<Float> = fit.rotationMatrix.transposed * upAxis.vector
        let sum: SIMD3<Float> = floorPoints.reduce(SIMD3<Float>(0, 0, 0), +)
        let centre: SIMD3<Float> = sum / Float(floorPoints.count)
        return FloorPreview(centre: centre, normal: normal)
    }

    /// Stores the floor: the fit through the picks becomes the first step of the scene transform, so the ground
    /// lies at height 0 in the up axis. Returns `false` with fewer than three picks.
    @discardableResult
    func applyFloor() -> Bool {
        guard let fit = SceneTransform.fromFloor(points: floorPoints, upAxis: upAxis) else { return false }
        floorPoints = []
        setSceneTransform(fit.composing(sceneTransform))
        return true
    }

    /// Removes the floor fit: the rotation and the translation go, the scale stays.
    func resetFloor() {
        floorPoints = []
        setSceneTransform(SceneTransform(rotation: SceneTransform.identity.rotation,
                                         translation: SceneTransform.identity.translation,
                                         scale: sceneTransform.scale,
                                         isMetric: sceneTransform.isMetric))
    }

    // MARK: - Measure and scale

    /// Makes the measured segment `realLength` long in `unit` by scaling the scene about the origin. The scene
    /// becomes metric. Returns `false` unless two points are measured and the length is positive.
    @discardableResult
    func applyMeasuredScale(realLength: Double, unit: LengthUnit) -> Bool {
        guard measurePoints.count == 2, realLength.isFinite, realLength > 0 else { return false }
        let metres: Float = Float(realLength * unit.metresPerUnit)
        guard let factor = SceneTransform.calibrationScale(pointA: measurePoints[0],
                                                           pointB: measurePoints[1],
                                                           realLength: metres) else {
            return false
        }
        var next: SceneTransform = sceneTransform.scaled(by: factor)
        next.isMetric = true
        measurePoints = []
        setSceneTransform(next)
        return true
    }

    /// Removes the scale calibration: the scene returns to scene units.
    func resetScale() {
        var next: SceneTransform = sceneTransform.scaled(by: 1 / sceneTransform.scale)
        next.scale = 1
        next.isMetric = false
        measurePoints = []
        setSceneTransform(next)
    }

    // MARK: - Viewport input

    /// A press in the viewport while a tool is active. Crop grabs a handle when one is under the pointer; the
    /// floor and measure tools wait for the release, so that a drag still orbits.
    func toolPress(atViewPoint point: CGPoint, viewSize: CGSize) -> ToolPressOutcome {
        switch activeTool {
        case .none:
            return .passThrough
        case .floor, .measure:
            return .pendingClick
        case .crop:
            guard let box = crop, let handle = cropHandle(near: point, box: box, viewSize: viewSize) else {
                return .passThrough
            }
            cropDrag = CropDragState(handle: handle, startBox: box, startPoint: point)
            return .dragHandle
        }
    }

    /// A drag of a grabbed crop handle. Shift keeps the proportions, Option resizes about the centre.
    func toolDrag(atViewPoint point: CGPoint, viewSize: CGSize, shift: Bool, option: Bool) {
        guard activeTool == .crop, let drag = cropDrag else { return }
        crop = draggedCrop(drag, to: point, viewSize: viewSize, uniform: shift, fromCentre: option)
    }

    /// The end of a drag, which stores the crop box.
    func toolRelease() {
        guard cropDrag != nil else { return }
        cropDrag = nil
        onSceneEditsChanged?(sceneTransform, crop)
    }

    /// A click that did not drag: the floor or measure tool picks the surface under the pointer.
    func toolClick(atViewPoint point: CGPoint, viewSize: CGSize) {
        guard activeTool == .floor || activeTool == .measure else { return }
        guard let camera = projectionCamera(for: viewSize) else { return }
        let pixel: SIMD2<Float> = SIMD2<Float>(Float(point.x), Float(point.y))
        guard let pick = CursorDepth.pick(cloud: scene.cloud, camera: camera, pixel: pixel) else { return }
        switch activeTool {
        case .floor:
            floorPoints.append(pick.worldPoint)
            if floorPoints.count > 6 {
                floorPoints.removeFirst()
            }
        case .measure:
            if measurePoints.count == 2 {
                measurePoints = []
            }
            measurePoints.append(pick.worldPoint)
        case .none, .crop:
            break
        }
    }

    // MARK: - Crop handles

    /// The handle nearest to `point` within 9 points. Faces are tried first, then the move handle.
    func cropHandle(near point: CGPoint, box: CropBox, viewSize: CGSize) -> CropHandle? {
        guard let camera = projectionCamera(for: viewSize) else { return nil }
        var best: CropHandle?
        var bestDistance: Float = 9
        for (handle, position) in Self.cropHandlePositions(box, camera: camera) {
            let dx: Float = Float(point.x) - position.x
            let dy: Float = Float(point.y) - position.y
            let distance: Float = (dx * dx + dy * dy).squareRoot()
            if distance <= bestDistance {
                bestDistance = distance
                best = handle
            }
        }
        return best
    }

    /// Projected position of every handle of `box`, in points.
    static func cropHandlePositions(_ box: CropBox, camera: Camera) -> [(CropHandle, SIMD2<Float>)] {
        let rotation: Matrix3 = Matrix3.rotation(quaternion: box.rotation)
        let axes: [SIMD3<Float>] = [rotation.c0, rotation.c1, rotation.c2]
        var result: [(CropHandle, SIMD2<Float>)] = []
        for axis in 0..<3 {
            for positive in [true, false] {
                let sign: Float = positive ? 1 : -1
                let world: SIMD3<Float> = box.center + axes[axis] * (box.halfExtents[axis] * sign)
                if let projected = OverlayGeometry.projectPoint(world, camera: camera) {
                    result.append((.face(axis: axis, positive: positive), projected))
                }
            }
        }
        if let projected = OverlayGeometry.projectPoint(box.center, camera: camera) {
            result.append((.move, projected))
        }
        return result
    }

    /// The crop box after dragging `drag.handle` to `point`.
    private func draggedCrop(_ drag: CropDragState,
                             to point: CGPoint,
                             viewSize: CGSize,
                             uniform: Bool,
                             fromCentre: Bool) -> CropBox {
        let start: CropBox = drag.startBox
        let minimum: Float = 1e-4
        let deltaX: Float = Float(point.x - drag.startPoint.x)
        let deltaY: Float = Float(point.y - drag.startPoint.y)
        switch drag.handle {
        case .move:
            // Pixels to scene units at the depth of the box centre, along the camera's right and down axes.
            let height: Float = max(Float(viewSize.height), 1)
            let eye: SIMD3<Float> = view.eye
            let offsetFromEye: SIMD3<Float> = start.center - eye
            let depth: Float = max(sceneDot(offsetFromEye, view.forward), 1e-3)
            let perPixel: Float = 2 * depth * tan(view.fovY / 2) / height
            var moved: CropBox = start
            moved.center = start.center + view.right * (deltaX * perPixel) + view.down * (deltaY * perPixel)
            return moved
        case .face(let axis, let positive):
            let rotation: Matrix3 = Matrix3.rotation(quaternion: start.rotation)
            let columns: [SIMD3<Float>] = [rotation.c0, rotation.c1, rotation.c2]
            let sign: Float = positive ? 1 : -1
            let outward: SIMD3<Float> = columns[axis] * sign
            let extent: Float = max(start.halfExtents.max(), minimum)
            let probe: Float = 0.01 * extent
            let faceCentre: SIMD3<Float> = start.center + outward * start.halfExtents[axis]
            guard let camera = projectionCamera(for: viewSize),
                  let origin = OverlayGeometry.projectPoint(faceCentre, camera: camera),
                  let moved = OverlayGeometry.projectPoint(faceCentre + outward * probe, camera: camera) else {
                return start
            }
            let screenStep: SIMD2<Float> = moved - origin
            let stepLength: Float = sceneDot(screenStep, screenStep)
            guard stepLength > 1e-12 else { return start }
            let dragPixels: SIMD2<Float> = SIMD2<Float>(deltaX, deltaY)
            // Distance the face moves along its outward direction, in scene units.
            let distance: Float = sceneDot(dragPixels, screenStep) / stepLength * probe

            var result: CropBox = start
            if fromCentre {
                // Both faces of the axis move: the half extent grows by the distance, the centre stays.
                let grown: Float = max(start.halfExtents[axis] + distance, minimum)
                result.halfExtents[axis] = grown
                if uniform, start.halfExtents[axis] > minimum {
                    let factor: Float = grown / start.halfExtents[axis]
                    scaleOtherAxes(&result, start: start, except: axis, factor: factor, minimum: minimum)
                }
            } else {
                // Only this face moves: the centre and the half extent each take half of the distance.
                let grown: Float = max(start.halfExtents[axis] + distance / 2, minimum)
                result.halfExtents[axis] = grown
                result.center = start.center + outward * (distance / 2)
                if uniform, start.halfExtents[axis] > minimum {
                    let factor: Float = grown / start.halfExtents[axis]
                    scaleOtherAxes(&result, start: start, except: axis, factor: factor, minimum: minimum)
                }
            }
            return result
        }
    }

    private func scaleOtherAxes(_ result: inout CropBox, start: CropBox, except axis: Int, factor: Float, minimum: Float) {
        for other in 0..<3 where other != axis {
            result.halfExtents[other] = max(start.halfExtents[other] * factor, minimum)
        }
    }

    /// The camera that projects the viewport tools: the viewer's view, sized in points like the overlay.
    private func projectionCamera(for viewSize: CGSize) -> Camera? {
        let width: Int = Int(viewSize.width.rounded())
        let height: Int = Int(viewSize.height.rounded())
        guard width > 0, height > 0 else { return nil }
        return view.camera(width: width, height: height)
    }
}
