import AppKit
import Foundation
import Observation
import SplatCore
import SplatIO
import SplatMetal

/// Metal objects for one window. Deliberately not observable.
final class MetalSession {
    let context: MetalContext
    let renderer: SplatRenderer

    init(cloud: GaussianCloud) throws(MetalError) {
        context = try MetalContext.makeDefault()
        renderer = try SplatRenderer(context: context, cloud: cloud)
    }
}

@Observable
final class ViewerModel {
    enum Background: String, CaseIterable, Identifiable {
        case black, gray, white

        var id: String { rawValue }

        var title: String {
            switch self {
            case .black: "Black"
            case .gray: "Gray"
            case .white: "White"
            }
        }

        var color: SIMD3<Float> {
            switch self {
            case .black: .zero
            case .gray: SIMD3(repeating: 0.5)
            case .white: SIMD3(repeating: 1)
            }
        }
    }

    enum RenderState {
        case preparing
        case ready(MetalSession)
        case unavailable
    }

    /// Seconds of flight between two neighbouring cameras during playback.
    static let secondsPerSegment: Double = 1.5

    /// The scene shown by the viewport. Replaced during training by `replaceCloud(_:)`.
    /// Written by the editing extension (`ViewerModel+Editing.swift`) too, hence not `private(set)`.
    var scene: LoadedScene

    /// Incremented on every `replaceCloud(_:)` and every edit. `ViewportView` reads it so SwiftUI re-renders the view.
    var cloudRevision = 0

    /// The camera the viewport renders. Changed by input, by flights and by playback.
    private(set) var view: ViewCamera
    var shDegreeLimit: Int
    var background: Background = .black
    var isInspectorPresented = true
    var renderState: RenderState = .preparing

    // MARK: Viewport overlays. Drawn by `ViewportOverlay`; per session only, not stored in the project.

    /// Ground grid below the scene.
    var showsGrid = true
    /// Axes gizmo, and the axis lines through the origin.
    var showsAxes = true
    /// Frustum pyramids of the estimated cameras.
    var showsCameras = false
    /// Camera names at the camera centres. Only shown together with `showsCameras`.
    var showsCameraNames = false

    // MARK: Display modes. The behaviour is in ViewerModel+Display.swift.

    /// What the viewport draws: the splats, points, depth, shapes or a heatmap.
    var displayMode: ViewerDisplayMode = .splat
    /// The value the heatmap mode colours.
    var heatmapMetric: HeatmapMetric = .size
    /// Disc radius in rendered pixels for the points mode.
    var pointSize: Double = 3
    /// Per-Gaussian colours computed for the displayed cloud. Each set is valid for the `cloudRevision` it was made for.
    @ObservationIgnored var displayColorSets: [DisplayColorKey: DisplayColorSet] = [:]
    /// The heatmap colours last shown, kept as the fallback while another metric is computed.
    @ObservationIgnored var lastHeatmapSet: (key: DisplayColorKey, set: DisplayColorSet)?
    /// Colour sets being computed, with the cloud revision each was requested for.
    var pendingDisplayColors: [DisplayColorKey: Int] = [:]
    /// Incremented when a colour set arrives, so the viewport redraws.
    var displayColorRevision = 0

    // MARK: Editing state. The behaviour is in ViewerModel+Editing.swift.

    /// The base cloud the edit history refers to. Replaced by `replaceCloud(_:)`.
    @ObservationIgnored var baseCloud: GaussianCloud
    /// Deletions of the base cloud, with undo and redo.
    var history: EditHistory
    /// Base index of each displayed Gaussian: element `i` is the base index of display index `i`.
    var displayIndexToBase: [Int]
    /// Selected Gaussians, over display indices.
    var selection: SelectionMask
    /// Set by the project model: false while a training preview is shown, which disables editing.
    var isEditingAllowed = true {
        didSet {
            if !isEditingAllowed {
                isEditing = false
                isFlying = false
            }
        }
    }
    /// Edit mode: a left drag paints the brush selection instead of orbiting.
    var isEditing = false {
        didSet {
            if !isEditing {
                brushCursor = nil
            }
            if isEditing {
                isFlying = false
            }
        }
    }

    // MARK: Cursor depth and fly mode. The behaviour is below in the "Cursor depth" and "Fly mode" sections.

    /// Pivot at cursor depth: gestures orbit, pan and zoom about the surface under the cursor.
    var anchorsToCursorDepth = true
    /// World point under the cursor at the start of the current gesture. Orbit gestures rotate about it.
    @ObservationIgnored private var cursorOrbitAnchor: SIMD3<Float>?
    /// True while fly mode owns the mouse and the keys. The view captures the cursor while this is set.
    private(set) var isFlying = false
    /// Base fly speed multiplier, changed by the scroll wheel in fly mode. The base speed is 0.5 scene radii per second.
    var flySpeedScale: Float = 1
    /// Short text shown over the viewport after a speed change, cleared after about a second.
    private(set) var flySpeedNotice: String?
    @ObservationIgnored private var flySpeedNoticeTask: Task<Void, Never>?
    /// Brush radius in points.
    var brushRadius: Double = 40
    var brushMode: BrushMode = .add
    /// Front-only brush: selects only the visible surface under the brush instead of every depth (x-ray).
    var brushFrontOnly: Bool = false
    /// Sigmoid opacity of each Gaussian of the displayed cloud, computed on first use for front-only selection.
    /// Cleared in `presentCloud(_:removedNonFinite:)` whenever the displayed cloud changes.
    @ObservationIgnored var displayOpacityCache: [Float]?
    /// Brush cursor in view points with a top-left origin. `nil` when the pointer is outside the viewport.
    var brushCursor: CGPoint?
    /// Called after every delete, undo and redo with the cumulative deletions as base indices.
    @ObservationIgnored var onDeletionsChanged: ((_ baseCount: Int, _ deleted: [UInt32]) -> Void)?

    /// Estimated cameras of the project, in the order the viewer lists them. Empty for a plain .ply scene.
    private(set) var cameras: [NamedCamera]
    /// The camera last selected or reached by a flight. `nil` when there are no cameras.
    private(set) var activeCameraIndex: Int?
    /// Fractional position along the camera flight, `0...(cameras.count − 1)`.
    var flightPosition: Double = 0
    /// True while playback advances along the camera flight.
    private(set) var isPlaying = false

    var upAxis: UpAxis {
        didSet {
            guard upAxis != oldValue else { return }
            stopMotion()
            view = ViewCamera.framing(bounds, upAxis: upAxis)
            rebuildCameraViews()
        }
    }

    /// Latest frame statistics, assigned at most every 0.5 s.
    private(set) var stats: FrameStats?

    /// Bounds of the scene the camera framing and the display normalisation use. Set once at init.
    @ObservationIgnored let bounds: Bounds
    @ObservationIgnored private var lastStatsTime: ContinuousClock.Instant?
    /// One view per camera, built from `cameras` and the current up axis.
    @ObservationIgnored private var cameraViews: [ViewCamera] = []
    /// Overlay geometry for the bounds, the up axis and the cameras. Built by `viewportOverlayScene`, cleared
    /// whenever the camera views are rebuilt.
    @ObservationIgnored var overlayCache: ViewportOverlayScene?
    /// Smooth path through `cameraViews`. `nil` when there are no cameras.
    @ObservationIgnored private var flight: ViewFlight?
    /// The running flight or playback task. Cancelled by every new request and by manual input.
    @ObservationIgnored private var motionTask: Task<Void, Never>?

    init(scene: LoadedScene, upAxis: UpAxis = .negativeY, cameras: [NamedCamera] = []) {
        self.scene = scene
        let count: Int = scene.cloud.count
        self.baseCloud = scene.cloud
        self.history = EditHistory(baseCount: count)
        self.displayIndexToBase = Array(0..<count)
        self.selection = SelectionMask(count: count)
        let bounds = scene.cloud.robustBounds()
        self.bounds = bounds
        self.upAxis = upAxis
        self.cameras = cameras
        self.shDegreeLimit = scene.cloud.shDegree
        self.view = ViewCamera.framing(bounds, upAxis: upAxis)
        rebuildCameraViews()
        if !cameraViews.isEmpty {
            view = cameraViews[0]
            activeCameraIndex = 0
        }
    }

    /// Creates the Metal context and renderer once per window.
    func prepareRenderer() {
        guard case .preparing = renderState else { return }
        do throws(MetalError) {
            renderState = .ready(try MetalSession(cloud: scene.cloud))
        } catch {
            renderState = .unavailable
        }
    }

    /// Replaces the base cloud, for example with a training preview or the final result of a run.
    /// The camera keeps its view. The edit state is reset: the new cloud has no deletions, no selection
    /// and editing is off. Before the renderer exists the new cloud is used when it is created.
    func replaceCloud(_ cloud: GaussianCloud) {
        resetEditing(baseCloud: cloud)
        shDegreeLimit = min(shDegreeLimit, cloud.shDegree)
        presentCloud(cloud, removedNonFinite: 0)
    }

    func report(_ newStats: FrameStats) {
        let now = ContinuousClock.now
        if let last = lastStatsTime, now - last < .milliseconds(500) { return }
        lastStatsTime = now
        stats = newStats
    }

    // MARK: - Cameras

    /// The view of camera `index`: its pose, with the pivot placed in front of the eye at the distance
    /// to the scene centre along the view direction (clamped to 0.2 R ... 3 R, or R when the scene lies
    /// behind the camera). R is the bounding-sphere radius of the scene.
    func cameraView(at index: Int) -> ViewCamera {
        Self.cameraView(cameras[index].camera, bounds: bounds, upAxis: upAxis)
    }

    private static func cameraView(_ camera: Camera, bounds: Bounds, upAxis: UpAxis) -> ViewCamera {
        let reference = ViewCamera.framing(bounds, upAxis: upAxis)
        let extent: SIMD3<Float> = bounds.extent
        let halfDiagonal: Float = 0.5 * (extent * extent).sum().squareRoot()
        let radius: Float = halfDiagonal > 0 ? halfDiagonal : 1
        let probe = ViewCamera(camera: camera, pivotDistance: 0, upAxis: upAxis,
                               minDistance: reference.minDistance, maxDistance: reference.maxDistance)
        let toCenter: SIMD3<Float> = bounds.center - camera.center
        let along: Float = (toCenter * probe.forward).sum()
        let pivotDistance: Float = along > 0 ? Swift.min(Swift.max(along, 0.2 * radius), 3 * radius) : radius
        return ViewCamera(camera: camera, pivotDistance: pivotDistance, upAxis: upAxis,
                          minDistance: reference.minDistance, maxDistance: reference.maxDistance)
    }

    /// Rebuilds the cached camera views and the flight through them. Called at init and when the up axis changes.
    private func rebuildCameraViews() {
        overlayCache = nil
        cameraViews = cameras.indices.map { cameraView(at: $0) }
        flight = cameraViews.isEmpty ? nil : ViewFlight(keyframes: cameraViews)
    }

    /// Replaces the camera list, for example with the refined poses a training run stored. Stops any flight
    /// or playback still running through the old cameras.
    func replaceCameras(_ cameras: [NamedCamera]) {
        stopMotion()
        self.cameras = cameras
        if let index = activeCameraIndex, index >= cameras.count {
            activeCameraIndex = nil
        }
        rebuildCameraViews()
    }

    /// Flies to camera `index`, wrapping around the list.
    func goToCamera(_ index: Int) {
        let count = cameraViews.count
        guard count > 0 else { return }
        let wrapped = ((index % count) + count) % count
        activeCameraIndex = wrapped
        flightPosition = Double(wrapped)
        fly(to: cameraViews[wrapped])
    }

    func nextCamera() {
        guard !cameraViews.isEmpty else { return }
        goToCamera((activeCameraIndex ?? -1) + 1)
    }

    func previousCamera() {
        guard !cameraViews.isEmpty else { return }
        goToCamera((activeCameraIndex ?? 0) - 1)
    }

    /// Moves the view to a fractional position along the camera flight, without animation.
    func scrub(to position: Double) {
        stopMotion()
        cursorOrbitAnchor = nil
        guard let flight else { return }
        let last = Double(flight.count - 1)
        let clamped = Swift.min(Swift.max(position, 0), last)
        flightPosition = clamped
        view = flight.camera(at: Float(clamped))
        activeCameraIndex = Int(clamped.rounded())
    }

    func togglePlayback() {
        if isPlaying {
            pause()
        } else {
            play()
        }
    }

    /// Plays the camera flight from the current position, or from the start when it is at the end.
    func play() {
        guard let flight, flight.count >= 2, !isPlaying else { return }
        stopMotion()
        let last = Double(flight.count - 1)
        if flightPosition >= last {
            flightPosition = 0
        }
        isPlaying = true
        let startPosition = flightPosition
        motionTask = Task { await self.runPlayback(from: startPosition, flight: flight) }
    }

    func pause() {
        stopMotion()
    }

    // MARK: - Manual input

    /// Orbit from a key step. Cancels any flight or playback. Rotates about the pivot.
    func orbit(deltaYaw: Float, deltaPitch: Float) {
        stopMotion()
        view.orbit(deltaYaw: deltaYaw, deltaPitch: deltaPitch)
    }

    /// Orbit from a mouse drag or a trackpad scroll. Rotates about the cursor anchor of the gesture when
    /// there is one, otherwise about the pivot.
    func orbitGesture(deltaYaw: Float, deltaPitch: Float) {
        guard anchorsToCursorDepth, let anchor = cursorOrbitAnchor else {
            orbit(deltaYaw: deltaYaw, deltaPitch: deltaPitch)
            return
        }
        stopMotion()
        view.orbit(around: anchor, deltaYaw: deltaYaw, deltaPitch: deltaPitch)
    }

    // MARK: - Cursor depth

    /// Starts a gesture at `point` (view points, top-left origin, in a view of `viewSize` points). Picks the
    /// surface under the cursor, moves the pivot to its depth (`setDepth`) and remembers its world point for
    /// orbit gestures. Stops any flight or playback when a surface is picked. Without a surface the anchor
    /// is cleared and the view is left alone.
    func beginInteraction(atViewPoint point: CGPoint, viewSize: CGSize) {
        guard anchorsToCursorDepth else {
            cursorOrbitAnchor = nil
            return
        }
        let width: Int = Int(viewSize.width.rounded())
        let height: Int = Int(viewSize.height.rounded())
        guard width > 0, height > 0 else {
            cursorOrbitAnchor = nil
            return
        }
        let camera: Camera = view.camera(width: width, height: height)
        let pixel: SIMD2<Float> = SIMD2(Float(point.x), Float(point.y))
        guard let pick = CursorDepth.pick(cloud: scene.cloud, camera: camera, pixel: pixel) else {
            cursorOrbitAnchor = nil
            return
        }
        stopMotion()
        view.setDepth(pick.depth)
        cursorOrbitAnchor = pick.worldPoint
    }

    /// Pan from a gesture. `deltaPixels` uses the same convention as `ViewCamera.pan`.
    func pan(deltaPixels: SIMD2<Float>, viewportHeight: Float) {
        stopMotion()
        view.pan(deltaPixels: deltaPixels, viewportHeight: viewportHeight)
    }

    func zoom(factor: Float) {
        stopMotion()
        view.zoom(factor: factor)
    }

    /// Runs a key command. Animated commands fly; the step commands change the view at once.
    func apply(_ command: ViewCommand) {
        let stepRadians: Float = 15 * Float.pi / 180
        switch command {
        case .preset(let preset):
            // Unlike Blender the numpad views frame the whole scene: a camera view's pivot can lie inside it.
            var target = ViewCamera.aligned(preset, like: view)
            target.frame(bounds)
            fly(to: target, duration: 0.35)
        case .stepOrbit(let yaw, let pitch):
            orbit(deltaYaw: Float(yaw) * stepRadians, deltaPitch: Float(pitch) * stepRadians)
        case .stepPan(let x, let y):
            let step: SIMD2<Float> = SIMD2(Float(x), Float(y))
            let deltaPixels: SIMD2<Float> = step * 60
            pan(deltaPixels: deltaPixels, viewportHeight: 800)
        case .cameraView:
            if !cameras.isEmpty {
                goToCamera(activeCameraIndex ?? 0)
            }
        case .previousCamera:
            previousCamera()
        case .nextCamera:
            nextCamera()
        case .frameAll:
            var target = view
            target.frame(bounds)
            fly(to: target, duration: 0.4)
        case .togglePlayback:
            togglePlayback()
        case .cycleDisplayMode:
            displayMode = displayMode.next
        }
    }

    /// Resets to the camera view when there are cameras, otherwise to the framing of the scene.
    func resetView() {
        if !cameras.isEmpty {
            goToCamera(0)
        } else {
            fly(to: ViewCamera.framing(bounds, upAxis: upAxis))
        }
    }

    // MARK: - Animation

    /// Animates the view to `target`. Reduce Motion jumps to the target instead.
    func fly(to target: ViewCamera, duration: Double = 0.6) {
        stopMotion()
        cursorOrbitAnchor = nil
        if NSWorkspace.shared.accessibilityDisplayShouldReduceMotion {
            view = target
            return
        }
        let start = view
        motionTask = Task { await self.runFlight(from: start, to: target, duration: duration) }
    }

    private func stopMotion() {
        motionTask?.cancel()
        motionTask = nil
        if isPlaying {
            isPlaying = false
        }
    }

    private func runFlight(from start: ViewCamera, to target: ViewCamera, duration: Double) async {
        let started = ContinuousClock.now
        while !Task.isCancelled {
            let elapsed = Self.seconds(ContinuousClock.now - started)
            let progress = Float(Swift.min(elapsed / duration, 1))
            if progress >= 1 { break }
            view = ViewFlight.transition(from: start, to: target, progress: progress)
            try? await Task.sleep(for: .milliseconds(16))
        }
        if !Task.isCancelled {
            view = target
        }
    }

    private func runPlayback(from startPosition: Double, flight: ViewFlight) async {
        let last = Double(flight.count - 1)
        let started = ContinuousClock.now
        while !Task.isCancelled {
            let elapsed = Self.seconds(ContinuousClock.now - started)
            let position = Swift.min(startPosition + elapsed / Self.secondsPerSegment, last)
            flightPosition = position
            activeCameraIndex = Int(position.rounded())
            view = flight.camera(at: Float(position))
            if position >= last {
                break
            }
            try? await Task.sleep(for: .milliseconds(16))
        }
        if !Task.isCancelled {
            isPlaying = false
            motionTask = nil
        }
    }

    private static func seconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) * 1e-18
    }

    // MARK: - Fly mode

    /// Enters or leaves fly mode. Entering is ignored while editing is active. Leaving keeps the view where it is.
    func setFlying(_ flying: Bool) {
        if flying {
            guard isEditingActive else { return }
            stopMotion()
            cursorOrbitAnchor = nil
            isFlying = true
        } else {
            isFlying = false
            flySpeedNoticeTask?.cancel()
            flySpeedNotice = nil
        }
    }

    func toggleFlying() {
        setFlying(!isFlying)
    }

    /// Mouse movement in fly mode turns the view. `deltaX` and `deltaY` are in points, y downward.
    /// Moving right turns the view right and moving down looks down, whatever the up axis.
    func look(deltaX: Float, deltaY: Float) {
        guard isFlying else { return }
        let radiansPerPoint: Float = 0.003
        let up: SIMD3<Float> = upAxis.vector
        let forward: SIMD3<Float> = view.forward
        let right: SIMD3<Float> = view.right
        // Derivative of the forward axis for positive yaw is up × forward. Its component along right gives the
        // direction of positive yaw as seen by the user.
        let turn: SIMD3<Float> = SIMD3(up.y * forward.z - up.z * forward.y,
                                       up.z * forward.x - up.x * forward.z,
                                       up.x * forward.y - up.y * forward.x)
        let towardRight: Float = turn.x * right.x + turn.y * right.y + turn.z * right.z
        let yawSign: Float = towardRight < 0 ? -1 : 1
        view.lookAround(deltaYaw: deltaX * radiansPerPoint * yawSign, deltaPitch: deltaY * radiansPerPoint)
    }

    /// Moves the eye at the fly speed. `movement` has right, up and forward components in -1...1 (already normalised
    /// by the caller). `boost` multiplies the speed (Shift 4, Option 1/4). `seconds` is the elapsed time.
    func flyStep(movement: SIMD3<Float>, boost: Float, seconds: Double) {
        guard isFlying else { return }
        let extent: SIMD3<Float> = bounds.extent
        let halfDiagonal: Float = 0.5 * (extent * extent).sum().squareRoot()
        let radius: Float = halfDiagonal > 0 ? halfDiagonal : 1
        let distance: Float = 0.5 * radius * flySpeedScale * boost * Float(seconds)
        view.moveEye(right: movement.x * distance, up: movement.y * distance, forward: movement.z * distance)
    }

    /// Changes the base fly speed by the scroll amount (about 10 % per unit) and shows it for about a second.
    func changeFlySpeed(scrollAmount: Float) {
        guard isFlying else { return }
        let factor: Float = exp(scrollAmount * 0.1)
        flySpeedScale = Swift.min(Swift.max(flySpeedScale * factor, 0.01), 100)
        flySpeedNotice = "Fly speed " + String(format: "%.2f", flySpeedScale) + "\u{00D7}"
        flySpeedNoticeTask?.cancel()
        flySpeedNoticeTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(1))
            guard !Task.isCancelled else { return }
            self?.flySpeedNotice = nil
        }
    }
}

extension UpAxis {
    var displayName: String {
        switch self {
        case .negativeY: "\u{2212}Y Up (COLMAP)"
        case .positiveY: "+Y Up"
        case .positiveZ: "+Z Up"
        }
    }
}
