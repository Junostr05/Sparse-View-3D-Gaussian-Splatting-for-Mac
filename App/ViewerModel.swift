import AppKit
import Foundation
import Observation
import OSLog
import SplatCore
import SplatIO
import SplatMetal

private let logger = Logger(subsystem: "io.github.junostr05.SparseSplat", category: "Viewer")

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
    private(set) var scene: LoadedScene

    /// Incremented on every `replaceCloud(_:)`. `ViewportView` reads it so SwiftUI re-renders the view.
    private(set) var cloudRevision = 0

    /// The camera the viewport renders. Changed by input, by flights and by playback.
    private(set) var view: ViewCamera
    var shDegreeLimit: Int
    var background: Background = .black
    var isInspectorPresented = true
    var renderState: RenderState = .preparing

    /// Estimated cameras of the project, in the order the viewer lists them. Empty for a plain .ply scene.
    let cameras: [NamedCamera]
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

    @ObservationIgnored private let bounds: Bounds
    @ObservationIgnored private var lastStatsTime: ContinuousClock.Instant?
    /// One view per camera, built from `cameras` and the current up axis.
    @ObservationIgnored private var cameraViews: [ViewCamera] = []
    /// Smooth path through `cameraViews`. `nil` when there are no cameras.
    @ObservationIgnored private var flight: ViewFlight?
    /// The running flight or playback task. Cancelled by every new request and by manual input.
    @ObservationIgnored private var motionTask: Task<Void, Never>?

    init(scene: LoadedScene, upAxis: UpAxis = .negativeY, cameras: [NamedCamera] = []) {
        self.scene = scene
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

    /// Replaces the displayed Gaussians, for example with a training preview. The camera keeps
    /// its view. Before the renderer exists the new cloud is used when it is created.
    func replaceCloud(_ cloud: GaussianCloud) {
        scene = LoadedScene(cloud: cloud, removedNonFinite: 0)
        shDegreeLimit = min(shDegreeLimit, cloud.shDegree)
        if case .ready(let session) = renderState {
            do throws(MetalError) {
                try session.renderer.setCloud(cloud)
            } catch {
                logger.error("Replacing the cloud failed: \(String(describing: error))")
            }
        }
        cloudRevision += 1
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
        cameraViews = cameras.indices.map { cameraView(at: $0) }
        flight = cameraViews.isEmpty ? nil : ViewFlight(keyframes: cameraViews)
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

    /// Orbit from a mouse or trackpad gesture. Cancels any flight or playback.
    func orbit(deltaYaw: Float, deltaPitch: Float) {
        stopMotion()
        view.orbit(deltaYaw: deltaYaw, deltaPitch: deltaPitch)
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
