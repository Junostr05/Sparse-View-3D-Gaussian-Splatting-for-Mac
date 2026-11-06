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

    /// The scene shown by the viewport. Replaced during training by `replaceCloud(_:)`.
    private(set) var scene: LoadedScene

    /// Incremented on every `replaceCloud(_:)`. `ViewportView` reads it so SwiftUI re-renders the view.
    private(set) var cloudRevision = 0

    var orbit: OrbitCamera
    var shDegreeLimit: Int
    var background: Background = .black
    var isInspectorPresented = true
    var renderState: RenderState = .preparing

    var upAxis: UpAxis {
        didSet {
            guard upAxis != oldValue else { return }
            orbit = OrbitCamera.framing(bounds, upAxis: upAxis)
        }
    }

    /// Latest frame statistics, assigned at most every 0.5 s.
    private(set) var stats: FrameStats?

    @ObservationIgnored private let bounds: Bounds
    @ObservationIgnored private var lastStatsTime: ContinuousClock.Instant?

    init(scene: LoadedScene) {
        self.scene = scene
        let bounds = scene.cloud.robustBounds()
        self.bounds = bounds
        self.upAxis = .negativeY
        self.orbit = OrbitCamera.framing(bounds, upAxis: .negativeY)
        self.shDegreeLimit = scene.cloud.shDegree
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

    /// Replaces the displayed Gaussians, for example with a training preview. The orbit camera keeps
    /// its framing. Before the renderer exists the new cloud is used when it is created.
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

    func resetView() {
        orbit = OrbitCamera.framing(bounds, upAxis: upAxis)
    }

    func report(_ newStats: FrameStats) {
        let now = ContinuousClock.now
        if let last = lastStatsTime, now - last < .milliseconds(500) { return }
        lastStatsTime = now
        stats = newStats
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
