import Foundation
import Metal
import SplatCore
import SplatMetal

/// Which per-Gaussian colour set a display mode needs.
enum DisplayColorKey: Hashable, Sendable {
    case points
    case shapes
    case heatmap(HeatmapMetric)

    var isHeatmap: Bool {
        if case .heatmap = self { return true }
        return false
    }
}

/// The colours of one `DisplayColorKey`, computed for the cloud of `revision`.
struct DisplayColorSet {
    let revision: Int
    let colors: [SIMD4<Float>]
}

extension ViewerDisplayMode {
    var title: String {
        switch self {
        case .splat: "Splat"
        case .points: "Points"
        case .depth: "Depth"
        case .shapes: "Shapes"
        case .heatmap: "Heatmap"
        }
    }
}

extension HeatmapMetric {
    var title: String {
        switch self {
        case .size: "Size"
        case .opacity: "Opacity"
        case .isolation: "Isolation"
        }
    }
}

/// Display modes of the viewport. Colour sets are computed off the main actor when first needed, for the cloud
/// that is displayed at the time. Until a set is ready the viewport shows the splats; the heatmap shows the metric
/// it showed last.
extension ViewerModel {
    /// The colour set the current mode needs. `nil` for the splat and depth modes, which use no colour override.
    var displayColorKey: DisplayColorKey? {
        switch displayMode {
        case .splat, .depth: nil
        case .points: .points
        case .shapes: .shapes
        case .heatmap: .heatmap(heatmapMetric)
        }
    }

    /// True while the colour set of the current mode is being computed for the displayed cloud.
    var isComputingDisplayColors: Bool {
        guard let key = displayColorKey, let revision = pendingDisplayColors[key] else { return false }
        return revision == cloudRevision
    }

    /// Radius of the scene for the normalisation of the heatmap: half the diagonal of the framing bounds, or 1.
    var sceneRadius: Float {
        let extent: SIMD3<Float> = bounds.extent
        let halfDiagonal: Float = 0.5 * (extent * extent).sum().squareRoot()
        return halfDiagonal > 0 ? halfDiagonal : 1
    }

    /// The display of the next frame. Requests the colour set of the mode when it is missing.
    func frameDisplay() -> ViewerDisplay {
        guard let key = displayColorKey else {
            return ViewerDisplay(mode: .splat)
        }
        requestDisplayColors(key)
        if let set = displayColorSets[key], set.revision == cloudRevision {
            return makeDisplay(colors: set.colors)
        }
        if key.isHeatmap, let last = lastHeatmapSet, last.set.revision == cloudRevision {
            return makeDisplay(colors: last.set.colors)
        }
        return ViewerDisplay(mode: .splat)
    }

    /// Draws one frame of the viewer with `renderer`, in the display mode in effect, the SH limit and the background.
    /// The viewport and the image and video export both draw through here, so an export shows what the viewport
    /// shows (without the overlays). Synchronous: the GPU has finished when this returns.
    @discardableResult
    func renderFrame(with renderer: SplatRenderer, camera: Camera, to texture: MTLTexture) throws(MetalError) -> FrameStats {
        // The crop clip is viewer-only: it is set here and nowhere in training.
        renderer.setCrop(cropClip)
        var settings = RasterSettings()
        settings.shDegreeLimit = shDegreeLimit
        settings.background = background.color
        if displayMode == .depth {
            return try renderer.renderDepthColorized(camera: camera, settings: settings, to: texture)
        }
        try renderer.setDisplay(frameDisplay())
        return try renderer.render(camera: camera, settings: settings, to: texture)
    }

    /// Waits until the colour set of the current display mode is ready for the displayed cloud. Frames rendered
    /// before that would fall back to the splats, so an export calls this before its first frame. Returns at once in
    /// the splat and depth modes.
    func awaitDisplayColors() async {
        while let key = displayColorKey, displayColorSets[key]?.revision != cloudRevision {
            _ = frameDisplay()
            try? await Task.sleep(for: .milliseconds(20))
            if Task.isCancelled { return }
        }
    }

    private func makeDisplay(colors: [SIMD4<Float>]) -> ViewerDisplay {
        ViewerDisplay(mode: displayMode, pointRadiusPixels: Float(pointSize), colorOverride: colors)
    }

    /// Starts computing the colour set for `key` for the displayed cloud, unless it is ready or on its way.
    private func requestDisplayColors(_ key: DisplayColorKey) {
        let revision: Int = cloudRevision
        if let set = displayColorSets[key], set.revision == revision {
            return
        }
        if pendingDisplayColors[key] == revision {
            return
        }
        pendingDisplayColors[key] = revision
        let cloud: GaussianCloud = scene.cloud
        let radius: Float = sceneRadius
        Task { [weak self] in
            let colors: [SIMD4<Float>] = await Task.detached(priority: .userInitiated) {
                ViewerModel.displayColors(for: key, cloud: cloud, sceneRadius: radius)
            }.value
            self?.finishDisplayColors(key, revision: revision, colors: colors)
        }
    }

    /// The colour set of `key`. Pure, so it runs off the main actor.
    nonisolated static func displayColors(for key: DisplayColorKey, cloud: GaussianCloud,
                                          sceneRadius: Float) -> [SIMD4<Float>] {
        switch key {
        case .points:
            return ViewerColors.pointColors(cloud)
        case .shapes:
            return ViewerColors.shapeColors(count: cloud.count)
        case .heatmap(let metric):
            return ViewerColors.heatmap(cloud, metric: metric, sceneRadius: sceneRadius)
        }
    }

    private func finishDisplayColors(_ key: DisplayColorKey, revision: Int, colors: [SIMD4<Float>]) {
        if pendingDisplayColors[key] == revision {
            pendingDisplayColors[key] = nil
        }
        // The cloud changed while the colours were computed: discard them. The next frame requests the new set.
        guard revision == cloudRevision else { return }
        let set = DisplayColorSet(revision: revision, colors: colors)
        // Memory: only sets of the current revision are kept, and of the heatmap only the metric in use.
        displayColorSets = displayColorSets.filter { entry in
            entry.value.revision == revision && !(entry.key.isHeatmap && entry.key != key)
        }
        displayColorSets[key] = set
        if key.isHeatmap {
            lastHeatmapSet = (key: key, set: set)
        }
        displayColorRevision += 1
    }
}
