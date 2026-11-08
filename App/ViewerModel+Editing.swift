import Foundation
import OSLog
import SplatCore
import SplatIO
import SplatMetal

private let editingLogger = Logger(subsystem: "io.github.junostr05.SparseSplat", category: "Editing")

/// What a brush stroke does to the selection.
enum BrushMode: String, CaseIterable, Identifiable {
    case add
    case subtract

    var id: String { rawValue }

    var title: String {
        switch self {
        case .add: "Add"
        case .subtract: "Subtract"
        }
    }
}

/// A selection or history command. Sent by the keyboard in edit mode and by the brush controls.
enum EditCommand: Sendable {
    case deleteSelection
    case selectAll
    case clearSelection
    case invertSelection
    case undo
    case redo
    case toggleFrontOnly
}

/// Non-destructive editing: the brush selection, deletion through `EditHistory`, undo and redo,
/// and export of the edited cloud. The displayed cloud is always `baseCloud` minus the deletions.
extension ViewerModel {
    /// Edit mode is on and the model allows it (no training preview is shown, no video export is rendering).
    var isEditingActive: Bool { isEditing && canEdit }

    /// Editing is possible at all. False while a training preview is shown, and while a flight video is written:
    /// a deletion would change the frames that follow it.
    var canEdit: Bool { isEditingAllowed && videoExport?.state != .running }

    /// Number of Gaussians in the displayed cloud.
    var displayedCount: Int { scene.cloud.count }

    var selectedCount: Int { selection.selectedCount }

    var deletedCount: Int { history.deleted.selectedCount }

    var canUndo: Bool { history.canUndo }

    var canRedo: Bool { history.canRedo }

    // MARK: - Brush

    /// Adds the Gaussians under a brush stroke to the selection (`.add`) or removes them from it.
    ///
    /// - Parameters:
    ///   - stroke: brush centres in drawable pixels, top-left origin: the pixel space of `view.camera(width:height:)`.
    ///   - viewSizePixels: drawable size in pixels, the image size the view renders.
    ///   - backingScale: pixels per point, used to convert `brushRadius` to pixels.
    ///   - mode: add or subtract. The view passes `.subtract` while Shift is held.
    func applyBrush(stroke: [SIMD2<Float>], viewSizePixels: SIMD2<Float>, backingScale: Float, mode: BrushMode) {
        guard isEditingActive, !stroke.isEmpty else { return }
        let width: Int = Int(viewSizePixels.x.rounded())
        let height: Int = Int(viewSizePixels.y.rounded())
        guard width > 0, height > 0 else { return }
        let camera: Camera = view.camera(width: width, height: height)
        let radiusPixels: Float = Float(brushRadius) * backingScale
        let frontOnly: FrontOnlySettings? = brushFrontOnly
            ? FrontOnlySettings(opacities: displayOpacities())
            : nil
        let hit: SelectionMask = BrushSelector.select(positions: scene.cloud.positions,
                                                      camera: camera,
                                                      stroke: stroke,
                                                      radiusPixels: radiusPixels,
                                                      frontOnly: frontOnly)
        switch mode {
        case .add:
            selection.formUnion(hit)
        case .subtract:
            selection.formSubtraction(hit)
        }
    }

    // MARK: - Selection

    func selectAll() {
        guard isEditingActive else { return }
        selection = SelectionMask.all(count: displayedCount)
    }

    func clearSelection() {
        guard isEditingActive else { return }
        selection = SelectionMask(count: displayedCount)
    }

    func invertSelection() {
        guard isEditingActive else { return }
        selection.invert()
    }

    /// Runs a selection or history command. Commands are ignored when editing is not allowed.
    func performEdit(_ command: EditCommand) {
        switch command {
        case .deleteSelection:
            deleteSelection()
        case .selectAll:
            selectAll()
        case .clearSelection:
            clearSelection()
        case .invertSelection:
            invertSelection()
        case .undo:
            undo()
        case .redo:
            redo()
        case .toggleFrontOnly:
            brushFrontOnly.toggle()
        }
    }

    /// Sigmoid opacity of each Gaussian in the displayed cloud, cached until the displayed cloud changes.
    private func displayOpacities() -> [Float] {
        if let cached = displayOpacityCache {
            return cached
        }
        let logits: [Float] = scene.cloud.opacityLogits
        var opacities: [Float] = []
        opacities.reserveCapacity(logits.count)
        for logit in logits {
            let negated: Float = -logit
            let opacity: Float = 1 / (1 + exp(negated))
            opacities.append(opacity)
        }
        displayOpacityCache = opacities
        return opacities
    }

    // MARK: - Deletion and history

    /// Deletes the selected Gaussians. The selection is mapped to base indices, recorded as one undo step
    /// and cleared. The displayed cloud is rebuilt on the GPU; the edit state is kept.
    func deleteSelection() {
        guard canEdit, !selection.isEmpty else { return }
        var baseSelection = SelectionMask(count: baseCloud.count)
        for displayIndex in selection.selectedIndices() {
            baseSelection[displayIndexToBase[displayIndex]] = true
        }
        history.delete(baseSelection)
        commitHistoryChange()
    }

    func undo() {
        guard canEdit, history.canUndo else { return }
        history.undo()
        commitHistoryChange()
    }

    func redo() {
        guard canEdit, history.canRedo else { return }
        history.redo()
        commitHistoryChange()
    }

    /// Applies stored deletions to the base cloud without undo steps. Indices outside the base are ignored.
    func restoreDeletions(_ deleted: [UInt32]) {
        let count: Int = baseCloud.count
        var mask = SelectionMask(count: count)
        for index in deleted where Int(index) < count {
            mask[Int(index)] = true
        }
        history = EditHistory(baseCount: count, deleted: mask)
        displayIndexToBase = history.baseIndices()
        rebuildDisplayCloud()
    }

    /// Stores the deletions, rebuilds the displayed cloud and notifies the persistence hook.
    /// Any structural change clears the selection, because display indices change with it.
    private func commitHistoryChange() {
        displayIndexToBase = history.baseIndices()
        rebuildDisplayCloud()
        let deleted: [UInt32] = history.deleted.selectedIndices().map { UInt32($0) }
        onDeletionsChanged?(baseCloud.count, deleted)
    }

    /// Builds the displayed cloud from the base and the deletions, and clears the selection.
    private func rebuildDisplayCloud() {
        let display: GaussianCloud
        if history.deleted.isEmpty {
            display = baseCloud
        } else {
            do {
                display = try baseCloud.removing(history.deleted)
            } catch {
                editingLogger.error("Removing the deleted Gaussians failed: \(String(describing: error))")
                return
            }
        }
        presentCloud(display, removedNonFinite: scene.removedNonFinite)
        selection = SelectionMask(count: display.count)
    }

    // MARK: - Base and display clouds

    /// Starts a new edit state for `cloud`, which becomes the base: no deletions, no selection, editing off.
    func resetEditing(baseCloud cloud: GaussianCloud) {
        baseCloud = cloud
        history = EditHistory(baseCount: cloud.count)
        displayIndexToBase = Array(0..<cloud.count)
        selection = SelectionMask(count: cloud.count)
        isEditing = false
        brushCursor = nil
    }

    /// Makes `cloud` the displayed cloud and uploads it to the renderer when the renderer exists.
    /// The camera and the edit state are untouched.
    func presentCloud(_ cloud: GaussianCloud, removedNonFinite: Int) {
        scene = LoadedScene(cloud: cloud, removedNonFinite: removedNonFinite)
        displayOpacityCache = nil
        if case .ready(let session) = renderState {
            do throws(MetalError) {
                try session.renderer.setCloud(cloud)
            } catch {
                editingLogger.error("Uploading the cloud failed: \(String(describing: error))")
            }
        }
        cloudRevision += 1
    }

    // MARK: - Export

    /// Encodes the displayed cloud (with the deletions applied) in `format`, off the main actor.
    func exportData(_ format: ExportFormat) async throws -> Data {
        let cloud: GaussianCloud = scene.cloud
        return try await Task.detached(priority: .userInitiated) {
            try CloudExporter.export(cloud, as: format)
        }.value
    }

    /// Encodes the displayed cloud (with the deletions applied) as a self-contained web page, off the main actor.
    /// The page starts from the current view and uses the viewer's background colour.
    func exportWebPage(maxGaussians: Int, title: String) async throws -> WebViewerExport {
        let cloud: GaussianCloud = scene.cloud
        let camera = WebViewerCamera(position: view.eye, target: view.pivot, up: upAxis.vector, fovY: view.fovY)
        let options = WebViewerOptions(maxGaussians: maxGaussians,
                                       title: title,
                                       camera: camera,
                                       background: background.color,
                                       upAxis: upAxis)
        return try await Task.detached(priority: .userInitiated) {
            try WebViewerExporter.export(cloud: cloud, options: options)
        }.value
    }
}
