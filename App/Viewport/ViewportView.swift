import SwiftUI
import MetalKit

struct ViewportView: NSViewRepresentable {
    let model: ViewerModel
    let session: MetalSession

    func makeCoordinator() -> ViewportCoordinator {
        ViewportCoordinator(model: model, session: session)
    }

    func makeNSView(context: Context) -> SplatMTKView {
        let view = SplatMTKView(device: session.context.device)
        view.delegate = context.coordinator
        context.coordinator.attach(to: view)
        return view
    }

    func updateNSView(_ nsView: SplatMTKView, context: Context) {
        // Read everything that affects the image so SwiftUI observation re-runs this method.
        _ = model.view
        _ = model.shDegreeLimit
        _ = model.background
        _ = model.cloudRevision
        // Display mode, point size, heatmap metric and freshly computed colour sets change the image.
        _ = model.displayMode
        _ = model.pointSize
        _ = model.heatmapMetric
        _ = model.displayColorRevision
        // Edit mode changes how the mouse and the keys are read, so it is part of the update too.
        nsView.isEditing = model.isEditingActive
        // The viewport tools claim clicks while one is active; the crop handles are read from the model too.
        _ = model.activeTool
        _ = model.crop
        nsView.isToolActive = model.activeTool != .none
        // Fly mode is owned by the model (menu, inspector, Shift + `); the view starts or stops the capture.
        _ = model.isFlying
        nsView.isFlying = model.isFlying
        nsView.needsDisplay = true
    }

    static func dismantleNSView(_ nsView: SplatMTKView, coordinator: ViewportCoordinator) {
        // Restore the cursor if the view goes away in fly mode, and keep the model in step.
        nsView.stopFlyingForRemoval()
        coordinator.model.setFlying(false)
    }
}
