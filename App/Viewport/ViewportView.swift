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
        // Edit mode changes how the mouse and the keys are read, so it is part of the update too.
        nsView.isEditing = model.isEditingActive
        nsView.needsDisplay = true
    }
}
