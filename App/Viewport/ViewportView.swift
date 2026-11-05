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
        _ = model.orbit
        _ = model.shDegreeLimit
        _ = model.background
        nsView.needsDisplay = true
    }
}
