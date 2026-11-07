import SwiftUI

/// The 3D viewport for a `ViewerModel`, including its preparing and unavailable states.
/// Shared by the viewer window and the project window. It is content, so it carries no glass.
struct ViewerContent: View {
    let model: ViewerModel

    var body: some View {
        // Read here so that the brush bar appears and disappears with edit mode.
        let isEditing: Bool = model.isEditingActive
        Group {
            switch model.renderState {
            case .preparing:
                Color.clear
            case .ready(let session):
                ViewportView(model: model, session: session)
                    .accessibilityLabel("3D viewport")
                    .overlay {
                        EditOverlay(model: model)
                    }
                    .overlay(alignment: .bottom) {
                        if isEditing {
                            BrushControls(model: model)
                                .padding(.bottom, 16)
                        }
                    }
            case .unavailable:
                ContentUnavailableView(
                    "Metal Not Available",
                    systemImage: "exclamationmark.triangle",
                    description: Text("This Mac cannot render the scene because no Metal device could be set up.")
                )
            }
        }
        .task(id: ObjectIdentifier(model)) {
            model.prepareRenderer()
        }
    }
}
