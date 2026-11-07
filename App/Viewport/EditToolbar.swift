import SwiftUI
import SplatIO

/// Toggles edit mode. Disabled while the model does not allow editing (during training).
struct EditModeToggle: View {
    @Bindable var model: ViewerModel

    var body: some View {
        Toggle(isOn: $model.isEditing) {
            Label("Edit Mode", systemImage: "paintbrush")
        }
        .toggleStyle(.button)
        .disabled(!model.canEdit)
        .help("Paint a selection with the brush (left drag). Delete removes it without changing the source file.")
    }
}

/// Opens the export sheet for the displayed cloud, with the deletions applied.
struct ExportButton: View {
    let model: ViewerModel
    /// The file name without extension, offered by the save panel.
    let defaultName: String
    /// Receives a message when the export or the save fails.
    let onFailure: (String) -> Void

    @State private var isSheetPresented = false

    var body: some View {
        Button {
            isSheetPresented = true
        } label: {
            Label("Export\u{2026}", systemImage: "square.and.arrow.up")
        }
        .help("Export the scene as .ply, .splat or .spz")
        .sheet(isPresented: $isSheetPresented) {
            ExportSheet(model: model, defaultName: defaultName, onFailure: onFailure)
        }
    }
}
