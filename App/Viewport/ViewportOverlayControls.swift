import SwiftUI

/// Switches for the viewport overlays: the ground grid, the axes, the estimated cameras and their names.
/// The rows go into the Viewport section of the project inspector. The camera rows appear only when the
/// project has cameras, and the names need the cameras to be shown.
struct ViewportOverlayControls: View {
    @Bindable var model: ViewerModel

    var body: some View {
        Toggle("Grid", isOn: $model.showsGrid)
        Toggle("Axes", isOn: $model.showsAxes)
        if !model.cameras.isEmpty {
            Toggle("Cameras", isOn: $model.showsCameras)
            Toggle("Camera Names", isOn: $model.showsCameraNames)
                .disabled(!model.showsCameras)
        }
    }
}
