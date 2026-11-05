import SwiftUI
import SplatCore
import SplatIO

struct SceneWindow: View {
    @State private var model: ViewerModel

    init(scene: LoadedScene) {
        _model = State(initialValue: ViewerModel(scene: scene))
    }

    var body: some View {
        viewport
            .task { model.prepareRenderer() }
            .toolbar {
                ToolbarItem {
                    Button {
                        model.resetView()
                    } label: {
                        Label("Reset View", systemImage: "arrow.counterclockwise")
                    }
                    .keyboardShortcut("0", modifiers: .command)
                }
                ToolbarItem {
                    Menu {
                        Picker("Up Axis", selection: $model.upAxis) {
                            ForEach(UpAxis.allCases, id: \.self) { axis in
                                Text(axis.displayName).tag(axis)
                            }
                        }
                        .pickerStyle(.inline)
                    } label: {
                        Label("Up Axis", systemImage: "arrow.up.and.down.and.arrow.left.and.right")
                    }
                }
                ToolbarItem {
                    Toggle(isOn: $model.isInspectorPresented) {
                        Label("Inspector", systemImage: "sidebar.trailing")
                    }
                }
            }
            .inspector(isPresented: $model.isInspectorPresented) {
                SceneInspector(model: model)
                    .inspectorColumnWidth(min: 240, ideal: 280, max: 360)
            }
    }

    @ViewBuilder
    private var viewport: some View {
        switch model.renderState {
        case .preparing:
            Color.clear
        case .ready(let session):
            ViewportView(model: model, session: session)
                .accessibilityLabel("3D viewport")
        case .unavailable:
            ContentUnavailableView(
                "Metal Not Available",
                systemImage: "exclamationmark.triangle",
                description: Text("This Mac cannot render the scene because no Metal device could be set up.")
            )
        }
    }
}
