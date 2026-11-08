import SwiftUI
import SplatCore
import SplatIO

struct SceneWindow: View {
    /// The file name without extension, used as the default export name.
    let name: String
    @State private var model: ViewerModel
    @State private var exportError: String?

    init(scene: LoadedScene, name: String) {
        self.name = name
        _model = State(initialValue: ViewerModel(scene: scene))
    }

    var body: some View {
        ViewerContent(model: model)
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
                    EditModeToggle(model: model)
                }
                ToolbarItem {
                    CleanUpButton(model: model)
                }
                ToolbarItem {
                    ExportButton(model: model, defaultName: name, onFailure: { exportError = $0 })
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
                    ViewerShortcutsButton()
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
            .alert("Export Failed", isPresented: Binding(
                get: { exportError != nil },
                set: { if !$0 { exportError = nil } }
            )) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(exportError ?? "")
            }
            .commandPalette { openWindow in
                CommandRegistry.sceneWindowCommands(viewer: model, openWindow: openWindow)
            }
    }
}
