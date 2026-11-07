import SwiftUI
import SplatCore
import SplatIO

struct SceneWindow: View {
    @State private var model: ViewerModel

    init(scene: LoadedScene) {
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
    }
}
