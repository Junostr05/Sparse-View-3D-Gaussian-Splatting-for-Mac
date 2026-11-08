import SwiftUI
import SplatCore
import SplatIO

/// The window of a `.gsproj` project: image list, live viewport and the training inspector.
struct ProjectWindow: View {
    let document: Binding<ProjectDocument>
    /// The package name without extension, used as the default export name.
    let name: String
    @State private var model = ProjectModel()
    @State private var pipeline = PipelineModel()
    @State private var isInspectorPresented = true

    var body: some View {
        let contents: ProjectContents = document.wrappedValue.contents
        NavigationSplitView {
            List(contents.manifest.imageFiles, id: \.self) { name in
                Label(name, systemImage: "photo")
            }
            .navigationSplitViewColumnWidth(min: 180, ideal: 220)
        } detail: {
            detail(for: contents)
        }
        .toolbar {
            if let viewer = model.viewer {
                ToolbarItem {
                    EditModeToggle(model: viewer)
                }
                ToolbarItem {
                    ExportButton(model: viewer, defaultName: name, onFailure: { model.errorMessage = $0 })
                }
            }
            ToolbarItem {
                Toggle(isOn: $isInspectorPresented) {
                    Label("Inspector", systemImage: "sidebar.trailing")
                }
            }
        }
        .inspector(isPresented: $isInspectorPresented) {
            ProjectInspector(document: document, model: model, pipeline: pipeline)
                .inspectorColumnWidth(min: 240, ideal: 300, max: 380)
        }
        .task {
            await model.restoreCheckpoint(contents, document: document)
        }
        .onDisappear {
            pipeline.close()
        }
        .commandPalette { openWindow in
            CommandRegistry.projectWindowCommands(viewer: model.viewer, project: model, document: document,
                                                  openWindow: openWindow)
        }
    }

    /// The viewport is content (no glass). Until a model exists, the guided card names the current step, and the
    /// frame strip of pending frames sits below it.
    @ViewBuilder
    private func detail(for contents: ProjectContents) -> some View {
        if let viewer = model.viewer {
            ViewerContent(model: viewer)
        } else if contents.manifest.imageFiles.isEmpty && pipeline.hasCandidates {
            VStack(spacing: 0) {
                GuidedStartView(document: document, model: model, pipeline: pipeline)
                    .frame(maxHeight: 340)
                Divider()
                FrameStripView(pipeline: pipeline)
            }
        } else {
            GuidedStartView(document: document, model: model, pipeline: pipeline)
        }
    }
}
