import SwiftUI
import SplatCore
import SplatIO

/// The window of a `.gsproj` project: image list, live viewport and the training inspector.
struct ProjectWindow: View {
    let document: Binding<ProjectDocument>
    @State private var model = ProjectModel()
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
            ToolbarItem {
                Toggle(isOn: $isInspectorPresented) {
                    Label("Inspector", systemImage: "sidebar.trailing")
                }
            }
        }
        .inspector(isPresented: $isInspectorPresented) {
            ProjectInspector(document: document, model: model)
                .inspectorColumnWidth(min: 240, ideal: 300, max: 380)
        }
        .task {
            await model.restoreCheckpoint(contents)
        }
    }

    /// The viewport is content (no glass). Before a model exists, an empty state explains what to do.
    @ViewBuilder
    private func detail(for contents: ProjectContents) -> some View {
        if let viewer = model.viewer {
            ViewerContent(model: viewer)
        } else if contents.manifest.imageFiles.isEmpty {
            ContentUnavailableView(
                "Import a COLMAP or ARKit capture",
                systemImage: "square.and.arrow.down",
                description: Text("""
                    COLMAP: a folder with sparse/0 (cameras.bin, images.bin, optional points3D.bin) and an images folder.
                    ARKit: a folder with capture.json and the images it names.
                    To open a single .ply or .splat scene, use File > Open.
                    """)
            )
        } else {
            ContentUnavailableView(
                "No model yet",
                systemImage: "cube.transparent",
                description: Text("Press Train in the inspector to reconstruct the scene.")
            )
        }
    }
}
