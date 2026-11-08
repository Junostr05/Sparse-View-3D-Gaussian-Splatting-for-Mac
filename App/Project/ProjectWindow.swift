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
    /// True while a drag hovers over the window.
    @State private var isDropTargeted = false
    /// A drop that replaces the current capture, waiting for the user's confirmation.
    @State private var pendingDrop: DropAction?

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
        .dropDestination(for: URL.self) { urls, _ in
            handleDrop(urls)
            return true
        } isTargeted: { isDropTargeted = $0 }
        .overlay {
            if isDropTargeted {
                ProjectDropHighlight()
            }
        }
        .alert("Replace the current capture?", isPresented: isConfirmingReplace) {
            Button("Replace", role: .destructive) {
                if let action = pendingDrop {
                    pendingDrop = nil
                    ProjectDropRouter.perform(action, replacing: true, document: document, model: model, pipeline: pipeline)
                }
            }
            Button("Cancel", role: .cancel) {
                pendingDrop = nil
            }
        } message: {
            Text("The images, poses and trained models in this project are replaced by the dropped items. Training settings stay.")
        }
    }

    private var isConfirmingReplace: Binding<Bool> {
        Binding(get: { pendingDrop != nil },
                set: { isPresented in
                    if !isPresented {
                        pendingDrop = nil
                    }
                })
    }

    /// Classifies a drop. An empty project imports at once; a project with images asks before it is replaced.
    private func handleDrop(_ urls: [URL]) {
        let action: DropAction = ProjectDropRouter.classify(urls)
        let decision: ProjectDropRouter.Decision = ProjectDropRouter.decide(
            action,
            hasCapture: !document.wrappedValue.contents.manifest.imageFiles.isEmpty,
            isTraining: model.isTraining
        )
        switch decision {
        case .perform(let performed):
            ProjectDropRouter.perform(performed, replacing: false, document: document, model: model, pipeline: pipeline)
        case .confirmReplace(let pending):
            pendingDrop = pending
        case .refuse(let message):
            model.errorMessage = message
        }
    }

    /// The viewport is content (no glass). Before a model exists, the frame strip shows pending frames,
    /// and an empty state explains what to do.
    @ViewBuilder
    private func detail(for contents: ProjectContents) -> some View {
        if let viewer = model.viewer {
            ViewerContent(model: viewer)
        } else if contents.manifest.imageFiles.isEmpty && pipeline.hasCandidates {
            FrameStripView(pipeline: pipeline)
        } else if contents.manifest.imageFiles.isEmpty {
            ContentUnavailableView(
                "Add a photo folder, a video or a pose capture",
                systemImage: "square.and.arrow.down",
                description: Text("""
                    Photos: a folder of images. Video: a movie file, sampled into frames.
                    COLMAP: a folder with sparse/0 (cameras.bin, images.bin, optional points3D.bin) and an images folder.
                    ARKit: a folder with capture.json and the images it names.
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
