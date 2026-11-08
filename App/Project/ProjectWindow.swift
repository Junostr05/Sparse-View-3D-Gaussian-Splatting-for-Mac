import SwiftUI
import SplatCore
import SplatIO

/// The window of a `.gsproj` project: image list, live viewport and the training inspector.
struct ProjectWindow: View {
    let document: Binding<ProjectDocument>
    /// The package name without extension, used as the default export name.
    let name: String
    /// The package's file URL, nil for an unsaved document. Used to tell the training queue about running training.
    let url: URL?
    /// The app's training queue, used by the command palette.
    let queue: TrainingQueue
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
            if let url {
                ActiveProjectTraining.set(url, training: false)
            }
        }
        .onChange(of: model.isTraining, initial: false) { (_, training: Bool) in
            // The training queue skips a project while its window trains it.
            if let url {
                ActiveProjectTraining.set(url, training: training)
            }
        }
        .commandPalette { openWindow in
            CommandRegistry.projectWindowCommands(viewer: model.viewer, project: model, document: document,
                                                  openWindow: openWindow, queue: queue, projectURL: url)
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
