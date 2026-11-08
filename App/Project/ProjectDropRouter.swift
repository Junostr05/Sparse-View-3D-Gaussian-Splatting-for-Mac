import SwiftUI
import SplatIO

/// Routes items dropped onto the project window to the import functions the inspector buttons use.
/// The classification is `DropClassifier` (SplatIO); this type adds the security scope, the training and capture
/// checks, and the replacement. The window holds the confirmation state.
enum ProjectDropRouter {
    /// What the window does with a classified drop.
    enum Decision {
        /// Import now.
        case perform(DropAction)
        /// The project already has images: ask first, then replace.
        case confirmReplace(DropAction)
        /// Do nothing and show the reason in the error message.
        case refuse(String)
    }

    /// Classifies the dropped URLs. Their security scope is held for the classification only; each import keeps
    /// or takes its own scope.
    static func classify(_ urls: [URL]) -> DropAction {
        withAccess(urls) {
            DropClassifier.classify(urls: urls, fileSystem: LocalFileProbe())
        }
    }

    /// Refuses unsupported drops and drops during training. A project with images asks before it is replaced.
    static func decide(_ action: DropAction, hasCapture: Bool, isTraining: Bool) -> Decision {
        switch action {
        case .unsupported(let reason):
            return .refuse(reason)
        case .project:
            return .refuse("Open a project from the File menu instead of dropping it here.")
        case .colmapFolder, .arkitFolder, .photos, .video:
            break
        }
        if isTraining {
            return .refuse("Wait until training finishes before importing.")
        }
        if hasCapture {
            return .confirmReplace(action)
        }
        return .perform(action)
    }

    /// Starts the import with the functions the inspector uses. A replacement first empties the capture (images,
    /// poses, checkpoints), keeping the training settings.
    static func perform(_ action: DropAction,
                        replacing: Bool,
                        document: Binding<ProjectDocument>,
                        model: ProjectModel,
                        pipeline: PipelineModel) {
        if replacing {
            var cleared: ProjectManifest = ProjectManifest()
            cleared.training = document.wrappedValue.contents.manifest.training
            model.adopt(ProjectContents(manifest: cleared, files: [:]), into: document)
        }
        switch action {
        case .colmapFolder(let folder):
            model.importCOLMAP(folder: folder, into: document)
        case .arkitFolder(let folder):
            model.importARKit(folder: folder, into: document)
        case .photos(let urls):
            // One folder is read as a photo folder; loose files are read one by one.
            let isSingleFolder: Bool = urls.count == 1 && withAccess(urls, { LocalFileProbe().isDirectory(urls[0]) })
            if isSingleFolder {
                pipeline.importPhotos(folder: urls[0])
            } else {
                pipeline.importPhotos(urls: urls)
            }
        case .video(let url):
            pipeline.importVideo(url: url)
        case .project, .unsupported:
            break
        }
    }

    /// Runs `body` with the security scope of each URL that grants one, and releases exactly those scopes afterwards.
    private static func withAccess<T>(_ urls: [URL], _ body: () -> T) -> T {
        var scoped: [URL] = []
        for url in urls where url.startAccessingSecurityScopedResource() {
            scoped.append(url)
        }
        defer {
            for url in scoped {
                url.stopAccessingSecurityScopedResource()
            }
        }
        return body()
    }
}

/// The highlight shown while a drag hovers over the project window: an accent border and a short label.
/// Content layer, so no glass. It never takes the pointer, so the drop still reaches the window.
struct ProjectDropHighlight: View {
    var body: some View {
        RoundedRectangle(cornerRadius: 16, style: .continuous)
            .strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 3, dash: [10, 6]))
            .background(Color.accentColor.opacity(0.06))
            .overlay {
                Text("Drop photos, a video or a capture folder")
                    .font(.headline)
                    .padding(.horizontal, 18)
                    .padding(.vertical, 10)
                    .background {
                        Capsule().fill(Color(nsColor: .windowBackgroundColor))
                    }
                    .accessibilityLabel("Drop photos, a video or a capture folder to import it")
            }
            .padding(8)
            .allowsHitTesting(false)
    }
}
