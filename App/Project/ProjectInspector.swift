import SwiftUI
import SplatCore
import SplatIO
import SplatTraining

/// How much of the project inspector is shown. Basic keeps the essentials; Expert shows every setting.
/// Stored in `@AppStorage("inspectorMode")`.
enum InspectorMode: String, CaseIterable, Identifiable {
    case basic
    case expert

    var id: Self { self }

    var title: String {
        switch self {
        case .basic: "Basic"
        case .expert: "Expert"
        }
    }
}

/// The project inspector. Basic mode: pipeline steps, the capture summary, training and the viewport quick controls.
/// Expert mode adds the depth and pose-free models, the sparse-view and quality settings and every viewport control,
/// in groups that collapse. The input stage (no images yet) shows the pipeline inspector in both modes.
struct ProjectInspector: View {
    let document: Binding<ProjectDocument>
    let model: ProjectModel
    let pipeline: PipelineModel
    @AppStorage("inspectorMode") private var mode: InspectorMode = .basic
    @AppStorage("inspector.depthModelExpanded") private var isDepthModelExpanded: Bool = false
    @AppStorage("inspector.poseFreeModelExpanded") private var isPoseFreeModelExpanded: Bool = false
    @AppStorage("inspector.sparseViewExpanded") private var isSparseViewExpanded: Bool = true
    @AppStorage("inspector.qualityExpanded") private var isQualityExpanded: Bool = false
    @AppStorage("inspector.viewportExpanded") private var isViewportExpanded: Bool = true
    /// Only used by the footer's "Check for Updates" button. Nothing is checked until the user presses it.
    @State private var updateChecker = UpdateChecker()

    var body: some View {
        let manifest: ProjectManifest = document.wrappedValue.contents.manifest
        let hasImages: Bool = !manifest.imageFiles.isEmpty
        Form {
            modeSection
            if hasErrors {
                errorSection
            }
            PipelineStepList(manifest: manifest, pipeline: pipeline, project: model)
            if hasImages {
                captureSection(manifest: manifest)
                TrainingSection(document: document, model: model, mode: mode)
            } else {
                PipelineInspector(document: document, project: model, pipeline: pipeline)
            }
            if mode == .expert {
                DepthModelSection(pipeline: pipeline,
                                  document: document,
                                  isTraining: model.isTraining,
                                  isExpanded: $isDepthModelExpanded)
                PoseFreeModelSection(pipeline: pipeline, isExpanded: $isPoseFreeModelExpanded)
                if hasImages {
                    SparseViewSettingsSection(document: document,
                                              model: model,
                                              pipeline: pipeline,
                                              isExpanded: $isSparseViewExpanded)
                    QualitySettingsSection(document: document, model: model, isExpanded: $isQualityExpanded)
                }
            }
            if let viewer = model.viewer {
                if mode == .expert {
                    expertViewportSection(viewer)
                } else {
                    basicViewportSection(viewer)
                }
            }
            if let message = model.warningMessage {
                Section {
                    Text(message)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            footerSection
        }
        .formStyle(.grouped)
    }

    // MARK: - Mode and errors

    private var modeSection: some View {
        Section {
            Picker("Mode", selection: $mode) {
                ForEach(InspectorMode.allCases) { option in
                    Text(option.title).tag(option)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .help("Basic shows the essentials. Expert shows every setting, in groups you can collapse.")
            Text(mode == .basic
                 ? "Showing the essentials. Expert shows every setting."
                 : "Showing every setting. Groups can be collapsed.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    private var hasErrors: Bool {
        model.errorMessage != nil || pipeline.errorMessage != nil
    }

    /// Errors from the project and the pipeline in one red banner. Each message has a Dismiss button.
    private var errorSection: some View {
        Section {
            if let message = model.errorMessage {
                errorRow(message) {
                    model.errorMessage = nil
                }
            }
            if let message = pipeline.errorMessage {
                errorRow(message) {
                    pipeline.errorMessage = nil
                }
            }
        }
    }

    private func errorRow(_ message: String, dismiss: @escaping () -> Void) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .accessibilityHidden(true)
                Text(message)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Button("Dismiss", action: dismiss)
                .buttonStyle(.bordered)
                .accessibilityHint("Hides this error message.")
        }
    }

    // MARK: - Footer

    private var footerSection: some View {
        Section {
            VStack(spacing: 4) {
                Text(Self.versionText)
                    .font(.footnote)
                    .foregroundStyle(.tertiary)
                Button("Check for Updates") {
                    Task { await updateChecker.check(currentVersion: Self.currentVersion) }
                }
                .buttonStyle(.borderless)
                .font(.footnote)
                .disabled(updateChecker.state == .checking)
                .help("Looks for a newer release on GitHub. Nothing is checked until you press this.")
                updateStatus
            }
            .frame(maxWidth: .infinity, alignment: .center)
        }
    }

    /// The result of the last "Check for Updates" press. Empty until the button has been pressed.
    @ViewBuilder
    private var updateStatus: some View {
        switch updateChecker.state {
        case .idle:
            EmptyView()
        case .checking:
            statusText("Checking\u{2026}")
        case .upToDate(let current):
            statusText("You have the latest version (\(current)).")
        case .updateAvailable(let latest, let url):
            VStack(spacing: 2) {
                statusText("Version \(latest) is available.")
                Link("Open Release Page", destination: url)
                    .font(.footnote)
            }
        case .noReleases:
            statusText("No releases published yet.")
        case .failed(let message):
            statusText(message)
        }
    }

    private func statusText(_ text: String) -> some View {
        Text(text)
            .font(.footnote)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
    }

    /// "SparseSplat 1.2.0 (1)" from the bundle's version and build number.
    private static var versionText: String {
        let info: [String: Any] = Bundle.main.infoDictionary ?? [:]
        let version: String = info["CFBundleShortVersionString"] as? String ?? "?"
        let build: String = info["CFBundleVersion"] as? String ?? "?"
        return "SparseSplat \(version) (\(build))"
    }

    /// The bundle's short version string, "1.2.0", compared against the latest release.
    private static var currentVersion: String {
        let info: [String: Any] = Bundle.main.infoDictionary ?? [:]
        return info["CFBundleShortVersionString"] as? String ?? "?"
    }

    // MARK: - Capture

    private func captureSection(manifest: ProjectManifest) -> some View {
        Section("Capture") {
            LabeledContent("Images", value: manifest.imageFiles.count.formatted())
            LabeledContent("Source", value: sourceName(manifest))
            if let summary = pipeline.poseSummary {
                LabeledContent("Poses", value: summary)
            }
            if manifest.poseMethod == .depthAnything3 {
                Text("Poses are pose-free estimates (Depth Anything 3). Novel views will be less sharp than with Apple Photogrammetry.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Viewport

    /// Basic viewport: the up axis, the display mode, and the camera picker with the fly-through slider.
    private func basicViewportSection(_ viewer: ViewerModel) -> some View {
        Section("Viewport") {
            Picker("Up Axis", selection: upAxisBinding(viewer)) {
                ForEach(UpAxis.allCases, id: \.self) { axis in
                    Text(axis.displayName).tag(axis)
                }
            }
            .help("The direction that points up in the scene. The start value follows the capture source.")
            Picker("Display", selection: displayModeBinding(viewer)) {
                ForEach(ViewerDisplayMode.allCases, id: \.self) { displayMode in
                    Text(displayMode.title).tag(displayMode)
                }
            }
            .help("How the scene is drawn. Expert mode adds point size, heatmap and the other display options.")
            cameraRows(viewer, includesPlayback: false)
        }
    }

    /// Expert viewport: every viewport control in one group that collapses. The order follows the basic one.
    private func expertViewportSection(_ viewer: ViewerModel) -> some View {
        Section(isExpanded: $isViewportExpanded) {
            Text("Scene view, camera flights, overlays and export.")
                .font(.footnote)
                .foregroundStyle(.secondary)
            Picker("Up Axis", selection: upAxisBinding(viewer)) {
                ForEach(UpAxis.allCases, id: \.self) { axis in
                    Text(axis.displayName).tag(axis)
                }
            }
            ViewportDisplayControls(viewer: viewer)
            Menu("View") {
                Button("Front") { viewer.apply(.preset(.front)) }
                Button("Back") { viewer.apply(.preset(.back)) }
                Button("Right") { viewer.apply(.preset(.right)) }
                Button("Left") { viewer.apply(.preset(.left)) }
                Button("Top") { viewer.apply(.preset(.top)) }
                Button("Bottom") { viewer.apply(.preset(.bottom)) }
                Divider()
                Button(viewer.isFlying ? "Leave Fly Mode" : "Fly Mode") { viewer.toggleFlying() }
            }
            .help("Jump to a standard view, or fly through the scene with the keyboard.")
            Toggle("Pivot at cursor depth", isOn: Binding(
                get: { viewer.anchorsToCursorDepth },
                set: { viewer.anchorsToCursorDepth = $0 }
            ))
            .help("Orbit and zoom around the surface under the cursor instead of the centre of the scene.")
            ViewerShortcutsButton()
            ViewportOverlayControls(model: viewer)
            ViewportExportControls(viewer: viewer)
            cameraRows(viewer, includesPlayback: true)
        } header: {
            Text("Viewport")
        }
    }

    private func upAxisBinding(_ viewer: ViewerModel) -> Binding<UpAxis> {
        Binding(get: { viewer.upAxis }, set: { viewer.upAxis = $0 })
    }

    private func displayModeBinding(_ viewer: ViewerModel) -> Binding<ViewerDisplayMode> {
        Binding(get: { viewer.displayMode }, set: { viewer.displayMode = $0 })
    }

    /// The camera picker and the fly-through slider, and in expert mode the playback button. Shown only when the
    /// project has estimated cameras.
    @ViewBuilder
    private func cameraRows(_ viewer: ViewerModel, includesPlayback: Bool) -> some View {
        if !viewer.cameras.isEmpty {
            let cameraSelection = Binding<Int>(get: { viewer.activeCameraIndex ?? 0 },
                                               set: { viewer.goToCamera($0) })
            let flightSelection = Binding<Double>(get: { viewer.flightPosition },
                                                  set: { viewer.scrub(to: $0) })
            let lastPosition: Double = Double(max(viewer.cameras.count - 1, 0))
            Picker("Camera", selection: cameraSelection) {
                ForEach(viewer.cameras.indices, id: \.self) { index in
                    Text(viewer.cameras[index].name).tag(index)
                }
            }
            .help("Look through one of the estimated cameras.")
            if viewer.cameras.count > 1 {
                Slider(value: flightSelection, in: 0...lastPosition) {
                    Text("Fly through")
                }
                .help("Move the view along the camera path.")
            }
            if includesPlayback {
                Button {
                    viewer.togglePlayback()
                } label: {
                    if viewer.isPlaying {
                        Label("Pause Flight", systemImage: "pause.fill")
                    } else {
                        Label("Play Flight", systemImage: "play.fill")
                    }
                }
                .buttonStyle(.bordered)
                .disabled(viewer.cameras.count < 2)
                .help(viewer.cameras.count < 2 ? "Needs at least two cameras." : "Play the camera path as a flight.")
            }
        }
    }

    // MARK: - Helpers

    private func sourceName(_ manifest: ProjectManifest) -> String {
        switch manifest.inputKind {
        case .video?: return "Video frames"
        case .photos?: return "Photos"
        case .colmap?, .arkit?, nil: break
        }
        switch manifest.source {
        case .colmap?: return "COLMAP"
        case .arkit?: return "ARKit"
        case nil: return "\u{2014}"
        }
    }
}
