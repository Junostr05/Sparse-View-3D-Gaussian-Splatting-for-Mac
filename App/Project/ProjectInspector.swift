import SwiftUI
import SplatCore
import SplatIO
import SplatTraining

/// The project inspector: pipeline steps and input sections while the project has no images, the capture
/// summary, the Sparse-View Quality settings and training controls after that. The depth model row sits between them.
struct ProjectInspector: View {
    let document: Binding<ProjectDocument>
    let model: ProjectModel
    let pipeline: PipelineModel
    /// Only used by the footer's "Check for Updates" button. Nothing is checked until the user presses it.
    @State private var updateChecker = UpdateChecker()

    var body: some View {
        let manifest: ProjectManifest = document.wrappedValue.contents.manifest
        Form {
            PipelineStepList(manifest: manifest, pipeline: pipeline, project: model)
            if manifest.imageFiles.isEmpty {
                PipelineInspector(document: document, project: model, pipeline: pipeline)
            } else {
                captureSection(manifest: manifest)
            }
            DepthModelSection(pipeline: pipeline, document: document, isTraining: model.isTraining)
            PoseFreeModelSection(pipeline: pipeline)
            if !manifest.imageFiles.isEmpty {
                SparseViewSettingsSection(document: document, model: model, pipeline: pipeline)
                QualitySettingsSection(document: document, model: model)
                trainingSection
            }
            if let viewer = model.viewer {
                viewportSection(viewer)
            }
            if let message = model.errorMessage {
                Section {
                    Text(message)
                        .foregroundStyle(.red)
                }
            }
            if let message = model.warningMessage {
                Section {
                    Text(message)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            if let message = pipeline.errorMessage {
                Section {
                    Text(message)
                        .foregroundStyle(.red)
                }
            }
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
                    updateStatus
                }
                .frame(maxWidth: .infinity, alignment: .center)
            }
        }
        .formStyle(.grouped)
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

    // MARK: - Sections

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

    /// Up axis, standard views and, when the project has estimated cameras, the camera picker,
    /// the fly-through slider and playback. The start up axis follows the capture source.
    private func viewportSection(_ viewer: ViewerModel) -> some View {
        let selection = Binding<UpAxis>(get: { viewer.upAxis }, set: { viewer.upAxis = $0 })
        let cameraSelection = Binding<Int>(get: { viewer.activeCameraIndex ?? 0 },
                                           set: { viewer.goToCamera($0) })
        let flightSelection = Binding<Double>(get: { viewer.flightPosition },
                                              set: { viewer.scrub(to: $0) })
        let lastPosition: Double = Double(max(viewer.cameras.count - 1, 0))
        return Section("Viewport") {
            Picker("Up Axis", selection: selection) {
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
            Toggle("Pivot at cursor depth", isOn: Binding(
                get: { viewer.anchorsToCursorDepth },
                set: { viewer.anchorsToCursorDepth = $0 }
            ))
            ViewerShortcutsButton()
            if !viewer.cameras.isEmpty {
                Picker("Camera", selection: cameraSelection) {
                    ForEach(viewer.cameras.indices, id: \.self) { index in
                        Text(viewer.cameras[index].name).tag(index)
                    }
                }
                if viewer.cameras.count > 1 {
                    Slider(value: flightSelection, in: 0...lastPosition) {
                        Text("Fly through")
                    }
                }
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
            }
        }
    }

    private var trainingSection: some View {
        Section("Training") {
            presetPicker
            trainingControls
            if model.isContinuing {
                LabeledContent("Round", value: "\(model.round)")
                LabeledContent("Total iterations", value: model.totalIterations.formatted())
            }
            if let progress = model.progress {
                ProgressView(value: Double(progress.iteration),
                             total: Double(max(progress.totalIterations, 1)))
                LabeledContent("Iteration", value: "\(progress.iteration) of \(progress.totalIterations)")
                LabeledContent("Loss", value: progress.loss.map { String(format: "%.4f", $0) } ?? "\u{2014}")
                LabeledContent("PSNR", value: progress.psnr.map { String(format: "%.2f dB", $0) } ?? "\u{2014}")
                LabeledContent("Gaussians", value: progress.gaussianCount.formatted())
                LabeledContent("Iterations/s", value: String(format: "%.1f", progress.iterationsPerSecond))
                LabeledContent("Phase", value: Self.phaseLabel(progress.phase))
                if let heldOut = progress.heldOut {
                    LabeledContent("Held-out PSNR", value: String(format: "%.2f dB", heldOut.psnr))
                    LabeledContent("Held-out SSIM", value: String(format: "%.3f", heldOut.ssim))
                }
                if let best = progress.bestIteration {
                    LabeledContent("Best iteration", value: best.formatted())
                }
                if let depthLoss = progress.depthLoss {
                    LabeledContent("Depth loss", value: String(format: "%.4f", depthLoss))
                }
            }
        }
    }

    /// Fast, Standard or Quality. Shows "Custom" when the stored settings match none of them.
    private var presetPicker: some View {
        let configuration: TrainingConfiguration = document.wrappedValue.contents.manifest.training
        return Picker("Preset", selection: presetBinding) {
            ForEach(TrainingPreset.allCases, id: \.self) { preset in
                Text(Self.presetTitle(preset)).tag(Optional(preset))
            }
            if Self.preset(for: configuration) == nil {
                Text("Custom").tag(TrainingPreset?.none)
            }
        }
        .disabled(model.isTraining)
    }

    private var presetBinding: Binding<TrainingPreset?> {
        Binding(
            get: { Self.preset(for: document.wrappedValue.contents.manifest.training) },
            set: { newValue in
                guard let preset = newValue else { return }
                preset.apply(to: &document.wrappedValue.contents.manifest.training)
            }
        )
    }

    /// One primary action (Train). While a run is active, Pause or Resume and Stop use the default style.
    /// With a stored checkpoint, Continue Training is the primary action and Retrain from Scratch is bordered.
    @ViewBuilder
    private var trainingControls: some View {
        if model.isTraining {
            HStack {
                Button(model.isPaused ? "Resume" : "Pause") {
                    model.togglePause()
                }
                Button("Stop") {
                    model.stop()
                }
            }
        } else if model.canContinue(document.wrappedValue.contents) {
            Button("Continue Training") {
                model.continueTraining(document: document)
            }
            .buttonStyle(.glassProminent)
            Button("Retrain from Scratch") {
                model.startTraining(document: document)
            }
            .buttonStyle(.bordered)
            continueOptions
        } else {
            Button("Train") {
                model.startTraining(document: document)
            }
            .buttonStyle(.glassProminent)
        }
    }

    /// Round length, endless mode and Gaussian growth for "Continue Training".
    private var continueOptions: some View {
        let settings = Bindable(model)
        return DisclosureGroup("Continue Options") {
            Picker("Iterations per round", selection: settings.continuation.roundIterations) {
                ForEach(Self.roundChoices, id: \.self) { iterations in
                    Text(iterations.formatted()).tag(iterations)
                }
            }
            Toggle("Train until stopped", isOn: settings.continuation.isEndless)
            Text("Rounds repeat until you press Stop. The model is saved after every round.")
                .font(.footnote)
                .foregroundStyle(.secondary)
            Toggle("Add more Gaussians", isOn: settings.continuation.addGaussians)
            Picker("Gaussian limit", selection: settings.continuation.maxGaussians) {
                ForEach(Self.gaussianLimitChoices, id: \.self) { limit in
                    Text(limit.formatted()).tag(limit)
                }
            }
            .disabled(!model.continuation.addGaussians)
            Text("The device memory budget can still cap the limit.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .disabled(model.isTraining)
    }

    // MARK: - Helpers

    /// Iterations per continuation round offered in the picker.
    private static let roundChoices: [Int] = [1_000, 2_000, 5_000, 10_000, 20_000]
    /// Gaussian ceilings offered when "Add more Gaussians" is on. The device memory budget can still cap them.
    private static let gaussianLimitChoices: [Int] = [1_000_000, 2_000_000, 3_000_000, 5_000_000]

    private static func preset(for configuration: TrainingConfiguration) -> TrainingPreset? {
        TrainingPreset.allCases.first { preset in
            preset.maxImageSide == configuration.maxImageSide && preset.iterations == configuration.iterations
        }
    }

    /// What the run is doing now: the plain run, the held-out iteration choice, or the final retrain.
    private static func phaseLabel(_ phase: TrainingPhase) -> String {
        switch phase {
        case .single:
            return "Training"
        case .selection(let heldOutView):
            return "Choosing iteration (holding out \(heldOutView))"
        case .final(let stopIteration):
            return "Final training to \(stopIteration.formatted())"
        }
    }

    private static func presetTitle(_ preset: TrainingPreset) -> String {
        "\(preset.rawValue.capitalized) (\(preset.maxImageSide) px, \(preset.iterations.formatted()) iterations)"
    }

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
