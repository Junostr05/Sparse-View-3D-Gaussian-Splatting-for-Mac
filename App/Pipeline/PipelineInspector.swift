import AppKit
import SwiftUI
import SplatCore
import SplatDepth
import SplatIO
import UniformTypeIdentifiers

/// The kinds of capture the user can add. Each one opens its importer and runs its import.
enum CaptureKind: Hashable {
    case photos
    case video
    case colmap
    case arkit
}

/// Presents the capture importers for `choice` and runs the matching import. The inspector's input section and the
/// guided start card both attach it, so both entry points run the same imports.
struct CaptureImporters: ViewModifier {
    @Binding var choice: CaptureKind?
    let document: Binding<ProjectDocument>
    let project: ProjectModel
    let pipeline: PipelineModel

    func body(content: Content) -> some View {
        content
            .fileImporter(isPresented: isPresented(.photos), allowedContentTypes: [.folder]) { result in
                handlePhotos(result)
            }
            .fileImporter(isPresented: isPresented(.video), allowedContentTypes: [.movie]) { result in
                handleVideo(result)
            }
            .fileImporter(isPresented: isPresented(.colmap), allowedContentTypes: [.folder]) { result in
                handlePoseFolder(result, arkit: false)
            }
            .fileImporter(isPresented: isPresented(.arkit), allowedContentTypes: [.folder]) { result in
                handlePoseFolder(result, arkit: true)
            }
    }

    /// `true` while `kind` is the chosen capture. Dismissing the panel clears the choice.
    private func isPresented(_ kind: CaptureKind) -> Binding<Bool> {
        Binding(
            get: { choice == kind },
            set: { presented in
                if !presented, choice == kind {
                    choice = nil
                }
            }
        )
    }

    private func handlePhotos(_ result: Result<URL, Error>) {
        switch result {
        case .success(let url):
            pipeline.importPhotos(folder: url)
        case .failure(let error):
            pipeline.errorMessage = error.localizedDescription
        }
    }

    private func handleVideo(_ result: Result<URL, Error>) {
        switch result {
        case .success(let url):
            pipeline.importVideo(url: url)
        case .failure(let error):
            pipeline.errorMessage = error.localizedDescription
        }
    }

    private func handlePoseFolder(_ result: Result<URL, Error>, arkit: Bool) {
        switch result {
        case .success(let url):
            if arkit {
                project.importARKit(folder: url, into: document)
            } else {
                project.importCOLMAP(folder: url, into: document)
            }
        case .failure(let error):
            project.errorMessage = error.localizedDescription
        }
    }
}

extension View {
    /// Attaches the capture importers. See `CaptureImporters`.
    func captureImporters(choice: Binding<CaptureKind?>,
                          document: Binding<ProjectDocument>,
                          project: ProjectModel,
                          pipeline: PipelineModel) -> some View {
        modifier(CaptureImporters(choice: choice, document: document, project: project, pipeline: pipeline))
    }
}

extension PipelineModel {
    /// Why "Estimate Poses" is unavailable, or `nil` when it can run. Shown as a footnote under the button.
    var poseDisabledReason: String? {
        if isScanning {
            return "Wait until the frames are read."
        }
        if isEstimating {
            return nil
        }
        if scanFailed {
            return "The frames could not be read. Choose the capture again."
        }
        if plan == nil || (plan?.selectedCount ?? 0) == 0 {
            return "Select at least one frame first."
        }
        return nil
    }
}

/// Sections of the inspector for the input stage: the entry points while nothing is pending, then the frame
/// selection and the pose estimation. Shown while the project has no images.
struct PipelineInspector: View {
    let document: Binding<ProjectDocument>
    let project: ProjectModel
    let pipeline: PipelineModel
    @State private var choice: CaptureKind?

    var body: some View {
        Group {
            if pipeline.source == nil {
                entrySection
            } else {
                framesSection
                posesSection
            }
        }
        .captureImporters(choice: $choice, document: document, project: project, pipeline: pipeline)
    }

    // MARK: - Sections

    private var entrySection: some View {
        Section("Add a Capture") {
            Button("Photos\u{2026}") {
                choice = .photos
            }
            .help("A folder of photos. Frames are picked from them.")
            Button("Video\u{2026}") {
                choice = .video
            }
            .help("A movie file. It is sampled into frames.")
            Button("Import COLMAP\u{2026}") {
                choice = .colmap
            }
            .help("A COLMAP folder with sparse/0 and an images folder. Its poses are used as they are.")
            Button("Import ARKit Capture\u{2026}") {
                choice = .arkit
            }
            .help("A folder with capture.json and the images it names.")
            Text("Photos and video need poses. COLMAP and ARKit captures bring their own.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    private var framesSection: some View {
        Section("Frames") {
            if pipeline.isScanning {
                HStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.small)
                    Text(scanningText)
                }
            }
            if pipeline.plan != nil {
                if case .video = pipeline.source {
                    Picker("Sampling interval", selection: intervalBinding) {
                        ForEach(PipelineModel.sampleIntervals, id: \.self) { interval in
                            Text("\(interval.formatted()) s").tag(interval)
                        }
                    }
                    .disabled(pipeline.isScanning || pipeline.isEstimating)
                }
                Toggle("Use all frames", isOn: usesAllBinding)
                    .disabled(pipeline.isEstimating)
                if !pipeline.usesAllFrames {
                    Stepper(value: bestCountBinding, in: 1...max(1, pipeline.candidates.count)) {
                        LabeledContent("Best frames", value: pipeline.bestCount.formatted())
                    }
                    .disabled(pipeline.isEstimating)
                }
                LabeledContent("Selected", value: selectionText)
                if let warning = pipeline.plan?.warning {
                    Text(PipelineModel.warningText(warning))
                        .foregroundStyle(.secondary)
                }
                Text("Changing the mode or the interval resets manual changes.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                Button("Discard Frames") {
                    pipeline.discard()
                }
                .buttonStyle(.bordered)
            }
            if pipeline.scanFailed {
                Text("Choose the capture again or pick a different one.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var posesSection: some View {
        Section("Poses") {
            Picker("Method", selection: methodBinding) {
                ForEach(PoseMethodPreference.allCases, id: \.self) { preference in
                    Text(preference.title)
                        .disabled(preference == .depthAnything3 && pipeline.da3PackageURL == nil)
                        .tag(preference)
                }
            }
            .disabled(pipeline.isEstimating)
            .help("How the camera poses are found. Automatic falls back to Depth Anything 3 when its package is installed.")
            if pipeline.isEstimating {
                HStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.small)
                    Text(pipeline.estimationStatus ?? "Estimating poses\u{2026}")
                }
            } else {
                Button("Estimate Poses") {
                    pipeline.estimatePoses(document: document, project: project)
                }
                .buttonStyle(.glassProminent)
                .disabled(!pipeline.canEstimatePoses)
                .help("Finds the camera pose of every selected frame.")
                if let reason = pipeline.poseDisabledReason {
                    Text(reason)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            Text("Apple Photogrammetry registers the selected frames on this Mac. Automatic falls back to Depth Anything 3 when its package is installed. If the frames still cannot be registered, import a COLMAP folder instead.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Bindings and text

    private var methodBinding: Binding<PoseMethodPreference> {
        Binding(get: { pipeline.poseMethodPreference },
                set: { pipeline.setPoseMethodPreference($0) })
    }

    private var usesAllBinding: Binding<Bool> {
        Binding(get: { pipeline.usesAllFrames },
                set: { pipeline.setUsesAllFrames($0) })
    }

    private var bestCountBinding: Binding<Int> {
        Binding(get: { pipeline.bestCount },
                set: { pipeline.setBestCount($0) })
    }

    private var intervalBinding: Binding<Double> {
        Binding(get: { pipeline.sampleInterval },
                set: { pipeline.setSampleInterval($0) })
    }

    private var scanningText: String {
        if case .video = pipeline.source {
            return "Scoring the video frames\u{2026}"
        }
        return "Reading the photos\u{2026}"
    }

    private var selectionText: String {
        let total: Int = pipeline.candidates.count
        let selected: Int = pipeline.plan?.selectedCount ?? 0
        return "\(selected.formatted()) of \(total.formatted())"
    }

}

/// The four pipeline steps with a status each. Steps that a COLMAP or ARKit import does without are skipped.
struct PipelineStepList: View {
    let manifest: ProjectManifest
    let pipeline: PipelineModel
    let project: ProjectModel

    var body: some View {
        Section("Steps") {
            ForEach(entries) { entry in
                StepRow(entry: entry)
            }
        }
    }

    private var entries: [StepEntry] {
        let fromFrames: Bool = manifest.source != nil
            && manifest.inputKind != .video
            && manifest.inputKind != .photos
        let posesPresent: Bool = manifest.source != nil && !manifest.imageFiles.isEmpty

        let framesState: StepState
        if fromFrames {
            framesState = .skipped
        } else if pipeline.isScanning {
            framesState = .running
        } else if pipeline.scanFailed {
            framesState = .failed
        } else if pipeline.hasCandidates || posesPresent {
            framesState = .done
        } else {
            framesState = .pending
        }

        let posesState: StepState
        if fromFrames {
            posesState = .skipped
        } else if pipeline.isEstimating {
            posesState = .running
        } else if pipeline.estimationFailed {
            posesState = .failed
        } else if posesPresent {
            posesState = .done
        } else {
            posesState = .pending
        }

        let depthState: StepState
        if pipeline.isDepthDownloading {
            depthState = .running
        } else if pipeline.anyDepthModelReady {
            depthState = .done
        } else if pipeline.depthErrorText != nil {
            depthState = .failed
        } else {
            depthState = .pending
        }

        let trainState: StepState
        if project.isTraining {
            trainState = .running
        } else if case .failed? = project.progress?.state {
            trainState = .failed
        } else if !manifest.checkpoints.isEmpty {
            trainState = .done
        } else {
            trainState = .pending
        }

        return [
            StepEntry(id: 0, title: "Frames", state: framesState),
            StepEntry(id: 1, title: "Poses", state: posesState),
            StepEntry(id: 2, title: "Depth", state: depthState),
            StepEntry(id: 3, title: "Train", state: trainState)
        ]
    }
}

/// Status of one pipeline step.
enum StepState: Equatable {
    case pending
    case running
    case done
    case failed
    case skipped

    var label: String {
        switch self {
        case .pending: "Waiting"
        case .running: "Running"
        case .done: "Done"
        case .failed: "Failed"
        case .skipped: "Skipped"
        }
    }
}

private struct StepEntry: Identifiable {
    let id: Int
    let title: String
    let state: StepState
}

private struct StepRow: View {
    let entry: StepEntry

    var body: some View {
        HStack(spacing: 8) {
            indicator
                .frame(width: 18)
            Text(entry.title)
            Spacer()
            Text(entry.state.label)
                .foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder
    private var indicator: some View {
        switch entry.state {
        case .pending:
            Image(systemName: "circle")
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
        case .running:
            ProgressView()
                .controlSize(.small)
                .accessibilityHidden(true)
        case .done:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .accessibilityHidden(true)
        case .failed:
            Image(systemName: "exclamationmark.circle.fill")
                .foregroundStyle(.red)
                .accessibilityHidden(true)
        case .skipped:
            Image(systemName: "minus.circle")
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
        }
    }
}

/// Depth Model section: the depth estimator picker and one row per downloadable depth model, with its state,
/// download with progress and cancel, and removal. Training uses the picked estimator when the depth prior is on.
extension DepthEstimatorKind {
    /// Short name for the estimator picker, so it fits the inspector width. The full model name and size are in the rows.
    var pickerTitle: String {
        switch self {
        case .depthAnythingV2Small: "Depth Anything V2"
        case .moge2Small: "MoGe-2"
        }
    }
}

struct DepthModelSection: View {
    let pipeline: PipelineModel
    let document: Binding<ProjectDocument>
    let isTraining: Bool
    let isExpanded: Binding<Bool>

    var body: some View {
        let selected: DepthEstimatorKind = document.wrappedValue.contents.manifest.training.depthPrior.estimator
        Section(isExpanded: isExpanded) {
            Picker("Depth estimator", selection: estimatorBinding) {
                ForEach(DepthEstimatorKind.allCases, id: \.self) { kind in
                    Text(kind.pickerTitle).tag(kind)
                }
            }
            .disabled(isTraining)
            .help("The model that estimates depth for each photo. Its name and size are listed below.")
            if isTraining {
                Text("Stop the run to change the depth estimator.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else if !pipeline.isDepthReady(selected) {
                Text("Download the model first to use it for depth.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            Text("Gives depth guidance to projects without depth maps. Used by training when the depth prior is on.")
                .font(.footnote)
                .foregroundStyle(.secondary)
            ForEach(PipelineModel.depthDescriptors, id: \.id) { descriptor in
                DepthModelRow(pipeline: pipeline, descriptor: descriptor, isTraining: isTraining)
            }
        } header: {
            Text("Depth Model")
        }
        .task {
            await pipeline.refreshDepthStatus()
        }
    }

    /// The estimator stored in the project's training settings.
    private var estimatorBinding: Binding<DepthEstimatorKind> {
        Binding(
            get: { document.wrappedValue.contents.manifest.training.depthPrior.estimator },
            set: { kind in
                document.wrappedValue.contents.manifest.training.depthPrior.estimator = kind
            }
        )
    }
}

/// One depth model: name, size and licence, its state, and the Download, Cancel and Remove controls.
private struct DepthModelRow: View {
    let pipeline: PipelineModel
    let descriptor: ModelDescriptor
    let isTraining: Bool
    @State private var isConfirmingRemoval = false

    var body: some View {
        let size: String = ByteCountFormatter.string(fromByteCount: descriptor.totalSizeBytes, countStyle: .file)
        let state: DepthModelState = pipeline.depthState(descriptor)
        VStack(alignment: .leading, spacing: 6) {
            LabeledContent(descriptor.displayName, value: statusText(state))
            Text("\(size) \u{00B7} \(descriptor.license)")
                .font(.footnote)
                .foregroundStyle(.secondary)
            if let progress = state.progress {
                ProgressView(value: progress, total: 1)
                Button("Cancel Download") {
                    pipeline.cancelDepthDownload(descriptor)
                }
                .buttonStyle(.bordered)
            } else {
                if pipeline.depthCanDownload(descriptor) {
                    Button(downloadTitle(state: state, size: size)) {
                        pipeline.downloadDepthModel(descriptor)
                    }
                    .buttonStyle(.bordered)
                    .disabled(isTraining || pipeline.isDepthDownloading)
                }
                if case .ready? = state.status {
                    Button("Remove", role: .destructive) {
                        isConfirmingRemoval = true
                    }
                    .disabled(isTraining || pipeline.isDepthDownloading)
                }
            }
            if let error = state.error {
                Text(error)
                    .foregroundStyle(.red)
            }
        }
        .confirmationDialog("Remove \(descriptor.displayName)?", isPresented: $isConfirmingRemoval) {
            Button("Remove", role: .destructive) {
                pipeline.removeDepthModel(descriptor)
            }
        } message: {
            Text("The downloaded model is deleted. Download it again to use it for depth.")
        }
    }

    private func statusText(_ state: DepthModelState) -> String {
        if let progress = state.progress {
            return "Downloading \(Int((progress * 100).rounded()))%"
        }
        if state.error != nil, !isInstalled(state) {
            return "Failed"
        }
        switch state.status {
        case nil:
            return "Checking\u{2026}"
        case .notConfigured?:
            return "Not available in this build"
        case .missing?:
            return "Not installed"
        case .corrupt?:
            return "Damaged"
        case .ready?:
            return "Installed"
        }
    }

    private func isInstalled(_ state: DepthModelState) -> Bool {
        if case .ready? = state.status {
            return true
        }
        return false
    }

    private func downloadTitle(state: DepthModelState, size: String) -> String {
        if case .corrupt? = state.status {
            return "Download Again (\(size))"
        }
        return "Download (\(size))"
    }
}

/// Pose-free model row: whether the converted Depth Anything 3 package is installed, import from a folder chosen
/// by the user, and removal. The model is not published, so it can only be imported.
struct PoseFreeModelSection: View {
    let pipeline: PipelineModel
    let isExpanded: Binding<Bool>
    @State private var isConfirmingRemoval = false

    /// `.mlpackage` directory packages. Falls back to the generic package type when the extension is not registered.
    private static let packageType: UTType = UTType(filenameExtension: "mlpackage") ?? .package

    var body: some View {
        let descriptor: ModelDescriptor = PipelineModel.da3Descriptor
        Section(isExpanded: isExpanded) {
            LabeledContent(descriptor.displayName, value: statusText)
            Text("Fallback when Apple Photogrammetry cannot register the photos. It estimates poses without a calibrated camera and is less accurate than Apple Photogrammetry. It is not published yet: convert it with Tools/da3/convert_da3.py and import the package.")
                .font(.footnote)
                .foregroundStyle(.secondary)
            Button("Import Package\u{2026}") {
                chooseAndImport()
            }
            .buttonStyle(.bordered)
            .disabled(pipeline.isImportingDA3 || pipeline.isEstimating)
            .help("Import a converted Depth Anything 3 package (.mlpackage) from disk.")
            if pipeline.isEstimating {
                Text("Wait until the pose estimation finishes.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            if pipeline.da3PackageURL != nil {
                Button("Remove", role: .destructive) {
                    isConfirmingRemoval = true
                }
                .buttonStyle(.bordered)
                .disabled(pipeline.isImportingDA3 || pipeline.isEstimating)
            }
            if let error = pipeline.da3Error {
                Text(error)
                    .foregroundStyle(.red)
            }
        } header: {
            Text("Pose-Free Model")
        }
        .task {
            await pipeline.refreshDA3Package()
        }
        .confirmationDialog("Remove the Pose-Free Model?", isPresented: $isConfirmingRemoval) {
            Button("Remove", role: .destructive) {
                pipeline.removeDA3Model()
            }
        } message: {
            Text("The imported package is deleted. Import it again to use Depth Anything 3 poses.")
        }
    }

    private var statusText: String {
        if pipeline.isImportingDA3 {
            return "Importing\u{2026}"
        }
        return pipeline.da3PackageURL == nil ? "Not installed" : "Installed"
    }

    /// Opens a panel that accepts both package directories and files, so the `.mlpackage` can be selected.
    /// `runModal` keeps the panel on the main actor.
    private func chooseAndImport() {
        let panel = NSOpenPanel()
        panel.message = "Choose the converted Depth Anything 3 package (.mlpackage)."
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [Self.packageType]
        guard panel.runModal() == .OK, let url = panel.url else {
            return
        }
        Task {
            await pipeline.importDA3Package(from: url)
        }
    }
}
