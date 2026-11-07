import AppKit
import SwiftUI
import SplatCore
import SplatIO
import UniformTypeIdentifiers

/// Sections of the inspector for the input stage: the entry points while nothing is pending, then the frame
/// selection and the pose estimation. Shown while the project has no images.
struct PipelineInspector: View {
    let document: Binding<ProjectDocument>
    let project: ProjectModel
    let pipeline: PipelineModel
    @State private var isChoosingPhotos = false
    @State private var isChoosingVideo = false
    @State private var isChoosingCOLMAP = false
    @State private var isChoosingARKit = false

    var body: some View {
        Group {
            if pipeline.source == nil {
                entrySection
            } else {
                framesSection
                posesSection
            }
        }
        .fileImporter(isPresented: $isChoosingPhotos, allowedContentTypes: [.folder]) { result in
            handlePhotos(result)
        }
        .fileImporter(isPresented: $isChoosingVideo, allowedContentTypes: [.movie]) { result in
            handleVideo(result)
        }
        .fileImporter(isPresented: $isChoosingCOLMAP, allowedContentTypes: [.folder]) { result in
            handlePoseFolder(result, arkit: false)
        }
        .fileImporter(isPresented: $isChoosingARKit, allowedContentTypes: [.folder]) { result in
            handlePoseFolder(result, arkit: true)
        }
    }

    // MARK: - Sections

    private var entrySection: some View {
        Section("Add a Capture") {
            Button("Photos\u{2026}") {
                isChoosingPhotos = true
            }
            Button("Video\u{2026}") {
                isChoosingVideo = true
            }
            Button("Import COLMAP\u{2026}") {
                isChoosingCOLMAP = true
            }
            Button("Import ARKit Capture\u{2026}") {
                isChoosingARKit = true
            }
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

    // MARK: - Import handlers

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
        if pipeline.depthProgress != nil {
            depthState = .running
        } else if pipeline.depthIsReady {
            depthState = .done
        } else if pipeline.depthError != nil {
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

/// Depth model row: state, download with progress and cancel, and the note that training does not use it yet.
struct DepthModelSection: View {
    let pipeline: PipelineModel

    var body: some View {
        let descriptor: ModelDescriptor = PipelineModel.depthDescriptor
        let size: String = ByteCountFormatter.string(fromByteCount: descriptor.totalSizeBytes, countStyle: .file)
        Section("Depth Model") {
            LabeledContent(descriptor.displayName, value: statusText)
            Text("Gives depth guidance to projects without depth maps. Used by training when the depth prior is on.")
                .font(.footnote)
                .foregroundStyle(.secondary)
            if let progress = pipeline.depthProgress {
                ProgressView(value: progress, total: 1)
                Button("Cancel Download") {
                    pipeline.cancelDepthDownload()
                }
                .buttonStyle(.bordered)
            } else if pipeline.depthCanDownload {
                Button(downloadTitle(size: size)) {
                    pipeline.downloadDepthModel()
                }
                .buttonStyle(.bordered)
            }
            if let error = pipeline.depthError {
                Text(error)
                    .foregroundStyle(.red)
            }
        }
        .task {
            await pipeline.refreshDepthStatus()
        }
    }

    private var statusText: String {
        switch pipeline.depthStatus {
        case nil:
            return pipeline.depthProgress == nil ? "Checking\u{2026}" : "Downloading"
        case .notConfigured?:
            return "Not available in this build"
        case .missing?:
            return "Not downloaded"
        case .corrupt?:
            return "Damaged"
        case .ready?:
            return "Installed"
        }
    }

    private func downloadTitle(size: String) -> String {
        if case .corrupt? = pipeline.depthStatus {
            return "Download Again (\(size))"
        }
        return "Download (\(size))"
    }
}

/// Pose-free model row: whether the converted Depth Anything 3 package is installed, import from a folder chosen
/// by the user, and removal. The model is not published, so it can only be imported.
struct PoseFreeModelSection: View {
    let pipeline: PipelineModel
    @State private var isConfirmingRemoval = false

    /// `.mlpackage` directory packages. Falls back to the generic package type when the extension is not registered.
    private static let packageType: UTType = UTType(filenameExtension: "mlpackage") ?? .package

    var body: some View {
        let descriptor: ModelDescriptor = PipelineModel.da3Descriptor
        Section("Pose-Free Model") {
            LabeledContent(descriptor.displayName, value: statusText)
            Text("Fallback when Apple Photogrammetry cannot register the photos. It estimates poses without a calibrated camera and is less accurate than Apple Photogrammetry. It is not published yet: convert it with Tools/da3/convert_da3.py and import the package.")
                .font(.footnote)
                .foregroundStyle(.secondary)
            Button("Import Package\u{2026}") {
                chooseAndImport()
            }
            .buttonStyle(.bordered)
            .disabled(pipeline.isImportingDA3 || pipeline.isEstimating)
            if pipeline.da3PackageURL != nil {
                Button("Remove", role: .destructive) {
                    isConfirmingRemoval = true
                }
                .disabled(pipeline.isImportingDA3 || pipeline.isEstimating)
            }
            if let error = pipeline.da3Error {
                Text(error)
                    .foregroundStyle(.red)
            }
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
