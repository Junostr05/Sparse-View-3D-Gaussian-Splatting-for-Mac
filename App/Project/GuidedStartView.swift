import SwiftUI
import SplatCore
import SplatIO

/// The four steps of the guided start card, in order.
private enum GuideStep: Int, CaseIterable, Identifiable {
    case addCapture = 1
    case estimatePoses
    case train
    case explore

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .addCapture: "Add your capture"
        case .estimatePoses: "Estimate poses"
        case .train: "Train the splat"
        case .explore: "Explore, edit and export"
        }
    }

    var sentence: String {
        switch self {
        case .addCapture: "Add photos, a video, or a COLMAP or ARKit capture."
        case .estimatePoses: "Poses are estimated automatically from the photos on this Mac."
        case .train: "Training fits the 3D scene to your photos. Progress shows here and in the inspector."
        case .explore: "Orbit the scene, delete what you do not want, and save an image, a video or a .ply, .splat or .spz file."
        }
    }

    var symbol: String {
        switch self {
        case .addCapture: "photo.on.rectangle.angled"
        case .estimatePoses: "camera.viewfinder"
        case .train: "sparkles"
        case .explore: "cube.transparent"
        }
    }
}

/// Where a step stands relative to the current one.
private enum StepPosition {
    case done
    case current
    case upcoming

    var spokenName: String {
        switch self {
        case .done: "Done"
        case .current: "Current step"
        case .upcoming: "Not started"
        }
    }
}

/// The guided start card shown in the viewport area while the project has no model. It names the current step and
/// offers its action. The actions call the same imports, pose estimation and training as the inspector does.
struct GuidedStartView: View {
    let document: Binding<ProjectDocument>
    let model: ProjectModel
    let pipeline: PipelineModel
    @State private var choice: CaptureKind?

    var body: some View {
        let contents: ProjectContents = document.wrappedValue.contents
        let current: GuideStep = currentStep(contents: contents)
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Create a splat from your photos")
                    .font(.title2.weight(.semibold))
                    .accessibilityAddTraits(.isHeader)
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(GuideStep.allCases) { step in
                        stepRow(step, current: current)
                    }
                }
                actionArea(for: current, contents: contents)
                Text("You can also drag photos, a video or a folder onto this window.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            .padding(24)
            .frame(maxWidth: 520, alignment: .leading)
            .background(.quinary, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .padding(24)
            .frame(maxWidth: .infinity)
        }
        .captureImporters(choice: $choice, document: document, project: model, pipeline: pipeline)
    }

    // MARK: - Current step

    /// The first step that is not finished yet. Photos in the project put training next; frames or a pending
    /// pose estimate put pose estimation next; an empty project starts at the capture.
    private func currentStep(contents: ProjectContents) -> GuideStep {
        if !contents.manifest.imageFiles.isEmpty {
            return .train
        }
        if pipeline.source != nil || pipeline.hasCandidates {
            return .estimatePoses
        }
        return .addCapture
    }

    // MARK: - Step rows

    private func stepRow(_ step: GuideStep, current: GuideStep) -> some View {
        let position: StepPosition
        if step.rawValue < current.rawValue {
            position = .done
        } else if step == current {
            position = .current
        } else {
            position = .upcoming
        }
        let titleStyle: Color = position == .upcoming ? Color.secondary : Color.primary
        return HStack(alignment: .firstTextBaseline, spacing: 12) {
            stepBadge(step, position: position)
            Image(systemName: step.symbol)
                .foregroundStyle(titleStyle)
                .frame(width: 24)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(step.title)
                    .font(.headline)
                    .foregroundStyle(titleStyle)
                Text(step.sentence)
                    .font(.subheadline)
                    .foregroundStyle(Color.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            position == .current ? Color.accentColor.opacity(0.12) : Color.clear,
            in: RoundedRectangle(cornerRadius: 10, style: .continuous)
        )
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Step \(step.rawValue) of \(GuideStep.allCases.count), \(step.title). \(position.spokenName).")
    }

    @ViewBuilder
    private func stepBadge(_ step: GuideStep, position: StepPosition) -> some View {
        let fill: Color = position == .current ? Color.accentColor : Color.secondary.opacity(0.2)
        ZStack {
            Circle()
                .fill(fill)
            if position == .done {
                Image(systemName: "checkmark")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(Color.white)
            } else {
                Text("\(step.rawValue)")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(position == .current ? Color.white : Color.secondary)
            }
        }
        .frame(width: 24, height: 24)
        .accessibilityHidden(true)
    }

    // MARK: - Actions of the current step

    @ViewBuilder
    private func actionArea(for step: GuideStep, contents: ProjectContents) -> some View {
        switch step {
        case .addCapture:
            captureChoices
        case .estimatePoses:
            poseAction
        case .train:
            trainAction(contents: contents)
        case .explore:
            EmptyView()
        }
    }

    /// The four importers. Photos is the prominent choice; the other captures are bordered.
    private var captureChoices: some View {
        VStack(alignment: .leading, spacing: 10) {
            Button("Choose Photos\u{2026}") {
                choice = .photos
            }
            .buttonStyle(.glassProminent)
            .help("A folder of photos. Frames are picked from them, then poses are estimated.")
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 170), spacing: 8, alignment: .leading)],
                      alignment: .leading,
                      spacing: 8) {
                Button("Choose Video\u{2026}") {
                    choice = .video
                }
                .help("A movie file. It is sampled into frames.")
                Button("Choose COLMAP Folder\u{2026}") {
                    choice = .colmap
                }
                .help("A COLMAP folder with sparse/0 and an images folder. Its poses are used as they are.")
                Button("Choose ARKit Capture\u{2026}") {
                    choice = .arkit
                }
                .help("A folder with capture.json and the images it names.")
            }
            .buttonStyle(.bordered)
            Text("Photos: a folder of images. Video: a movie file. COLMAP: a folder with sparse/0 and an images folder. ARKit: a folder with capture.json and its images.")
                .font(.footnote)
                .foregroundStyle(Color.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private var poseAction: some View {
        VStack(alignment: .leading, spacing: 10) {
            if pipeline.isScanning {
                progressLine("Reading the photos\u{2026}")
            } else if pipeline.isEstimating {
                progressLine(pipeline.estimationStatus ?? "Estimating poses\u{2026}")
            } else {
                Button("Estimate Poses") {
                    pipeline.estimatePoses(document: document, project: model)
                }
                .buttonStyle(.glassProminent)
                .disabled(!pipeline.canEstimatePoses)
                .help("Finds the camera pose of every selected frame.")
                if let reason = pipeline.poseDisabledReason {
                    Text(reason)
                        .font(.footnote)
                        .foregroundStyle(Color.secondary)
                }
            }
        }
    }

    @ViewBuilder
    private func trainAction(contents: ProjectContents) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            if model.isTraining {
                progressLine(trainingStatus)
                Text("Pause or stop the run in the inspector.")
                    .font(.footnote)
                    .foregroundStyle(Color.secondary)
            } else if model.canContinue(contents) {
                Button("Continue Training") {
                    model.continueTraining(document: document)
                }
                .buttonStyle(.glassProminent)
                .help("Keep optimizing the saved model for more iterations.")
            } else {
                Button("Train") {
                    model.startTraining(document: document)
                }
                .buttonStyle(.glassProminent)
                .help("Reconstruct the scene from the imported photos with the current preset.")
            }
        }
    }

    private func progressLine(_ text: String) -> some View {
        HStack(spacing: 8) {
            ProgressView()
                .controlSize(.small)
            Text(text)
                .foregroundStyle(Color.secondary)
        }
    }

    private var trainingStatus: String {
        guard let progress = model.progress else {
            return "Preparing the training run\u{2026}"
        }
        return "Training: iteration \(progress.iteration.formatted()) of \(progress.totalIterations.formatted())"
    }
}
