import SwiftUI
import SplatCore
import SplatTraining

/// The Training section of the project inspector: the preset, the primary action and the run's progress.
/// Basic mode shows the preset, the action and the essential progress rows. Expert mode adds the full
/// progress rows and the Continue Options. Both modes share this section, so a run is controlled the same way.
struct TrainingSection: View {
    let document: Binding<ProjectDocument>
    let model: ProjectModel
    let mode: InspectorMode

    var body: some View {
        Section("Training") {
            presetPicker
            trainingControls
            if model.isContinuing {
                LabeledContent("Round", value: "\(model.round)")
                LabeledContent("Total iterations", value: model.totalIterations.formatted())
            }
            if let progress = model.progress {
                progressRows(progress)
            }
            // DASHBOARD INSERTION POINT: the training dashboard (`TrainingDashboardView`) goes here, after the progress rows.
        }
    }

    // MARK: - Progress

    /// The progress bar and the iteration and PSNR rows in both modes. Expert mode adds the remaining measurements.
    @ViewBuilder
    private func progressRows(_ progress: TrainingProgress) -> some View {
        ProgressView(value: Double(progress.iteration),
                     total: Double(max(progress.totalIterations, 1)))
            .accessibilityLabel("Training progress")
        LabeledContent("Iteration", value: "\(progress.iteration) of \(progress.totalIterations)")
        LabeledContent("PSNR", value: progress.psnr.map { String(format: "%.2f dB", $0) } ?? "\u{2014}")
        if mode == .expert {
            LabeledContent("Loss", value: progress.loss.map { String(format: "%.4f", $0) } ?? "\u{2014}")
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

    // MARK: - Preset

    /// Fast, Standard or Quality. Shows "Custom" when the stored settings match none of them.
    private var presetPicker: some View {
        let configuration: TrainingConfiguration = document.wrappedValue.contents.manifest.training
        let current: TrainingPreset? = Self.preset(for: configuration)
        return Group {
            Picker("Preset", selection: presetBinding) {
                ForEach(TrainingPreset.allCases, id: \.self) { preset in
                    Text(Self.presetName(preset)).tag(Optional(preset))
                }
                if current == nil {
                    Text("Custom").tag(TrainingPreset?.none)
                }
            }
            .disabled(model.isTraining)
            .help("Fast, Standard and Quality set the image size and the number of iterations.")
            if let preset = current {
                Text("\(preset.maxImageSide) px images, \(preset.iterations.formatted()) iterations.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else {
                Text("Custom settings from this project.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            if model.isTraining {
                Text("Stop the run to change the preset.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
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

    // MARK: - Primary action

    /// One primary action (Train). While a run is active, Pause or Resume and Stop use bordered buttons.
    /// With a stored checkpoint, Continue Training is the primary action and Retrain from Scratch is bordered.
    @ViewBuilder
    private var trainingControls: some View {
        if model.isTraining {
            HStack {
                Button(model.isPaused ? "Resume" : "Pause") {
                    model.togglePause()
                }
                .buttonStyle(.bordered)
                .help(model.isPaused ? "Continue the paused run." : "Pause the run. Resume continues it.")
                Button("Stop") {
                    model.stop()
                }
                .buttonStyle(.bordered)
                .help("End the run. The model reached so far is kept.")
            }
        } else if model.canContinue(document.wrappedValue.contents) {
            Button("Continue Training") {
                model.continueTraining(document: document)
            }
            .buttonStyle(.glassProminent)
            .help("Keep optimizing the saved model for more iterations.")
            Button("Retrain from Scratch") {
                model.startTraining(document: document)
            }
            .buttonStyle(.bordered)
            .help("Discard the saved model and train again with the current preset.")
            if mode == .expert {
                continueOptions
            }
        } else {
            Button("Train") {
                model.startTraining(document: document)
            }
            .buttonStyle(.glassProminent)
            .help("Reconstruct the scene from the imported photos with the current preset.")
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
                .help("Run rounds until you press Stop.")
            Text("Rounds repeat until you press Stop. The model is saved after every round.")
                .font(.footnote)
                .foregroundStyle(.secondary)
            Toggle("Add more Gaussians", isOn: settings.continuation.addGaussians)
                .help("Let the model grow beyond its current number of Gaussians, up to the limit below.")
            Picker("Gaussian limit", selection: settings.continuation.maxGaussians) {
                ForEach(Self.gaussianLimitChoices, id: \.self) { limit in
                    Text(limit.formatted()).tag(limit)
                }
            }
            .disabled(!model.continuation.addGaussians)
            Text(model.continuation.addGaussians
                 ? "The device memory budget can still cap the limit."
                 : "Turn on Add more Gaussians to choose a limit.")
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

    /// The picker label: "Fast", "Standard" or "Quality". The image size and iterations are shown below the picker.
    private static func presetName(_ preset: TrainingPreset) -> String {
        preset.rawValue.capitalized
    }
}
