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

    var body: some View {
        let manifest: ProjectManifest = document.wrappedValue.contents.manifest
        Form {
            PipelineStepList(manifest: manifest, pipeline: pipeline, project: model)
            if manifest.imageFiles.isEmpty {
                PipelineInspector(document: document, project: model, pipeline: pipeline)
            } else {
                captureSection(manifest: manifest)
            }
            DepthModelSection(pipeline: pipeline)
            if !manifest.imageFiles.isEmpty {
                SparseViewSettingsSection(document: document, model: model, pipeline: pipeline)
                trainingSection
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
        }
        .formStyle(.grouped)
    }

    // MARK: - Sections

    private func captureSection(manifest: ProjectManifest) -> some View {
        Section("Capture") {
            LabeledContent("Images", value: manifest.imageFiles.count.formatted())
            LabeledContent("Source", value: sourceName(manifest))
            if let summary = pipeline.poseSummary {
                LabeledContent("Poses", value: summary)
            }
        }
    }

    private var trainingSection: some View {
        Section("Training") {
            presetPicker
            trainingControls
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
        } else {
            Button("Train") {
                model.startTraining(document: document)
            }
            .buttonStyle(.glassProminent)
        }
    }

    // MARK: - Helpers

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
