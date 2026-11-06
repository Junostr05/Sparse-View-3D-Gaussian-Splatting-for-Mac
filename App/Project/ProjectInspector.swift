import SwiftUI
import SplatIO
import SplatTraining

/// Import buttons while the project has no images; the Train controls and live statistics after that.
struct ProjectInspector: View {
    let document: Binding<ProjectDocument>
    let model: ProjectModel
    @State private var isChoosingCOLMAP = false
    @State private var isChoosingARKit = false

    var body: some View {
        let manifest: ProjectManifest = document.wrappedValue.contents.manifest
        Form {
            if manifest.imageFiles.isEmpty {
                importSection
            } else {
                captureSection(manifest: manifest)
                trainingSection
            }
            if let message = model.errorMessage {
                Section {
                    Text(message)
                        .foregroundStyle(.red)
                }
            }
        }
        .formStyle(.grouped)
        .fileImporter(isPresented: $isChoosingCOLMAP, allowedContentTypes: [.folder]) { result in
            handleImport(result, arkit: false)
        }
        .fileImporter(isPresented: $isChoosingARKit, allowedContentTypes: [.folder]) { result in
            handleImport(result, arkit: true)
        }
    }

    // MARK: - Sections

    private var importSection: some View {
        Section("Import") {
            Button("Import COLMAP\u{2026}") {
                isChoosingCOLMAP = true
            }
            Button("Import ARKit Capture\u{2026}") {
                isChoosingARKit = true
            }
        }
    }

    private func captureSection(manifest: ProjectManifest) -> some View {
        Section("Capture") {
            LabeledContent("Images", value: manifest.imageFiles.count.formatted())
            LabeledContent("Source", value: sourceName(manifest.source))
        }
    }

    private var trainingSection: some View {
        Section("Training") {
            trainingControls
            if let progress = model.progress {
                ProgressView(value: Double(progress.iteration),
                             total: Double(max(progress.totalIterations, 1)))
                LabeledContent("Iteration", value: "\(progress.iteration) of \(progress.totalIterations)")
                LabeledContent("Loss", value: progress.loss.map { String(format: "%.4f", $0) } ?? "\u{2014}")
                LabeledContent("PSNR", value: progress.psnr.map { String(format: "%.2f dB", $0) } ?? "\u{2014}")
                LabeledContent("Gaussians", value: progress.gaussianCount.formatted())
                LabeledContent("Iterations/s", value: String(format: "%.1f", progress.iterationsPerSecond))
            }
        }
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

    private func handleImport(_ result: Result<URL, Error>, arkit: Bool) {
        switch result {
        case .success(let url):
            if arkit {
                model.importARKit(folder: url, into: document)
            } else {
                model.importCOLMAP(folder: url, into: document)
            }
        case .failure(let error):
            model.errorMessage = error.localizedDescription
        }
    }

    private func sourceName(_ source: PoseSource?) -> String {
        switch source {
        case .colmap?: "COLMAP"
        case .arkit?: "ARKit"
        case nil: "\u{2014}"
        }
    }
}
