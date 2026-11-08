import SwiftUI
import SplatCore
import SplatDepth
import SplatIO
import SplatTraining

/// The "Sparse-View Quality" inspector section: depth prior, artifact regularizers, pseudo-views and
/// leave-one-out early stopping, plus the depth import and the per-image depth status. Plain `Form`
/// controls; the Train button stays the only prominent action. Expert mode only; collapses as one group.
struct SparseViewSettingsSection: View {
    let document: Binding<ProjectDocument>
    let model: ProjectModel
    let pipeline: PipelineModel
    let isExpanded: Binding<Bool>

    @State private var isImportingDepth: Bool = false

    /// Anisotropy and isolation pruning switch on and off together. The scale cap is its own toggle
    /// and off by default: it hurt reconstruction quality in the measurements.
    private static let artifactKinds: [RegularizerKind] = [.anisotropy, .isolationPrune]

    var body: some View {
        let contents: ProjectContents = document.wrappedValue.contents
        let configuration: TrainingConfiguration = contents.manifest.training
        // Supplied maps, or the cache of the selected estimator. Another estimator's cache does not count.
        let depthPrefix: String = ProjectLayout.depthDirectory + "/"
        let selectedCachePrefix: String = ProjectLayout.estimatedDepthDirectory(for: configuration.depthPrior.estimator) + "/"
        let hasDepthMaps: Bool = contents.files.keys.contains { key in
            key.hasPrefix(depthPrefix) && (!key.hasPrefix(depthPrefix + "estimated/") || key.hasPrefix(selectedCachePrefix))
        }
        let depthAvailable: Bool = hasDepthMaps || pipeline.isDepthReady(configuration.depthPrior.estimator)
        let depthOn: Bool = configuration.isEnabled(.depthPearson)
        let imageCount: Int = contents.manifest.imageFiles.count
        let minimumViews: Int = configuration.earlyStopping.minimumViews

        Section(isExpanded: isExpanded) {
            Toggle("Depth prior", isOn: depthBinding)
                .disabled(model.isTraining || (!depthAvailable && !depthOn))
                .help("Uses a depth estimate for each view to keep the shapes plausible when there are few photos.")
            if depthAvailable {
                Text("Guides the shape of each view with depth.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else {
                Text("Needs depth. Import depth maps, or download the depth model under Depth Model.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            LabeledContent("Depth weight", value: String(format: "%.3f", configuration.depthPrior.pearsonWeight))
            Slider(value: weightBinding, in: 0...0.2, step: 0.005) {
                Text("Depth weight")
            }
            .labelsHidden()
            .disabled(model.isTraining || !depthOn)
            .help("How strongly depth guides the shapes. Only used while the depth prior is on.")
            if !depthOn {
                Text("Turn on the depth prior to change the weight.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Toggle("Artifact regularizers", isOn: artifactBinding)
                .disabled(model.isTraining)
                .help("Penalizes needle-like and isolated Gaussians, which show up as floaters.")
            Text("Limits needle-like and isolated Gaussians.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Toggle("Scale cap", isOn: kindBinding(.scaleCap))
                .disabled(model.isTraining)
                .help("Limits how large a single Gaussian can grow. Off by default because it hurt sharpness.")
            Text("Caps oversized Gaussians. Can cost sharpness.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Toggle("Pseudo-views", isOn: kindBinding(.pseudoViews))
                .disabled(model.isTraining)
                .help("Adds training views between neighbouring cameras, so the sides of the scene get constrained.")
            Text("Trains on views between neighbouring cameras. Adds training time.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Toggle("Early stopping (leave-one-out)", isOn: earlyStoppingBinding)
                .disabled(model.isTraining || imageCount < minimumViews)
                .help("Holds out one photo at a time to find the iteration that generalizes best.")
            if imageCount < minimumViews {
                Text("Needs at least \(minimumViews) images. Early stopping is unavailable for this project.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else {
                Text("Holds out one view to choose the best iteration.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Button("Import Depth Maps\u{2026}") {
                isImportingDepth = true
            }
            .buttonStyle(.bordered)
            .disabled(model.isTraining)
            .help("Use depth maps you already have, one per photo, instead of the depth model.")
            if model.isTraining {
                Text("Import depth maps after the run finishes.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            DisclosureGroup("Depth status") {
                depthStatusRows
            }
        } header: {
            Text("Sparse-View Quality")
        }
        .fileImporter(isPresented: $isImportingDepth, allowedContentTypes: [.folder]) { result in
            handleDepthFolder(result)
        }
    }

    // MARK: - Depth status

    @ViewBuilder
    private var depthStatusRows: some View {
        if let rows = model.depthStatuses {
            if rows.isEmpty {
                Text("No images.")
                    .foregroundStyle(.secondary)
            }
            ForEach(rows) { row in
                VStack(alignment: .leading, spacing: 2) {
                    Text(row.name)
                    Text(Self.statusText(row.status))
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
        } else {
            Text("Not checked yet. Depth is checked when training starts.")
                .foregroundStyle(.secondary)
        }
    }

    /// Plain-language description of one view's depth outcome.
    static func statusText(_ status: ViewDepthStatus) -> String {
        switch status {
        case .none:
            return "No depth for this view"
        case .measuredMetric:
            return "Measured depth, used directly"
        case .aligned(let result):
            let agreement: String = "\(result.inliers) of \(result.samples) points agree"
            return result.kindCorrected
                ? "Aligned to the sparse points (\(agreement); kind corrected)"
                : "Aligned to the sparse points (\(agreement))"
        case .unaligned(let failure):
            return "Not aligned (\(failureText(failure))). Only the depth correlation applies."
        case .rejected(let reason):
            return "Not used: \(reason)"
        }
    }

    private static func failureText(_ failure: AlignmentFailure) -> String {
        switch failure {
        case .tooFewPoints(let count):
            return "only \(count) matching sparse points"
        case .noPositiveFit:
            return "no consistent scale"
        case .tooFewInliers(let inliers, let of):
            return "only \(inliers) of \(of) points agree"
        }
    }

    // MARK: - Import

    private func handleDepthFolder(_ result: Result<URL, Error>) {
        switch result {
        case .success(let folder):
            model.importDepthMaps(folder: folder, into: document)
        case .failure(let error):
            model.errorMessage = ProjectModel.describe(error)
        }
    }

    // MARK: - Bindings into the stored training settings

    private var depthBinding: Binding<Bool> {
        Binding(
            get: { document.wrappedValue.contents.manifest.training.isEnabled(.depthPearson) },
            set: { enabled in
                setKinds([.depthPearson, .depthL1], enabled: enabled)
            }
        )
    }

    private var weightBinding: Binding<Double> {
        Binding(
            get: { Double(document.wrappedValue.contents.manifest.training.depthPrior.pearsonWeight) },
            set: { value in
                document.wrappedValue.contents.manifest.training.depthPrior.pearsonWeight = Float(value)
            }
        )
    }

    private var artifactBinding: Binding<Bool> {
        Binding(
            get: {
                let configuration: TrainingConfiguration = document.wrappedValue.contents.manifest.training
                return Self.artifactKinds.allSatisfy { configuration.isEnabled($0) }
            },
            set: { enabled in
                setKinds(Self.artifactKinds, enabled: enabled)
            }
        )
    }

    private var earlyStoppingBinding: Binding<Bool> {
        Binding(
            get: { document.wrappedValue.contents.manifest.training.earlyStopping.isEnabled },
            set: { enabled in
                document.wrappedValue.contents.manifest.training.earlyStopping.isEnabled = enabled
            }
        )
    }

    private func kindBinding(_ kind: RegularizerKind) -> Binding<Bool> {
        Binding(
            get: { document.wrappedValue.contents.manifest.training.isEnabled(kind) },
            set: { enabled in
                setKinds([kind], enabled: enabled)
            }
        )
    }

    private func setKinds(_ kinds: [RegularizerKind], enabled: Bool) {
        for kind in kinds {
            if enabled {
                document.wrappedValue.contents.manifest.training.regularizers.insert(kind)
            } else {
                document.wrappedValue.contents.manifest.training.regularizers.remove(kind)
            }
        }
    }
}
