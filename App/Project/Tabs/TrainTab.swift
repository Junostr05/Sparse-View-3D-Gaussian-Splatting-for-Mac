import SwiftUI
import SplatCore

/// The Train tab: the training section (preset, Train, Continue, Pause and Stop, progress and the dashboard) and,
/// in Expert mode, the Sparse-View Quality and Quality groups. Needs photos in the project.
struct TrainTab: View {
    let document: Binding<ProjectDocument>
    let model: ProjectModel
    let pipeline: PipelineModel
    let mode: InspectorMode
    @AppStorage("inspector.sparseViewExpanded") private var isSparseViewExpanded: Bool = true
    @AppStorage("inspector.qualityExpanded") private var isQualityExpanded: Bool = false

    var body: some View {
        let hasImages: Bool = !document.wrappedValue.contents.manifest.imageFiles.isEmpty
        Form {
            if hasImages {
                TrainingSection(document: document, model: model, mode: mode)
                if mode == .expert {
                    SparseViewSettingsSection(document: document,
                                              model: model,
                                              pipeline: pipeline,
                                              isExpanded: $isSparseViewExpanded)
                    QualitySettingsSection(document: document, model: model, isExpanded: $isQualityExpanded)
                }
            } else {
                Section {
                    Text("Add a capture in the Capture tab first. Training needs photos.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            // INTEGRATION POINT: train tab (batch queue goes here)
        }
        .formStyle(.grouped)
    }
}
