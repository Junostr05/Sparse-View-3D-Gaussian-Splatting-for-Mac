import SwiftUI
import SplatCore
import SplatIO

/// The Capture tab: the pipeline steps, the capture summary and photo triage once the project has photos, the
/// frame and pose entry points while it has none, and the depth and pose-free models in Expert mode.
struct CaptureTab: View {
    let document: Binding<ProjectDocument>
    let model: ProjectModel
    let pipeline: PipelineModel
    let mode: InspectorMode
    @AppStorage("inspector.depthModelExpanded") private var isDepthModelExpanded: Bool = false
    @AppStorage("inspector.poseFreeModelExpanded") private var isPoseFreeModelExpanded: Bool = false

    var body: some View {
        let manifest: ProjectManifest = document.wrappedValue.contents.manifest
        let hasImages: Bool = !manifest.imageFiles.isEmpty
        Form {
            PipelineStepList(manifest: manifest, pipeline: pipeline, project: model)
            if hasImages {
                captureSection(manifest: manifest)
                PhotoTriageSection(document: document, model: model)
            } else {
                PipelineInspector(document: document, project: model, pipeline: pipeline)
            }
            if mode == .expert {
                DepthModelSection(pipeline: pipeline,
                                  document: document,
                                  isTraining: model.isTraining,
                                  isExpanded: $isDepthModelExpanded)
                PoseFreeModelSection(pipeline: pipeline, isExpanded: $isPoseFreeModelExpanded)
            }
        }
        .formStyle(.grouped)
    }

    /// Photo count (with the excluded ones taken out), source and poses.
    private func captureSection(manifest: ProjectManifest) -> some View {
        let total: Int = manifest.imageFiles.count
        let excluded: Set<String> = Set(manifest.excludedImages)
        let used: Int = manifest.imageFiles.filter { !excluded.contains($0) }.count
        let imageText: String = excluded.isEmpty
            ? total.formatted()
            : "\(used.formatted()) of \(total.formatted()) used"
        return Section("Capture") {
            LabeledContent("Images", value: imageText)
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
