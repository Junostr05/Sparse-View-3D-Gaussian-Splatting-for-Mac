import SwiftUI
import SplatCore

struct SceneInspector: View {
    @Bindable var model: ViewerModel

    var body: some View {
        Form {
            Section("Scene") {
                LabeledContent("Gaussians", value: model.scene.cloud.count.formatted())
                LabeledContent("SH Degree", value: model.scene.cloud.shDegree.formatted())
                if model.scene.removedNonFinite > 0 {
                    LabeledContent("Removed Invalid", value: model.scene.removedNonFinite.formatted())
                }
            }
            Section("Rendering") {
                Picker("SH Degree Limit", selection: $model.shDegreeLimit) {
                    ForEach(0...model.scene.cloud.shDegree, id: \.self) { degree in
                        Text(degree.formatted()).tag(degree)
                    }
                }
                Picker("Background", selection: $model.background) {
                    ForEach(ViewerModel.Background.allCases) { background in
                        Text(background.title).tag(background)
                    }
                }
                Picker("Up Axis", selection: $model.upAxis) {
                    ForEach(UpAxis.allCases, id: \.self) { axis in
                        Text(axis.displayName).tag(axis)
                    }
                }
            }
            Section("Performance") {
                LabeledContent("Frame Time", value: frameTimeText)
                LabeledContent("Visible Gaussians", value: model.stats.map { $0.visibleGaussians.formatted() } ?? "\u{2014}")
                LabeledContent("Tile Intersections", value: model.stats.map { $0.tileIntersections.formatted() } ?? "\u{2014}")
            }
        }
        .formStyle(.grouped)
    }

    private var frameTimeText: String {
        guard let stats = model.stats else { return "\u{2014}" }
        return String(format: "%.1f ms", stats.gpuTimeMilliseconds)
    }
}
