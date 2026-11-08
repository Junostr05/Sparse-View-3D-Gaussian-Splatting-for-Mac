import SplatCore
import SwiftUI

/// The Display controls of the Viewport section: what the viewport draws, with the point size and the heatmap metric
/// shown only for the modes that use them. Plain inspector controls; the 3D content itself carries no glass.
struct ViewportDisplayControls: View {
    @Bindable var viewer: ViewerModel

    var body: some View {
        Picker("Display", selection: $viewer.displayMode) {
            ForEach(ViewerDisplayMode.allCases, id: \.self) { mode in
                Text(mode.title).tag(mode)
            }
        }
        .pickerStyle(.menu)
        Text(footnote)
            .font(.footnote)
            .foregroundStyle(.secondary)
        if viewer.displayMode == .points {
            LabeledContent("Point size", value: "\(Int(viewer.pointSize)) px")
            Slider(value: $viewer.pointSize, in: 1...12, step: 1)
                .accessibilityLabel("Point size")
        }
        if viewer.displayMode == .heatmap {
            Picker("Heatmap", selection: $viewer.heatmapMetric) {
                ForEach(HeatmapMetric.allCases, id: \.self) { metric in
                    Text(metric.title).tag(metric)
                }
            }
            .pickerStyle(.menu)
        }
    }

    /// One line for the mode. While a colour set is being computed the footnote says so.
    private var footnote: String {
        if viewer.isComputingDisplayColors {
            return "Computing\u{2026}"
        }
        switch viewer.displayMode {
        case .splat:
            return "The Gaussians as soft splats."
        case .points:
            return "Each Gaussian drawn as a dot in its colour."
        case .depth:
            return "Depth from the camera, near = warm."
        case .shapes:
            return "Ellipses of the Gaussians, translucent."
        case .heatmap:
            return "Red = suspicious (huge, nearly transparent or isolated Gaussians)."
        }
    }
}
