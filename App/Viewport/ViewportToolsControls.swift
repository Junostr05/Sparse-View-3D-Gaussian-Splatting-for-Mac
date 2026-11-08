import SwiftUI
import SplatCore

/// The panel of the viewport tools: the tool picker, the numeric fields, and the apply and reset actions of the
/// active tool. It floats over the viewport, so it is functional chrome and carries glass; the 3D content does not.
/// Every action calls the `ViewerModel` method that the command palette uses.
struct ViewportToolsControls: View {
    @Bindable var model: ViewerModel
    /// The real length typed for the measured segment, in `model.measureUnit`.
    @State private var realLength: Double = 1
    @Namespace private var glass

    var body: some View {
        GlassEffectContainer(spacing: 10) {
            VStack(alignment: .leading, spacing: 12) {
                toolPicker
                switch model.activeTool {
                case .none:
                    EmptyView()
                case .crop:
                    cropControls
                case .floor:
                    floorControls
                case .measure:
                    measureControls
                }
                Text(scaleStatus)
                    .font(.footnote)
                    .foregroundStyle(.primary)
            }
            .padding(12)
            .frame(width: 280, alignment: .leading)
            .glassEffect(.regular, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .glassEffectID("tool-panel", in: glass)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Viewport tools")
    }

    // MARK: - Picker

    private var toolPicker: some View {
        HStack(spacing: 6) {
            ForEach([ViewerTool.crop, .floor, .measure], id: \.self) { tool in
                Button(tool.title) {
                    model.activeTool = model.activeTool == tool ? .none : tool
                }
                .buttonStyle(.glass)
                .tint(model.activeTool == tool ? Color.accentColor : nil)
                .accessibilityLabel(tool.title + (model.activeTool == tool ? ", active" : ""))
            }
            Spacer(minLength: 0)
            Button("Done") {
                model.activeTool = .none
            }
            .buttonStyle(.glass)
            .disabled(model.activeTool == .none)
            .help("Leave the tool (Escape)")
        }
    }

    // MARK: - Crop

    private var cropControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Drag a face to resize the box, its centre to move it. Shift keeps the proportions, Option resizes from the centre.")
                .font(.footnote)
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
            Toggle("Hide outside the box", isOn: $model.clipsToCropBox)
            HStack {
                Button("Delete Outside") {
                    model.deleteOutsideCrop()
                }
                .disabled(!model.canEdit || model.crop == nil)
                .help("Delete the Gaussians outside the box. Undoable.")
                Button("Remove Box") {
                    model.setCrop(nil)
                }
                .disabled(model.crop == nil)
            }
        }
    }

    // MARK: - Floor

    private var floorControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Click three to six points on the ground. \(model.floorPoints.count) of 6 set.")
                .font(.footnote)
                .foregroundStyle(.primary)
            HStack {
                Button("Apply Floor") {
                    model.applyFloor()
                }
                .buttonStyle(.glassProminent)
                .disabled(model.floorPoints.count < 3)
                Button("Clear Points") {
                    model.floorPoints = []
                }
                .disabled(model.floorPoints.isEmpty)
            }
            Button("Reset Floor") {
                model.resetFloor()
            }
        }
    }

    // MARK: - Measure and scale

    private var measureControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let label = model.measureLabel {
                Text("Distance: \(label)")
                    .font(.headline.monospacedDigit())
                    .foregroundStyle(.primary)
            } else {
                Text("Click two points on the scene.")
                    .font(.footnote)
                    .foregroundStyle(.primary)
            }
            HStack {
                TextField("Real length", value: $realLength, format: .number)
                    .frame(width: 90)
                    .accessibilityLabel("Real length of the measured segment")
                Picker("Unit", selection: $model.measureUnit) {
                    ForEach(LengthUnit.allCases) { unit in
                        Text(unit.title).tag(unit)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }
            HStack {
                Button("Set Scale") {
                    model.applyMeasuredScale(realLength: realLength, unit: model.measureUnit)
                }
                .buttonStyle(.glassProminent)
                .disabled(model.measurePoints.count != 2 || realLength <= 0)
                Button("Reset Scale") {
                    model.resetScale()
                }
            }
        }
    }

    private var scaleStatus: String {
        let transform: SceneTransform = model.sceneTransform
        if transform.isMetric {
            return "Scale: metric (measurements in \(model.measureUnit.title))"
        }
        if transform.scale != 1 {
            return "Scale: scene units, x" + String(format: "%.3f", transform.scale)
        }
        return "Scale: scene units (not calibrated)"
    }
}
