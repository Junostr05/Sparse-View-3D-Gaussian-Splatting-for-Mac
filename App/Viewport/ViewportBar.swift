import SplatCore
import SwiftUI

/// The floating bar over the bottom centre of the viewport: display mode, overlays, fly mode, framing, camera
/// views and Save Image. Every control calls the `ViewerModel` action that the inspector and the command palette
/// use. It is functional chrome, so its controls are glass in one container; the 3D content gets none.
struct ViewportBar: View {
    @Bindable var model: ViewerModel
    @Namespace private var glass

    var body: some View {
        GlassEffectContainer(spacing: 10) {
            HStack(spacing: 10) {
                displayControls
                overlayMenu
                toolsMenu
                flyButton
                frameAllButton
                cameraMenu
                saveImageButton
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
    }

    // MARK: - Display mode

    private var displayControls: some View {
        HStack(spacing: 8) {
            Menu {
                Picker("Display Mode", selection: $model.displayMode) {
                    ForEach(ViewerDisplayMode.allCases, id: \.self) { mode in
                        Text(mode.title).tag(mode)
                    }
                }
                .pickerStyle(.inline)
                Menu("Heatmap Metric") {
                    Picker("Heatmap Metric", selection: $model.heatmapMetric) {
                        ForEach(HeatmapMetric.allCases, id: \.self) { metric in
                            Text(metric.title).tag(metric)
                        }
                    }
                    .pickerStyle(.inline)
                }
            } label: {
                Label("Display Mode", systemImage: "paintpalette")
            }
            .menuIndicator(.hidden)
            .buttonStyle(.glass)
            .labelStyle(.iconOnly)
            .help("Display mode: splats, points, depth, shapes or heatmap")
            .glassEffectID("display", in: glass)

            Text(model.displayMode.title)
                .foregroundStyle(.primary)
                .lineLimit(1)
                .fixedSize()
                .accessibilityLabel("Display mode: \(model.displayMode.title)")
        }
    }

    // MARK: - Overlays

    private var overlayMenu: some View {
        Menu {
            Toggle("Grid", isOn: $model.showsGrid)
            Toggle("Axes", isOn: $model.showsAxes)
            if !model.cameras.isEmpty {
                Toggle("Cameras", isOn: $model.showsCameras)
                Toggle("Camera Names", isOn: $model.showsCameraNames)
                    .disabled(!model.showsCameras)
            }
        } label: {
            Label("Overlays", systemImage: "square.grid.3x3")
        }
        .menuIndicator(.hidden)
        .buttonStyle(.glass)
        .labelStyle(.iconOnly)
        .help("Show or hide the grid, the axes and the cameras")
        .glassEffectID("overlays", in: glass)
    }

    // MARK: - Tools

    /// Crop box, floor and measuring tools. The tool panel appears beside the viewport while one is active.
    private var toolsMenu: some View {
        Menu {
            Button {
                model.activeTool = model.activeTool == .crop ? .none : .crop
            } label: {
                Label("Crop Box", systemImage: model.activeTool == .crop ? "checkmark" : "crop")
            }
            Button {
                model.activeTool = model.activeTool == .floor ? .none : .floor
            } label: {
                Label("Set Floor", systemImage: model.activeTool == .floor ? "checkmark" : "arrow.down.to.line")
            }
            Button {
                model.activeTool = model.activeTool == .measure ? .none : .measure
            } label: {
                Label("Measure Distance", systemImage: model.activeTool == .measure ? "checkmark" : "ruler")
            }
            if model.activeTool != .none {
                Divider()
                Button("Leave Tool") {
                    model.activeTool = .none
                }
            }
        } label: {
            Label("Tools", systemImage: "wrench.and.screwdriver")
        }
        .menuIndicator(.hidden)
        .buttonStyle(.glass)
        .labelStyle(.iconOnly)
        .help("Crop box, floor and measuring tools. Escape leaves the active tool.")
        .glassEffectID("tools", in: glass)
    }

    // MARK: - Fly mode and framing

    private var flyButton: some View {
        Toggle(isOn: Binding(get: { model.isFlying }, set: { model.setFlying($0) })) {
            Label("Fly Mode", systemImage: "airplane")
        }
        .toggleStyle(.button)
        .buttonStyle(.glass)
        .labelStyle(.iconOnly)
        .help("Fly mode: W A S D move, Q E down and up, Esc to exit")
        .glassEffectID("fly", in: glass)
    }

    private var frameAllButton: some View {
        Button {
            model.apply(.frameAll)
        } label: {
            Label("Frame All", systemImage: "viewfinder")
        }
        .buttonStyle(.glass)
        .labelStyle(.iconOnly)
        .help("Frame the whole scene (Home or period)")
        .glassEffectID("frame", in: glass)
    }

    // MARK: - Camera views

    private var cameraMenu: some View {
        Menu {
            presetButton(.front, "Front")
            presetButton(.back, "Back")
            presetButton(.left, "Left")
            presetButton(.right, "Right")
            presetButton(.top, "Top")
            presetButton(.bottom, "Bottom")
            if !model.cameras.isEmpty {
                Divider()
                ForEach(model.cameras.indices, id: \.self) { index in
                    Button {
                        model.goToCamera(index)
                    } label: {
                        Label(model.cameras[index].name,
                              systemImage: model.activeCameraIndex == index ? "checkmark" : "camera")
                    }
                }
            }
        } label: {
            Label("Camera Views", systemImage: "camera.viewfinder")
        }
        .menuIndicator(.hidden)
        .buttonStyle(.glass)
        .labelStyle(.iconOnly)
        .help("Axis-aligned views and the estimated cameras")
        .glassEffectID("cameras", in: glass)
    }

    private func presetButton(_ preset: ViewPreset, _ title: String) -> some View {
        Button(title) {
            model.apply(.preset(preset))
        }
    }

    // MARK: - Export

    private var saveImageButton: some View {
        Button {
            model.presentedExport = .image
        } label: {
            Label("Save Image", systemImage: "photo")
        }
        .buttonStyle(.glass)
        .labelStyle(.iconOnly)
        .help("Save the current view as a PNG image")
        .glassEffectID("save", in: glass)
    }
}
