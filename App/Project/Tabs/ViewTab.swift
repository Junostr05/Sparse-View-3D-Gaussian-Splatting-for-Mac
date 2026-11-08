import SwiftUI
import SplatCore

/// The View tab: up axis, display mode, overlays, camera views and flight, pivot and fly toggles, and the shortcuts.
/// Basic shows the up axis, the display mode and the cameras. Expert adds every viewport control in one group.
/// Export controls live in the Export tab.
struct ViewTab: View {
    let model: ProjectModel
    let mode: InspectorMode
    @AppStorage("inspector.viewportExpanded") private var isViewportExpanded: Bool = true

    var body: some View {
        Form {
            if let viewer = model.viewer {
                if mode == .expert {
                    expertViewportSection(viewer)
                } else {
                    basicViewportSection(viewer)
                }
            } else {
                Section {
                    Text("Train a model to view it here. The display modes, overlays and camera views appear once a model exists.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            if let viewer = model.viewer {
                Section("Clean Up") {
                    Button("Clean Up Assistant\u{2026}") {
                        viewer.presentedCleanup = true
                    }
                    .disabled(!viewer.canEdit)
                    .help("Find huge, transparent or isolated Gaussians and delete them in one undoable step.")
                    Text(viewer.canEdit ? "Proposes floaters and artifacts; nothing is deleted until you confirm."
                                        : "Not available while a run or an export is in progress.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            // INTEGRATION POINT: view tab (crop and floor tools)
        }
        .formStyle(.grouped)
    }

    // MARK: - Sections

    /// Basic viewport: the up axis, the display mode, and the camera picker with the fly-through slider.
    private func basicViewportSection(_ viewer: ViewerModel) -> some View {
        Section("Viewport") {
            Picker("Up Axis", selection: upAxisBinding(viewer)) {
                ForEach(UpAxis.allCases, id: \.self) { axis in
                    Text(axis.displayName).tag(axis)
                }
            }
            .help("The direction that points up in the scene. The start value follows the capture source.")
            Picker("Display", selection: displayModeBinding(viewer)) {
                ForEach(ViewerDisplayMode.allCases, id: \.self) { displayMode in
                    Text(displayMode.title).tag(displayMode)
                }
            }
            .help("How the scene is drawn. Expert mode adds point size, heatmap and the other display options.")
            cameraRows(viewer, includesPlayback: false)
        }
    }

    /// Expert viewport: every viewport control in one group that collapses. The order follows the basic one.
    private func expertViewportSection(_ viewer: ViewerModel) -> some View {
        Section(isExpanded: $isViewportExpanded) {
            Text("Scene view, camera flights and overlays.")
                .font(.footnote)
                .foregroundStyle(.secondary)
            Picker("Up Axis", selection: upAxisBinding(viewer)) {
                ForEach(UpAxis.allCases, id: \.self) { axis in
                    Text(axis.displayName).tag(axis)
                }
            }
            ViewportDisplayControls(viewer: viewer)
            Menu("View") {
                Button("Front") { viewer.apply(.preset(.front)) }
                Button("Back") { viewer.apply(.preset(.back)) }
                Button("Right") { viewer.apply(.preset(.right)) }
                Button("Left") { viewer.apply(.preset(.left)) }
                Button("Top") { viewer.apply(.preset(.top)) }
                Button("Bottom") { viewer.apply(.preset(.bottom)) }
                Divider()
                Button(viewer.isFlying ? "Leave Fly Mode" : "Fly Mode") { viewer.toggleFlying() }
            }
            .help("Jump to a standard view, or fly through the scene with the keyboard.")
            Toggle("Pivot at cursor depth", isOn: Binding(
                get: { viewer.anchorsToCursorDepth },
                set: { viewer.anchorsToCursorDepth = $0 }
            ))
            .help("Orbit and zoom around the surface under the cursor instead of the centre of the scene.")
            ViewerShortcutsButton()
            ViewportOverlayControls(model: viewer)
            cameraRows(viewer, includesPlayback: true)
        } header: {
            Text("Viewport")
        }
    }

    // MARK: - Bindings and camera rows

    private func upAxisBinding(_ viewer: ViewerModel) -> Binding<UpAxis> {
        Binding(get: { viewer.upAxis }, set: { viewer.upAxis = $0 })
    }

    private func displayModeBinding(_ viewer: ViewerModel) -> Binding<ViewerDisplayMode> {
        Binding(get: { viewer.displayMode }, set: { viewer.displayMode = $0 })
    }

    /// The camera picker and the fly-through slider, and in expert mode the playback button. Shown only when the
    /// project has estimated cameras.
    @ViewBuilder
    private func cameraRows(_ viewer: ViewerModel, includesPlayback: Bool) -> some View {
        if !viewer.cameras.isEmpty {
            let cameraSelection = Binding<Int>(get: { viewer.activeCameraIndex ?? 0 },
                                               set: { viewer.goToCamera($0) })
            let flightSelection = Binding<Double>(get: { viewer.flightPosition },
                                                  set: { viewer.scrub(to: $0) })
            let lastPosition: Double = Double(max(viewer.cameras.count - 1, 0))
            Picker("Camera", selection: cameraSelection) {
                ForEach(viewer.cameras.indices, id: \.self) { index in
                    Text(viewer.cameras[index].name).tag(index)
                }
            }
            .help("Look through one of the estimated cameras.")
            if viewer.cameras.count > 1 {
                Slider(value: flightSelection, in: 0...lastPosition) {
                    Text("Fly through")
                }
                .help("Move the view along the camera path.")
            }
            if includesPlayback {
                Button {
                    viewer.togglePlayback()
                } label: {
                    if viewer.isPlaying {
                        Label("Pause Flight", systemImage: "pause.fill")
                    } else {
                        Label("Play Flight", systemImage: "play.fill")
                    }
                }
                .buttonStyle(.bordered)
                .disabled(viewer.cameras.count < 2)
                .help(viewer.cameras.count < 2 ? "Needs at least two cameras." : "Play the camera path as a flight.")
            }
        }
    }
}
