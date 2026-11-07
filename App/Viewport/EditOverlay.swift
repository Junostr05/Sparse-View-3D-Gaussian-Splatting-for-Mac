import SwiftUI
import SplatCore

/// Brush ring and the selected Gaussians, drawn over the viewport. It is part of the content layer, so it
/// carries no glass, and it never takes input: the viewport underneath keeps every mouse event.
struct EditOverlay: View {
    let model: ViewerModel

    /// Most selected Gaussians drawn as dots. Larger selections are sampled evenly.
    private static let maximumDots: Int = 20_000
    /// Radius of a dot in points.
    private static let dotRadius: CGFloat = 1.5
    /// Nearest depth that is projected, in camera units. Matches the brush selector.
    private static let nearDepth: Float = 0.01

    var body: some View {
        // Read the observed state here, so that SwiftUI runs this body again when any of it changes.
        let isActive: Bool = model.isEditingActive
        let camera: ViewCamera = model.view
        let selection: SelectionMask = model.selection
        let cloud: GaussianCloud = model.scene.cloud
        let cursor: CGPoint? = model.brushCursor
        let radius: CGFloat = CGFloat(model.brushRadius)

        Canvas { context, size in
            guard isActive else { return }
            Self.drawSelection(into: &context, size: size, camera: camera, selection: selection, cloud: cloud)
            if let cursor {
                Self.drawBrushRing(into: &context, center: cursor, radius: radius)
            }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    /// Draws one small dot per selected Gaussian centre, at the position the current camera gives it.
    private static func drawSelection(into context: inout GraphicsContext,
                                      size: CGSize,
                                      camera: ViewCamera,
                                      selection: SelectionMask,
                                      cloud: GaussianCloud) {
        let width: Int = Int(size.width.rounded())
        let height: Int = Int(size.height.rounded())
        guard width > 0, height > 0, selection.selectedCount > 0 else { return }
        // Projected in points: the projection scales with the image size, so this matches the pixel view.
        let projection: Camera = camera.camera(width: width, height: height)
        let indices: [Int] = selection.selectedIndices()
        let total: Int = indices.count
        let step: Int = max(1, (total + maximumDots - 1) / maximumDots)

        var dots = Path()
        var sample: Int = 0
        while sample < total {
            let index: Int = indices[sample]
            let base: Int = 3 * index
            let center: SIMD3<Float> = SIMD3(cloud.positions[base], cloud.positions[base + 1], cloud.positions[base + 2])
            if let projected = DepthGeometry.project(center, camera: projection, near: nearDepth) {
                let u: CGFloat = CGFloat(projected.u)
                let v: CGFloat = CGFloat(projected.v)
                if u >= 0, u <= size.width, v >= 0, v <= size.height {
                    dots.addEllipse(in: CGRect(x: u - dotRadius, y: v - dotRadius,
                                               width: 2 * dotRadius, height: 2 * dotRadius))
                }
            }
            sample += step
        }
        context.fill(dots, with: .color(.accentColor))
    }

    /// A dark outline under a white ring, so the brush reads over both bright and dark content.
    private static func drawBrushRing(into context: inout GraphicsContext, center: CGPoint, radius: CGFloat) {
        let ring = Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius,
                                          width: 2 * radius, height: 2 * radius))
        context.stroke(ring, with: .color(.black.opacity(0.6)), lineWidth: 3)
        context.stroke(ring, with: .color(.white), lineWidth: 1.5)
    }
}

/// The floating brush bar at the bottom of the viewport, shown in edit mode. It is the functional layer:
/// one glass container with the brush settings, the selection and history actions and Delete.
struct BrushControls: View {
    @Bindable var model: ViewerModel

    var body: some View {
        GlassEffectContainer(spacing: 10) {
            // One row when it fits, otherwise two rows so that no label wraps.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) {
                    modeAndRadius
                    selectionButtons
                    statusText
                    deleteButton
                }
                .fixedSize(horizontal: true, vertical: false)
                VStack(spacing: 8) {
                    HStack(spacing: 10) {
                        modeAndRadius
                        statusText
                    }
                    HStack(spacing: 10) {
                        selectionButtons
                        deleteButton
                    }
                }
                .fixedSize(horizontal: true, vertical: false)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
    }

    private var modeAndRadius: some View {
        HStack(spacing: 10) {
            Picker("Brush Mode", selection: $model.brushMode) {
                ForEach(BrushMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(width: 150)
            .accessibilityLabel("Brush Mode")
            .help("Add to or subtract from the selection. Hold Shift to subtract.")

            Slider(value: $model.brushRadius, in: 8...200, step: 1) {
                Text("Brush Radius")
            }
            .labelsHidden()
            .frame(width: 120)
            .help("Brush radius in points")
        }
    }

    private var selectionButtons: some View {
        HStack(spacing: 10) {
            actionButton("Select All", systemImage: "checkmark.circle", enabled: model.canEdit) {
                model.performEdit(.selectAll)
            }
            actionButton("Clear Selection", systemImage: "xmark.circle", enabled: model.canEdit) {
                model.performEdit(.clearSelection)
            }
            actionButton("Invert Selection", systemImage: "circle.lefthalf.filled", enabled: model.canEdit) {
                model.performEdit(.invertSelection)
            }
            actionButton("Undo", systemImage: "arrow.uturn.backward", enabled: model.canUndo) {
                model.performEdit(.undo)
            }
            actionButton("Redo", systemImage: "arrow.uturn.forward", enabled: model.canRedo) {
                model.performEdit(.redo)
            }
        }
    }

    private var statusText: some View {
        Text("\(model.selectedCount.formatted()) selected \u{00B7} \(model.deletedCount.formatted()) deleted")
            .monospacedDigit()
            .lineLimit(1)
            .fixedSize()
            .foregroundStyle(.primary)
    }

    private var deleteButton: some View {
        Button {
            model.performEdit(.deleteSelection)
        } label: {
            Label("Delete", systemImage: "trash")
        }
        .buttonStyle(.glassProminent)
        .disabled(!model.canEdit || model.selectedCount == 0)
        .help("Delete the selected Gaussians. Undo restores them.")
    }

    /// A glass icon button with an accessibility label taken from its title.
    private func actionButton(_ title: String,
                              systemImage: String,
                              enabled: Bool,
                              action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
        }
        .buttonStyle(.glass)
        .labelStyle(.iconOnly)
        .disabled(!enabled)
        .help(title)
    }
}
