import SwiftUI
import SplatCore

/// The "Quality" inspector group: four experimental training features, all off by default. Plain `Form`
/// controls, edited through the same document bindings as the Sparse-View section and disabled while training.
/// Expert mode only; collapses as one group.
struct QualitySettingsSection: View {
    let document: Binding<ProjectDocument>
    let model: ProjectModel
    let isExpanded: Binding<Bool>

    var body: some View {
        Section(isExpanded: isExpanded) {
            Text("Experimental: measured on synthetic scenes only, not yet on real photos.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Toggle("Exposure compensation", isOn: appearanceBinding)
                .disabled(model.isTraining)
                .help("Learns a colour gain and bias per photo, so exposure changes do not smear into the scene.")
            Text("Learns a colour gain and bias per photo, so exposure and white-balance changes do not smear into the scene. On by default. Measured: +3.5 dB on test scenes with exposure changes, neutral without.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Toggle("Anti-aliasing", isOn: antiAliasingBinding)
                .disabled(model.isTraining)
                .help("Mip-Splatting filter that reduces shimmering when zooming out.")
            Text("Mip-Splatting filter: fewer artifacts when zooming out. Exports contain the filter. No measurable change in PSNR on test scenes.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Toggle("MCMC densification", isOn: mcmcBinding)
                .disabled(model.isTraining)
                .help("Moves and adds Gaussians within the budget instead of cloning and splitting them.")
            Text("Moves and adds Gaussians within the budget instead of cloning and splitting. No reliable PSNR change on test scenes; it uses about three times more Gaussians.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Toggle("Refine camera poses", isOn: poseRefinementBinding)
                .disabled(model.isTraining)
                .help("Corrects small errors in the camera poses while training.")
            Text("Corrects small pose errors while training. No measurable gain on test scenes yet; leave off unless the poses are known to be inaccurate.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            if model.isTraining {
                Text("Stop the run to change these settings.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        } header: {
            Text("Quality")
        }
    }

    // MARK: - Bindings into the stored training settings

    private var appearanceBinding: Binding<Bool> {
        Binding(
            get: { document.wrappedValue.contents.manifest.training.appearance.isEnabled },
            set: { enabled in
                document.wrappedValue.contents.manifest.training.appearance.isEnabled = enabled
            }
        )
    }

    private var antiAliasingBinding: Binding<Bool> {
        Binding(
            get: { document.wrappedValue.contents.manifest.training.antiAliasing.isEnabled },
            set: { enabled in
                document.wrappedValue.contents.manifest.training.antiAliasing.isEnabled = enabled
            }
        )
    }

    private var mcmcBinding: Binding<Bool> {
        Binding(
            get: { document.wrappedValue.contents.manifest.training.mcmc.isEnabled },
            set: { enabled in
                document.wrappedValue.contents.manifest.training.mcmc.isEnabled = enabled
            }
        )
    }

    private var poseRefinementBinding: Binding<Bool> {
        Binding(
            get: { document.wrappedValue.contents.manifest.training.poseRefinement.isEnabled },
            set: { enabled in
                document.wrappedValue.contents.manifest.training.poseRefinement.isEnabled = enabled
            }
        )
    }
}
