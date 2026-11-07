import SwiftUI
import SplatCore

/// The "Quality" inspector group: four experimental training features, all off by default. Plain `Form`
/// controls, edited through the same document bindings as the Sparse-View section and disabled while training.
struct QualitySettingsSection: View {
    let document: Binding<ProjectDocument>
    let model: ProjectModel

    var body: some View {
        Section {
            Toggle("Exposure compensation", isOn: appearanceBinding)
                .disabled(model.isTraining)
            Text("Learns a colour gain and bias per photo, so exposure and white-balance changes do not smear into the scene. Measured: +3.5 dB on test scenes with exposure changes, neutral without. Can stop training with a non-finite loss on real photos; a fix is in progress.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Toggle("Anti-aliasing", isOn: antiAliasingBinding)
                .disabled(model.isTraining)
            Text("Mip-Splatting filter: fewer artifacts when zooming out. Exports contain the filter. No measurable change in PSNR on test scenes.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Toggle("MCMC densification", isOn: mcmcBinding)
                .disabled(model.isTraining)
            Text("Moves and adds Gaussians within the budget instead of cloning and splitting. No reliable PSNR change on test scenes; it uses about three times more Gaussians.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Toggle("Refine camera poses", isOn: poseRefinementBinding)
                .disabled(model.isTraining)
            Text("Corrects small pose errors while training. No measurable gain on test scenes yet; leave off unless the poses are known to be inaccurate.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        } header: {
            Text("Quality")
        } footer: {
            Text("Experimental: measured on synthetic scenes only, not yet on real photos.")
                .font(.footnote)
                .foregroundStyle(.secondary)
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
