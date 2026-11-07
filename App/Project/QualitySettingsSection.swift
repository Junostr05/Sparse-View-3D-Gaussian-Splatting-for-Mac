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
            Text("Learns a colour gain and bias per photo, so exposure and white-balance changes do not smear into the scene.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Toggle("Anti-aliasing", isOn: antiAliasingBinding)
                .disabled(model.isTraining)
            Text("Mip-Splatting filter: fewer artifacts when zooming out. Exports contain the filter.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Toggle("MCMC densification", isOn: mcmcBinding)
                .disabled(model.isTraining)
            Text("Moves and adds Gaussians within the budget instead of cloning and splitting.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Toggle("Refine camera poses", isOn: poseRefinementBinding)
                .disabled(model.isTraining)
            Text("Corrects small pose errors while training. Helps pose-free captures.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        } header: {
            Text("Quality")
        } footer: {
            Text("Experimental: not yet measured on your scene.")
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
