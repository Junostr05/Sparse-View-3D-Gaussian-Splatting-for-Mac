import SwiftUI
import SplatIO

/// Offers to recover an interrupted training run. Shown at the top of the project window when the package holds an
/// autosave that is newer than its latest checkpoint. Recover stores the autosave as the latest checkpoint, with its
/// optimizer state; Discard deletes the autosave files.
struct RecoveryBanner: View {
    let model: ProjectModel
    let document: Binding<ProjectDocument>

    var body: some View {
        if let pending = model.pendingRecovery {
            HStack(alignment: .center, spacing: 12) {
                Image(systemName: "arrow.counterclockwise.circle")
                    .foregroundStyle(.secondary)
                    .accessibilityHidden(true)
                Text("An interrupted training run from \(pending.date.formatted(date: .abbreviated, time: .shortened)) was found (iteration \(pending.iteration)). Recover it?")
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Button("Discard", role: .destructive) {
                    model.discardAutosave()
                }
                Button("Recover") {
                    Task { await model.recoverAutosave(document: document) }
                }
                .buttonStyle(.glassProminent)
            }
            .padding(12)
            .background(.bar)
        }
    }
}
