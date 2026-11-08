import SwiftUI
import SplatCore

/// Clean Up Assistant sheet: finds floaters and artifacts in the viewer's current cloud and proposes them.
///
/// Nothing is deleted without an explicit click on Delete Selected. The proposal is shown through the viewport's
/// selection tint, and deletion goes through `EditHistory`, so Undo and persistence work as for manual deletions.
/// A proposal belongs to the cloud it was computed for: after any edit it is stale and must be found again.
struct CleanupAssistantView: View {
    let viewer: ViewerModel
    @State private var assistant = CleanupAssistantModel()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        Form {
            Section {
                ForEach(CleanupCriterion.allCases) { criterion in
                    criterionRow(criterion)
                }
            } header: {
                Text("Criteria")
            } footer: {
                Text("Each criterion is compared with the scene. Lower sensitivity flags fewer Gaussians.")
            }

            Section {
                HStack(spacing: 12) {
                    Button("Find") {
                        assistant.find(on: viewer)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!canFind)
                    .help("Compute the proposal for the cloud in the viewer")
                    if assistant.isFinding {
                        ProgressView()
                            .controlSize(.small)
                        Text("Analysing \(viewer.displayedCount.formatted()) Gaussians\u{2026}")
                            .foregroundStyle(.secondary)
                    }
                }
                if let note = lockedNote {
                    Text(note)
                        .foregroundStyle(.secondary)
                }
            } header: {
                Text("Find")
            }

            if let proposal = assistant.proposal {
                resultsSections(proposal)
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 480, idealWidth: 520, minHeight: 540)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Done") {
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
            }
        }
    }

    // MARK: - Criteria

    private func criterionRow(_ criterion: CleanupCriterion) -> some View {
        let isEnabled: Bool = assistant.criteria[criterion].isEnabled
        return HStack(spacing: 12) {
            Toggle(isOn: assistant.enabledBinding(for: criterion)) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(criterion.title)
                    Text(criterion.summary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .toggleStyle(.checkbox)
            Picker("Sensitivity", selection: assistant.sensitivityBinding(for: criterion)) {
                ForEach(CleanupSensitivity.allCases) { sensitivity in
                    Text(sensitivity.title).tag(sensitivity)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(width: 180)
            .disabled(!isEnabled)
            .accessibilityLabel("Sensitivity of \(criterion.title)")
        }
    }

    // MARK: - Results

    @ViewBuilder
    private func resultsSections(_ proposal: CleanupProposal) -> some View {
        let isStale: Bool = assistant.isStale(for: viewer)
        Section {
            if isStale {
                Text("The cloud changed after this proposal. Find again to see current results.")
                    .foregroundStyle(.secondary)
            } else if proposal.isEmpty {
                Text("The enabled criteria flag no Gaussians.")
                    .foregroundStyle(.secondary)
            }
            ForEach(proposal.results) { result in
                resultRow(result, cloudCount: proposal.cloudCount)
            }
            LabeledContent("Proposal") {
                Text("\(proposal.count.formatted()) (\(percent(proposal.fraction)))")
                    .monospacedDigit()
            }
            if proposal.wasLimited {
                Text("The safety cap keeps at most \(CleanupAssistant.safetyPercent) % of the Gaussians in one step. "
                     + "The most extreme ones were kept.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        } header: {
            Text("Results")
        }

        Section {
            Toggle(isOn: previewBinding) {
                Text("Show in Viewport")
            }
            .disabled(!canAct(isStale: isStale) || proposal.isEmpty)
            .help("Tint the proposed Gaussians in the viewport")

            Button("Select Proposal") {
                assistant.select(on: viewer)
            }
            .disabled(!canAct(isStale: isStale) || proposal.isEmpty)
            .help("Select the proposed Gaussians. Nothing is deleted.")

            Button(role: .destructive) {
                viewer.performEdit(.deleteSelection)
            } label: {
                Text("Delete Selected (\(viewer.selectedCount.formatted()))")
            }
            .disabled(!viewer.canEdit || viewer.selectedCount == 0)
            .help("Delete the selected Gaussians. Undo restores them.")

            Button("Undo") {
                viewer.performEdit(.undo)
            }
            .disabled(!viewer.canEdit || !viewer.canUndo)
            .help("Restore the Gaussians of the last deletion (\u{2318}Z)")
        } header: {
            Text("Act")
        } footer: {
            Text("Nothing is deleted until you click Delete Selected. Undo (\u{2318}Z) restores deleted Gaussians.")
        }
    }

    private func resultRow(_ result: CleanupCriterionResult, cloudCount: Int) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("\(result.criterion.title): \(result.count.formatted()) (\(percent(share(result.count, of: cloudCount))))")
                .monospacedDigit()
            if result.wasLimited {
                Text("Limited by the safety cap: \(result.matchCount.formatted()) flagged.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - State

    private var canFind: Bool {
        viewer.canEdit && !assistant.isFinding && assistant.hasEnabledCriterion
    }

    /// Actions on the proposal need editing to be allowed and a proposal that still matches the cloud.
    private func canAct(isStale: Bool) -> Bool {
        viewer.canEdit && !isStale
    }

    private var previewBinding: Binding<Bool> {
        Binding(
            get: { assistant.isPreviewing(on: viewer) },
            set: { assistant.setPreview($0, on: viewer) }
        )
    }

    /// Why finding and acting are unavailable, or nil when they are available.
    private var lockedNote: String? {
        if !viewer.isEditingAllowed {
            return "Not available while a training preview is shown."
        }
        if viewer.videoExport?.state == .running {
            return "Not available while a flight video is written."
        }
        return nil
    }

    private func share(_ count: Int, of total: Int) -> Double {
        total > 0 ? Double(count) / Double(total) : 0
    }

    private func percent(_ fraction: Double) -> String {
        fraction.formatted(.percent.precision(.fractionLength(0...1)))
    }
}

/// State of the Clean Up Assistant sheet: the criteria, the last proposal and the cloud revision it belongs to.
@Observable
final class CleanupAssistantModel {
    var criteria: CleanupCriteria = .standard
    /// The last computed proposal. `nil` before the first Find.
    private(set) var proposal: CleanupProposal?
    /// `ViewerModel.cloudRevision` the proposal was computed for. Any later revision makes the proposal stale.
    private(set) var proposalRevision: Int = -1
    private(set) var isFinding = false
    @ObservationIgnored private var findTask: Task<Void, Never>?

    var hasEnabledCriterion: Bool {
        CleanupCriterion.allCases.contains { criteria[$0].isEnabled }
    }

    func isStale(for viewer: ViewerModel) -> Bool {
        proposal != nil && proposalRevision != viewer.cloudRevision
    }

    /// Computes the proposal for the viewer's displayed cloud off the main actor.
    func find(on viewer: ViewerModel) {
        findTask?.cancel()
        let cloud: GaussianCloud = viewer.scene.cloud
        let revision: Int = viewer.cloudRevision
        let rules: CleanupCriteria = criteria
        isFinding = true
        findTask = Task { [weak self] in
            let result: CleanupProposal = await Task.detached(priority: .userInitiated) {
                CleanupAssistant.propose(cloud: cloud, criteria: rules)
            }.value
            guard let self, !Task.isCancelled else { return }
            self.proposal = result
            self.proposalRevision = revision
            self.isFinding = false
        }
    }

    /// True while the viewer's selection is exactly the proposal, so that the proposal is shown in the viewport.
    func isPreviewing(on viewer: ViewerModel) -> Bool {
        guard let proposal, !isStale(for: viewer), viewer.isEditingActive, !proposal.isEmpty else { return false }
        return viewer.selection == proposal.selection
    }

    /// Shows the proposal as the viewport's selection tint (edit mode on), or removes it again when it is still
    /// the selection. A selection the user made by hand is never replaced by turning the preview off.
    func setPreview(_ isOn: Bool, on viewer: ViewerModel) {
        guard let proposal, !isStale(for: viewer), viewer.canEdit else { return }
        if isOn {
            viewer.isEditing = true
            viewer.setSelection(proposal.selection)
        } else if viewer.selection == proposal.selection {
            viewer.setSelection(SelectionMask(count: viewer.displayedCount))
        }
    }

    /// Makes the proposal the viewer's selection. Nothing is deleted.
    func select(on viewer: ViewerModel) {
        guard let proposal, !isStale(for: viewer) else { return }
        viewer.setSelection(proposal.selection)
    }

    func enabledBinding(for criterion: CleanupCriterion) -> Binding<Bool> {
        Binding(
            get: { self.criteria[criterion].isEnabled },
            set: { isOn in
                var rule: CleanupRule = self.criteria[criterion]
                rule.isEnabled = isOn
                self.criteria[criterion] = rule
            }
        )
    }

    func sensitivityBinding(for criterion: CleanupCriterion) -> Binding<CleanupSensitivity> {
        Binding(
            get: { self.criteria[criterion].sensitivity },
            set: { sensitivity in
                var rule: CleanupRule = self.criteria[criterion]
                rule.sensitivity = sensitivity
                self.criteria[criterion] = rule
            }
        )
    }
}

extension CleanupCriterion {
    /// The name of the criterion, also the start of its result line.
    var title: String {
        switch self {
        case .huge: "Huge Gaussians"
        case .transparent: "Transparent Gaussians"
        case .isolated: "Isolated Gaussians"
        case .needle: "Needle Gaussians"
        case .outsideCore: "Outside-core Gaussians"
        }
    }

    /// One line on what the criterion looks for.
    var summary: String {
        switch self {
        case .huge: "Very large splats compared with the scene"
        case .transparent: "Almost invisible"
        case .isolated: "Far from every neighbour"
        case .needle: "Stretched into thin needles"
        case .outsideCore: "Far outside the dense centre"
        }
    }
}

extension CleanupSensitivity {
    var title: String {
        switch self {
        case .low: "Low"
        case .medium: "Medium"
        case .high: "High"
        }
    }
}
