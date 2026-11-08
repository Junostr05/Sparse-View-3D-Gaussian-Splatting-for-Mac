import SwiftUI
import SplatCore
import SplatIO
import SplatTraining

/// How much of the project inspector is shown. Basic keeps the essentials; Expert shows every setting.
/// Stored in `@AppStorage("inspectorMode")`.
enum InspectorMode: String, CaseIterable, Identifiable {
    case basic
    case expert

    var id: Self { self }

    var title: String {
        switch self {
        case .basic: "Basic"
        case .expert: "Expert"
        }
    }
}

/// The four tabs of the project inspector. The selected tab is kept per window in `@SceneStorage`.
enum InspectorTab: String, CaseIterable, Identifiable {
    case capture
    case train
    case view
    case export

    var id: Self { self }

    var title: String {
        switch self {
        case .capture: "Capture"
        case .train: "Train"
        case .view: "View"
        case .export: "Export"
        }
    }
}

/// The project inspector: a tab bar over four tabs (Capture, Train, View, Export) and the Basic/Expert switch.
/// Each tab is its own view in `App/Project/Tabs`. Errors and the training warning sit above the tabs, so they
/// show whichever tab is open. Starting a training run selects the Train tab; nothing else changes the tab.
struct ProjectInspector: View {
    let document: Binding<ProjectDocument>
    let model: ProjectModel
    let pipeline: PipelineModel
    @AppStorage("inspectorMode") private var mode: InspectorMode = .basic
    @SceneStorage("project.inspectorTab") private var selectedTab: InspectorTab = .capture

    var body: some View {
        VStack(spacing: 0) {
            header
            if hasErrors {
                errorBanners
            }
            if let message = model.warningMessage {
                Text(message)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16)
                    .padding(.bottom, 8)
            }
            tabContent
        }
        .onChange(of: model.isTraining) { _, isTraining in
            if isTraining {
                selectedTab = .train
            }
        }
    }

    // MARK: - Header

    /// The tab bar and the Basic/Expert switch. Both are system segmented controls.
    private var header: some View {
        VStack(spacing: 8) {
            Picker("Inspector section", selection: $selectedTab) {
                ForEach(InspectorTab.allCases) { tab in
                    Text(tab.title).tag(tab)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .accessibilityLabel("Inspector section")

            HStack(spacing: 8) {
                Picker("Detail", selection: $mode) {
                    ForEach(InspectorMode.allCases) { option in
                        Text(option.title).tag(option)
                    }
                }
                .pickerStyle(.segmented)
                .controlSize(.small)
                .labelsHidden()
                .fixedSize()
                .help("Basic shows the essentials. Expert shows every setting, in groups you can collapse.")
                Text(mode == .basic ? "Essentials" : "Every setting")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                Spacer(minLength: 0)
            }
        }
        .padding(.horizontal, 16)
        .padding(.top, 12)
        .padding(.bottom, 8)
    }

    // MARK: - Tabs

    @ViewBuilder
    private var tabContent: some View {
        switch selectedTab {
        case .capture:
            CaptureTab(document: document, model: model, pipeline: pipeline, mode: mode)
        case .train:
            TrainTab(document: document, model: model, pipeline: pipeline, mode: mode)
        case .view:
            ViewTab(model: model, mode: mode)
        case .export:
            ExportTab(model: model)
        }
    }

    // MARK: - Errors

    private var hasErrors: Bool {
        model.errorMessage != nil || pipeline.errorMessage != nil
    }

    /// Errors from the project and the pipeline. Each message has a Dismiss button.
    private var errorBanners: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let message = model.errorMessage {
                errorRow(message) {
                    model.errorMessage = nil
                }
            }
            if let message = pipeline.errorMessage {
                errorRow(message) {
                    pipeline.errorMessage = nil
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.bottom, 8)
    }

    private func errorRow(_ message: String, dismiss: @escaping () -> Void) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .accessibilityHidden(true)
                Text(message)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Button("Dismiss", action: dismiss)
                .buttonStyle(.bordered)
                .accessibilityHint("Hides this error message.")
        }
    }
}
