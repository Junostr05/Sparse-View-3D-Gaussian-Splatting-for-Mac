import SwiftUI

/// The Export tab: Save Image, Export Flight Video and the PLY, SPLAT and SPZ exports, then the project info with the
/// app version and the update check. Available in both modes.
struct ExportTab: View {
    let model: ProjectModel
    /// Only used by the "Check for Updates" button. Nothing is checked until the user presses it.
    @State private var updateChecker = UpdateChecker()

    var body: some View {
        Form {
            Section("Export") {
                if let viewer = model.viewer {
                    ViewportExportControls(viewer: viewer)
                } else {
                    Text("Train a model to export it as an image, a video or a splat file.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            Section("Project") {
                Text(Self.versionText)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                Button("Check for Updates") {
                    Task { await updateChecker.check(currentVersion: Self.currentVersion) }
                }
                .buttonStyle(.borderless)
                .disabled(updateChecker.state == .checking)
                .help("Looks for a newer release on GitHub. Nothing is checked until you press this.")
                updateStatus
            }
            // INTEGRATION POINT: export tab (web export goes here)
        }
        .formStyle(.grouped)
    }

    /// The result of the last "Check for Updates" press. Empty until the button has been pressed.
    @ViewBuilder
    private var updateStatus: some View {
        switch updateChecker.state {
        case .idle:
            EmptyView()
        case .checking:
            statusText("Checking\u{2026}")
        case .upToDate(let current):
            statusText("You have the latest version (\(current)).")
        case .updateAvailable(let latest, let url):
            VStack(alignment: .leading, spacing: 2) {
                statusText("Version \(latest) is available.")
                Link("Open Release Page", destination: url)
                    .font(.footnote)
            }
        case .noReleases:
            statusText("No releases published yet.")
        case .failed(let message):
            statusText(message)
        }
    }

    private func statusText(_ text: String) -> some View {
        Text(text)
            .font(.footnote)
            .foregroundStyle(.secondary)
    }

    /// "SparseSplat 1.4.0 (1)" from the bundle's version and build number.
    private static var versionText: String {
        let info: [String: Any] = Bundle.main.infoDictionary ?? [:]
        let version: String = info["CFBundleShortVersionString"] as? String ?? "?"
        let build: String = info["CFBundleVersion"] as? String ?? "?"
        return "SparseSplat \(version) (\(build))"
    }

    /// The bundle's short version string, "1.4.0", compared against the latest release.
    private static var currentVersion: String {
        let info: [String: Any] = Bundle.main.infoDictionary ?? [:]
        return info["CFBundleShortVersionString"] as? String ?? "?"
    }
}
