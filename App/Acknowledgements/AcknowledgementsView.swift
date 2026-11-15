import SwiftUI

/// Identifier of the Acknowledgements window scene.
enum AcknowledgementsWindow {
    static let id: String = "acknowledgements"
}

/// Opens the Acknowledgements window from the application menu, after About.
struct AcknowledgementsCommands: Commands {
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        CommandGroup(after: .appInfo) {
            Button("Acknowledgements") {
                openWindow(id: AcknowledgementsWindow.id)
            }
        }
    }
}

/// Third-party notices of the app: every `.txt` file in `Contents/Resources/Licenses` of the bundle, in name order.
/// The folder is only present in builds that bundle a third-party model (see `Tools/release/make_dmg.sh`).
struct AcknowledgementsView: View {
    /// One notice file: its name and its text.
    struct Notice: Identifiable, Equatable {
        let id: String
        let text: String
    }

    @State private var notices: [Notice] = []

    var body: some View {
        ScrollView {
            if notices.isEmpty {
                Text("No bundled third-party models.")
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            } else {
                VStack(alignment: .leading, spacing: 24) {
                    ForEach(notices) { notice in
                        VStack(alignment: .leading, spacing: 8) {
                            Text(notice.id)
                                .font(.headline)
                            Text(notice.text)
                                .font(.system(.footnote, design: .monospaced))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
                .padding()
            }
        }
        .frame(minWidth: 560, minHeight: 380)
        .navigationTitle("Acknowledgements")
        .task {
            notices = Self.loadNotices()
        }
    }

    /// The notices in the bundle's `Licenses` folder. Files that cannot be read as UTF-8 text are skipped.
    static func loadNotices() -> [Notice] {
        guard let folder: URL = Bundle.main.resourceURL?.appending(path: "Licenses", directoryHint: .isDirectory) else {
            return []
        }
        let files: [URL] = (try? FileManager.default.contentsOfDirectory(at: folder,
                                                                         includingPropertiesForKeys: nil)) ?? []
        var notices: [Notice] = []
        for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) where file.pathExtension == "txt" {
            guard let text = try? String(contentsOf: file, encoding: .utf8) else {
                continue
            }
            notices.append(Notice(id: file.lastPathComponent, text: text))
        }
        return notices
    }
}
