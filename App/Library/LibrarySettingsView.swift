import AppKit
import SwiftUI

/// Settings (Cmd-,): the library folder, with Change and Reveal in Finder.
struct LibrarySettingsView: View {
    let folder: LibraryFolder

    var body: some View {
        Form {
            LabeledContent("Library Folder") {
                Text(folder.displayPath ?? "Not chosen")
                    .textSelection(.enabled)
                    .lineLimit(2)
                    .truncationMode(.middle)
                    .multilineTextAlignment(.trailing)
            }
            HStack {
                Button("Change\u{2026}") {
                    folder.chooseWithPanel()
                }
                Button("Reveal in Finder") {
                    if let url = folder.readyURL {
                        NSWorkspace.shared.activateFileViewerSelecting([url])
                    }
                }
                .disabled(folder.readyURL == nil)
            }
            Text("Every project in this folder appears in the library. A project is a .gsproj package.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .formStyle(.grouped)
        .padding()
        .frame(width: 460)
    }
}
