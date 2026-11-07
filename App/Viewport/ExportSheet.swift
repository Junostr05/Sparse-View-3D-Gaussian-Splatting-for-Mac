import AppKit
import SwiftUI
import UniformTypeIdentifiers
import SplatIO

/// Chooses an export format, encodes the displayed cloud and then asks where to save it.
///
/// The save step uses `NSSavePanel` rather than `fileExporter`: in the current SDK `fileExporter` only
/// accepts `WritableDocument` types, which a `FileDocument` wrapper does not satisfy.
struct ExportSheet: View {
    let model: ViewerModel
    let defaultName: String
    let onFailure: (String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var format: ExportFormat = .ply
    @State private var isWorking = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Export")
                .font(.headline)
            Picker("Format", selection: $format) {
                ForEach(ExportFormat.allCases, id: \.self) { format in
                    Text(format.displayName).tag(format)
                }
            }
            .pickerStyle(.inline)
            if format == .splat {
                Text("The .splat format keeps only the base colour of each Gaussian. View-dependent colour from the higher-order spherical harmonics is dropped.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)
                Button("Export\u{2026}") {
                    Task { await prepare() }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(isWorking)
            }
        }
        .padding(20)
        .frame(width: 380)
    }

    private var contentType: UTType {
        UTType(filenameExtension: format.fileExtension) ?? .data
    }

    /// Encodes the cloud off the main actor, then opens the save panel.
    private func prepare() async {
        isWorking = true
        defer { isWorking = false }
        let data: Data
        do {
            data = try await model.exportData(format)
        } catch {
            onFailure("The scene could not be exported: \(error.localizedDescription)")
            dismiss()
            return
        }
        save(data)
    }

    /// Asks for a location. The bytes are written when the person confirms, and the sheet closes after a write.
    private func save(_ data: Data) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [contentType]
        panel.nameFieldStringValue = "\(defaultName).\(format.fileExtension)"
        panel.canCreateDirectories = true
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            Task {
                do {
                    try await Self.write(data, to: url)
                    dismiss()
                } catch {
                    onFailure("The file could not be saved: \(error.localizedDescription)")
                }
            }
        }
    }

    /// Writes off the main actor, because an export can be hundreds of megabytes.
    private static func write(_ data: Data, to url: URL) async throws {
        try await Task.detached(priority: .utility) {
            try data.write(to: url, options: .atomic)
        }.value
    }
}
