import AppKit
import SwiftUI
import UniformTypeIdentifiers
import SplatIO

/// One choice in the export sheet: a splat file format, or a self-contained web page.
enum ExportChoice: Hashable, Identifiable {
    case cloud(ExportFormat)
    case webPage

    /// The choices in the order the sheet lists them.
    static let all: [ExportChoice] = ExportFormat.allCases.map { ExportChoice.cloud($0) } + [ExportChoice.webPage]

    var id: Self { self }

    var displayName: String {
        switch self {
        case .cloud(let format): format.displayName
        case .webPage: "Web Page (.html)"
        }
    }

    var fileExtension: String {
        switch self {
        case .cloud(let format): format.fileExtension
        case .webPage: "html"
        }
    }
}

/// Chooses an export format, encodes the displayed cloud and then asks where to save it.
///
/// The save step uses `NSSavePanel` rather than `fileExporter`: in the current SDK `fileExporter` only
/// accepts `WritableDocument` types, which a `FileDocument` wrapper does not satisfy.
struct ExportSheet: View {
    let model: ViewerModel
    let defaultName: String
    let onFailure: (String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var choice: ExportChoice
    @State private var isWorking = false
    @State private var maxGaussians: Int = WebViewerOptions.defaultMaxGaussians

    init(model: ViewerModel,
         defaultName: String,
         onFailure: @escaping (String) -> Void,
         initialChoice: ExportChoice = .cloud(.ply)) {
        self.model = model
        self.defaultName = defaultName
        self.onFailure = onFailure
        _choice = State(initialValue: initialChoice)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Export")
                .font(.headline)
            Picker("Format", selection: $choice) {
                ForEach(ExportChoice.all) { option in
                    Text(option.displayName).tag(option)
                }
            }
            .pickerStyle(.inline)
            if choice == .cloud(.splat) {
                Text("The .splat format keeps only the base colour of each Gaussian. View-dependent colour from the higher-order spherical harmonics is dropped.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if choice == .webPage {
                webPageOptions
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

    /// The limit on Gaussians, the size estimate and what the page leaves out.
    private var webPageOptions: some View {
        let total: Int = model.scene.cloud.count
        let included: Int = min(total, max(1, maxGaussians))
        let estimate: String = ByteCountFormatter.string(
            fromByteCount: Int64(WebViewerExporter.estimatedByteCount(gaussianCount: included)),
            countStyle: .file)
        return VStack(alignment: .leading, spacing: 8) {
            LabeledContent("Maximum Gaussians") {
                TextField("Maximum Gaussians", value: $maxGaussians, format: .number)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 110)
            }
            Text("The page includes \(included.formatted()) of \(total.formatted()) Gaussians. When the scene is larger, the least important ones are left out. Estimated size: about \(estimate).")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Text("Only the base colour of each Gaussian is exported. View-dependent shading from the higher-order spherical harmonics is not included. The page opens in any current browser with WebGL 2 and needs no network connection.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var contentType: UTType {
        UTType(filenameExtension: choice.fileExtension) ?? .data
    }

    /// Encodes the cloud off the main actor, then opens the save panel.
    private func prepare() async {
        isWorking = true
        defer { isWorking = false }
        let data: Data
        do {
            data = try await encode()
        } catch {
            onFailure("The scene could not be exported: \(error.localizedDescription)")
            dismiss()
            return
        }
        save(data)
    }

    /// The bytes of the chosen format.
    private func encode() async throws -> Data {
        switch choice {
        case .cloud(let format):
            return try await model.exportData(format)
        case .webPage:
            let export = try await model.exportWebPage(maxGaussians: max(1, maxGaussians), title: defaultName)
            return export.data
        }
    }

    /// Asks for a location. The bytes are written when the person confirms, and the sheet closes after a write.
    private func save(_ data: Data) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [contentType]
        panel.nameFieldStringValue = "\(defaultName).\(choice.fileExtension)"
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
