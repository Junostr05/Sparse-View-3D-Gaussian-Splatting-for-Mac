import AppKit
import SplatCore
import SplatMetal
import SwiftUI
import UniformTypeIdentifiers

/// Width presets of the image export. The height follows the aspect ratio chosen next to them.
private enum ImageSizePreset: String, CaseIterable, Identifiable {
    case current, fullHD, fourK, twiceCurrent

    var id: String { rawValue }

    var title: String {
        switch self {
        case .current: "Current size"
        case .fullHD: "1920 \u{00D7} 1080"
        case .fourK: "3840 \u{00D7} 2160"
        case .twiceCurrent: "2\u{00D7} current"
        }
    }
}

/// Aspect ratio of the image export: the viewport's, or 16:9.
private enum ImageAspect: String, CaseIterable, Identifiable {
    case matchView, widescreen

    var id: String { rawValue }

    var title: String {
        switch self {
        case .matchView: "Match view aspect"
        case .widescreen: "16:9"
        }
    }
}

private extension ExportSizeError {
    var message: String {
        switch self {
        case .empty:
            "The image size is empty."
        case .tooLarge(let maxSide, _):
            "The largest image is \(maxSide) \u{00D7} \(maxSide) pixels. Choose a smaller size."
        }
    }
}

/// Saves the current view of the viewer as a PNG. The image is rendered like the viewport (display mode, colours,
/// deletions, SH limit, background) without the overlays. The save panel comes first, then the frame is rendered
/// and written; errors stay in the sheet.
struct ViewerImageExportSheet: View {
    let viewer: ViewerModel

    @Environment(\.dismiss) private var dismiss
    @State private var preset: ImageSizePreset = .current
    @State private var aspect: ImageAspect = .matchView
    @State private var errorMessage: String?
    @State private var isWriting = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Save Image")
                .font(.headline)
            Picker("Size", selection: $preset) {
                ForEach(ImageSizePreset.allCases) { preset in
                    Text(preset.title).tag(preset)
                }
            }
            .pickerStyle(.menu)
            .accessibilityLabel("Image size")
            Picker("Aspect", selection: $aspect) {
                ForEach(ImageAspect.allCases) { aspect in
                    Text(aspect.title).tag(aspect)
                }
            }
            .pickerStyle(.segmented)
            .accessibilityLabel("Aspect ratio")
            LabeledContent("Output", value: "\(planned.width) \u{00D7} \(planned.height) px")
            Text("Renders the current view with the viewport's display mode, background and deletions. No overlays. Width presets: with Match view aspect the height follows the viewport.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if let errorMessage {
                Text(errorMessage)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)
                Button("Save\u{2026}") {
                    save()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(isWriting || ExportGeometry.validate(planned) != nil)
            }
        }
        .padding(20)
        .frame(width: 400)
    }

    /// The pixel size of the viewport, or 1920 × 1080 before the viewport has drawn once.
    private var viewportSize: PixelSize {
        viewer.viewportPixelSize ?? PixelSize(width: 1920, height: 1080)
    }

    /// The size the preset and the aspect choice give, with even sides.
    private var planned: PixelSize {
        let width: Int
        switch preset {
        case .current: width = viewportSize.width
        case .fullHD: width = 1920
        case .fourK: width = 3840
        case .twiceCurrent: width = 2 * viewportSize.width
        }
        let ratio: Double
        switch aspect {
        case .matchView: ratio = Double(viewportSize.width) / Double(max(viewportSize.height, 1))
        case .widescreen: ratio = 16.0 / 9.0
        }
        return ExportGeometry.size(width: width, aspect: ratio)
    }

    private func save() {
        errorMessage = nil
        let size = planned
        if let problem = ExportGeometry.validate(size) {
            errorMessage = problem.message
            return
        }
        // About three frames of BGRA: the GPU texture, the copy and the PNG encoder's input.
        let physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory
        guard ExportGeometry.fitsInMemory(size, copies: 3, physicalMemory: physicalMemory) else {
            let needed: String = ByteCountFormatter.string(
                fromByteCount: Int64(ExportGeometry.byteCount(of: size, copies: 3)), countStyle: .memory)
            errorMessage = "The image needs about \(needed) of memory, more than this Mac can spare. Choose a smaller size."
            return
        }
        guard let renderer = ViewerImageRenderer(model: viewer) else {
            errorMessage = "The viewer is not ready yet."
            return
        }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "Viewer Image.png"
        panel.canCreateDirectories = true
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            Task {
                await write(to: url, size: size, renderer: renderer)
            }
        }
    }

    /// Renders the current view and writes the PNG. The frame is rendered on the main actor, the PNG encoding is not.
    private func write(to url: URL, size: PixelSize, renderer: ViewerImageRenderer) async {
        isWriting = true
        defer { isWriting = false }
        do {
            let texture = try renderer.render(view: viewer.view, size: size)
            let bytes: Data = renderer.bgraData(of: texture)
            try await Task.detached(priority: .utility) {
                try FrameExport.writePNG(bgra: bytes, width: size.width, height: size.height, to: url)
            }.value
            dismiss()
        } catch {
            errorMessage = "The image could not be saved: \(error.localizedDescription)"
        }
    }
}
