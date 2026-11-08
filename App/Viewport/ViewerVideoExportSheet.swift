import AppKit
import SplatCore
import SplatMetal
import SwiftUI
import UniformTypeIdentifiers

/// Output sizes of the flight video (16:9).
private enum VideoResolution: String, CaseIterable, Identifiable {
    case hd720, fullHD

    var id: String { rawValue }

    var size: PixelSize {
        switch self {
        case .hd720: PixelSize(width: 1280, height: 720)
        case .fullHD: PixelSize(width: 1920, height: 1080)
        }
    }

    var title: String {
        "\(size.width) \u{00D7} \(size.height)"
    }
}

/// Exports the flight video: through the estimated cameras in order, as "Play Flight" shows them, or a 360° turntable
/// around the pivot when the project has no cameras. The display mode in effect is used for every frame and there are
/// no overlays. Progress and cancel stay in the sheet until the export finishes or fails.
struct ViewerVideoExportSheet: View {
    let viewer: ViewerModel

    @Environment(\.dismiss) private var dismiss
    @State private var resolution: VideoResolution = .fullHD
    @State private var seconds: Double
    @State private var errorMessage: String?

    init(viewer: ViewerModel) {
        self.viewer = viewer
        _seconds = State(initialValue: ExportGeometry.defaultVideoDuration(cameraCount: viewer.cameraViews.count))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Export Flight Video")
                .font(.headline)
            if let job = viewer.videoExport, job.state == .running {
                progressContent(job)
            } else {
                settingsContent
            }
        }
        .padding(20)
        .frame(width: 420)
        .interactiveDismissDisabled(isWriting)
        .onChange(of: viewer.videoExport?.state) { _, newState in
            if newState == .finished {
                dismiss()
            }
        }
    }

    private var isWriting: Bool {
        viewer.videoExport?.state == .running
    }

    private var frameCount: Int {
        ExportGeometry.frameCount(seconds: seconds, fps: ExportGeometry.videoFrameRate)
    }

    private var settingsContent: some View {
        VStack(alignment: .leading, spacing: 16) {
            Picker("Resolution", selection: $resolution) {
                ForEach(VideoResolution.allCases) { resolution in
                    Text(resolution.title).tag(resolution)
                }
            }
            .pickerStyle(.segmented)
            .accessibilityLabel("Video resolution")
            LabeledContent("Duration", value: "\(Int(seconds.rounded())) s")
            Slider(value: $seconds, in: ExportGeometry.durationRange, step: 1)
                .accessibilityLabel("Video duration in seconds")
            Text(pathDescription)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Text("\(frameCount) frames at \(ExportGeometry.videoFrameRate) fps, H.264 in an MP4. Display: \(viewer.displayMode.title). No overlays.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if let failure = failureMessage {
                Text(failure)
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
                Button("Export\u{2026}") {
                    chooseDestination()
                }
                .keyboardShortcut(.defaultAction)
            }
        }
    }

    private func progressContent(_ job: ViewerVideoExportJob) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            ProgressView(value: Double(job.framesWritten), total: Double(max(job.frameCount, 1)))
                .accessibilityLabel("Video export progress")
            Text("Frame \(job.framesWritten) of \(job.frameCount)")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .monospacedDigit()
            HStack {
                Spacer()
                Button("Cancel Export", role: .cancel) {
                    job.cancel()
                }
                .keyboardShortcut(.cancelAction)
            }
        }
    }

    /// The failure of the last export, or the reason the export could not start.
    private var failureMessage: String? {
        if let errorMessage {
            return errorMessage
        }
        if case .failed(let message) = viewer.videoExport?.state {
            return "The video could not be exported: \(message)"
        }
        return nil
    }

    private var pathDescription: String {
        if viewer.cameraViews.isEmpty {
            return "No estimated cameras: a 360\u{00B0} turn around the pivot at the current elevation."
        }
        let count: Int = viewer.cameraViews.count
        return "Through the \(count) estimated \(count == 1 ? "camera" : "cameras") in order, as Play Flight shows them."
    }

    private func chooseDestination() {
        errorMessage = nil
        guard let renderer = ViewerImageRenderer(model: viewer) else {
            errorMessage = "The viewer is not ready yet."
            return
        }
        let size = resolution.size
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.mpeg4Movie]
        panel.nameFieldStringValue = "Flight.mp4"
        panel.canCreateDirectories = true
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            Task {
                startExport(to: url, size: size, renderer: renderer)
            }
        }
    }

    private func startExport(to url: URL, size: PixelSize, renderer: ViewerImageRenderer) {
        let count: Int = frameCount
        let keyframes: [ViewCamera] = viewer.cameraViews
        let path: [ViewCamera] = keyframes.isEmpty
            ? FlightPathSampler.turntable(around: viewer.view, frameCount: count)
            : FlightPathSampler.flight(through: keyframes, frameCount: count)
        viewer.videoExport = ViewerVideoExportJob(destination: url, path: path, size: size,
                                                  frameRate: ExportGeometry.videoFrameRate,
                                                  renderer: renderer, model: viewer)
    }
}
