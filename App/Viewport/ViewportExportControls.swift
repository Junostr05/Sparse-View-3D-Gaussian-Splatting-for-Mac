import SwiftUI

/// The export buttons of the Viewport section: a PNG of the current view and a flight video. Each opens its sheet.
/// Both are disabled while a video is written. The video is also unavailable while a training preview is shown,
/// because the preview's cloud changes during the export.
struct ViewportExportControls: View {
    let viewer: ViewerModel

    @State private var isShowingImageSheet = false
    @State private var isShowingVideoSheet = false

    var body: some View {
        let isWritingVideo: Bool = viewer.videoExport?.state == .running
        Button("Save Image\u{2026}") {
            isShowingImageSheet = true
        }
        .disabled(isWritingVideo)
        .accessibilityLabel("Save image of the current view")
        .sheet(isPresented: $isShowingImageSheet) {
            ViewerImageExportSheet(viewer: viewer)
        }
        Button("Export Flight Video\u{2026}") {
            isShowingVideoSheet = true
        }
        .disabled(isWritingVideo || !viewer.isEditingAllowed)
        .help(viewer.isEditingAllowed
              ? "Renders the cameras in order, or a turntable without cameras."
              : "Not available while a training preview is shown.")
        .accessibilityLabel("Export flight video")
        .sheet(isPresented: $isShowingVideoSheet) {
            ViewerVideoExportSheet(viewer: viewer)
        }
    }
}
