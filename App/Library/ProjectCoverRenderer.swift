import Foundation
import SplatCore
import SplatIO

/// Renders the cover a project stores after each checkpoint: the viewer's first camera, or the default framing,
/// at 640 px wide in 4:3. The library shows it instead of a photo.
@MainActor
enum ProjectCoverRenderer {
    /// JPEG bytes of the viewer's view, or `nil` when the viewer has no renderer yet or the frame cannot be drawn.
    static func jpeg(of viewer: ViewerModel) -> Data? {
        guard let renderer = ViewerImageRenderer(model: viewer) else {
            return nil
        }
        let pixelSize: PixelSize = ExportGeometry.size(width: CoverJPEG.maxPixelSide, aspect: 4.0 / 3.0)
        guard ExportGeometry.validate(pixelSize) == nil else {
            return nil
        }
        let view: ViewCamera = viewer.cameraViews.first
            ?? ViewCamera.framing(viewer.bounds, upAxis: viewer.upAxis)
        do {
            let texture = try renderer.render(view: view, size: pixelSize)
            let bgra: Data = renderer.bgraData(of: texture)
            guard let image = CoverJPEG.image(bgra: bgra, width: pixelSize.width, height: pixelSize.height) else {
                return nil
            }
            return CoverJPEG.encode(image)
        } catch {
            return nil
        }
    }
}
