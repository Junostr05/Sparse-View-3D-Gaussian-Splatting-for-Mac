import Foundation
import Metal
import SplatCore
import SplatMetal

/// Renders frames of the viewer offscreen for the image and video export. A frame uses the viewer's own renderer and
/// the viewport's display path (`ViewerModel.renderFrame`), so it shows the display mode, colours, deletions, SH limit
/// and background of the viewport, without the overlays.
///
/// Threading: `SplatRenderer` is not thread-safe, and the viewport draws from the main actor. Every frame is therefore
/// one synchronous call on the main actor, and the callers yield between frames so the viewport keeps drawing. The
/// export does not build a second renderer, so the cloud and the colour overrides stay the viewer's own.
@MainActor
final class ViewerImageRenderer {
    private let model: ViewerModel
    private let session: MetalSession
    /// Reused while the size stays the same. Its pixels are valid until the next `render` call.
    private var texture: MTLTexture?

    /// `nil` until the viewer's renderer exists.
    init?(model: ViewerModel) {
        guard case .ready(let session) = model.renderState else { return nil }
        self.model = model
        self.session = session
    }

    /// Renders `view` at `size`. The GPU has finished when this returns.
    func render(view: ViewCamera, size: PixelSize) throws(MetalError) -> MTLTexture {
        let target = try readbackTexture(size: size)
        let camera = view.camera(width: size.width, height: size.height)
        try model.renderFrame(with: session.renderer, camera: camera, to: target)
        return target
    }

    /// The BGRA8 pixels of `texture`, rows packed. Call after `render`.
    func bgraData(of texture: MTLTexture) -> Data {
        let rowBytes: Int = texture.width * 4
        var data = Data(count: rowBytes * texture.height)
        data.withUnsafeMutableBytes { bytes in
            if let base = bytes.baseAddress {
                FrameExport.copyBGRA(texture, into: base, bytesPerRow: rowBytes)
            }
        }
        return data
    }

    private func readbackTexture(size: PixelSize) throws(MetalError) -> MTLTexture {
        if let texture, texture.width == size.width, texture.height == size.height {
            return texture
        }
        let created = try FrameExport.makeReadbackTexture(device: session.context.device,
                                                          width: size.width, height: size.height)
        texture = created
        return created
    }
}
