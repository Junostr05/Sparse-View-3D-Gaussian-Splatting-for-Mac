import MetalKit
import OSLog
import SplatCore
import SplatMetal

private let logger = Logger(subsystem: "io.github.junostr05.SparseSplat", category: "Viewport")

final class ViewportCoordinator: NSObject, MTKViewDelegate {
    private let model: ViewerModel
    private let session: MetalSession
    private var offscreen: MTLTexture?
    private var hasLoggedRenderError = false

    init(model: ViewerModel, session: MetalSession) {
        self.model = model
        self.session = session
    }

    /// Wires the view's input closures to the viewer model, which owns the camera.
    func attach(to view: SplatMTKView) {
        let model = self.model
        view.onOrbit = { delta in
            // Dragging right turns the scene right; dragging down views it from above.
            model.orbit(deltaYaw: -delta.x * 0.01, deltaPitch: delta.y * 0.01)
        }
        view.onPan = { [weak view] delta in
            guard let view else { return }
            model.pan(deltaPixels: delta, viewportHeight: Float(max(view.bounds.height, 1)))
        }
        view.onZoom = { factor in
            model.zoom(factor: factor)
        }
        view.onCommand = { command in
            model.apply(command)
        }
        view.onBrushStroke = { [weak view] stroke, shift in
            guard let view else { return }
            let size: CGSize = view.drawableSize
            let backingScale: Float = Float(size.width / max(view.bounds.width, 1))
            let viewSize: SIMD2<Float> = SIMD2(Float(size.width), Float(size.height))
            let mode: BrushMode = shift ? .subtract : model.brushMode
            model.applyBrush(stroke: stroke, viewSizePixels: viewSize, backingScale: backingScale, mode: mode)
        }
        view.onBrushCursor = { point in
            model.brushCursor = point
        }
        view.onEditCommand = { command in
            model.performEdit(command)
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        view.needsDisplay = true
    }

    func draw(in view: MTKView) {
        let size = view.drawableSize
        let width = Int(size.width)
        let height = Int(size.height)
        guard width > 0, height > 0 else { return }
        guard let texture = offscreenTexture(width: width, height: height) else { return }

        let camera = model.view.camera(width: width, height: height)
        var settings = RasterSettings()
        settings.shDegreeLimit = model.shDegreeLimit
        settings.background = model.background.color

        let frameStats: FrameStats
        do {
            frameStats = try session.renderer.render(camera: camera, settings: settings, to: texture)
        } catch {
            if !hasLoggedRenderError {
                hasLoggedRenderError = true
                logger.error("Render failed: \(String(describing: error))")
            }
            return
        }

        guard let drawable = view.currentDrawable,
              let commandBuffer = session.context.queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        let copyWidth = min(width, drawable.texture.width)
        let copyHeight = min(height, drawable.texture.height)
        blit.copy(from: texture,
                  sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: copyWidth, height: copyHeight, depth: 1),
                  to: drawable.texture,
                  destinationSlice: 0, destinationLevel: 0,
                  destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blit.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()

        model.report(frameStats)
    }

    private func offscreenTexture(width: Int, height: Int) -> MTLTexture? {
        if let offscreen, offscreen.width == width, offscreen.height == height {
            return offscreen
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderWrite, .shaderRead]
        descriptor.storageMode = .private
        offscreen = session.context.device.makeTexture(descriptor: descriptor)
        return offscreen
    }
}
