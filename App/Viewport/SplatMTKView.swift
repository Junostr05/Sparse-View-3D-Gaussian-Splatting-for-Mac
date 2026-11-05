import AppKit
import MetalKit

/// `MTKView` that turns mouse, scroll and magnify events into camera deltas.
/// The coordinator provides the closures; the view holds no scene state.
final class SplatMTKView: MTKView {
    /// Drag delta in points; positive y is downward.
    var onOrbit: ((SIMD2<Float>) -> Void)?
    /// Drag delta in points; positive y is downward.
    var onPan: ((SIMD2<Float>) -> Void)?
    /// Multiplicative zoom factor.
    var onZoom: ((Float) -> Void)?

    init(device: MTLDevice) {
        super.init(frame: .zero, device: device)
        colorPixelFormat = .bgra8Unorm
        framebufferOnly = false
        isPaused = true
        enableSetNeedsDisplay = true
    }

    required init(coder: NSCoder) {
        fatalError("init(coder:) is not supported")
    }

    override var acceptsFirstResponder: Bool { true }

    private func delta(of event: NSEvent) -> SIMD2<Float> {
        SIMD2(Float(event.deltaX), Float(event.deltaY))
    }

    override func mouseDragged(with event: NSEvent) {
        if event.modifierFlags.contains(.option) {
            onPan?(delta(of: event))
        } else {
            onOrbit?(delta(of: event))
        }
    }

    override func rightMouseDragged(with event: NSEvent) {
        onPan?(delta(of: event))
    }

    override func scrollWheel(with event: NSEvent) {
        let scale: CGFloat = event.hasPreciseScrollingDeltas ? 1 : 8
        let amount = Float(event.scrollingDeltaY * scale)
        guard amount != 0 else { return }
        onZoom?(exp(-amount * 0.01))
    }

    override func magnify(with event: NSEvent) {
        onZoom?(exp(-Float(event.magnification)))
    }
}
