import AppKit
import MetalKit
import SplatCore

/// `MTKView` that turns mouse, trackpad, scroll, magnify and key events into viewer actions,
/// using the Blender mapping. The coordinator provides the closures; the view holds no scene state.
final class SplatMTKView: MTKView {
    /// Drag delta in points; positive y is downward.
    var onOrbit: ((SIMD2<Float>) -> Void)?
    /// Drag delta in points; positive y is downward.
    var onPan: ((SIMD2<Float>) -> Void)?
    /// Multiplicative zoom factor.
    var onZoom: ((Float) -> Void)?
    /// A command from a key press.
    var onCommand: ((ViewCommand) -> Void)?

    /// Key code of the Home key.
    private static let homeKeyCode: UInt16 = 115

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

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        window?.makeFirstResponder(self)
    }

    private func delta(of event: NSEvent) -> SIMD2<Float> {
        SIMD2(Float(event.deltaX), Float(event.deltaY))
    }

    /// Scroll deltas as a drag delta. The larger of X and Y is used when `dominantOnly` is set.
    private func scrollDelta(of event: NSEvent, dominantOnly: Bool) -> SIMD2<Float> {
        let x = Float(event.scrollingDeltaX)
        let y = Float(event.scrollingDeltaY)
        guard dominantOnly else { return SIMD2(x, y) }
        return abs(x) > abs(y) ? SIMD2(x, 0) : SIMD2(0, y)
    }

    // MARK: - Mouse

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        super.mouseDown(with: event)
    }

    override func mouseDragged(with event: NSEvent) {
        if event.modifierFlags.contains(.option) {
            onPan?(delta(of: event))
        } else {
            onOrbit?(delta(of: event))
        }
    }

    override func rightMouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        super.rightMouseDown(with: event)
    }

    override func rightMouseDragged(with event: NSEvent) {
        onPan?(delta(of: event))
    }

    override func otherMouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        super.otherMouseDown(with: event)
    }

    /// Middle button: orbit, Shift pans, Control zooms with the vertical drag.
    override func otherMouseDragged(with event: NSEvent) {
        guard event.buttonNumber == 2 else {
            super.otherMouseDragged(with: event)
            return
        }
        let flags = event.modifierFlags
        let drag = delta(of: event)
        if flags.contains(.control) {
            let factor: Float = exp(drag.y * 0.01)
            onZoom?(factor)
        } else if flags.contains(.shift) {
            onPan?(drag)
        } else {
            onOrbit?(drag)
        }
    }

    // MARK: - Scroll

    override func scrollWheel(with event: NSEvent) {
        let flags = event.modifierFlags
        if event.hasPreciseScrollingDeltas {
            trackpadScroll(event, flags: flags)
        } else {
            wheelScroll(event, flags: flags)
        }
    }

    /// Classic mouse wheel: zoom, Control pans horizontally, Shift pans vertically.
    private func wheelScroll(_ event: NSEvent, flags: NSEvent.ModifierFlags) {
        let lineScale: Float = 8
        if flags.contains(.control) || flags.contains(.shift) {
            // A Shift scroll may arrive on the X axis, so only the dominant axis is used.
            let dominant: SIMD2<Float> = scrollDelta(of: event, dominantOnly: true)
            let amount: Float = (dominant.x + dominant.y) * lineScale
            guard amount != 0 else { return }
            let pan: SIMD2<Float> = flags.contains(.control) ? SIMD2(amount, 0) : SIMD2(0, amount)
            onPan?(pan)
        } else {
            let amountY: Float = Float(event.scrollingDeltaY) * lineScale
            guard amountY != 0 else { return }
            onZoom?(exp(-amountY * 0.01))
        }
    }

    /// Trackpad: orbit, Shift pans, Control or Command zooms.
    private func trackpadScroll(_ event: NSEvent, flags: NSEvent.ModifierFlags) {
        let raw = scrollDelta(of: event, dominantOnly: false)
        if flags.contains(.control) || flags.contains(.command) {
            let amountY: Float = raw.y
            guard amountY != 0 else { return }
            onZoom?(exp(-amountY * 0.01))
        } else if flags.contains(.shift) {
            onPan?(raw)
        } else {
            let half: SIMD2<Float> = raw * 0.5
            onOrbit?(half)
        }
    }

    override func magnify(with event: NSEvent) {
        onZoom?(exp(-Float(event.magnification)))
    }

    // MARK: - Keyboard

    override func keyDown(with event: NSEvent) {
        if event.modifierFlags.contains(.command) {
            super.keyDown(with: event)
            return
        }
        if event.keyCode == Self.homeKeyCode {
            onCommand?(.frameAll)
            return
        }
        let flags = event.modifierFlags
        guard let character = event.charactersIgnoringModifiers?.lowercased().first,
              let command = ViewCommand.command(character: character,
                                                control: flags.contains(.control),
                                                shift: flags.contains(.shift),
                                                keypad: flags.contains(.numericPad)) else {
            super.keyDown(with: event)
            return
        }
        onCommand?(command)
    }
}
