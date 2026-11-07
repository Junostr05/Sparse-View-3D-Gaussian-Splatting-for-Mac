import AppKit
import MetalKit
import SplatCore

/// `MTKView` that turns mouse, trackpad, scroll, magnify and key events into viewer actions,
/// using the Blender mapping. The coordinator provides the closures; the view holds no scene state.
///
/// In edit mode a left drag paints the brush selection instead of orbiting. The brush works in
/// drawable pixels with a top-left origin, the pixel space that the renderer draws in.
final class SplatMTKView: MTKView {
    /// Drag delta in points; positive y is downward.
    var onOrbit: ((SIMD2<Float>) -> Void)?
    /// Drag delta in points; positive y is downward.
    var onPan: ((SIMD2<Float>) -> Void)?
    /// Multiplicative zoom factor.
    var onZoom: ((Float) -> Void)?
    /// A command from a key press.
    var onCommand: ((ViewCommand) -> Void)?
    /// Set from the model before each update. Enables brush strokes, cursor tracking and edit keys.
    var isEditing = false
    /// Brush points in drawable pixels, top-left origin, for one mouse event (interpolated since the last one).
    /// The flag is true while Shift is held, which subtracts from the selection.
    var onBrushStroke: ((_ stroke: [SIMD2<Float>], _ shift: Bool) -> Void)?
    /// Brush cursor in view points with a top-left origin, or `nil` when the pointer leaves the view.
    var onBrushCursor: ((CGPoint?) -> Void)?
    /// A selection or history command from an edit key.
    var onEditCommand: ((EditCommand) -> Void)?

    /// Key code of the Home key.
    private static let homeKeyCode: UInt16 = 115
    /// Key code of the Delete (backspace) key.
    private static let deleteKeyCode: UInt16 = 51
    /// Key code of the forward Delete key.
    private static let forwardDeleteKeyCode: UInt16 = 117
    /// Distance in drawable pixels between the brush samples of one stroke segment.
    private static let strokeSampleSpacing: Float = 4
    /// Upper bound on the samples of one stroke segment, so one fast drag cannot flood the selector.
    private static let maximumSamplesPerSegment: Int = 256

    /// Last brush point of the stroke in progress. `nil` when no stroke is in progress.
    private var strokeLast: SIMD2<Float>?

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

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        for area in trackingAreas where area.owner === self {
            removeTrackingArea(area)
        }
        addTrackingArea(NSTrackingArea(rect: .zero,
                                       options: [.mouseMoved, .mouseEnteredAndExited, .activeAlways, .inVisibleRect],
                                       owner: self,
                                       userInfo: nil))
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

    /// The event location in drawable pixels with a top-left origin. This matches the image that
    /// `draw(in:)` renders, because the renderer's texture row 0 is the top of the drawable.
    private func drawablePoint(of event: NSEvent) -> SIMD2<Float> {
        let point: NSPoint = convert(event.locationInWindow, from: nil)
        let width: CGFloat = max(bounds.width, 1)
        let height: CGFloat = max(bounds.height, 1)
        let scaleX: CGFloat = drawableSize.width / width
        let scaleY: CGFloat = drawableSize.height / height
        let pixelX: Float = Float(point.x * scaleX)
        let pixelY: Float = Float((height - point.y) * scaleY)
        return SIMD2(pixelX, pixelY)
    }

    /// Points from `start` (exclusive) to `end` (inclusive), evenly spaced, so a fast drag leaves no gaps.
    private static func segment(from start: SIMD2<Float>, to end: SIMD2<Float>) -> [SIMD2<Float>] {
        let delta: SIMD2<Float> = end - start
        let length: Float = (delta * delta).sum().squareRoot()
        let wanted: Int = Int((length / strokeSampleSpacing).rounded(.up))
        let samples: Int = min(max(wanted, 1), maximumSamplesPerSegment)
        var points: [SIMD2<Float>] = []
        points.reserveCapacity(samples)
        for sample in 1...samples {
            let fraction: Float = Float(sample) / Float(samples)
            let point: SIMD2<Float> = start + delta * fraction
            points.append(point)
        }
        return points
    }

    // MARK: - Mouse

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        if isEditing, !event.modifierFlags.contains(.option) {
            let point: SIMD2<Float> = drawablePoint(of: event)
            strokeLast = point
            onBrushStroke?([point], event.modifierFlags.contains(.shift))
            return
        }
        super.mouseDown(with: event)
    }

    override func mouseDragged(with event: NSEvent) {
        if let last = strokeLast {
            let point: SIMD2<Float> = drawablePoint(of: event)
            strokeLast = point
            onBrushStroke?(Self.segment(from: last, to: point), event.modifierFlags.contains(.shift))
        } else if event.modifierFlags.contains(.option) {
            onPan?(delta(of: event))
        } else {
            onOrbit?(delta(of: event))
        }
    }

    override func mouseUp(with event: NSEvent) {
        strokeLast = nil
        super.mouseUp(with: event)
    }

    override func mouseMoved(with event: NSEvent) {
        guard isEditing else { return }
        let point: NSPoint = convert(event.locationInWindow, from: nil)
        onBrushCursor?(CGPoint(x: point.x, y: bounds.height - point.y))
    }

    override func mouseExited(with event: NSEvent) {
        if isEditing {
            onBrushCursor?(nil)
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
        if isEditing, let edit = Self.editCommand(for: event) {
            onEditCommand?(edit)
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

    /// Undo and redo in edit mode. Other Command keys go to the menu system.
    override func performKeyEquivalent(with event: NSEvent) -> Bool {
        guard isEditing,
              event.modifierFlags.contains(.command),
              event.charactersIgnoringModifiers?.lowercased() == "z" else {
            return super.performKeyEquivalent(with: event)
        }
        onEditCommand?(event.modifierFlags.contains(.shift) ? .redo : .undo)
        return true
    }

    /// Edit keys: Delete or x deletes, a selects all, Option+a clears, Control+i inverts, f toggles front-only.
    private static func editCommand(for event: NSEvent) -> EditCommand? {
        if event.keyCode == deleteKeyCode || event.keyCode == forwardDeleteKeyCode {
            return .deleteSelection
        }
        guard let character = event.charactersIgnoringModifiers?.lowercased().first else {
            return nil
        }
        let flags = event.modifierFlags
        let control: Bool = flags.contains(.control)
        let option: Bool = flags.contains(.option)
        switch character {
        case "x" where !control && !option:
            return .deleteSelection
        case "a" where option:
            return .clearSelection
        case "a" where !control:
            return .selectAll
        case "i" where control && !option:
            return .invertSelection
        case "f" where !control && !option:
            return .toggleFrontOnly
        default:
            return nil
        }
    }
}
