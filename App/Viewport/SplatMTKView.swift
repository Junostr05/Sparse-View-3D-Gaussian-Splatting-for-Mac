import AppKit
import MetalKit
import SplatCore

/// `MTKView` that turns mouse, trackpad, scroll, magnify and key events into viewer actions,
/// using the Blender mapping. The coordinator provides the closures; the view holds no scene state.
///
/// In edit mode a left drag paints the brush selection instead of orbiting. The brush works in
/// drawable pixels with a top-left origin, the pixel space that the renderer draws in.
///
/// Fly mode captures the cursor, reads W A S D Q E from the keys and integrates the movement with a 60 Hz
/// timer on the main run loop. It ends with Escape, Return, a click, Shift + ` again, or when the window loses key status.
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
    /// Start of a gesture at a point (view points, top-left origin) in a view of the given size. Called on every
    /// press and on the first event of a drag or scroll after 0.3 s without one.
    var onBeginInteraction: ((CGPoint, CGSize) -> Void)?
    /// Shift + ` pressed outside fly mode.
    var onFlyToggle: (() -> Void)?
    /// Fly mode was left from the view (Escape, Return, click, window lost key status or view removed).
    var onFlyExit: (() -> Void)?
    /// Mouse movement in fly mode, in points, y downward.
    var onFlyLook: ((SIMD2<Float>) -> Void)?
    /// Scroll amount in fly mode, used to change the base fly speed.
    var onFlySpeed: ((Float) -> Void)?
    /// Movement in fly mode: right, up and forward components in -1...1 (normalised), boost and elapsed seconds.
    var onFlyStep: ((SIMD3<Float>, Float, Double) -> Void)?

    /// Fly mode state. Set from the model before each update. Setting it starts or stops the cursor capture and the timer.
    var isFlying = false {
        didSet {
            guard isFlying != oldValue else { return }
            if isFlying {
                startFlying()
            } else {
                stopFlying()
            }
        }
    }

    /// Key code of the Home key.
    private static let homeKeyCode: UInt16 = 115
    /// Key code of the Delete (backspace) key.
    private static let deleteKeyCode: UInt16 = 51
    /// Key code of the forward Delete key.
    private static let forwardDeleteKeyCode: UInt16 = 117
    /// Key code of the ` (grave) key on an ANSI keyboard. Shift + this key toggles fly mode.
    private static let graveKeyCode: UInt16 = 50
    /// Keys that leave fly mode: Escape, Return and the keypad Enter.
    private static let flyExitKeyCodes: Set<UInt16> = [53, 36, 76]
    /// Physical keys for fly movement (ANSI positions, so WASD stay in place on any layout).
    private static let keyA: UInt16 = 0
    private static let keyS: UInt16 = 1
    private static let keyD: UInt16 = 2
    private static let keyQ: UInt16 = 12
    private static let keyW: UInt16 = 13
    private static let keyE: UInt16 = 14
    /// Distance in drawable pixels between the brush samples of one stroke segment.
    private static let strokeSampleSpacing: Float = 4
    /// Upper bound on the samples of one stroke segment, so one fast drag cannot flood the selector.
    private static let maximumSamplesPerSegment: Int = 256
    /// After this much time without a gesture event, the next one picks the cursor anchor again.
    private static let gestureIdleLimit: Duration = .milliseconds(300)

    /// Last brush point of the stroke in progress. `nil` when no stroke is in progress.
    private var strokeLast: SIMD2<Float>?
    /// Time of the last gesture event (press, drag or scroll) that reached the model.
    private var lastInteractionTime: ContinuousClock.Instant?
    /// Fly mode: keys currently held, by key code.
    private var pressedKeys: Set<UInt16> = []
    /// Fly mode: the 60 Hz movement timer on the main run loop.
    private var flyTimer: Timer?
    /// Fly mode: time of the last movement tick.
    private var lastFlyTick: ContinuousClock.Instant?
    /// Fly mode: true while the cursor is hidden and disassociated, so it must be restored.
    private var isCursorCaptured = false
    /// Fly mode: observer for the window losing key status.
    private var resignKeyObserver: NSObjectProtocol?
    /// Fly mode: the window's `acceptsMouseMovedEvents` before fly mode changed it.
    private var previousAcceptsMouseMoved: Bool?

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

    override func viewWillMove(toWindow newWindow: NSWindow?) {
        super.viewWillMove(toWindow: newWindow)
        if newWindow == nil, isFlying {
            exitFlyFromView()
        }
    }

    /// Called when SwiftUI removes the view: restores the cursor and leaves fly mode in the model.
    func stopFlyingForRemoval() {
        if isFlying {
            exitFlyFromView()
        }
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

    /// The event location in view points with a top-left origin, for the cursor depth pick.
    private func topLeftPoint(of event: NSEvent) -> CGPoint {
        let point: NSPoint = convert(event.locationInWindow, from: nil)
        return CGPoint(x: point.x, y: bounds.height - point.y)
    }

    /// Reports a gesture event to the model. A press always picks the anchor. Other events pick only when
    /// no gesture event came within the last 0.3 s, so a drag or a trackpad gesture keeps one anchor.
    private func noteInteraction(_ event: NSEvent, isPress: Bool) {
        let now: ContinuousClock.Instant = ContinuousClock.now
        let isIdle: Bool
        if let last = lastInteractionTime {
            isIdle = now - last > Self.gestureIdleLimit
        } else {
            isIdle = true
        }
        lastInteractionTime = now
        guard isPress || isIdle else { return }
        onBeginInteraction?(topLeftPoint(of: event), bounds.size)
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
        if isFlying {
            exitFlyFromView()
            return
        }
        window?.makeFirstResponder(self)
        if isEditing, !event.modifierFlags.contains(.option) {
            let point: SIMD2<Float> = drawablePoint(of: event)
            strokeLast = point
            onBrushStroke?([point], event.modifierFlags.contains(.shift))
            return
        }
        noteInteraction(event, isPress: true)
        super.mouseDown(with: event)
    }

    override func mouseDragged(with event: NSEvent) {
        if let last = strokeLast {
            let point: SIMD2<Float> = drawablePoint(of: event)
            strokeLast = point
            onBrushStroke?(Self.segment(from: last, to: point), event.modifierFlags.contains(.shift))
        } else if event.modifierFlags.contains(.option) {
            noteInteraction(event, isPress: false)
            onPan?(delta(of: event))
        } else {
            noteInteraction(event, isPress: false)
            onOrbit?(delta(of: event))
        }
    }

    override func mouseUp(with event: NSEvent) {
        strokeLast = nil
        super.mouseUp(with: event)
    }

    override func mouseMoved(with event: NSEvent) {
        if isFlying {
            onFlyLook?(delta(of: event))
            return
        }
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
        if isFlying {
            exitFlyFromView()
            return
        }
        window?.makeFirstResponder(self)
        noteInteraction(event, isPress: true)
        super.rightMouseDown(with: event)
    }

    override func rightMouseDragged(with event: NSEvent) {
        noteInteraction(event, isPress: false)
        onPan?(delta(of: event))
    }

    override func otherMouseDown(with event: NSEvent) {
        if isFlying {
            exitFlyFromView()
            return
        }
        window?.makeFirstResponder(self)
        noteInteraction(event, isPress: true)
        super.otherMouseDown(with: event)
    }

    /// Middle button: orbit, Shift pans, Control zooms with the vertical drag.
    override func otherMouseDragged(with event: NSEvent) {
        guard event.buttonNumber == 2 else {
            super.otherMouseDragged(with: event)
            return
        }
        noteInteraction(event, isPress: false)
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
        if isFlying {
            onFlySpeed?(Float(event.scrollingDeltaY))
            return
        }
        noteInteraction(event, isPress: false)
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
        guard !isFlying else { return }
        noteInteraction(event, isPress: false)
        onZoom?(exp(-Float(event.magnification)))
    }

    // MARK: - Keyboard

    override func keyDown(with event: NSEvent) {
        if event.modifierFlags.contains(.command) {
            super.keyDown(with: event)
            return
        }
        if event.keyCode == Self.graveKeyCode, event.modifierFlags.contains(.shift) {
            if isFlying {
                exitFlyFromView()
            } else {
                onFlyToggle?()
            }
            return
        }
        if isFlying {
            if Self.flyExitKeyCodes.contains(event.keyCode) {
                exitFlyFromView()
            } else {
                pressedKeys.insert(event.keyCode)
            }
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

    override func keyUp(with event: NSEvent) {
        if isFlying {
            pressedKeys.remove(event.keyCode)
            return
        }
        super.keyUp(with: event)
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

    // MARK: - Fly mode

    /// Leaves fly mode from the view and tells the model.
    private func exitFlyFromView() {
        isFlying = false
        onFlyExit?()
    }

    /// Captures the cursor, makes the window send mouse moves, and starts the 60 Hz movement timer.
    private func startFlying() {
        pressedKeys.removeAll()
        if let window {
            previousAcceptsMouseMoved = window.acceptsMouseMovedEvents
            window.acceptsMouseMovedEvents = true
            window.makeFirstResponder(self)
            resignKeyObserver = NotificationCenter.default.addObserver(
                forName: NSWindow.didResignKeyNotification,
                object: window,
                queue: .main
            ) { [weak self] _ in
                Task { @MainActor [weak self] in
                    self?.exitFlyFromView()
                }
            }
        }
        CGAssociateMouseAndMouseCursorPosition(0)
        NSCursor.hide()
        isCursorCaptured = true
        lastFlyTick = ContinuousClock.now
        flyTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.flyTick()
            }
        }
    }

    /// Stops the timer and gives the cursor back. Safe to call more than once.
    private func stopFlying() {
        flyTimer?.invalidate()
        flyTimer = nil
        lastFlyTick = nil
        pressedKeys.removeAll()
        if let observer = resignKeyObserver {
            NotificationCenter.default.removeObserver(observer)
            resignKeyObserver = nil
        }
        if let window, let previous = previousAcceptsMouseMoved {
            window.acceptsMouseMovedEvents = previous
        }
        previousAcceptsMouseMoved = nil
        if isCursorCaptured {
            NSCursor.unhide()
            CGAssociateMouseAndMouseCursorPosition(1)
            isCursorCaptured = false
        }
    }

    /// One movement tick: integrates the held keys over the real elapsed time (clamped to 0.1 s).
    private func flyTick() {
        guard isFlying else { return }
        let now: ContinuousClock.Instant = ContinuousClock.now
        let previous: ContinuousClock.Instant = lastFlyTick ?? now
        lastFlyTick = now
        let elapsed: Duration = now - previous
        let parts = elapsed.components
        let rawSeconds: Double = Double(parts.seconds) + Double(parts.attoseconds) * 1e-18
        let seconds: Double = Swift.min(Swift.max(rawSeconds, 0), 0.1)

        var right: Float = 0
        var up: Float = 0
        var forward: Float = 0
        if pressedKeys.contains(Self.keyD) { right += 1 }
        if pressedKeys.contains(Self.keyA) { right -= 1 }
        if pressedKeys.contains(Self.keyE) { up += 1 }
        if pressedKeys.contains(Self.keyQ) { up -= 1 }
        if pressedKeys.contains(Self.keyW) { forward += 1 }
        if pressedKeys.contains(Self.keyS) { forward -= 1 }
        guard right != 0 || up != 0 || forward != 0 else { return }

        let length: Float = (right * right + up * up + forward * forward).squareRoot()
        let direction: SIMD3<Float> = SIMD3<Float>(right, up, forward)
        let movement: SIMD3<Float> = direction / length
        let flags: NSEvent.ModifierFlags = NSEvent.modifierFlags
        let boost: Float
        if flags.contains(.shift) {
            boost = 4
        } else if flags.contains(.option) {
            boost = 0.25
        } else {
            boost = 1
        }
        onFlyStep?(movement, boost, seconds)
    }
}
