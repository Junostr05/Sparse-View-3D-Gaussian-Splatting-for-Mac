import Foundation
import Testing
import SplatCore

private func flightDot(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
    a.x * b.x + a.y * b.y + a.z * b.z
}

/// Four keyframes with distinct pivots and orientations, spaced about 2 units apart.
private func flightKeyframes() -> [ViewCamera] {
    let bounds = Bounds(min: SIMD3(-1, -1, -1), max: SIMD3(1, 1, 1))
    let base = ViewCamera.framing(bounds, upAxis: .positiveZ)
    var frames: [ViewCamera] = []
    for i in 0..<4 {
        var camera = base
        camera.pivot = SIMD3(Float(i) * 2, Float(i % 2) * 1.5, 0)
        camera.orbit(deltaYaw: Float(i) * 0.8, deltaPitch: 0.1 * Float(i))
        frames.append(camera)
    }
    return frames
}

@Suite struct ViewFlightTests {
    @Test func passesThroughEveryKeyframeAtIntegerPositions() {
        let keyframes = flightKeyframes()
        let flight = ViewFlight(keyframes: keyframes)
        for (i, keyframe) in keyframes.enumerated() {
            let camera = flight.camera(at: Float(i))
            #expect(approxEqual(camera.eye, keyframe.eye, tol: 1e-5), "eye at \(i)")
            #expect(approxEqual(camera.forward, keyframe.forward, tol: 1e-5), "forward at \(i)")
            #expect(approxEqual(camera.right, keyframe.right, tol: 1e-5), "right at \(i)")
            #expect(abs(camera.fovY - keyframe.fovY) < 1e-6)
            #expect(abs(camera.distance - keyframe.distance) < 1e-5)
        }
    }

    @Test func pathIsContinuous() {
        let keyframes = flightKeyframes()
        let flight = ViewFlight(keyframes: keyframes)
        // 100 samples across three segments: consecutive samples are 3/99 positions apart.
        // Measured maxima for this geometry: eye step 0.225 units, orientation step 0.023 rad.
        // The bounds are about twice those; a jump between keyframes would be about 2 units
        // (eyes) or about 0.8 rad (orientation), so the bounds still catch discontinuities.
        var previousEye: SIMD3<Float>? = nil
        var previousForward: SIMD3<Float>? = nil
        var maxEyeStep: Float = 0
        var maxAngleStep: Float = 0
        for k in 0..<100 {
            let position = Float(k) * 3 / 99
            let camera = flight.camera(at: position)
            if let eye = previousEye, let forward = previousForward {
                maxEyeStep = Swift.max(maxEyeStep, flightLength(camera.eye - eye))
                let cosine = Swift.min(Swift.max(flightDot(camera.forward, forward), -1), 1)
                maxAngleStep = Swift.max(maxAngleStep, acos(cosine))
            }
            previousEye = camera.eye
            previousForward = camera.forward
        }
        #expect(maxEyeStep < 0.5, "largest eye step \(maxEyeStep)")
        #expect(maxAngleStep < 0.1, "largest orientation step \(maxAngleStep) rad")
    }

    @Test func positionsOutsideTheRangeAreClamped() {
        let keyframes = flightKeyframes()
        let flight = ViewFlight(keyframes: keyframes)
        #expect(flight.camera(at: -4).eye == keyframes[0].eye)
        #expect(flight.camera(at: 50).eye == keyframes[3].eye)
        #expect(flight.camera(at: 3.0001).eye == keyframes[3].eye)
    }

    @Test func singleKeyframeIsReturnedForAnyPosition() {
        let keyframes = flightKeyframes()
        let flight = ViewFlight(keyframes: [keyframes[1]])
        #expect(flight.camera(at: 0) == keyframes[1])
        #expect(flight.camera(at: 0.7) == keyframes[1])
        #expect(flight.camera(at: 9) == keyframes[1])
    }

    @Test func transitionEasesBetweenCameras() {
        let keyframes = flightKeyframes()
        let a = keyframes[0]
        let b = keyframes[2]
        #expect(approxEqual(ViewFlight.transition(from: a, to: b, progress: 0).eye, a.eye, tol: 1e-5))
        #expect(approxEqual(ViewFlight.transition(from: a, to: b, progress: 1).eye, b.eye, tol: 1e-5))
        #expect(approxEqual(ViewFlight.transition(from: a, to: b, progress: -3).eye, a.eye, tol: 1e-5))
        #expect(approxEqual(ViewFlight.transition(from: a, to: b, progress: 7).eye, b.eye, tol: 1e-5))
        // Smoothstep is symmetric, so progress 0.5 is the midpoint of the linear interpolation.
        let middle = ViewFlight.transition(from: a, to: b, progress: 0.5)
        #expect(approxEqual(middle.eye, (a.eye + b.eye) / 2, tol: 1e-4))
        // Eased: the first 10% of progress covers less than 10% of the distance.
        let early = ViewFlight.transition(from: a, to: b, progress: 0.1)
        let total = flightLength(b.eye - a.eye)
        #expect(flightLength(early.eye - a.eye) < 0.1 * total)
    }

    @Test func durationIsSegmentsTimesSeconds() {
        let flight = ViewFlight(keyframes: flightKeyframes())
        #expect(flight.duration(secondsPerSegment: 2.5) == 7.5)
        #expect(ViewFlight(keyframes: [flightKeyframes()[0]]).duration(secondsPerSegment: 2) == 0)
    }
}

private func flightLength(_ v: SIMD3<Float>) -> Float {
    (v.x * v.x + v.y * v.y + v.z * v.z).squareRoot()
}
