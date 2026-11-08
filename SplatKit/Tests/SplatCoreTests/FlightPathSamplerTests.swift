import Foundation
import Testing
import SplatCore

private func dot(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
    a.x * b.x + a.y * b.y + a.z * b.z
}

private func cross(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> SIMD3<Float> {
    SIMD3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
}

private func unit(_ v: SIMD3<Float>) -> SIMD3<Float> {
    v / dot(v, v).squareRoot()
}

/// Four keyframes with distinct pivots and orientations, as in the ViewFlight tests.
private func pathKeyframes() -> [ViewCamera] {
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

@Suite struct FlightPathSamplerTests {
    @Test func flightHasOneCameraPerFrameAndEndsOnTheKeyframes() {
        let keyframes = pathKeyframes()
        let frames = FlightPathSampler.flight(through: keyframes, frameCount: 37)
        #expect(frames.count == 37)
        #expect(approxEqual(frames[0].eye, keyframes[0].eye, tol: 1e-5))
        #expect(approxEqual(frames[0].forward, keyframes[0].forward, tol: 1e-5))
        let last = frames[frames.count - 1]
        #expect(approxEqual(last.eye, keyframes[3].eye, tol: 1e-5))
        #expect(approxEqual(last.forward, keyframes[3].forward, tol: 1e-5))
    }

    @Test func flightPositionsAreMonotoneAndSpanTheKeyframes() {
        let positions = FlightPathSampler.flightPositions(keyframeCount: 4, frameCount: 50)
        #expect(positions.count == 50)
        #expect(positions.first == 0)
        #expect(positions.last == 3)
        for i in 1..<positions.count {
            #expect(positions[i] >= positions[i - 1], "position not monotone at frame \(i)")
        }
    }

    @Test func flightSkipsEverythingWithoutKeyframesOrFrames() {
        #expect(FlightPathSampler.flight(through: [], frameCount: 10).isEmpty)
        #expect(FlightPathSampler.flight(through: pathKeyframes(), frameCount: 0).isEmpty)
        let single = FlightPathSampler.flight(through: [pathKeyframes()[1]], frameCount: 5)
        #expect(single.count == 5)
        #expect(approxEqual(single[4].eye, pathKeyframes()[1].eye, tol: 1e-5))
    }

    @Test func turntableMakesAFullTurnAroundThePivotAtConstantElevation() {
        let bounds = Bounds(min: SIMD3(-1, -1, -1), max: SIMD3(1, 1, 1))
        var start = ViewCamera.framing(bounds, upAxis: .negativeY)
        start.orbit(deltaYaw: 0.3, deltaPitch: 0.2)
        let frameCount: Int = 24
        let frames = FlightPathSampler.turntable(around: start, frameCount: frameCount)
        #expect(frames.count == frameCount)

        let up: SIMD3<Float> = start.upAxis.vector
        let stepAngle: Float = Float(2 * Double.pi / Double(frameCount))
        for (index, frame) in frames.enumerated() {
            #expect(approxEqual(frame.pivot, start.pivot, tol: 1e-4), "pivot moved at \(index)")
            #expect(abs(frame.distance - start.distance) < 1e-5, "distance changed at \(index)")
            #expect(abs(dot(frame.forward, up) - dot(start.forward, up)) < 1e-4, "elevation changed at \(index)")
            #expect(abs(frame.fovY - start.fovY) < 1e-6)
        }
        // Every step, including the one from the last frame back to the first, is the same yaw about the up axis.
        // The yaw is measured on the horizontal projection of the eye offset: the 3D angle between two views on the
        // same cone around the up axis is not the yaw step.
        func horizontal(_ index: Int) -> SIMD3<Float> {
            let offset: SIMD3<Float> = frames[index].eye - frames[index].pivot
            return unit(offset - up * dot(offset, up))
        }
        for index in 0..<frameCount {
            let current: SIMD3<Float> = horizontal(index)
            let next: SIMD3<Float> = horizontal((index + 1) % frameCount)
            let cosine: Float = min(max(dot(current, next), -1), 1)
            #expect(abs(acos(cosine) - stepAngle) < 1e-3, "step \(index) is not 2π/N")
            #expect(dot(cross(current, next), up) > 0, "step \(index) turns the wrong way")
        }
    }

    @Test func turntableFirstFrameIsTheStartView() {
        let bounds = Bounds(min: SIMD3(-1, -1, -1), max: SIMD3(1, 1, 1))
        let start = ViewCamera.framing(bounds, upAxis: .positiveZ)
        let frames = FlightPathSampler.turntable(around: start, frameCount: 8)
        #expect(approxEqual(frames[0].forward, start.forward, tol: 1e-6))
        #expect(approxEqual(frames[0].eye, start.eye, tol: 1e-5))
        #expect(FlightPathSampler.turntable(around: start, frameCount: 0).isEmpty)
    }
}
