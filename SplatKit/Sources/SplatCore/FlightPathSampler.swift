import Foundation

/// Per-frame cameras of an exported video. Pure, so it is tested on Linux.
public enum FlightPathSampler {
    /// The camera of each of `frameCount` frames along the smooth flight through `keyframes`. It is the spline the
    /// viewer's flight and playback follow (`ViewFlight`), so the video matches what "Play Flight" shows. Frame 0 is
    /// the first keyframe and the last frame is the last keyframe. Empty `keyframes` or a `frameCount` below 1 give
    /// no frames.
    public static func flight(through keyframes: [ViewCamera], frameCount: Int) -> [ViewCamera] {
        guard !keyframes.isEmpty, frameCount >= 1 else { return [] }
        let flight = ViewFlight(keyframes: keyframes)
        return flightPositions(keyframeCount: keyframes.count, frameCount: frameCount).map { position in
            flight.camera(at: position)
        }
    }

    /// The flight position of each frame: linear in the frame index, from 0 (first keyframe) to `keyframeCount − 1`
    /// (last keyframe). A single frame sits at the first keyframe.
    public static func flightPositions(keyframeCount: Int, frameCount: Int) -> [Float] {
        guard keyframeCount >= 1, frameCount >= 1 else { return [] }
        guard frameCount > 1 else { return [0] }
        let last: Double = Double(keyframeCount - 1)
        return (0..<frameCount).map { index in
            let fraction: Double = Double(index) / Double(frameCount - 1)
            return Float(fraction * last)
        }
    }

    /// A full turn around the pivot of `start` in `frameCount` frames, at the distance and elevation of `start`. The
    /// camera keeps its pitch and roll. Frame i is yawed by 2π · i / frameCount, so the step from the last frame back
    /// to the first is the same as every other step and the video loops without a jump.
    public static func turntable(around start: ViewCamera, frameCount: Int) -> [ViewCamera] {
        guard frameCount >= 1 else { return [] }
        return (0..<frameCount).map { index in
            let yaw: Float = Float(2 * Double.pi * Double(index) / Double(frameCount))
            return start.orbited(deltaYaw: yaw, deltaPitch: 0)
        }
    }
}
