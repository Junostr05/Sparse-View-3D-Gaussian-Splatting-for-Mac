#if canImport(AVFoundation) && canImport(ImageIO) && canImport(CoreGraphics)
import AVFoundation
import CoreGraphics
import Foundation
import SplatCore

/// Samples frames from a video file and returns them as PNG `CapturedFrame`s in time order.
///
/// Sandbox: `url` must be readable by the caller. A caller that got the URL from a file picker holds
/// the security scope (`startAccessingSecurityScopedResource`) and releases it after this call. This
/// function does not start or stop access itself.
///
/// Memory: `.count` decodes every candidate twice (once to score a small copy, once at full size for
/// the selected frames) so only one full-size image is alive at a time. `.all` encodes each frame as it
/// is decoded, so the result holds all of them at once; keep `maxCandidates` moderate.
public struct VideoFrameExtractor: Sendable {
    /// Longer side of the copy used for sharpness scoring.
    static let scoringMaxSide: Int = 480

    /// - Parameters:
    ///   - url: The video file.
    ///   - mode: `.all` returns every candidate (`minimumSpacing` is not applied). `.count(n)` scores
    ///     candidates by sharpness and keeps `n` of them spread over the timeline (see `FrameSelector`).
    ///   - sampleInterval: Seconds between candidate times, starting at `interval / 2`.
    ///   - maxSide: Longer side of the output frames in pixels.
    ///   - maxCandidates: Upper bound on candidate times; they are thinned evenly.
    ///   - minimumSpacing: Minimum time between selected frames, used by `.count` only.
    /// - Throws: `FrameExtractionError` for unreadable files, missing video tracks, zero duration, or
    ///   when no frame decodes.
    public static func extract(from url: URL,
                               mode: FrameSelectionMode,
                               sampleInterval: TimeInterval = 0.25,
                               maxSide: Int = 1600,
                               maxCandidates: Int = 240,
                               minimumSpacing: Double = 0) async throws -> [CapturedFrame] {
        let sourceName: String = url.lastPathComponent
        let asset: AVURLAsset = AVURLAsset(url: url)

        let duration: Double
        let videoTracks: [AVAssetTrack]
        do {
            let durationTime: CMTime = try await asset.load(.duration)
            duration = CMTimeGetSeconds(durationTime)
            videoTracks = try await asset.loadTracks(withMediaType: .video)
        } catch {
            throw FrameExtractionError.unreadableVideo(sourceName)
        }
        guard !videoTracks.isEmpty else {
            throw FrameExtractionError.noVideoTrack(sourceName)
        }
        guard duration.isFinite, duration > 0 else {
            throw FrameExtractionError.zeroDuration(sourceName)
        }

        var times: [Double] = CaptureSampling.candidateTimes(duration: duration,
                                                             interval: sampleInterval,
                                                             maxCount: max(1, maxCandidates))
        if times.isEmpty {
            // Very short clips: a single candidate in the middle.
            times = [duration / 2]
        }

        let generator: AVAssetImageGenerator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        let bound: CGFloat = CGFloat(max(1, maxSide))
        generator.maximumSize = CGSize(width: bound, height: bound)

        var frames: [CapturedFrame] = []
        switch mode {
        case .all:
            for requested in times {
                let cmTime: CMTime = CMTime(seconds: requested, preferredTimescale: 600)
                guard let result = try? await generator.image(at: cmTime) else {
                    continue
                }
                let data: Data = try CaptureImageSupport.pngData(result.image, sourceName: sourceName)
                let timestamp: Double = frameTime(result.actualTime, fallback: requested)
                let name: String = FrameNaming.name(prefix: "frame", index: frames.count + 1, ext: "png")
                frames.append(CapturedFrame(name: name, data: data, sourceName: sourceName, timestamp: timestamp))
            }

        case .count:
            // Pass 1: score every candidate on a small copy. Only scores and times are kept.
            var requestedTimes: [Double] = []
            var candidates: [FrameCandidate] = []
            for requested in times {
                let cmTime: CMTime = CMTime(seconds: requested, preferredTimescale: 600)
                guard let result = try? await generator.image(at: cmTime) else {
                    continue
                }
                let small: RGBImage = try CaptureImageSupport.rgbImage(result.image,
                                                                       maxSide: scoringMaxSide,
                                                                       sourceName: sourceName)
                let sharpness: Double = FrameSharpness.laplacianVariance(small)
                let timestamp: Double = frameTime(result.actualTime, fallback: requested)
                let index: Int = requestedTimes.count
                requestedTimes.append(requested)
                candidates.append(FrameCandidate(index: index, time: timestamp, sharpness: sharpness))
            }

            // Pass 2: decode only the selected candidates at full size.
            let chosen: [Int] = FrameSelector.select(candidates, mode: mode, minimumSpacing: minimumSpacing)
            for index in chosen {
                let cmTime: CMTime = CMTime(seconds: requestedTimes[index], preferredTimescale: 600)
                guard let result = try? await generator.image(at: cmTime) else {
                    continue
                }
                let data: Data = try CaptureImageSupport.pngData(result.image, sourceName: sourceName)
                let name: String = FrameNaming.name(prefix: "frame", index: frames.count + 1, ext: "png")
                frames.append(CapturedFrame(name: name,
                                            data: data,
                                            sourceName: sourceName,
                                            timestamp: candidates[index].time))
            }
        }

        guard !frames.isEmpty else {
            throw FrameExtractionError.noDecodableFrames(sourceName)
        }
        return frames
    }

    /// Seconds of a decoded frame's actual time, or `fallback` when the time is not finite.
    private static func frameTime(_ actual: CMTime, fallback: Double) -> Double {
        let seconds: Double = CMTimeGetSeconds(actual)
        return seconds.isFinite ? seconds : fallback
    }
}
#endif
