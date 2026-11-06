#if canImport(AVFoundation) && canImport(ImageIO) && canImport(CoreGraphics)
import AVFoundation
import CoreGraphics
import Foundation
import SplatCore

/// One decoded candidate frame of a video, scored for sharpness, with a small preview.
public struct VideoCandidate: Sendable, Equatable {
    /// Position among the decoded candidates, 0-based, ascending by time.
    public var index: Int
    /// The time asked from the image generator.
    public var requestedTime: Double
    /// Actual presentation time of the decoded frame.
    public var time: Double
    /// `FrameSharpness.laplacianVariance` of the scoring copy. Higher = sharper.
    public var sharpness: Double
    /// PNG preview, longer side at most `thumbnailSide`.
    public var thumbnail: Data

    public init(index: Int, requestedTime: Double, time: Double, sharpness: Double, thumbnail: Data) {
        self.index = index
        self.requestedTime = requestedTime
        self.time = time
        self.sharpness = sharpness
        self.thumbnail = thumbnail
    }
}

/// Samples frames from a video file and returns them as PNG `CapturedFrame`s in time order.
///
/// Sandbox: `url` must be readable by the caller. A caller that got the URL from a file picker holds
/// the security scope (`startAccessingSecurityScopedResource`) and releases it after this call. This
/// function does not start or stop access itself.
///
/// Memory: `.count` decodes every candidate twice (once to score a small copy, once at full size for
/// the selected frames) so only one full-size image is alive at a time. `.all` encodes each frame as it
/// is decoded, so the result holds all of them at once; keep `maxCandidates` moderate.
///
/// Two-step use for a frame picker: `scoreCandidates(from:)` scores every candidate and returns light
/// previews; `frames(from:atRequestedTimes:)` then decodes only the chosen ones at full size.
public struct VideoFrameExtractor: Sendable {
    /// Longer side of the copy used for sharpness scoring.
    static let scoringMaxSide: Int = 480

    /// Decode size used by `scoreCandidates`. It equals the default `maxSide` of `extract`, so the
    /// candidates' sharpness (and therefore the selection) matches `extract` at its defaults.
    static let defaultDecodeMaxSide: Int = 1600

    /// A frame to decode: the time asked from the generator, and the timestamp to report (nil = the
    /// actual decoded time).
    private struct FrameRequest {
        let requested: Double
        let timestamp: Double?
    }

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
        let duration: Double = try await videoDuration(of: asset, sourceName: sourceName)
        let times: [Double] = candidateRequests(duration: duration,
                                                sampleInterval: sampleInterval,
                                                maxCandidates: maxCandidates)
        let generator: AVAssetImageGenerator = makeGenerator(asset, maxSide: maxSide)

        var frames: [CapturedFrame] = []
        switch mode {
        case .all:
            let requests: [FrameRequest] = times.map { FrameRequest(requested: $0, timestamp: nil) }
            frames = try await fullSizeFrames(generator, requests: requests, sourceName: sourceName)

        case .count:
            // Pass 1: score every candidate on a small copy. Only scores and times are kept.
            let candidates: [VideoCandidate] = try await scoreRequests(times,
                                                                       generator: generator,
                                                                       sourceName: sourceName,
                                                                       thumbnailSide: nil)
            // Pass 2: decode only the selected candidates at full size.
            let chosen: [Int] = FrameSelector.select(frameCandidates(candidates),
                                                     mode: mode,
                                                     minimumSpacing: minimumSpacing)
            let requests: [FrameRequest] = chosen.map { index in
                FrameRequest(requested: candidates[index].requestedTime, timestamp: candidates[index].time)
            }
            frames = try await fullSizeFrames(generator, requests: requests, sourceName: sourceName)
        }

        guard !frames.isEmpty else {
            throw FrameExtractionError.noDecodableFrames(sourceName)
        }
        return frames
    }

    /// Decodes every candidate once on a small copy: scores it and keeps a PNG thumbnail. No full-size
    /// image is kept. Candidate times match `extract` (`sampleInterval`, `maxCandidates`).
    ///
    /// Each candidate is decoded at the same size `extract` uses by default, so `sharpness` agrees with
    /// the score `extract` computes; a caller that passes a different `maxSide` to `extract` gets a
    /// slightly different selection.
    /// - Throws: `FrameExtractionError` for unreadable files, missing video tracks, zero duration, or
    ///   when no candidate decodes.
    public static func scoreCandidates(from url: URL,
                                       sampleInterval: TimeInterval = 0.25,
                                       maxCandidates: Int = 240,
                                       thumbnailSide: Int = 320) async throws -> [VideoCandidate] {
        let sourceName: String = url.lastPathComponent
        let asset: AVURLAsset = AVURLAsset(url: url)
        let duration: Double = try await videoDuration(of: asset, sourceName: sourceName)
        let times: [Double] = candidateRequests(duration: duration,
                                                sampleInterval: sampleInterval,
                                                maxCandidates: maxCandidates)
        let generator: AVAssetImageGenerator = makeGenerator(asset, maxSide: defaultDecodeMaxSide)
        let candidates: [VideoCandidate] = try await scoreRequests(times,
                                                                   generator: generator,
                                                                   sourceName: sourceName,
                                                                   thumbnailSide: max(1, thumbnailSide))
        guard !candidates.isEmpty else {
            throw FrameExtractionError.noDecodableFrames(sourceName)
        }
        return candidates
    }

    /// Decodes the frames at exactly these requested times at full size (longer side `maxSide`), in the
    /// order given, as PNG `CapturedFrame`s named `frame_0001.png`... with `timestamp` = actual time.
    /// Frames that fail to decode are skipped. An empty `times` returns `[]`.
    /// - Throws: `FrameExtractionError` for unreadable files, missing video tracks or zero duration, and
    ///   `noDecodableFrames` when times were given but none decodes.
    public static func frames(from url: URL,
                              atRequestedTimes times: [Double],
                              maxSide: Int = 1600) async throws -> [CapturedFrame] {
        let sourceName: String = url.lastPathComponent
        let asset: AVURLAsset = AVURLAsset(url: url)
        _ = try await videoDuration(of: asset, sourceName: sourceName)
        guard !times.isEmpty else {
            return []
        }
        let generator: AVAssetImageGenerator = makeGenerator(asset, maxSide: maxSide)
        let requests: [FrameRequest] = times.map { FrameRequest(requested: $0, timestamp: nil) }
        let frames: [CapturedFrame] = try await fullSizeFrames(generator, requests: requests, sourceName: sourceName)
        guard !frames.isEmpty else {
            throw FrameExtractionError.noDecodableFrames(sourceName)
        }
        return frames
    }

    // MARK: - Shared steps

    /// Checks that the asset is readable, has a video track and a positive finite duration; returns the
    /// duration in seconds.
    private static func videoDuration(of asset: AVURLAsset, sourceName: String) async throws -> Double {
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
        return duration
    }

    /// Candidate request times from `CaptureSampling`; a single middle time for very short clips.
    private static func candidateRequests(duration: Double,
                                          sampleInterval: TimeInterval,
                                          maxCandidates: Int) -> [Double] {
        let times: [Double] = CaptureSampling.candidateTimes(duration: duration,
                                                             interval: sampleInterval,
                                                             maxCount: max(1, maxCandidates))
        if times.isEmpty {
            // Very short clips: a single candidate in the middle.
            return [duration / 2]
        }
        return times
    }

    private static func makeGenerator(_ asset: AVURLAsset, maxSide: Int) -> AVAssetImageGenerator {
        let generator: AVAssetImageGenerator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        let bound: CGFloat = CGFloat(max(1, maxSide))
        generator.maximumSize = CGSize(width: bound, height: bound)
        return generator
    }

    /// Decodes each requested time once, scores it on a `scoringMaxSide` copy and keeps the actual time.
    /// Undecodable times are skipped, so `index` is the position in the returned array. Thumbnails are
    /// encoded only when `thumbnailSide` is given; otherwise `thumbnail` is empty.
    private static func scoreRequests(_ requests: [Double],
                                      generator: AVAssetImageGenerator,
                                      sourceName: String,
                                      thumbnailSide: Int?) async throws -> [VideoCandidate] {
        var candidates: [VideoCandidate] = []
        for requested in requests {
            let cmTime: CMTime = CMTime(seconds: requested, preferredTimescale: 600)
            guard let result = try? await generator.image(at: cmTime) else {
                continue
            }
            let small: RGBImage = try CaptureImageSupport.rgbImage(result.image,
                                                                   maxSide: scoringMaxSide,
                                                                   sourceName: sourceName)
            let sharpness: Double = FrameSharpness.laplacianVariance(small)
            var thumbnail: Data = Data()
            if let side = thumbnailSide {
                thumbnail = try CaptureImageSupport.thumbnailPNG(result.image, maxSide: side, sourceName: sourceName)
            }
            let time: Double = frameTime(result.actualTime, fallback: requested)
            candidates.append(VideoCandidate(index: candidates.count,
                                             requestedTime: requested,
                                             time: time,
                                             sharpness: sharpness,
                                             thumbnail: thumbnail))
        }
        return candidates
    }

    /// Decodes each request at full size and encodes it as PNG, in order. Requests that fail to decode
    /// are skipped. Names are numbered by position in the result.
    private static func fullSizeFrames(_ generator: AVAssetImageGenerator,
                                       requests: [FrameRequest],
                                       sourceName: String) async throws -> [CapturedFrame] {
        var frames: [CapturedFrame] = []
        for request in requests {
            let cmTime: CMTime = CMTime(seconds: request.requested, preferredTimescale: 600)
            guard let result = try? await generator.image(at: cmTime) else {
                continue
            }
            let data: Data = try CaptureImageSupport.pngData(result.image, sourceName: sourceName)
            let timestamp: Double = request.timestamp ?? frameTime(result.actualTime, fallback: request.requested)
            let name: String = FrameNaming.name(prefix: "frame", index: frames.count + 1, ext: "png")
            frames.append(CapturedFrame(name: name, data: data, sourceName: sourceName, timestamp: timestamp))
        }
        return frames
    }

    private static func frameCandidates(_ candidates: [VideoCandidate]) -> [FrameCandidate] {
        candidates.map { candidate in
            FrameCandidate(index: candidate.index, time: candidate.time, sharpness: candidate.sharpness)
        }
    }

    /// Seconds of a decoded frame's actual time, or `fallback` when the time is not finite.
    private static func frameTime(_ actual: CMTime, fallback: Double) -> Double {
        let seconds: Double = CMTimeGetSeconds(actual)
        return seconds.isFinite ? seconds : fallback
    }
}
#endif
