import Foundation

/// A size in pixels.
public struct PixelSize: Equatable, Sendable {
    public var width: Int
    public var height: Int

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
    }

    public var pixelCount: Int {
        width * height
    }
}

/// Why an export size is refused.
public enum ExportSizeError: Error, Equatable, Sendable {
    case empty
    case tooLarge(maxSide: Int, maxPixels: Int)
}

/// Sizes, frame counts and durations of the image and video export. Pure, so it is tested on Linux.
public enum ExportGeometry {
    /// Largest accepted side in pixels.
    public static let maxSide: Int = 8192
    /// Largest accepted pixel count (8192 × 8192).
    public static let maxPixels: Int = 8192 * 8192
    /// Frame rate of the exported videos.
    public static let videoFrameRate: Int = 30
    /// Allowed video durations in seconds.
    public static let durationRange: ClosedRange<Double> = 4...60
    /// Default video length per estimated camera, before clamping to `durationRange`.
    public static let secondsPerCamera: Double = 2
    /// Default video length of a turntable (a scene without estimated cameras).
    public static let turntableSeconds: Double = 8

    /// The largest even integer not above `value`, and at least 2. H.264 needs even sizes.
    public static func evenFloor(_ value: Double) -> Int {
        guard value.isFinite, value >= 2 else { return 2 }
        let capped: Double = min(value, 1_000_000)
        let whole: Int = Int(capped)
        return max(2, whole - whole % 2)
    }

    /// A size `width` pixels wide with the aspect ratio `aspect` (width / height). Both sides are rounded down to even
    /// numbers. A non-positive or non-finite aspect falls back to 16:9.
    public static func size(width: Int, aspect: Double) -> PixelSize {
        let safeAspect: Double = (aspect.isFinite && aspect > 0) ? aspect : 16.0 / 9.0
        let evenWidth: Int = evenFloor(Double(width))
        let height: Int = evenFloor(Double(evenWidth) / safeAspect)
        return PixelSize(width: evenWidth, height: height)
    }

    /// The reason `size` is refused, or `nil` when it is accepted: both sides positive, no side above `maxSide`
    /// and no more than `maxPixels` pixels.
    public static func validate(_ size: PixelSize) -> ExportSizeError? {
        guard size.width > 0, size.height > 0 else {
            return .empty
        }
        if size.width > maxSide || size.height > maxSide || size.pixelCount > maxPixels {
            return .tooLarge(maxSide: maxSide, maxPixels: maxPixels)
        }
        return nil
    }

    /// Bytes of `copies` BGRA frames of `size` (4 bytes per pixel).
    public static func byteCount(of size: PixelSize, copies: Int) -> UInt64 {
        let pixels: UInt64 = UInt64(max(size.pixelCount, 0))
        let frames: UInt64 = UInt64(max(copies, 0))
        return pixels * 4 * frames
    }

    /// True when `copies` BGRA frames of `size` take at most a quarter of `physicalMemory`. The rest is left for the app,
    /// the GPU and the encoder.
    public static func fitsInMemory(_ size: PixelSize, copies: Int, physicalMemory: UInt64) -> Bool {
        byteCount(of: size, copies: copies) <= physicalMemory / 4
    }

    /// Number of frames of a video of `seconds` at `fps`, at least 2 so that the video has a start and an end.
    public static func frameCount(seconds: Double, fps: Int) -> Int {
        guard seconds.isFinite, seconds > 0, fps > 0 else { return 2 }
        let frames: Double = min((seconds * Double(fps)).rounded(), 1_000_000)
        return max(2, Int(frames))
    }

    /// Default video length: `secondsPerCamera` per estimated camera, clamped to `durationRange`. Without cameras the
    /// default is `turntableSeconds`.
    public static func defaultVideoDuration(cameraCount: Int) -> Double {
        guard cameraCount > 0 else { return turntableSeconds }
        let wanted: Double = secondsPerCamera * Double(cameraCount)
        return min(max(wanted, durationRange.lowerBound), durationRange.upperBound)
    }

    /// `seconds` clamped to `durationRange`.
    public static func clampedDuration(_ seconds: Double) -> Double {
        guard seconds.isFinite else { return durationRange.lowerBound }
        return min(max(seconds, durationRange.lowerBound), durationRange.upperBound)
    }
}
