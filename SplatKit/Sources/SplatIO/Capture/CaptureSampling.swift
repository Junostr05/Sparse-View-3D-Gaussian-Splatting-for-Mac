import Foundation

/// Pure helpers for choosing candidate frame times. Linux-testable, so they live outside the
/// macOS-only capture code.
public enum CaptureSampling {
    /// Candidate times `t_i = (i + 0.5) * interval` with `t_i < duration`, ascending.
    ///
    /// If more than `maxCount` candidates remain they are thinned evenly to exactly `maxCount`
    /// times, taking the middle of each of `maxCount` equal groups. Returns `[]` when `duration` or
    /// `interval` is not a positive finite number, or when `maxCount <= 0`.
    public static func candidateTimes(duration: Double, interval: Double, maxCount: Int) -> [Double] {
        guard duration.isFinite, interval.isFinite, duration > 0, interval > 0, maxCount > 0 else {
            return []
        }
        let quotient: Double = duration / interval
        guard quotient < Double(Int.max / 2) else { return [] }

        // Number of i >= 0 with (i + 0.5) * interval < duration, corrected for rounding at the edge.
        var count: Int = max(0, Int((quotient - 0.5).rounded(.up)))
        while count > 0 && (Double(count - 1) + 0.5) * interval >= duration {
            count -= 1
        }
        while (Double(count) + 0.5) * interval < duration {
            count += 1
        }

        if count <= maxCount {
            var times: [Double] = []
            times.reserveCapacity(count)
            for i in 0..<count {
                let t: Double = (Double(i) + 0.5) * interval
                times.append(t)
            }
            return times
        }

        // Thin evenly: indices floor((k + 0.5) * count / maxCount) are strictly increasing and < count.
        var thinned: [Double] = []
        thinned.reserveCapacity(maxCount)
        let step: Double = Double(count) / Double(maxCount)
        for k in 0..<maxCount {
            let index: Int = Int((Double(k) + 0.5) * step)
            let t: Double = (Double(index) + 0.5) * interval
            thinned.append(t)
        }
        return thinned
    }
}

/// File naming for extracted frames.
public enum FrameNaming {
    /// `prefix_NNNN.ext`, zero-padded to at least four digits, e.g. `frame_0001.png`.
    /// `index` is written as given (callers pass 1-based numbers). A leading dot in `ext` is ignored.
    public static func name(prefix: String, index: Int, ext: String) -> String {
        precondition(index >= 0, "frame index must not be negative")
        let digits: String = String(index)
        var padded: String = digits
        if digits.count < 4 {
            padded = String(repeating: "0", count: 4 - digits.count) + digits
        }
        var cleanExt: String = ext
        if cleanExt.hasPrefix(".") {
            cleanExt = String(cleanExt.dropFirst())
        }
        return prefix + "_" + padded + "." + cleanExt
    }
}
