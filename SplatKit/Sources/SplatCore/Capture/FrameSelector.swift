import Foundation

/// A candidate frame extracted from a video, scored for sharpness.
public struct FrameCandidate: Sendable, Equatable {
    /// Caller's identifier (position among the extracted candidates).
    public var index: Int
    /// Presentation time in seconds.
    public var time: Double
    /// Higher = sharper.
    public var sharpness: Double

    public init(index: Int, time: Double, sharpness: Double) {
        self.index = index
        self.time = time
        self.sharpness = sharpness
    }
}

/// Chooses a fixed number of frames that are spread over the timeline and prefer sharp frames.
public enum FrameSelector {
    /// Returns the `index` values of the chosen candidates, sorted by time ascending.
    ///
    /// Algorithm (deterministic):
    /// 1. Sort by (time, index). `count <= 0` or no candidates gives `[]`.
    ///    If `count >= candidates.count` every candidate is returned (subject to `minimumSpacing`).
    /// 2. Split [tMin, tMax] into `count` equal bins; the last bin includes tMax. If the span is zero,
    ///    everything lands in bin 0. Each bin picks its sharpest candidate (ties: earlier time, then smaller index).
    /// 3. An empty bin takes the unchosen candidate nearest its centre (ties: earlier time, then smaller index).
    /// 4. With `minimumSpacing > 0`, no two chosen frames may be closer than that in time. A pick that
    ///    violates it is replaced by the next-sharpest candidate of the same bin that fits; if none
    ///    fits, the bin is skipped. Finally, unchosen candidates are backfilled sharpest-first while they fit.
    public static func select(_ candidates: [FrameCandidate], count: Int, minimumSpacing: Double = 0) -> [Int] {
        guard count > 0, !candidates.isEmpty else { return [] }

        let sorted: [FrameCandidate] = candidates.sorted { a, b in
            if a.time != b.time { return a.time < b.time }
            return a.index < b.index
        }
        let n: Int = sorted.count
        let spacing: Double = minimumSpacing > 0 ? minimumSpacing : 0

        if count >= n && spacing == 0 {
            return sorted.map { $0.index }
        }

        let target: Int = min(count, n)
        let tMin: Double = sorted[0].time
        let span: Double = sorted[n - 1].time - tMin

        // Positions in `sorted` are used internally; they are already ordered by (time, index).
        func binOf(_ time: Double) -> Int {
            guard span > 0 else { return 0 }
            let raw: Double = ((time - tMin) * Double(target) / span).rounded(.down)
            let clamped: Double = min(max(raw, 0), Double(target - 1))
            return Int(clamped)
        }

        var buckets: [[Int]] = [[Int]](repeating: [], count: target)
        for p in 0..<n {
            buckets[binOf(sorted[p].time)].append(p)
        }
        for k in 0..<target {
            buckets[k].sort { a, b in
                let sa: Double = sorted[a].sharpness
                let sb: Double = sorted[b].sharpness
                if sa != sb { return sa > sb }
                return a < b
            }
        }

        var chosen: [Int] = []
        var isChosen: [Bool] = [Bool](repeating: false, count: n)

        func fits(_ p: Int) -> Bool {
            guard spacing > 0 else { return true }
            for c in chosen where abs(sorted[p].time - sorted[c].time) < spacing {
                return false
            }
            return true
        }

        func take(_ p: Int) {
            chosen.append(p)
            isChosen[p] = true
        }

        // Steps 2 and 4: sharpest fitting candidate per non-empty bin.
        for k in 0..<target where !buckets[k].isEmpty {
            if let p = buckets[k].first(where: { fits($0) }) {
                take(p)
            }
        }

        // Step 3: empty bins take the nearest unchosen candidate to the bin centre.
        for k in 0..<target where buckets[k].isEmpty {
            let centre: Double = tMin + (Double(k) + 0.5) * span / Double(target)
            var best: Int? = nil
            for p in 0..<n where !isChosen[p] && fits(p) {
                guard let b = best else {
                    best = p
                    continue
                }
                if abs(sorted[p].time - centre) < abs(sorted[b].time - centre) {
                    best = p
                }
            }
            if let b = best {
                take(b)
            }
        }

        // Step 4: backfill sharpest-first while candidates fit the spacing.
        while chosen.count < target {
            var best: Int? = nil
            for p in 0..<n where !isChosen[p] && fits(p) {
                guard let b = best else {
                    best = p
                    continue
                }
                if sorted[p].sharpness > sorted[b].sharpness {
                    best = p
                }
            }
            guard let b = best else { break }
            take(b)
        }

        return chosen.sorted().map { sorted[$0].index }
    }
}
