import Foundation

/// Held-out view metrics for one evaluation.
public struct HeldOutMetrics: Sendable, Equatable {
    public var psnr: Double
    public var ssim: Double

    public init(psnr: Double, ssim: Double) {
        self.psnr = psnr
        self.ssim = ssim
    }
}

/// Chooses the held-out view for leave-one-out early stopping (P11).
public enum HeldOutSelection {
    /// Returns the index of the held-out view, or nil when early stopping is disabled, there are
    /// too few views, or the camera layout is degenerate. An `override` name wins whenever it is
    /// present, even below `minimumViews`, as long as there are at least two views.
    ///
    /// Automatic choice: PCA of the camera centres. The two extremes along the principal axis are
    /// never chosen; among the rest, the camera whose projection is nearest the median projection
    /// of all cameras is picked. Ties go to the lowest index.
    public static func select(names: [String], cameraCenters: [SIMD3<Float>], override: String?,
                              policy: EarlyStoppingPolicy) -> Int? {
        precondition(names.count == cameraCenters.count, "one name per camera centre")
        guard policy.isEnabled else {
            return nil
        }
        let count: Int = names.count

        if let override, let index = names.firstIndex(of: override) {
            return count >= 2 ? index : nil
        }

        // Three views are the minimum for an interior camera to exist.
        guard count >= max(policy.minimumViews, 3) else {
            return nil
        }

        // Centroid, accumulated in Double.
        var cx: Double = 0
        var cy: Double = 0
        var cz: Double = 0
        for center in cameraCenters {
            cx += Double(center.x)
            cy += Double(center.y)
            cz += Double(center.z)
        }
        let countD: Double = Double(count)
        cx /= countD
        cy /= countD
        cz /= countD

        // Covariance of the centres as a row-major 3×3 matrix.
        var cov = [Double](repeating: 0, count: 9)
        var offsets: [[Double]] = []
        offsets.reserveCapacity(count)
        for center in cameraCenters {
            let dx: Double = Double(center.x) - cx
            let dy: Double = Double(center.y) - cy
            let dz: Double = Double(center.z) - cz
            let d: [Double] = [dx, dy, dz]
            offsets.append(d)
            for r in 0..<3 {
                for k in 0..<3 {
                    cov[r * 3 + k] += d[r] * d[k] / countD
                }
            }
        }

        let trace: Double = cov[0] + cov[4] + cov[8]
        guard trace >= 1e-12 else {
            return nil
        }

        // Principal axis by power iteration from (1, 1, 1)/√3.
        let inverseRoot: Double = 1 / 3.0.squareRoot()
        var axis: [Double] = [inverseRoot, inverseRoot, inverseRoot]
        for _ in 0..<50 {
            var product: [Double] = multiply(cov, axis)
            var norm: Double = length(product)
            if norm == 0 {
                // Start vector orthogonal to the range of the covariance: restart from the
                // coordinate axis with the largest variance, which cannot be orthogonal to it.
                let diagonal: [Double] = [cov[0], cov[4], cov[8]]
                var best: Int = 0
                for k in 1..<3 where diagonal[k] > diagonal[best] {
                    best = k
                }
                axis = [0, 0, 0]
                axis[best] = 1
                product = multiply(cov, axis)
                norm = length(product)
            }
            axis = [product[0] / norm, product[1] / norm, product[2] / norm]
        }
        let eigenvalue: Double = dot(axis, multiply(cov, axis))
        guard eigenvalue >= 1e-12 else {
            return nil
        }

        // Projection of every centre onto the principal axis.
        var projections = [Double](repeating: 0, count: count)
        for i in 0..<count {
            projections[i] = dot(offsets[i], axis)
        }

        // Exclude both extremes; lowest index wins ties.
        var low: Int = 0
        var high: Int = 0
        for i in 0..<count {
            if projections[i] < projections[low] {
                low = i
            }
            if projections[i] > projections[high] {
                high = i
            }
        }
        guard low != high else {
            return nil
        }

        // Median of all projections (mean of the two middle values for even counts).
        let sorted: [Double] = projections.sorted()
        let median: Double
        if count % 2 == 1 {
            median = sorted[count / 2]
        } else {
            median = (sorted[count / 2 - 1] + sorted[count / 2]) / 2
        }

        var chosen: Int? = nil
        var chosenDistance: Double = .infinity
        for i in 0..<count where i != low && i != high {
            let distance: Double = abs(projections[i] - median)
            if distance < chosenDistance {
                chosenDistance = distance
                chosen = i
            }
        }
        return chosen
    }

    private static func multiply(_ m: [Double], _ v: [Double]) -> [Double] {
        var out: [Double] = [0, 0, 0]
        for r in 0..<3 {
            var acc: Double = 0
            for k in 0..<3 {
                acc += m[r * 3 + k] * v[k]
            }
            out[r] = acc
        }
        return out
    }

    private static func dot(_ a: [Double], _ b: [Double]) -> Double {
        var acc: Double = 0
        for k in 0..<3 {
            acc += a[k] * b[k]
        }
        return acc
    }

    private static func length(_ v: [Double]) -> Double {
        dot(v, v).squareRoot()
    }
}

/// Early-stopping scheduler for run 1 of leave-one-out training (P11).
///
/// The SSIM of each held-out evaluation is smoothed with an exponential moving average. A new best
/// requires the EMA to exceed the best EMA by `minDelta`, so a single lucky evaluation cannot end the
/// run on its own. The run stops once `patience` evaluations have passed without a new best, but
/// never before `minimumIterations`, and always at `maxIterations`.
public struct LeaveOneOutScheduler: Sendable {
    public enum Decision: Sendable, Equatable {
        case `continue`, stop
    }

    private let policy: EarlyStoppingPolicy
    private let maxIterations: Int
    private var bestEMA: Double = 0
    private var bestRecordIndex: Int = 0

    /// Iteration of the record that set the best EMA (the first record is always the best).
    public private(set) var bestIteration: Int? = nil
    public private(set) var bestMetrics: HeldOutMetrics? = nil
    public private(set) var history: [(iteration: Int, metrics: HeldOutMetrics, ema: Double)] = []

    public init(policy: EarlyStoppingPolicy, maxIterations: Int) {
        self.policy = policy
        self.maxIterations = maxIterations
    }

    /// Records one held-out evaluation and returns whether training should stop after this iteration.
    public mutating func record(iteration: Int, metrics: HeldOutMetrics) -> Decision {
        let ssim: Double = metrics.ssim
        let ema: Double
        if let previous = history.last?.ema {
            ema = policy.emaWeight * ssim + (1 - policy.emaWeight) * previous
        } else {
            ema = ssim
        }
        history.append((iteration: iteration, metrics: metrics, ema: ema))
        let index: Int = history.count - 1

        if bestIteration == nil || ema > bestEMA + policy.minDelta {
            bestEMA = ema
            bestIteration = iteration
            bestMetrics = metrics
            bestRecordIndex = index
        }

        if iteration >= maxIterations {
            return .stop
        }
        let recordsSinceBest: Int = index - bestRecordIndex
        if policy.isEnabled && iteration >= policy.minimumIterations && recordsSinceBest >= policy.patience {
            return .stop
        }
        return .continue
    }

    /// `round(best · N/(N−1))` clamped to `[evalInterval, maxIterations]`; `maxIterations` when no
    /// evaluation was recorded. The factor keeps the per-view visit count of the chosen iteration.
    public func finalIterations(viewCount: Int) -> Int {
        guard let best = bestIteration else {
            return maxIterations
        }
        let factor: Double = viewCount > 1 ? Double(viewCount) / Double(viewCount - 1) : 1
        let scaled: Int = Int((Double(best) * factor).rounded())
        return min(max(scaled, policy.evalInterval), maxIterations)
    }
}
