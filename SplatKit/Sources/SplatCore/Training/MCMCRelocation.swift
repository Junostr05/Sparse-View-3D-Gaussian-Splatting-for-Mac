import Foundation

// MCMC densification (v1.2), relocation and growth, after "3D Gaussian Splatting as Markov Chain Monte Carlo"
// (Kheradmand et al. 2024). This file is the pure CPU core: sampling, the relocation formulas and the growth
// count. It is Linux-testable. The GPU trainer applies the plan to its buffers at a wait point (`MCMCRelocator`
// in SplatTraining). Formulas are evaluated in Double; the results are stored as Float.

/// Result of relocating one source Gaussian: the new opacity logit and log-scales of every copy of it.
public struct MCMCRelocatedAppearance: Sendable, Equatable {
    public var opacityLogit: Float
    public var logScale: SIMD3<Float>

    public init(opacityLogit: Float, logScale: SIMD3<Float>) {
        self.opacityLogit = opacityLogit
        self.logScale = logScale
    }
}

/// One dead Gaussian that is overwritten with a copy of a live source.
public struct MCMCReplacement: Sendable, Equatable {
    /// Index of the dead Gaussian that receives the copy.
    public var target: Int
    /// Index of the live Gaussian that is copied.
    public var source: Int

    public init(target: Int, source: Int) {
        self.target = target
        self.source = source
    }
}

/// Copy count of one source Gaussian: `n = 1 + (times sampled as a replacement or as a growth copy)`.
public struct MCMCSourceCopies: Sendable, Equatable {
    public var source: Int
    public var count: Int

    public init(source: Int, count: Int) {
        self.source = source
        self.count = count
    }
}

/// What one relocation step does. Pure data; `MCMCRelocation.plan` decides it.
public struct MCMCRelocationPlan: Sendable, Equatable {
    /// Dead Gaussians in index order, each with the live source it copies.
    public var replacements: [MCMCReplacement]
    /// Source of each new Gaussian, in order. The new Gaussians are appended at `count`, `count + 1`, ...
    public var growthSources: [Int]
    /// Copy count of every source that is sampled at least once, sorted by source index.
    public var copies: [MCMCSourceCopies]

    public init(replacements: [MCMCReplacement], growthSources: [Int], copies: [MCMCSourceCopies]) {
        self.replacements = replacements
        self.growthSources = growthSources
        self.copies = copies
    }

    /// True when the step changes nothing.
    public var isEmpty: Bool {
        replacements.isEmpty && growthSources.isEmpty
    }
}

public enum MCMCRelocation {
    /// Largest copy count the formulas use. Larger counts are treated as this value.
    public static let maxCopies = 51

    /// `densityWeights[n][k]`, for `n` in `0...maxCopies` and `k` in `0..<n`, such that
    /// `D(n, o') = Σ_k densityWeights[n][k] · o'^(k+1)` with
    /// `D(n, o') = Σ_{i=1}^{n} Σ_{k=0}^{i−1} C(i−1, k) (−1)^k / sqrt(k+1) · o'^(k+1)`.
    /// Swapping the sums gives `w[n][k] = (−1)^k / sqrt(k+1) · Σ_{i=k+1}^{n} C(i−1, k)`.
    private static let densityWeights: [[Double]] = {
        // Pascal's triangle, exact in Double for the sizes used here.
        var binomial = [[Double]](repeating: [Double](repeating: 0, count: maxCopies + 1),
                                  count: maxCopies + 1)
        binomial[0][0] = 1
        for m in 1...maxCopies {
            binomial[m][0] = 1
            for k in 1...m {
                binomial[m][k] = binomial[m - 1][k - 1] + binomial[m - 1][k]
            }
        }
        var weights = [[Double]](repeating: [Double](repeating: 0, count: maxCopies), count: maxCopies + 1)
        for n in 1...maxCopies {
            for k in 0..<n {
                var binomialSum: Double = 0
                for i in (k + 1)...n {
                    binomialSum += binomial[i - 1][k]
                }
                let sign: Double = k % 2 == 0 ? 1 : -1
                weights[n][k] = sign * binomialSum / Double(k + 1).squareRoot()
            }
        }
        return weights
    }()

    /// Logistic function of a logit, in Float.
    public static func sigmoid(_ logit: Float) -> Float {
        1 / (1 + exp(-logit))
    }

    /// Copy count clamped to `1...maxCopies`.
    static func clampedCopies(_ copies: Int) -> Int {
        max(1, min(copies, maxCopies))
    }

    /// Opacity of each of `n` copies whose composite opacity equals `opacity`: `o' = 1 − (1 − o)^(1/n)`.
    /// Computed as `−expm1(log1p(−o) / n)`, which stays accurate for opacities near 0 and 1.
    public static func relocatedOpacity(opacity: Double, copies: Int) -> Double {
        let n = Double(clampedCopies(copies))
        return -expm1(log1p(-opacity) / n)
    }

    /// `D(n, o')` of the relocation scale formula, evaluated with the precomputed weights.
    public static func densityFactor(copies: Int, opacity: Double) -> Double {
        let n = clampedCopies(copies)
        var power: Double = 1
        var total: Double = 0
        for k in 0..<n {
            power *= opacity
            total += densityWeights[n][k] * power
        }
        return total
    }

    /// Opacity logit of a probability, in Float.
    public static func logit(_ probability: Double) -> Float {
        Float(log(probability) - log1p(-probability))
    }

    /// New opacity and log-scales of a source relocated to `copies` copies:
    /// `o' = 1 − (1 − o)^(1/n)` and `scale' = scale · o / D(n, o')`, so `logScale' = logScale + log(o / D)`.
    /// With one copy the Gaussian is unchanged: the log-scales are returned as given, the opacity logit is
    /// `logit(o)`. If `D` is not positive and finite, the scale is kept (not expected for valid inputs).
    public static func relocated(logScale: SIMD3<Float>, opacity: Double, copies: Int) -> MCMCRelocatedAppearance {
        let n = clampedCopies(copies)
        if n == 1 {
            return MCMCRelocatedAppearance(opacityLogit: logit(opacity), logScale: logScale)
        }
        let newOpacity = relocatedOpacity(opacity: opacity, copies: n)
        let density = densityFactor(copies: n, opacity: newOpacity)
        var newLogScale = logScale
        if density > 0, density.isFinite, opacity > 0 {
            let shift = Float(log(opacity) - log(density))
            newLogScale = logScale + SIMD3<Float>(repeating: shift)
        }
        return MCMCRelocatedAppearance(opacityLogit: logit(newOpacity), logScale: newLogScale)
    }

    /// Number of Gaussians added per relocation step: `min(maxGaussians − count, ceil(growthFraction · count))`,
    /// never negative. The result never takes the count above `maxGaussians`.
    public static func growthCount(count: Int, maxGaussians: Int, growthFraction: Float) -> Int {
        let room = maxGaussians - count
        guard room > 0, count > 0, growthFraction > 0 else {
            return 0
        }
        let wanted = Int((growthFraction * Float(count)).rounded(.up))
        return min(room, wanted)
    }

    /// Deterministic multinomial draws with probabilities proportional to `weights` (negative weights count as
    /// zero). Inverse-CDF sampling on a `SeededGenerator`, so the same seed gives the same draws. Returns an empty
    /// array when the total weight is not positive and finite.
    public static func sampleSources(weights: [Double], draws: Int, seed: UInt64) -> [Int] {
        guard draws > 0, !weights.isEmpty else {
            return []
        }
        var cumulative: [Double] = []
        cumulative.reserveCapacity(weights.count)
        var running: Double = 0
        for weight in weights {
            running += max(0, weight)
            cumulative.append(running)
        }
        guard running > 0, running.isFinite else {
            return []
        }
        var generator = SeededGenerator(seed: seed)
        var result: [Int] = []
        result.reserveCapacity(draws)
        for _ in 0..<draws {
            let target = Double.random(in: 0..<running, using: &generator)
            // The first index whose cumulative weight exceeds the target. Zero-weight entries never qualify.
            var low = 0
            var high = cumulative.count - 1
            while low < high {
                let middle = (low + high) / 2
                if cumulative[middle] > target {
                    high = middle
                } else {
                    low = middle + 1
                }
            }
            result.append(low)
        }
        return result
    }

    /// Plans one relocation step. Gaussians with opacity below `deadOpacity` are dead. Each dead one is paired with
    /// a live source drawn with probability proportional to its opacity, and `growth` further live sources are
    /// drawn the same way for the new Gaussians. A source's copy count is `1 +` the number of times it was drawn.
    /// Returns an empty plan when there is nothing to do or no live Gaussian to copy.
    public static func plan(opacities: [Float], deadOpacity: Float, growth: Int, seed: UInt64) -> MCMCRelocationPlan {
        var dead: [Int] = []
        var weights = [Double](repeating: 0, count: opacities.count)
        for index in 0..<opacities.count {
            let opacity = opacities[index]
            if opacity < deadOpacity {
                dead.append(index)
            } else {
                weights[index] = Double(max(0, opacity))
            }
        }
        let growthCount = max(0, growth)
        let draws = dead.count + growthCount
        guard draws > 0 else {
            return MCMCRelocationPlan(replacements: [], growthSources: [], copies: [])
        }
        let sampled = sampleSources(weights: weights, draws: draws, seed: seed)
        guard sampled.count == draws else {
            return MCMCRelocationPlan(replacements: [], growthSources: [], copies: [])
        }

        var replacements: [MCMCReplacement] = []
        replacements.reserveCapacity(dead.count)
        for (position, target) in dead.enumerated() {
            replacements.append(MCMCReplacement(target: target, source: sampled[position]))
        }
        let growthSources = Array(sampled[dead.count...])

        var counts: [Int: Int] = [:]
        for source in sampled {
            counts[source, default: 1] += 1
        }
        let copies = counts.keys.sorted().map { MCMCSourceCopies(source: $0, count: counts[$0] ?? 1) }
        return MCMCRelocationPlan(replacements: replacements, growthSources: growthSources, copies: copies)
    }
}
