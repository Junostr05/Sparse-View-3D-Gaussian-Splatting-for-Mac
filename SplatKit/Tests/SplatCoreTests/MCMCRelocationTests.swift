import Foundation
import Testing
@testable import SplatCore

// CPU core of MCMC relocation and growth (v1.2): density factor, relocation formulas, sampling, the plan and the
// growth count. Runs on Linux and macOS.

@Suite struct MCMCRelocationTests {

    // MARK: - Helpers

    /// C(m, k) from factorials, independent of the table in `MCMCRelocation`.
    private func binomial(_ m: Int, _ k: Int) -> Double {
        guard k >= 0, k <= m else { return 0 }
        var numerator: Double = 1
        var denominator: Double = 1
        for step in 0..<k {
            numerator *= Double(m - step)
            denominator *= Double(step + 1)
        }
        return numerator / denominator
    }

    /// D(n, o') evaluated literally from its definition: Σ_{i=1}^{n} Σ_{k=0}^{i−1} C(i−1, k) (−1)^k / sqrt(k+1) o'^(k+1).
    private func directDensity(n: Int, opacity: Double) -> Double {
        var total: Double = 0
        for i in 1...n {
            for k in 0...(i - 1) {
                let sign: Double = k % 2 == 0 ? 1 : -1
                let term: Double = binomial(i - 1, k) * sign / Double(k + 1).squareRoot()
                total += term * pow(opacity, Double(k + 1))
            }
        }
        return total
    }

    private func composite(_ opacity: Double, copies: Int) -> Double {
        1 - pow(1 - opacity, Double(copies))
    }

    // MARK: - Density factor and relocation formulas

    @Test func densityFactorMatchesDirectDefinition() {
        for n in 1...12 {
            for opacity in [0.01, 0.2, 0.5, 0.9] {
                let expected: Double = directDensity(n: n, opacity: opacity)
                let actual: Double = MCMCRelocation.densityFactor(copies: n, opacity: opacity)
                #expect(abs(actual - expected) <= 1e-12 * max(1, abs(expected)), "n=\(n) o=\(opacity)")
            }
        }
    }

    @Test func densityFactorHandComputedValues() {
        let o: Double = 0.4
        // D(1, o) = o
        #expect(abs(MCMCRelocation.densityFactor(copies: 1, opacity: o) - o) < 1e-14)
        // D(2, o) = 2o − o² / √2
        let d2: Double = 2 * o - o * o / 2.0.squareRoot()
        #expect(abs(MCMCRelocation.densityFactor(copies: 2, opacity: o) - d2) < 1e-14)
        // D(3, o) = 3o − (3/√2) o² + o³ / √3
        let d3: Double = 3 * o - (3 / 2.0.squareRoot()) * o * o + o * o * o / 3.0.squareRoot()
        #expect(abs(MCMCRelocation.densityFactor(copies: 3, opacity: o) - d3) < 1e-14)
    }

    @Test func relocationPreservesCompositeOpacity() {
        for n in 2...20 {
            for opacity in [0.005, 0.1, 0.5, 0.9] {
                let newOpacity: Double = MCMCRelocation.relocatedOpacity(opacity: opacity, copies: n)
                #expect(abs(composite(newOpacity, copies: n) - opacity) < 1e-5, "n=\(n) o=\(opacity)")

                // The stored (Float) logit gives the same composite opacity.
                let appearance = MCMCRelocation.relocated(logScale: .zero, opacity: opacity, copies: n)
                let stored: Double = 1 / (1 + exp(-Double(appearance.opacityLogit)))
                #expect(abs(composite(stored, copies: n) - opacity) < 1e-5, "stored n=\(n) o=\(opacity)")
            }
        }
    }

    @Test func singleCopyLeavesGaussianUnchanged() {
        let logScale = SIMD3<Float>(-2.3, -1.1, -0.4)
        let appearance = MCMCRelocation.relocated(logScale: logScale, opacity: 0.37, copies: 1)
        #expect(appearance.logScale == logScale)
        let stored: Double = 1 / (1 + exp(-Double(appearance.opacityLogit)))
        #expect(abs(stored - 0.37) < 1e-6)
    }

    @Test func scaleShiftFollowsOpacityOverDensity() {
        let logScale = SIMD3<Float>(-1.5, -0.7, -2.2)
        let opacity: Double = 0.3
        for n in [2, 5, 17] {
            let newOpacity: Double = MCMCRelocation.relocatedOpacity(opacity: opacity, copies: n)
            let density: Double = MCMCRelocation.densityFactor(copies: n, opacity: newOpacity)
            // scale' = scale · o / D  ⇔  logScale' = logScale + log(o / D)
            let expectedShift: Double = log(opacity / density)
            let appearance = MCMCRelocation.relocated(logScale: logScale, opacity: opacity, copies: n)
            for axis in 0..<3 {
                let shift = Double(appearance.logScale[axis] - logScale[axis])
                #expect(abs(shift - expectedShift) < 1e-5, "n=\(n) axis=\(axis)")
            }
        }
    }

    @Test func copyCountIsCappedAtFiftyOne() {
        let logScale = SIMD3<Float>(-1, -1, -1)
        let capped = MCMCRelocation.relocated(logScale: logScale, opacity: 0.2, copies: MCMCRelocation.maxCopies)
        let above = MCMCRelocation.relocated(logScale: logScale, opacity: 0.2, copies: 80)
        #expect(capped == above)
    }

    // MARK: - Sampling

    @Test func samplingIsDeterministic() {
        let weights: [Double] = [0.1, 0.0, 0.3, 0.6]
        let first = MCMCRelocation.sampleSources(weights: weights, draws: 1_000, seed: 7)
        let second = MCMCRelocation.sampleSources(weights: weights, draws: 1_000, seed: 7)
        let other = MCMCRelocation.sampleSources(weights: weights, draws: 1_000, seed: 8)
        #expect(first == second)
        #expect(first != other)
    }

    @Test func samplingIsProportionalToOpacity() {
        let weights: [Double] = [0.1, 0.0, 0.3, 0.6]
        let draws = 200_000
        let samples = MCMCRelocation.sampleSources(weights: weights, draws: draws, seed: 11)
        #expect(samples.count == draws)
        var counts = [Int](repeating: 0, count: weights.count)
        for index in samples {
            counts[index] += 1
        }
        // A zero weight is never drawn.
        #expect(counts[1] == 0)
        let total: Double = weights.reduce(0, +)
        for (index, weight) in weights.enumerated() {
            let frequency = Double(counts[index]) / Double(draws)
            #expect(abs(frequency - weight / total) < 0.005, "index \(index): \(frequency)")
        }
    }

    @Test func samplingWithoutPositiveWeightReturnsNothing() {
        #expect(MCMCRelocation.sampleSources(weights: [0, 0], draws: 5, seed: 1).isEmpty)
        #expect(MCMCRelocation.sampleSources(weights: [1, 2], draws: 0, seed: 1).isEmpty)
    }

    // MARK: - Plan

    @Test func planPairsDeadWithLiveSourcesAndCountsCopies() {
        // Dead (below 0.005): indices 1, 3, 5.
        let opacities: [Float] = [0.9, 0.001, 0.5, 0.002, 0.7, 0.0001, 0.3]
        let plan = MCMCRelocation.plan(opacities: opacities, deadOpacity: 0.005, growth: 2, seed: 3)
        #expect(plan.replacements.map(\.target) == [1, 3, 5])
        #expect(plan.growthSources.count == 2)

        let live: Set<Int> = [0, 2, 4, 6]
        for replacement in plan.replacements {
            #expect(live.contains(replacement.source))
        }
        for source in plan.growthSources {
            #expect(live.contains(source))
        }

        var occurrences: [Int: Int] = [:]
        for replacement in plan.replacements {
            occurrences[replacement.source, default: 0] += 1
        }
        for source in plan.growthSources {
            occurrences[source, default: 0] += 1
        }
        #expect(plan.copies.count == occurrences.count)
        for entry in plan.copies {
            #expect(entry.count == 1 + (occurrences[entry.source] ?? 0))
        }
        #expect(plan.copies.map(\.source) == plan.copies.map(\.source).sorted())
        #expect(MCMCRelocation.plan(opacities: opacities, deadOpacity: 0.005, growth: 2, seed: 3) == plan)
    }

    @Test func planIsEmptyWithoutLiveGaussians() {
        let plan = MCMCRelocation.plan(opacities: [0.001, 0.002], deadOpacity: 0.005, growth: 5, seed: 1)
        #expect(plan.isEmpty)
    }

    @Test func planIsEmptyWithoutDeadOrGrowth() {
        let plan = MCMCRelocation.plan(opacities: [0.5, 0.9], deadOpacity: 0.005, growth: 0, seed: 1)
        #expect(plan.isEmpty)
    }

    // MARK: - Growth

    @Test func growthRespectsTheBudgetExactly() {
        #expect(MCMCRelocation.growthCount(count: 100, maxGaussians: 110, growthFraction: 0.5) == 10)
        #expect(MCMCRelocation.growthCount(count: 100, maxGaussians: 1_000, growthFraction: 0.25) == 25)
        #expect(MCMCRelocation.growthCount(count: 100, maxGaussians: 100, growthFraction: 0.5) == 0)
        #expect(MCMCRelocation.growthCount(count: 100, maxGaussians: 90, growthFraction: 0.5) == 0)
        // ceil(0.1 × 101) = ceil(10.1) = 11
        #expect(MCMCRelocation.growthCount(count: 101, maxGaussians: 1_000, growthFraction: 0.1) == 11)
        for count in stride(from: 1, through: 400, by: 37) {
            for maxGaussians in [count, count + 1, 2 * count, 3 * count] {
                let added = MCMCRelocation.growthCount(count: count, maxGaussians: maxGaussians, growthFraction: 0.3)
                #expect(count + added <= maxGaussians)
                #expect(added >= 0)
            }
        }
    }
}
