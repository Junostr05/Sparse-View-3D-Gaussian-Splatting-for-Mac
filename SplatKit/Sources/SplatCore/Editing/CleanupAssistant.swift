import Foundation

/// A kind of floater or artifact the cleanup assistant looks for.
public enum CleanupCriterion: String, CaseIterable, Sendable, Identifiable {
    /// Largest axis is a large fraction of the scene radius (with a percentile guard).
    case huge
    /// Opacity is nearly zero.
    case transparent
    /// The distance to the 4th nearest neighbour is large compared with the scene's typical spacing.
    case isolated
    /// The axis ratio is extreme: a stretched needle.
    case needle
    /// Far outside the dense core of the cloud, measured from the opacity-weighted centre.
    case outsideCore

    public var id: String { rawValue }

    /// The threshold of the criterion for `sensitivity`. Its meaning depends on the criterion: a fraction of the
    /// scene radius (`huge`), an opacity (`transparent`), a multiple of the median neighbour distance (`isolated`),
    /// an axis ratio (`needle`) or a multiple of the 98th-percentile radius (`outsideCore`).
    func threshold(for sensitivity: CleanupSensitivity) -> Float {
        switch self {
        case .huge:
            switch sensitivity {
            case .low: 0.25
            case .medium: 0.15
            case .high: 0.08
            }
        case .transparent:
            switch sensitivity {
            case .low: 0.02
            case .medium: 0.05
            case .high: 0.10
            }
        case .isolated:
            switch sensitivity {
            case .low: 10
            case .medium: 6
            case .high: 4
            }
        case .needle:
            switch sensitivity {
            case .low: 40
            case .medium: 20
            case .high: 10
            }
        case .outsideCore:
            switch sensitivity {
            case .low: 4
            case .medium: 3
            case .high: 2
            }
        }
    }
}

/// How aggressively a criterion flags Gaussians.
public enum CleanupSensitivity: String, CaseIterable, Sendable, Identifiable {
    case low
    case medium
    case high

    public var id: String { rawValue }
}

/// Whether one criterion is used, and how sensitive it is.
public struct CleanupRule: Sendable, Equatable {
    public var isEnabled: Bool
    public var sensitivity: CleanupSensitivity

    public init(isEnabled: Bool, sensitivity: CleanupSensitivity = .medium) {
        self.isEnabled = isEnabled
        self.sensitivity = sensitivity
    }
}

/// The rules of all criteria. Every criterion has a rule; a missing one counts as disabled.
public struct CleanupCriteria: Sendable, Equatable {
    public var rules: [CleanupCriterion: CleanupRule]

    public init(rules: [CleanupCriterion: CleanupRule] = CleanupCriteria.defaultRules) {
        self.rules = rules
    }

    /// Huge, transparent, isolated and needle Gaussians at medium sensitivity. The outside-core criterion is off,
    /// because a sparse outskirt of a real scene is not an artifact.
    public static let defaultRules: [CleanupCriterion: CleanupRule] = [
        .huge: CleanupRule(isEnabled: true),
        .transparent: CleanupRule(isEnabled: true),
        .isolated: CleanupRule(isEnabled: true),
        .needle: CleanupRule(isEnabled: true),
        .outsideCore: CleanupRule(isEnabled: false),
    ]

    /// The default criteria.
    public static let standard = CleanupCriteria()

    public subscript(criterion: CleanupCriterion) -> CleanupRule {
        get { rules[criterion] ?? CleanupRule(isEnabled: false) }
        set { rules[criterion] = newValue }
    }
}

/// What one enabled criterion proposes.
public struct CleanupCriterionResult: Sendable, Equatable, Identifiable {
    public let criterion: CleanupCriterion
    /// Gaussians the criterion flags before any safety limit.
    public let matchCount: Int
    /// The part of the proposal this criterion contributes, over the cloud's display indices.
    public let mask: SelectionMask

    public var id: CleanupCriterion { criterion }

    /// Gaussians of this criterion that the proposal contains.
    public var count: Int { mask.selectedCount }

    /// True when the safety limit kept fewer Gaussians than the criterion flagged.
    public var wasLimited: Bool { count < matchCount }

    public init(criterion: CleanupCriterion, matchCount: Int, mask: SelectionMask) {
        self.criterion = criterion
        self.matchCount = matchCount
        self.mask = mask
    }
}

/// The Gaussians the assistant proposes to delete, per criterion and as one union.
public struct CleanupProposal: Sendable, Equatable {
    /// Number of Gaussians of the cloud the proposal was computed for.
    public let cloudCount: Int
    /// The most Gaussians one proposal may contain: `CleanupAssistant.safetyPercent` of `cloudCount`, rounded down.
    public let safetyLimit: Int
    /// One result per enabled criterion, in `CleanupCriterion.allCases` order.
    public let results: [CleanupCriterionResult]
    /// Union of the criteria after the limits. Its count never exceeds `safetyLimit`.
    public let selection: SelectionMask
    /// True when the union had more candidates than `safetyLimit` and the most extreme ones were kept.
    public let unionWasLimited: Bool

    public init(cloudCount: Int,
                safetyLimit: Int,
                results: [CleanupCriterionResult],
                selection: SelectionMask,
                unionWasLimited: Bool) {
        self.cloudCount = cloudCount
        self.safetyLimit = safetyLimit
        self.results = results
        self.selection = selection
        self.unionWasLimited = unionWasLimited
    }

    /// Number of Gaussians in the proposal.
    public var count: Int { selection.selectedCount }

    /// Share of the cloud in the proposal, 0 for an empty cloud.
    public var fraction: Double {
        cloudCount > 0 ? Double(count) / Double(cloudCount) : 0
    }

    /// True when any criterion or the union was cut down by the safety limit.
    public var wasLimited: Bool {
        unionWasLimited || results.contains { $0.wasLimited }
    }

    public var isEmpty: Bool { count == 0 }
}

/// Finds floaters and artifacts in a cloud. Pure and deterministic: the same cloud and criteria always give the
/// same proposal, and no Gaussian is proposed unless a criterion flags it.
public enum CleanupAssistant {
    /// Most Gaussians one proposal may contain, in percent of the cloud.
    public static let safetyPercent: Int = 15
    /// Neighbour rank of the isolation criterion, the same as the heatmap's isolation metric.
    static let isolationNeighbours: Int = 4
    /// Huge Gaussians must also exceed this multiple of the median largest axis, so that a scene whose Gaussians
    /// are generally large does not flag most of them.
    static let hugeMedianGuard: Float = 10
    /// Percentile of the Gaussian radii used as the core radius of `outsideCore`.
    static let coreRadiusFraction: Float = 0.98

    /// Proposes the Gaussians of `cloud` that the enabled `criteria` flag.
    ///
    /// Each criterion keeps at most the safety limit, the most extreme Gaussians first. The union is limited the same
    /// way, so one proposal never contains more than 15 % of the cloud. Runs in O(N log N) for N Gaussians.
    public static func propose(cloud: GaussianCloud, criteria: CleanupCriteria) -> CleanupProposal {
        let count: Int = cloud.count
        let limit: Int = count * safetyPercent / 100
        guard count > 0 else {
            return CleanupProposal(cloudCount: 0, safetyLimit: 0, results: [],
                                   selection: SelectionMask(count: 0), unionWasLimited: false)
        }
        let radius: Float = sceneRadius(of: cloud)

        // Per criterion: the retained Gaussians, how many it flagged, and the margin of each retained one.
        var retainedMasks: [(criterion: CleanupCriterion, matchCount: Int, mask: SelectionMask)] = []
        // Largest margin of a retained Gaussian over the criteria that kept it. Ranks the union when it is limited.
        var rank = [Float](repeating: 0, count: count)

        for criterion in CleanupCriterion.allCases {
            let rule: CleanupRule = criteria[criterion]
            guard rule.isEnabled else { continue }
            let threshold: Float = criterion.threshold(for: rule.sensitivity)
            guard let margins = margins(for: criterion, cloud: cloud, threshold: threshold, radius: radius) else {
                retainedMasks.append((criterion, 0, SelectionMask(count: count)))
                continue
            }
            // A margin above 1 means the criterion flags the Gaussian.
            var flagged: [Int] = []
            for index in 0..<count where margins[index] > 1 {
                flagged.append(index)
            }
            let matchCount: Int = flagged.count
            var kept: [Int] = flagged
            if flagged.count > limit {
                kept = flagged.sorted { (a: Int, b: Int) -> Bool in
                    margins[a] != margins[b] ? margins[a] > margins[b] : a < b
                }
                kept = Array(kept.prefix(limit))
            }
            for index in kept {
                rank[index] = max(rank[index], margins[index])
            }
            retainedMasks.append((criterion, matchCount, SelectionMask(count: count, indices: kept)))
        }

        var union = SelectionMask(count: count)
        for entry in retainedMasks {
            union.formUnion(entry.mask)
        }
        var unionWasLimited: Bool = false
        if union.selectedCount > limit {
            unionWasLimited = true
            let candidates: [Int] = union.selectedIndices().sorted { (a: Int, b: Int) -> Bool in
                rank[a] != rank[b] ? rank[a] > rank[b] : a < b
            }
            union = SelectionMask(count: count, indices: Array(candidates.prefix(limit)))
        }

        let results: [CleanupCriterionResult] = retainedMasks.map { entry in
            // Each criterion's share of the final union, so that the per-criterion counts add up to the proposal.
            let share: [Int] = entry.mask.selectedIndices().filter { union[$0] }
            return CleanupCriterionResult(criterion: entry.criterion,
                                          matchCount: entry.matchCount,
                                          mask: SelectionMask(count: count, indices: share))
        }
        return CleanupProposal(cloudCount: count,
                               safetyLimit: limit,
                               results: results,
                               selection: union,
                               unionWasLimited: unionWasLimited)
    }

    // MARK: - Margins

    /// Ratio of each Gaussian's value to the criterion's limit: above 1 is flagged, and a larger value is more
    /// extreme. `nil` when the criterion does not apply to this cloud (for example no spread to compare with).
    static func margins(for criterion: CleanupCriterion,
                        cloud: GaussianCloud,
                        threshold: Float,
                        radius: Float) -> [Float]? {
        switch criterion {
        case .huge:
            return hugeMargins(cloud: cloud, fraction: threshold, radius: radius)
        case .transparent:
            return transparentMargins(cloud: cloud, opacity: threshold)
        case .isolated:
            return isolatedMargins(cloud: cloud, multiple: threshold)
        case .needle:
            return needleMargins(cloud: cloud, ratio: threshold)
        case .outsideCore:
            return outsideCoreMargins(cloud: cloud, multiple: threshold)
        }
    }

    /// Largest axis over the threshold: `fraction` of the scene radius, but at least `hugeMedianGuard` times the
    /// median largest axis.
    static func hugeMargins(cloud: GaussianCloud, fraction: Float, radius: Float) -> [Float]? {
        let count: Int = cloud.count
        var largest = [Float](repeating: 0, count: count)
        for index in 0..<count {
            let base: Int = 3 * index
            let largestLog: Float = max(cloud.logScales[base], max(cloud.logScales[base + 1], cloud.logScales[base + 2]))
            largest[index] = exp(largestLog)
        }
        guard let median = percentile(largest, fraction: 0.5) else { return nil }
        let threshold: Float = max(fraction * radius, hugeMedianGuard * median)
        guard threshold > 0, threshold.isFinite else { return nil }
        return largest.map { sanitized($0 / threshold) }
    }

    /// Limit over the opacity: the margin is above 1 exactly when the opacity is below `opacity`.
    static func transparentMargins(cloud: GaussianCloud, opacity: Float) -> [Float] {
        var margins = [Float](repeating: 0, count: cloud.count)
        for index in 0..<cloud.count {
            let value: Float = 1 / (1 + exp(-cloud.opacityLogits[index]))
            margins[index] = sanitized(opacity / max(value, 1e-12))
        }
        return margins
    }

    /// Fourth-neighbour distance over `multiple` times the median fourth-neighbour distance of the cloud.
    static func isolatedMargins(cloud: GaussianCloud, multiple: Float) -> [Float]? {
        let distances: [Float] = ViewerColors.isolationDistances(positions: ViewerColors.positions(of: cloud),
                                                                 k: isolationNeighbours)
        guard let median = percentile(distances, fraction: 0.5), median > 0 else { return nil }
        let threshold: Float = multiple * median
        return distances.map { sanitized($0 / threshold) }
    }

    /// Largest over smallest axis, over `ratio`.
    static func needleMargins(cloud: GaussianCloud, ratio: Float) -> [Float] {
        var margins = [Float](repeating: 0, count: cloud.count)
        for index in 0..<cloud.count {
            let base: Int = 3 * index
            let logs: (Float, Float, Float) = (cloud.logScales[base], cloud.logScales[base + 1], cloud.logScales[base + 2])
            let largestLog: Float = max(logs.0, max(logs.1, logs.2))
            let smallestLog: Float = min(logs.0, min(logs.1, logs.2))
            margins[index] = sanitized(exp(largestLog - smallestLog) / ratio)
        }
        return margins
    }

    /// Distance from the opacity-weighted centre over `multiple` times the 98th-percentile radius.
    static func outsideCoreMargins(cloud: GaussianCloud, multiple: Float) -> [Float]? {
        let count: Int = cloud.count
        var weightSum: Double = 0
        var weighted: SIMD3<Double> = .zero
        var plain: SIMD3<Double> = .zero
        for index in 0..<count {
            let base: Int = 3 * index
            let position: SIMD3<Double> = SIMD3<Double>(Double(cloud.positions[base]),
                                                        Double(cloud.positions[base + 1]),
                                                        Double(cloud.positions[base + 2]))
            let weight: Double = 1 / (1 + exp(-Double(cloud.opacityLogits[index])))
            weightSum += weight
            weighted += weight * position
            plain += position
        }
        let centre: SIMD3<Double> = weightSum > 0 ? weighted / weightSum : plain / Double(max(count, 1))
        var radii = [Float](repeating: 0, count: count)
        for index in 0..<count {
            let base: Int = 3 * index
            let offset: SIMD3<Double> = SIMD3<Double>(Double(cloud.positions[base]),
                                                      Double(cloud.positions[base + 1]),
                                                      Double(cloud.positions[base + 2])) - centre
            radii[index] = Float((offset * offset).sum().squareRoot())
        }
        guard let core = percentile(radii, fraction: coreRadiusFraction), core > 0 else { return nil }
        let threshold: Float = multiple * core
        return radii.map { sanitized($0 / threshold) }
    }

    // MARK: - Helpers

    /// Half the diagonal of the robust bounds, the radius the size criteria are relative to. 1 for a degenerate cloud.
    static func sceneRadius(of cloud: GaussianCloud) -> Float {
        let extent: SIMD3<Float> = cloud.robustBounds().extent
        let radius: Float = 0.5 * (extent * extent).sum().squareRoot()
        return radius.isFinite && radius > 0 ? radius : 1
    }

    /// Nearest-rank percentile of the finite values, or `nil` without finite values.
    static func percentile(_ values: [Float], fraction: Float) -> Float? {
        var finite: [Float] = values.filter { $0.isFinite }
        guard !finite.isEmpty else { return nil }
        finite.sort()
        let last: Int = finite.count - 1
        let rank: Int = min(max(Int(Float(last) * fraction), 0), last)
        return finite[rank]
    }

    /// NaN becomes 0 (never flagged). Infinity is kept: an infinite margin is the most extreme value.
    private static func sanitized(_ value: Float) -> Float {
        value.isNaN ? 0 : value
    }
}
