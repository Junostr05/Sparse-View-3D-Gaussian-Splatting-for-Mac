import Foundation

// CPU reference densification (gsplat default strategy semantics, opacity reset off).
// The Metal kernels of Phase 3 Task 14 mirror these rules line by line.
//
// Interpretations of the plan, where it is literal:
// - Statistics are `Float`, as in the GPU accumulators. Each norm is computed in Float.
// - `plan` step 2 ("every prune → keep") turns all prunes into keeps when no Gaussian
//   survives otherwise; the flag `keptAll` records this.
// - Demoted growth candidates are reported as `.keep` in `DensifyPlan.actions`.
// - `base` counts keep + clone + split once per Gaussian (not their outputs). Hence the output
//   count is `base + granted growths`, which stays ≤ `maxGaussians` whenever the input count
//   is ≤ `maxGaussians`.
// - `apply` assigns offsets by an exclusive scan of `outputCount` and appends in input order,
//   which yields the same layout as writing each output at its offset.
// - Split offsets use `UInt32(truncatingIfNeeded: iteration)` for the seed; iterations are
//   non-negative, so this is the same value as `UInt32(iteration)`.

/// Accumulated screen-space statistics per Gaussian, used to decide densification.
public struct DensificationStatistics: Sendable, Equatable {
    /// Σ over views of ‖(dL/dmean2D.x · W/2, dL/dmean2D.y · H/2)‖.
    public var gradientNormSums: [Float]
    /// Number of views in which the Gaussian was visible.
    public var visibleCounts: [UInt32]

    public init(count: Int) {
        gradientNormSums = [Float](repeating: 0, count: count)
        visibleCounts = [UInt32](repeating: 0, count: count)
    }

    /// Adds one view. `mean2D` is `dL/dmean2D` in pixels; only visible Gaussians contribute.
    public mutating func accumulate(mean2D: [SIMD2<Float>], visible: [Bool], width: Int, height: Int) {
        let n = gradientNormSums.count
        precondition(mean2D.count == n && visible.count == n,
                     "screen-space arrays must match the statistics count")
        let halfWidth = Float(width) / 2
        let halfHeight = Float(height) / 2
        for i in 0..<n where visible[i] {
            let scaled = SIMD2<Float>(mean2D[i].x * halfWidth, mean2D[i].y * halfHeight)
            gradientNormSums[i] += (scaled * scaled).sum().squareRoot()
            visibleCounts[i] += 1
        }
    }

    public mutating func reset() {
        for i in 0..<gradientNormSums.count {
            gradientNormSums[i] = 0
            visibleCounts[i] = 0
        }
    }
}

/// Per-Gaussian densification decision. Raw values match the Metal kernel's output codes.
public enum DensifyAction: UInt32, Sendable {
    case prune = 0
    case keep = 1
    case clone = 2
    case split = 3

    /// Number of Gaussians this input produces: 0, 1, 2, 2.
    public var outputCount: Int {
        switch self {
        case .prune: 0
        case .keep: 1
        case .clone, .split: 2
        }
    }
}

/// Result of planning one densification step.
public struct DensifyPlan: Sendable, Equatable {
    public var actions: [DensifyAction]
    public var outputCount: Int
    public var pruned: Int
    public var cloned: Int
    public var split: Int
    /// Growth candidates that were turned into keeps because the budget was reached.
    public var demoted: Int
    /// True when every Gaussian was marked for pruning and the model was kept intact.
    public var keptAll: Bool
}

public enum Densification {
    /// Decision for one Gaussian, in the order prune (opacity, then scale after the start
    /// iteration), growth (clone or split by maximum scale), keep.
    public static func action(logScale: SIMD3<Float>,
                              opacityLogit: Float,
                              gradientNormSum: Float,
                              visibleCount: UInt32,
                              policy: DensificationPolicy,
                              sceneRadius: Float,
                              iteration: Int) -> DensifyAction {
        let maxScale = exp(max(logScale.x, max(logScale.y, logScale.z)))
        let opacity: Float = 1 / (1 + exp(-opacityLogit))

        if opacity < policy.pruneOpacity {
            return .prune
        }
        if iteration > policy.pruneScaleStartIteration && maxScale > policy.pruneScaleFraction * sceneRadius {
            return .prune
        }
        let average: Float = visibleCount > 0 ? gradientNormSum / Float(visibleCount) : 0
        guard average > policy.gradientThreshold else {
            return .keep
        }
        return maxScale <= policy.splitScaleFraction * sceneRadius ? .clone : .split
    }

    /// Decides every Gaussian's fate and enforces the budget. Growth candidates (clone or split)
    /// are granted in ascending index order until `maxGaussians` would be exceeded.
    public static func plan(_ cloud: GaussianCloud,
                            statistics: DensificationStatistics,
                            policy: DensificationPolicy,
                            sceneRadius: Float,
                            iteration: Int,
                            maxGaussians: Int) -> DensifyPlan {
        let n = cloud.count
        precondition(statistics.gradientNormSums.count == n && statistics.visibleCounts.count == n,
                     "statistics must match the cloud")

        // 1. Decide on the parent.
        var actions: [DensifyAction] = []
        actions.reserveCapacity(n)
        for i in 0..<n {
            actions.append(action(logScale: SIMD3(cloud.logScales[3 * i], cloud.logScales[3 * i + 1],
                                                  cloud.logScales[3 * i + 2]),
                                  opacityLogit: cloud.opacityLogits[i],
                                  gradientNormSum: statistics.gradientNormSums[i],
                                  visibleCount: statistics.visibleCounts[i],
                                  policy: policy,
                                  sceneRadius: sceneRadius,
                                  iteration: iteration))
        }

        // 2. Never empty the model: if nothing survives, keep everything.
        var keptAll = false
        if n > 0 && !actions.contains(where: { $0 != .prune }) {
            for i in 0..<n {
                actions[i] = .keep
            }
            keptAll = true
        }

        // 3. Budget: base slots plus granted growth must not exceed the cap.
        var base = 0
        for a in actions where a != .prune {
            base += 1
        }
        let allowed = max(0, maxGaussians - base)
        var candidateRank = 0
        var demoted = 0
        for i in 0..<n {
            switch actions[i] {
            case .clone, .split:
                if candidateRank >= allowed {
                    actions[i] = .keep
                    demoted += 1
                }
                candidateRank += 1
            case .prune, .keep:
                break
            }
        }

        // 4. Totals.
        var pruned = 0
        var cloned = 0
        var split = 0
        var outputCount = 0
        for a in actions {
            outputCount += a.outputCount
            switch a {
            case .prune: pruned += 1
            case .clone: cloned += 1
            case .split: split += 1
            case .keep: break
            }
        }

        return DensifyPlan(actions: actions, outputCount: outputCount, pruned: pruned,
                           cloned: cloned, split: split, demoted: demoted, keptAll: keptAll)
    }

    /// Materialises `plan`: the new cloud and the matching Adam moments (written back to
    /// `state`; the step counter is unchanged).
    public static func apply(_ plan: DensifyPlan,
                             to cloud: GaussianCloud,
                             state: inout AdamState,
                             policy: DensificationPolicy,
                             seed: UInt32,
                             iteration: Int) -> GaussianCloud {
        let n = cloud.count
        precondition(plan.actions.count == n, "plan must cover every Gaussian")
        precondition(state.m.opacityLogits.count == n && state.v.opacityLogits.count == n,
                     "Adam moments must match the cloud")

        let shStride = 3 * GaussianCloud.coefficientCount(degree: cloud.shDegree)
        let source = GaussianGradients(positions: cloud.positions, logScales: cloud.logScales,
                                       rotations: cloud.rotations,
                                       opacityLogits: cloud.opacityLogits, sh: cloud.sh)
        let oldM = state.m
        let oldV = state.v

        var params = SoABuilder(shStride: shStride)
        var newM = SoABuilder(shStride: shStride)
        var newV = SoABuilder(shStride: shStride)
        let logDivisor = log(policy.splitScaleDivisor)
        let streamSeed = seed &+ UInt32(truncatingIfNeeded: iteration)

        for i in 0..<n {
            switch plan.actions[i] {
            case .prune:
                break
            case .keep:
                params.appendCopy(of: source, row: i)
                newM.appendCopy(of: oldM, row: i)
                newV.appendCopy(of: oldV, row: i)
            case .clone:
                params.appendCopy(of: source, row: i)
                newM.appendCopy(of: oldM, row: i)
                newV.appendCopy(of: oldV, row: i)
                params.appendCopy(of: source, row: i)
                newM.appendZeros()
                newV.appendZeros()
            case .split:
                let parentPosition = SIMD3<Float>(source.positions[3 * i], source.positions[3 * i + 1],
                                                  source.positions[3 * i + 2])
                let parentLogScale = SIMD3<Float>(source.logScales[3 * i], source.logScales[3 * i + 1],
                                                  source.logScales[3 * i + 2])
                let rotation = SIMD4<Float>(source.rotations[4 * i], source.rotations[4 * i + 1],
                                            source.rotations[4 * i + 2], source.rotations[4 * i + 3])
                let rotationMatrix = Matrix3.rotation(quaternion: rotation)
                let scale = SIMD3<Float>(exp(parentLogScale.x), exp(parentLogScale.y), exp(parentLogScale.z))
                let childLogScale = parentLogScale - SIMD3<Float>(repeating: logDivisor)
                let shSlice = source.sh[(i * shStride)..<((i + 1) * shStride)]
                for j in 0..<2 {
                    let stream = UInt32(3 * j)
                    let eps = SIMD3<Float>(
                        SplatRandom.normal(seed: streamSeed, index: UInt32(i), stream: stream),
                        SplatRandom.normal(seed: streamSeed, index: UInt32(i), stream: stream + 1),
                        SplatRandom.normal(seed: streamSeed, index: UInt32(i), stream: stream + 2))
                    let offset = rotationMatrix * (scale * eps)
                    params.appendGaussian(position: parentPosition + offset,
                                          logScale: childLogScale,
                                          rotation: rotation,
                                          opacityLogit: source.opacityLogits[i],
                                          sh: shSlice)
                    newM.appendZeros()
                    newV.appendZeros()
                }
            }
        }

        state.m = GaussianMoments(positions: newM.positions, logScales: newM.logScales,
                                  rotations: newM.rotations, opacityLogits: newM.opacityLogits,
                                  sh: newM.sh)
        state.v = GaussianMoments(positions: newV.positions, logScales: newV.logScales,
                                  rotations: newV.rotations, opacityLogits: newV.opacityLogits,
                                  sh: newV.sh)

        // The layout is consistent by construction, so the checked initializer cannot fail.
        return try! GaussianCloud(shDegree: cloud.shDegree,
                                  positions: params.positions,
                                  logScales: params.logScales,
                                  rotations: params.rotations,
                                  opacityLogits: params.opacityLogits,
                                  sh: params.sh)
    }
}

/// Appends whole Gaussians to structure-of-arrays storage.
private struct SoABuilder {
    var positions: [Float] = []
    var logScales: [Float] = []
    var rotations: [Float] = []
    var opacityLogits: [Float] = []
    var sh: [Float] = []
    let shStride: Int

    init(shStride: Int) {
        self.shStride = shStride
    }

    mutating func appendCopy(of source: GaussianGradients, row i: Int) {
        positions.append(contentsOf: source.positions[(3 * i)..<(3 * i + 3)])
        logScales.append(contentsOf: source.logScales[(3 * i)..<(3 * i + 3)])
        rotations.append(contentsOf: source.rotations[(4 * i)..<(4 * i + 4)])
        opacityLogits.append(source.opacityLogits[i])
        sh.append(contentsOf: source.sh[(i * shStride)..<((i + 1) * shStride)])
    }

    mutating func appendGaussian(position: SIMD3<Float>, logScale: SIMD3<Float>,
                                 rotation: SIMD4<Float>, opacityLogit: Float,
                                 sh shSlice: ArraySlice<Float>) {
        positions += [position.x, position.y, position.z]
        logScales += [logScale.x, logScale.y, logScale.z]
        rotations += [rotation.x, rotation.y, rotation.z, rotation.w]
        opacityLogits.append(opacityLogit)
        sh.append(contentsOf: shSlice)
    }

    mutating func appendZeros() {
        positions.append(contentsOf: repeatElement(Float(0), count: 3))
        logScales.append(contentsOf: repeatElement(Float(0), count: 3))
        rotations.append(contentsOf: repeatElement(Float(0), count: 4))
        opacityLogits.append(0)
        sh.append(contentsOf: repeatElement(Float(0), count: shStride))
    }
}
