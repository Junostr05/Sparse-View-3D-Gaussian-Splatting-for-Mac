import Foundation
import SplatCore

/// Why a depth map could not be aligned to the sparse points of a view.
public enum AlignmentFailure: Error, Sendable, Equatable {
    case tooFewPoints(Int)
    case noPositiveFit
    case tooFewInliers(inliers: Int, of: Int)
}

/// Scale and shift that map a relative depth map onto the SfM points of one view.
public struct AlignmentResult: Sendable, Equatable {
    public var scale: Double
    public var shift: Double
    /// The kind the map was fitted as (may differ from the declared kind after a correction).
    public var fittedKind: DepthKind
    public var inliers: Int
    public var samples: Int
    public var medianRelativeError: Double
    /// True when the declared relative kind failed and the other one was used.
    public var kindCorrected: Bool

    public init(scale: Double, shift: Double, fittedKind: DepthKind, inliers: Int, samples: Int,
                medianRelativeError: Double, kindCorrected: Bool) {
        self.scale = scale
        self.shift = shift
        self.fittedKind = fittedKind
        self.inliers = inliers
        self.samples = samples
        self.medianRelativeError = medianRelativeError
        self.kindCorrected = kindCorrected
    }
}

/// Robust scale/shift alignment of depth maps to SfM points (Phase 4 P4, P5).
///
/// All arithmetic is `Double`. Randomness only comes from a `SeededGenerator`, and ties are
/// resolved by the lowest index, so a given input and seed always give the same result.
public enum DepthAlignment {
    /// Side of the occlusion pre-filter cell in pixels (P5).
    static let occlusionCellSize: Int = 8
    /// Points farther than this factor times the cell's nearest point are occluded (P5).
    static let occlusionDepthRatio: Double = 1.15
    /// Up to this many points use every pair as a candidate; above it, pairs are sampled (P5).
    static let allPairsLimit: Int = 64
    static let sampledPairCount: Int = 1024
    /// Robust scale of the median absolute deviation for a normal distribution.
    static let madScale: Double = 1.4826
    /// Inlier gate in robust standard deviations.
    static let inlierGate: Double = 2.5
    static let refinementPasses: Int = 2

    /// Prior values and camera-z depths of the SfM points that are visible in one view.
    ///
    /// Each point is projected into the map's grid. Points behind the camera or outside the image
    /// are dropped. Per 8×8 cell, a point is kept only if its depth is at most 1.15 times the
    /// nearest point of that cell, which removes points occluded by a closer surface. Points on
    /// invalid map pixels are dropped too.
    public static func samples(points: [SIMD3<Float>], camera: Camera, map: DepthMap) -> [(prior: Double, z: Double)] {
        guard camera.width > 0, camera.height > 0, map.width > 0, map.height > 0 else {
            return []
        }
        let mapPerCameraX: Double = Double(map.width) / Double(camera.width)
        let mapPerCameraY: Double = Double(map.height) / Double(camera.height)

        // Projected points: pixel column, pixel row, depth, source index.
        var columns: [Int] = []
        var rows: [Int] = []
        var depths: [Double] = []
        var sourceIndices: [Int] = []
        for index in 0..<points.count {
            guard let projection = DepthGeometry.project(points[index], camera: camera, near: 0) else {
                continue
            }
            let u: Double = Double(projection.u)
            let v: Double = Double(projection.v)
            guard u >= 0, u < Double(camera.width), v >= 0, v < Double(camera.height) else {
                continue
            }
            let column: Int = min(Int((u * mapPerCameraX).rounded(.down)), map.width - 1)
            let row: Int = min(Int((v * mapPerCameraY).rounded(.down)), map.height - 1)
            columns.append(column)
            rows.append(row)
            depths.append(Double(projection.z))
            sourceIndices.append(index)
        }

        // Nearest depth per occlusion cell.
        let cellsAcross: Int = (map.width + occlusionCellSize - 1) / occlusionCellSize
        var nearestPerCell: [Int: Double] = [:]
        for k in 0..<columns.count {
            let cellX: Int = columns[k] / occlusionCellSize
            let cellY: Int = rows[k] / occlusionCellSize
            let cell: Int = cellY * cellsAcross + cellX
            if let current = nearestPerCell[cell] {
                nearestPerCell[cell] = min(current, depths[k])
            } else {
                nearestPerCell[cell] = depths[k]
            }
        }

        var result: [(prior: Double, z: Double)] = []
        result.reserveCapacity(columns.count)
        for k in 0..<columns.count {
            let cellX: Int = columns[k] / occlusionCellSize
            let cellY: Int = rows[k] / occlusionCellSize
            let cell: Int = cellY * cellsAcross + cellX
            let nearest: Double = nearestPerCell[cell] ?? depths[k]
            if depths[k] > occlusionDepthRatio * nearest {
                continue
            }
            let pixelIndex: Int = rows[k] * map.width + columns[k]
            guard map.isValid(pixelIndex) else {
                continue
            }
            let prior: Double = Double(map.values[pixelIndex])
            result.append((prior: prior, z: depths[k]))
        }
        return result
    }

    /// Fits `q ≈ s·p + t` with `q = 1/z` for inverse depth and `q = z` otherwise. With `scaleOnly`
    /// the shift is fixed at 0 (COLMAP metric depth).
    public static func fit(_ samples: [(prior: Double, z: Double)], kind: DepthKind, scaleOnly: Bool,
                           minimumPoints: Int, minimumInlierFraction: Double, seed: UInt64) -> Result<AlignmentResult, AlignmentFailure> {
        let count: Int = samples.count
        guard count >= minimumPoints else {
            return .failure(.tooFewPoints(count))
        }

        let inverse: Bool = (kind == .relativeInverseDepth)
        var priors: [Double] = []
        var targets: [Double] = []
        priors.reserveCapacity(count)
        targets.reserveCapacity(count)
        for sample in samples {
            priors.append(sample.prior)
            let target: Double = inverse ? 1.0 / sample.z : sample.z
            targets.append(target)
        }

        // Candidate (scale, shift) pairs, in deterministic order.
        var candidates: [(scale: Double, shift: Double)] = []
        if scaleOnly {
            for k in 0..<count {
                let prior: Double = priors[k]
                guard prior > 0 else {
                    continue
                }
                let scale: Double = targets[k] / prior
                if scale > 0 {
                    candidates.append((scale: scale, shift: 0))
                }
            }
        } else {
            var pairs: [(first: Int, second: Int)] = []
            if count <= allPairsLimit {
                for first in 0..<count {
                    for second in (first + 1)..<count {
                        pairs.append((first: first, second: second))
                    }
                }
            } else {
                var generator = SeededGenerator(seed: seed)
                for _ in 0..<sampledPairCount {
                    let first: Int = Int(generator.next() % UInt64(count))
                    let second: Int = Int(generator.next() % UInt64(count))
                    if first != second {
                        pairs.append((first: first, second: second))
                    }
                }
            }
            for pair in pairs {
                let priorA: Double = priors[pair.first]
                let priorB: Double = priors[pair.second]
                let priorDifference: Double = priorA - priorB
                let magnitude: Double = max(abs(priorA), abs(priorB), 1e-12)
                guard abs(priorDifference) >= 1e-9 * magnitude else {
                    continue
                }
                let targetDifference: Double = targets[pair.first] - targets[pair.second]
                let scale: Double = targetDifference / priorDifference
                guard scale > 0 else {
                    continue
                }
                let shift: Double = targets[pair.first] - scale * priorA
                candidates.append((scale: scale, shift: shift))
            }
        }
        guard !candidates.isEmpty else {
            return .failure(.noPositiveFit)
        }

        // Candidate with the smallest median absolute residual over all samples (first wins ties).
        var scale: Double = candidates[0].scale
        var shift: Double = candidates[0].shift
        var bestMedian: Double = Double.infinity
        for candidate in candidates {
            var absoluteResiduals: [Double] = []
            absoluteResiduals.reserveCapacity(count)
            for k in 0..<count {
                let residual: Double = candidate.scale * priors[k] + candidate.shift - targets[k]
                absoluteResiduals.append(abs(residual))
            }
            let candidateMedian: Double = median(absoluteResiduals)
            if candidateMedian < bestMedian {
                bestMedian = candidateMedian
                scale = candidate.scale
                shift = candidate.shift
            }
        }

        // Robust refinement: gate inliers by a MAD-based threshold and refit by least squares.
        for _ in 0..<refinementPasses {
            let residuals: [Double] = signedResiduals(priors: priors, targets: targets, scale: scale, shift: shift)
            let inlier: [Bool] = inlierFlags(residuals: residuals)
            var inlierCount: Int = 0
            var sumP: Double = 0
            var sumQ: Double = 0
            var sumPP: Double = 0
            var sumPQ: Double = 0
            for k in 0..<count where inlier[k] {
                inlierCount += 1
                sumP += priors[k]
                sumQ += targets[k]
                sumPP += priors[k] * priors[k]
                sumPQ += priors[k] * targets[k]
            }
            if scaleOnly {
                if sumPP > 0 {
                    scale = sumPQ / sumPP
                }
            } else {
                let countDouble: Double = Double(inlierCount)
                let determinant: Double = countDouble * sumPP - sumP * sumP
                let reference: Double = countDouble * sumPP
                if inlierCount >= 2, reference > 0, determinant > 1e-12 * reference {
                    let newScale: Double = (countDouble * sumPQ - sumP * sumQ) / determinant
                    let newShift: Double = (sumQ - newScale * sumP) / countDouble
                    scale = newScale
                    shift = newShift
                }
            }
            guard scale > 0 else {
                return .failure(.noPositiveFit)
            }
        }

        // Inlier count with the final parameters.
        let finalResiduals: [Double] = signedResiduals(priors: priors, targets: targets, scale: scale, shift: shift)
        let finalInlier: [Bool] = inlierFlags(residuals: finalResiduals)
        var inliers: Int = 0
        var relativeErrors: [Double] = []
        for k in 0..<count where finalInlier[k] {
            inliers += 1
            relativeErrors.append(abs(finalResiduals[k]) / targets[k])
        }
        let fractionRequired: Int = Int((minimumInlierFraction * Double(count)).rounded(.up))
        let requiredInliers: Int = max(minimumPoints, fractionRequired)
        guard inliers >= requiredInliers else {
            return .failure(.tooFewInliers(inliers: inliers, of: count))
        }

        let result = AlignmentResult(scale: scale, shift: shift,
                                     fittedKind: kind,
                                     inliers: inliers,
                                     samples: count,
                                     medianRelativeError: median(relativeErrors),
                                     kindCorrected: false)
        return .success(result)
    }

    /// Metric camera-z depth from an aligned relative map. Invalid pixels, and pixels whose
    /// aligned value is not positive (or, for inverse depth, not above `1 / zMax`), become 0.
    public static func metricDepth(_ map: DepthMap, alignment: AlignmentResult, zMax: Double) -> [Float] {
        var output: [Float] = [Float](repeating: 0, count: map.values.count)
        let inverse: Bool = (alignment.fittedKind == .relativeInverseDepth)
        let inverseFloor: Double = zMax > 0 ? 1.0 / zMax : Double.infinity
        for index in 0..<map.values.count where map.isValid(index) {
            let prior: Double = Double(map.values[index])
            let aligned: Double = alignment.scale * prior + alignment.shift
            if inverse {
                if aligned > inverseFloor {
                    output[index] = Float(1.0 / aligned)
                }
            } else if aligned > 0 {
                output[index] = Float(aligned)
            }
        }
        return output
    }

    // MARK: - Helpers

    /// Median of the values; for an even count the two middle values are averaged. 0 when empty.
    static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else {
            return 0
        }
        let sorted: [Double] = values.sorted()
        let middle: Int = sorted.count / 2
        if sorted.count % 2 == 1 {
            return sorted[middle]
        }
        let lower: Double = sorted[middle - 1]
        let upper: Double = sorted[middle]
        return (lower + upper) * 0.5
    }

    private static func signedResiduals(priors: [Double], targets: [Double], scale: Double, shift: Double) -> [Double] {
        var residuals: [Double] = []
        residuals.reserveCapacity(priors.count)
        for k in 0..<priors.count {
            residuals.append(scale * priors[k] + shift - targets[k])
        }
        return residuals
    }

    /// Inlier flags: |r| ≤ 2.5 · (1.4826 · median|r| + tiny).
    private static func inlierFlags(residuals: [Double]) -> [Bool] {
        let absolute: [Double] = residuals.map { abs($0) }
        let sigma: Double = madScale * median(absolute) + 1e-12
        let gate: Double = inlierGate * sigma
        var flags: [Bool] = []
        flags.reserveCapacity(residuals.count)
        for value in absolute {
            flags.append(value <= gate)
        }
        return flags
    }
}
