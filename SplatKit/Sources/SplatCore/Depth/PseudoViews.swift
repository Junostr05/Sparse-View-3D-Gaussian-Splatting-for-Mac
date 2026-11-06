import Foundation

/// An interpolated camera between two metric views, with a depth prior obtained by forward-warping
/// the parents' metric depth (Phase 4 P9). Used for depth-only regularization; no RGB target.
public struct PseudoView: Sendable, Equatable {
    public var camera: Camera
    public var prior: DepthPrior
    /// Indices into the source view array of the two parents, ascending.
    public var parents: SIMD2<Int>
    /// Interpolation parameter between the parents, in (0, 1).
    public var t: Float
}

/// Why a candidate pseudo-view was not generated. One entry per rejected candidate.
public enum PseudoViewRejection: Sendable, Equatable {
    /// The parents' optical axes differ by more than `maxAngleDegrees`.
    case wideAngle
    /// Fraction of pseudo pixels that receive a warped depth is below `minimumCoverage`.
    case lowCoverage(Float)
    /// The 5th-percentile warped depth is below `minimumDepthFraction` × the parents' median depth.
    case tooClose
}

/// Pseudo-view generation for sparse-view training (Phase 4 P9). Pure CPU, deterministic.
public enum PseudoViews {

    // MARK: - Pairs

    /// Unique nearest-neighbour pairs of metric views by camera centre, as indices into `views`,
    /// sorted ascending. Ties are broken by the lowest index. Views without metric depth are skipped.
    public static func pairs(_ views: [TrainingView]) -> [SIMD2<Int>] {
        let metric: [Int] = views.indices.filter { hasMetricDepth(views[$0]) }
        var found: [SIMD2<Int>] = []
        for i in metric {
            let centerI: SIMD3<Float> = views[i].camera.center
            var best = -1
            var bestDistance = Double.infinity
            for j in metric where j != i {
                let distance: Double = squaredDistance(centerI, views[j].camera.center)
                if distance < bestDistance {
                    bestDistance = distance
                    best = j
                }
            }
            if best >= 0 {
                found.append(SIMD2<Int>(min(i, best), max(i, best)))
            }
        }
        found.sort { lhs, rhs in
            lhs.x != rhs.x ? lhs.x < rhs.x : lhs.y < rhs.y
        }
        var unique: [SIMD2<Int>] = []
        for pair in found where unique.last != pair {
            unique.append(pair)
        }
        return unique
    }

    // MARK: - Interpolation

    /// Camera at fraction `t` between `a` and `b`: centre lerp, camera-to-world rotation slerp,
    /// intrinsics lerp, then divided by `resolutionDivisor` (size is `a`'s size divided, rounded down,
    /// at least 1).
    public static func interpolate(_ a: Camera, _ b: Camera, t: Float, resolutionDivisor: Int) -> Camera {
        let divisor: Int = max(1, resolutionDivisor)
        let divisorFloat: Float = Float(divisor)

        let quaternionA: SIMD4<Float> = quaternion(fromRotation: a.rotation.transposed)
        let quaternionB: SIMD4<Float> = quaternion(fromRotation: b.rotation.transposed)
        let blended: SIMD4<Float> = slerp(quaternionA, quaternionB, t: t)
        let rotation: Matrix3 = Matrix3.rotation(quaternion: blended).transposed

        let centerA: SIMD3<Float> = a.center
        let centerB: SIMD3<Float> = b.center
        var centre = SIMD3<Float>(0, 0, 0)
        for axis in 0..<3 {
            centre[axis] = lerp(centerA[axis], centerB[axis], t)
        }
        let translation: SIMD3<Float> = -(rotation * centre)

        let fx: Float = lerp(a.fx, b.fx, t) / divisorFloat
        let fy: Float = lerp(a.fy, b.fy, t) / divisorFloat
        let cx: Float = lerp(a.cx, b.cx, t) / divisorFloat
        let cy: Float = lerp(a.cy, b.cy, t) / divisorFloat
        let width: Int = max(1, a.width / divisor)
        let height: Int = max(1, a.height / divisor)

        return Camera(width: width, height: height,
                      fx: fx, fy: fy, cx: cx, cy: cy,
                      rotation: rotation, translation: translation)
    }

    // MARK: - Warping

    /// Forward-warps the metric depth of every given view into `camera` with a z-buffer. Each target
    /// pixel keeps the smallest camera-z depth that lands in it (ties: first in view and pixel order).
    /// Returns row-major depths, with 0 for pixels that receive no sample.
    public static func warpedDepth(into camera: Camera, from views: [TrainingView], near: Float) -> [Float] {
        let width: Int = camera.width
        let height: Int = camera.height
        let count: Int = max(0, width * height)
        var values = [Float](repeating: 0, count: count)
        var nearest = [Float](repeating: Float.infinity, count: count)
        let widthFloat: Float = Float(width)
        let heightFloat: Float = Float(height)

        for view in views {
            guard hasMetricDepth(view), let metric = view.depth?.metricDepth else { continue }
            let sourceWidth: Int = view.camera.width
            for y in 0..<view.camera.height {
                for x in 0..<sourceWidth {
                    let z: Float = metric[y * sourceWidth + x]
                    guard z.isFinite, z > 0 else { continue }
                    let world: SIMD3<Float> = DepthGeometry.backProject(x: x, y: y, depth: z, camera: view.camera)
                    guard let projected = DepthGeometry.project(world, camera: camera, near: near) else { continue }
                    guard projected.u >= 0, projected.u < widthFloat,
                          projected.v >= 0, projected.v < heightFloat else { continue }
                    let index: Int = Int(projected.v) * width + Int(projected.u)
                    if projected.z < nearest[index] {
                        nearest[index] = projected.z
                        values[index] = projected.z
                    }
                }
            }
        }
        return values
    }

    // MARK: - Generation

    /// Builds the pseudo-views for every pair. Each pair yields `settings.perPair` candidates at
    /// t = k / (perPair + 1); every candidate ends up either in `views` or in `rejected`.
    /// Checks run in order: optical-axis angle (per pair), coverage, then the near-depth test.
    public static func make(_ views: [TrainingView],
                            settings: PseudoViewSettings,
                            near: Float) -> (views: [PseudoView], rejected: [PseudoViewRejection]) {
        var accepted: [PseudoView] = []
        var rejected: [PseudoViewRejection] = []
        let candidateCount: Int = max(0, settings.perPair)
        let denominator: Float = Float(candidateCount + 1)

        for pair in pairs(views) {
            let a: TrainingView = views[pair.x]
            let b: TrainingView = views[pair.y]

            let axisA: SIMD3<Float> = opticalAxis(a.camera)
            let axisB: SIMD3<Float> = opticalAxis(b.camera)
            var dot: Double = 0
            for component in 0..<3 {
                dot += Double(axisA[component]) * Double(axisB[component])
            }
            let clampedDot: Double = min(1, max(-1, dot))
            let angleDegrees: Double = Double(acos(clampedDot)) * 180 / Double.pi
            let angleRejected: Bool = angleDegrees > Double(settings.maxAngleDegrees)

            let parentMedianA: Float = DepthPrior.median(of: a.depth?.metricDepth)
            let parentMedianB: Float = DepthPrior.median(of: b.depth?.metricDepth)
            let parentMedian: Float = min(parentMedianA, parentMedianB)

            for step in 0..<candidateCount {
                let t: Float = Float(step + 1) / denominator
                if angleRejected {
                    rejected.append(.wideAngle)
                    continue
                }

                let camera: Camera = interpolate(a.camera, b.camera, t: t, resolutionDivisor: settings.resolutionDivisor)
                let warped: [Float] = warpedDepth(into: camera, from: [a, b], near: near)
                let pixelCount: Int = camera.width * camera.height

                var validValues: [Float] = []
                validValues.reserveCapacity(warped.count)
                for value in warped where value.isFinite && value > 0 {
                    validValues.append(value)
                }

                let coverage: Float = Float(Double(validValues.count) / Double(max(1, pixelCount)))
                if coverage < settings.minimumCoverage {
                    rejected.append(.lowCoverage(coverage))
                    continue
                }
                guard !validValues.isEmpty else {
                    rejected.append(.tooClose)
                    continue
                }

                validValues.sort()
                let percentileIndex: Int = Int((0.05 * Double(validValues.count - 1)).rounded(.down))
                let fifthPercentile: Float = validValues[percentileIndex]
                let threshold: Double = Double(settings.minimumDepthFraction) * Double(parentMedian)
                if Double(fifthPercentile) < threshold {
                    rejected.append(.tooClose)
                    continue
                }

                let prior = DepthPrior(width: camera.width, height: camera.height,
                                       correlationTarget: warped, domain: .depth,
                                       metricDepth: warped, isMeasuredMetric: false)
                accepted.append(PseudoView(camera: camera, prior: prior, parents: pair, t: t))
            }
        }
        return (accepted, rejected)
    }

    // MARK: - Sampling

    /// Deterministic pick in `0..<count` for an iteration (SplatRandom stream 7). Returns 0 when
    /// `count` is not positive.
    public static func index(iteration: Int, count: Int, seed: UInt64) -> Int {
        guard count > 0 else { return 0 }
        let seed32: UInt32 = UInt32(truncatingIfNeeded: seed ^ (seed >> 32))
        let index32: UInt32 = UInt32(truncatingIfNeeded: iteration)
        let uniform: Float = SplatRandom.uniform(seed: seed32, index: index32, stream: 7)
        let scaled: Int = Int(Double(uniform) * Double(count))
        return min(max(scaled, 0), count - 1)
    }

    // MARK: - Helpers

    /// A view contributes metric depth when its prior carries `metricDepth` matching the camera size.
    private static func hasMetricDepth(_ view: TrainingView) -> Bool {
        guard let prior = view.depth, let metric = prior.metricDepth else { return false }
        return prior.width == view.camera.width
            && prior.height == view.camera.height
            && metric.count == view.camera.width * view.camera.height
    }

    /// Third row of the world-to-camera rotation, i.e. the optical axis in world space.
    private static func opticalAxis(_ camera: Camera) -> SIMD3<Float> {
        SIMD3<Float>(camera.rotation.c0.z, camera.rotation.c1.z, camera.rotation.c2.z)
    }

    private static func squaredDistance(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Double {
        var sum: Double = 0
        for component in 0..<3 {
            let difference: Double = Double(a[component]) - Double(b[component])
            sum += difference * difference
        }
        return sum
    }

    private static func lerp(_ a: Float, _ b: Float, _ t: Float) -> Float {
        Float(Double(a) + Double(t) * (Double(b) - Double(a)))
    }

    /// Element of a column-major matrix at (row, column).
    private static func element(_ m: Matrix3, _ row: Int, _ column: Int) -> Double {
        let vector: SIMD3<Float> = column == 0 ? m.c0 : (column == 1 ? m.c1 : m.c2)
        return Double(vector[row])
    }

    /// Unit quaternion (w, x, y, z) of a rotation matrix (Shepperd's method, computed in Double).
    private static func quaternion(fromRotation m: Matrix3) -> SIMD4<Float> {
        let r00: Double = element(m, 0, 0)
        let r01: Double = element(m, 0, 1)
        let r02: Double = element(m, 0, 2)
        let r10: Double = element(m, 1, 0)
        let r11: Double = element(m, 1, 1)
        let r12: Double = element(m, 1, 2)
        let r20: Double = element(m, 2, 0)
        let r21: Double = element(m, 2, 1)
        let r22: Double = element(m, 2, 2)
        let trace: Double = r00 + r11 + r22

        var w: Double = 0
        var x: Double = 0
        var y: Double = 0
        var z: Double = 0
        if trace > 0 {
            let s: Double = (trace + 1).squareRoot() * 2
            w = 0.25 * s
            x = (r21 - r12) / s
            y = (r02 - r20) / s
            z = (r10 - r01) / s
        } else if r00 > r11 && r00 > r22 {
            let s: Double = (1 + r00 - r11 - r22).squareRoot() * 2
            w = (r21 - r12) / s
            x = 0.25 * s
            y = (r01 + r10) / s
            z = (r02 + r20) / s
        } else if r11 > r22 {
            let s: Double = (1 + r11 - r00 - r22).squareRoot() * 2
            w = (r02 - r20) / s
            x = (r01 + r10) / s
            y = 0.25 * s
            z = (r12 + r21) / s
        } else {
            let s: Double = (1 + r22 - r00 - r11).squareRoot() * 2
            w = (r10 - r01) / s
            x = (r02 + r20) / s
            y = (r12 + r21) / s
            z = 0.25 * s
        }
        return SIMD4<Float>(Float(w), Float(x), Float(y), Float(z))
    }

    /// Shortest-path slerp between unit quaternions; lerp and normalize when nearly parallel.
    private static func slerp(_ start: SIMD4<Float>, _ end: SIMD4<Float>, t: Float) -> SIMD4<Float> {
        var a: [Double] = [Double(start.x), Double(start.y), Double(start.z), Double(start.w)]
        var b: [Double] = [Double(end.x), Double(end.y), Double(end.z), Double(end.w)]
        let fraction: Double = Double(t)

        var dot: Double = 0
        for component in 0..<4 {
            dot += a[component] * b[component]
        }
        if dot < 0 {
            for component in 0..<4 {
                b[component] = -b[component]
            }
            dot = -dot
        }

        var result: [Double] = [0, 0, 0, 0]
        if dot > 0.9995 {
            var norm: Double = 0
            for component in 0..<4 {
                let value: Double = a[component] + fraction * (b[component] - a[component])
                result[component] = value
                norm += value * value
            }
            let scale: Double = 1 / norm.squareRoot()
            for component in 0..<4 {
                result[component] *= scale
            }
        } else {
            let theta0: Double = acos(min(1, dot))
            let sinTheta0: Double = sin(theta0)
            let theta: Double = theta0 * fraction
            let weightA: Double = sin(theta0 - theta) / sinTheta0
            let weightB: Double = sin(theta) / sinTheta0
            for component in 0..<4 {
                result[component] = weightA * a[component] + weightB * b[component]
            }
        }
        return SIMD4<Float>(Float(result[0]), Float(result[1]), Float(result[2]), Float(result[3]))
    }
}
