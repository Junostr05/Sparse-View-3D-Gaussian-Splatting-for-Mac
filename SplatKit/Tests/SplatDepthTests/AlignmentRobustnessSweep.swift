import Foundation
import SplatCore
import SplatDepth
import Testing

/// Opt-in diagnostic (`SPLAT_ALIGNMENT_SWEEP=1`): how often the consensus alignment picks the visible surface
/// on synthetic scenes, against an oracle that only keeps SfM points the ground-truth depth confirms as visible.
@Suite(.enabled(if: ProcessInfo.processInfo.environment["SPLAT_ALIGNMENT_SWEEP"] == "1"))
struct AlignmentRobustnessSweep {

    /// Median relative depth error of the aligned map over ground-truth pixels (1 when nothing is aligned).
    private func onSurface(_ metric: [Float], truth: [Float]) -> Double {
        var errors: [Double] = []
        for index in truth.indices where truth[index] > 0 {
            errors.append(metric[index] > 0 ? abs(Double(metric[index] / truth[index]) - 1) : 1)
        }
        errors.sort()
        return errors.isEmpty ? 1 : errors[errors.count / 2]
    }

    @Test
    func printsSweep() {
        for views in [3, 6] {
            var currentFailures = 0
            var oracleFailures = 0
            var total = 0
            var currentSum = 0.0
            var oracleSum = 0.0
            var exactSum = 0.0
            var sampleSum = 0
            var visibleSum = 0
            for seed in UInt64(1)...6 {
                let scene = SyntheticScene.make(viewCount: views, novelCount: 1, width: 160, height: 120,
                                                gaussians: 1_200, sparsePoints: 200, seed: seed)
                for index in 0..<views {
                    let view = scene.train[index]
                    let truth: [Float] = scene.trainDepth[index].values
                    // Affine relative inverse depth as a depth model would give it (a, b arbitrary but fixed).
                    let map = scene.relativeInverseDepth(view: index, a: 1.7, b: 0.05)
                    let all = DepthAlignment.samples(points: scene.points.positions, camera: view.camera, map: map)

                    // Oracle: keep only samples whose camera z agrees with the ground-truth depth at that pixel.
                    var oracle: [(prior: Double, z: Double)] = []
                    var exact: [(prior: Double, z: Double)] = []
                    for point in scene.points.positions {
                        guard let projection = DepthGeometry.project(point, camera: view.camera, near: 0) else { continue }
                        let column = Int(projection.u)
                        let row = Int(projection.v)
                        guard column >= 0, column < view.camera.width, row >= 0, row < view.camera.height else { continue }
                        let pixel = row * view.camera.width + column
                        let surface = truth[pixel]
                        guard surface > 0, abs(Double(projection.z / surface) - 1) <= 0.05, map.isValid(pixel) else { continue }
                        oracle.append((prior: Double(map.values[pixel]), z: Double(projection.z)))
                        exact.append((prior: Double(map.values[pixel]), z: Double(surface)))
                    }
                    sampleSum += all.count
                    visibleSum += oracle.count

                    func quality(_ samples: [(prior: Double, z: Double)]) -> Double {
                        let result = DepthAlignment.fit(samples, kind: .relativeInverseDepth, scaleOnly: false,
                                                        minimumPoints: 8, minimumInlierFraction: 0.3, seed: 0)
                        guard case .success(let alignment) = result else { return 1 }
                        let metric = DepthAlignment.metricDepth(map, alignment: alignment, zMax: 1_000)
                        return onSurface(metric, truth: truth)
                    }
                    let current = quality(all)
                    let best = quality(oracle)
                    let perfect = quality(exact)
                    exactSum += perfect
                    total += 1
                    currentSum += current
                    oracleSum += best
                    if current > 0.1 { currentFailures += 1 }
                    if best > 0.1 { oracleFailures += 1 }
                }
            }
            print("ALIGNSWEEP \(views) views: \(total) view fits, current mean median-rel-err \(String(format: "%.3f", currentSum / Double(total))), fits with error >10 % \(currentFailures); oracle mean \(String(format: "%.3f", oracleSum / Double(total))), exact-z oracle \(String(format: "%.3f", exactSum / Double(total))), failures \(oracleFailures); mean samples \(sampleSum / total), mean truly visible \(visibleSum / total)")
        }
    }
}
