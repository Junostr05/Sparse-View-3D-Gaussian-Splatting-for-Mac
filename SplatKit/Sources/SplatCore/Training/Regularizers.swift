import Foundation

/// CPU geometry regularizers (Phase 4 P10).
public enum Regularizers {
    /// Anisotropy hinge `λ/n · Σ max(0, s_max/s_min − maxRatio)` with `s = exp(logScale)`.
    ///
    /// Adds the analytic gradient with respect to `logScale` into `gradients.logScales` and returns
    /// the loss value. For a Gaussian whose ratio `r` exceeds `maxRatio`, `g[a_max] += λ·r/n` and
    /// `g[a_min] −= λ·r/n`. `a_max` and `a_min` are the lowest axis indices on ties. Gaussians with
    /// `a_max == a_min` contribute nothing. Accumulation is in Double.
    @discardableResult
    public static func addAnisotropyGradient(_ cloud: GaussianCloud,
                                             weight: Float,
                                             maxRatio: Float,
                                             into gradients: inout GaussianGradients) -> Double {
        precondition(gradients.logScales.count == cloud.logScales.count,
                     "gradient buffer does not match the cloud")
        let n = cloud.count
        guard n > 0 else { return 0 }
        let scale: Double = Double(weight) / Double(n)
        let limit: Double = Double(maxRatio)
        var loss: Double = 0
        for i in 0..<n {
            let base = 3 * i
            var axisMax = 0
            var axisMin = 0
            var maxLog: Double = Double(cloud.logScales[base])
            var minLog: Double = maxLog
            for axis in 1..<3 {
                let value: Double = Double(cloud.logScales[base + axis])
                if value > maxLog {
                    maxLog = value
                    axisMax = axis
                }
                if value < minLog {
                    minLog = value
                    axisMin = axis
                }
            }
            if axisMax == axisMin {
                continue
            }
            let ratio: Double = exp(maxLog - minLog)
            guard ratio > limit else { continue }
            loss += scale * (ratio - limit)
            let slope: Float = Float(scale * ratio)
            gradients.logScales[base + axisMax] += slope
            gradients.logScales[base + axisMin] -= slope
        }
        return loss
    }

    /// Scale cap projection: `logScale_a = min(logScale_a, log(fraction · max(R, ‖μ − c‖)))` for every
    /// axis, where `R` and `c` are the extent radius and centre. No gradient is involved.
    /// A Gaussian whose bound is not positive is left unchanged.
    public static func applyScaleCap(_ cloud: inout GaussianCloud, fraction: Float, extent: SceneExtent) {
        let n = cloud.count
        var logScales = cloud.logScales
        let fractionValue: Double = Double(fraction)
        let radius: Double = Double(extent.radius)
        let centerX: Double = Double(extent.center.x)
        let centerY: Double = Double(extent.center.y)
        let centerZ: Double = Double(extent.center.z)
        for i in 0..<n {
            let base = 3 * i
            let dx: Double = Double(cloud.positions[base]) - centerX
            let dy: Double = Double(cloud.positions[base + 1]) - centerY
            let dz: Double = Double(cloud.positions[base + 2]) - centerZ
            let distance: Double = (dx * dx + dy * dy + dz * dz).squareRoot()
            let bound: Double = fractionValue * max(radius, distance)
            guard bound > 0 else { continue }
            let cap: Float = Float(log(bound))
            for axis in 0..<3 where logScales[base + axis] > cap {
                logScales[base + axis] = cap
            }
        }
        cloud = GaussianCloud(validated: cloud.shDegree,
                              positions: cloud.positions,
                              logScales: logScales,
                              rotations: cloud.rotations,
                              opacityLogits: cloud.opacityLogits,
                              sh: cloud.sh)
    }
}
