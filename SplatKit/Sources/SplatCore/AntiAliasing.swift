import Foundation

// Training-time anti-aliasing after Mip-Splatting (Yu et al., CVPR 2024), implemented from the paper's math.
//
// Two filters, both controlled by `MipFilter`:
//   1. 2D filter: the projected covariance gets `v2·I` instead of the classic `eps2D·I`, and the opacity is
//      multiplied by the compensation c2 = sqrt(max(det Σ, 1e-12) / det(Σ + v2·I)).
//   2. 3D filter: every Gaussian's scales become s' = sqrt(s² + f²) with f² = v3 / ν², where ν is its sampling
//      rate (`SamplingRate`), and its opacity is multiplied by κ = Π sᵢ / s'ᵢ. The raw parameters stay what the
//      optimizer sees; the effective values are computed in the projection and in its backward pass.
//
// Classic rasterization (`isEnabled == false`, the default) does not touch any of this.

/// Anti-aliasing mode of one rasterization pass.
public struct MipFilter: Sendable, Equatable {
    /// True: Mip-Splatting filters (2D filter with compensation, 3D smoothing). False: classic rasterization.
    public var isEnabled: Bool
    /// v2: variance added to the projected 2D covariance in mip mode (Mip-Splatting: 0.1). It replaces the
    /// classic `eps2D` of `RasterSettings` while this filter is on.
    public var filter2DVariance: Float
    /// v3: variance of the 3D filter in squared pixels (Mip-Splatting: 0.2). The filter variance is v3 / ν².
    public var filter3DVariance: Float
    /// Per-Gaussian sampling rate ν (`SamplingRate.compute`), one entry per Gaussian. Read by the CPU reference
    /// only; the Metal path reads the sampling-rate buffer of its renderer and ignores this array. Must hold
    /// `count` entries when `isEnabled` is set for a CPU rasterization.
    public var samplingRates: [Float]

    public init(isEnabled: Bool = false,
                filter2DVariance: Float = 0.1,
                filter3DVariance: Float = 0.2,
                samplingRates: [Float] = []) {
        self.isEnabled = isEnabled
        self.filter2DVariance = filter2DVariance
        self.filter3DVariance = filter3DVariance
        self.samplingRates = samplingRates
    }

    /// Classic rasterization (no anti-aliasing).
    public static let classic = MipFilter()

    /// Smoothed scales s' and the opacity factor κ of Gaussian `index`, from its raw scales.
    /// Requires `isEnabled` and a sampling rate for `index`.
    func smoothing(index: Int, rawScale: SIMD3<Double>) -> (scale: SIMD3<Double>, opacityFactor: Double) {
        precondition(index < samplingRates.count, "mip filter needs one sampling rate per Gaussian")
        let nu = Double(samplingRates[index])
        let filterVariance = Double(filter3DVariance) / (nu * nu)
        let sx = (rawScale.x * rawScale.x + filterVariance).squareRoot()
        let sy = (rawScale.y * rawScale.y + filterVariance).squareRoot()
        let sz = (rawScale.z * rawScale.z + filterVariance).squareRoot()
        let factor = (rawScale.x / sx) * (rawScale.y / sy) * (rawScale.z / sz)
        return (SIMD3<Double>(sx, sy, sz), factor)
    }
}

/// Per-Gaussian sampling rate ν: the largest focal-length-over-depth ratio `fx / z` over the training cameras
/// that see the Gaussian. CPU reference of the Metal kernel `samplingRateCompute`.
public enum SamplingRate {
    /// A projected centre counts as inside a camera when it lies within the image enlarged by this fraction of
    /// its size on every side.
    public static let frustumMargin: Float = 0.15

    /// Sampling rate per Gaussian (`positions` holds `3·count` world-space floats).
    ///
    /// Gaussians that no camera sees (centre outside every enlarged image, or depth ≤ `near` in all cameras)
    /// get the mean ν over the visible Gaussians; when nothing is visible, 1. This is the documented fallback:
    /// an unseen Gaussian is not constrained by the training views, so only its scale matters.
    public static func compute(positions: [Float], cameras: [Camera], near: Float = 0.01) -> [Float] {
        precondition(positions.count % 3 == 0, "positions must hold 3 floats per Gaussian")
        let count = positions.count / 3
        var rates = [Float](repeating: 0, count: count)
        var sum: Double = 0
        var visibleCount = 0
        for i in 0..<count {
            let p = SIMD3<Float>(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2])
            var best: Float = 0
            for camera in cameras {
                let pc = camera.rotation * p + camera.translation
                let z = pc.z
                guard z > near else { continue }
                let meanX = camera.fx * pc.x / z + camera.cx
                let meanY = camera.fy * pc.y / z + camera.cy
                let width = Float(camera.width)
                let height = Float(camera.height)
                guard meanX >= -frustumMargin * width, meanX <= (1 + frustumMargin) * width,
                      meanY >= -frustumMargin * height, meanY <= (1 + frustumMargin) * height else { continue }
                best = max(best, camera.fx / z)
            }
            if best > 0 {
                rates[i] = best
                sum += Double(best)
                visibleCount += 1
            }
        }
        let fallback: Float = visibleCount > 0 ? Float(sum / Double(visibleCount)) : 1
        for i in 0..<count where rates[i] == 0 {
            rates[i] = fallback
        }
        return rates
    }
}

extension GaussianCloud {
    /// The 3D filter baked into the parameters, for export and viewing: logScale = log(s') and opacity logit =
    /// logit(o') with s' = sqrt(s² + f²) per axis and o' = o · Π sᵢ/s'ᵢ. `filterVariances[i]` is f² of
    /// Gaussian `i` (the mip filter's v3 / ν²).
    ///
    /// Known limitation: continuing training from a baked cloud applies the 3D filter a second time (a small
    /// drift). Nothing here un-bakes a cloud.
    public func bakingSmoothing3D(filterVariances: [Float]) -> GaussianCloud {
        precondition(filterVariances.count == count, "one filter variance per Gaussian")
        var baked = logScales
        var logits = opacityLogits
        for i in 0..<count {
            let f2 = Double(filterVariances[i])
            var factor: Double = 1
            for j in 0..<3 {
                let s = exp(Double(logScales[3 * i + j]))
                let smoothed = (s * s + f2).squareRoot()
                baked[3 * i + j] = Float(log(smoothed))
                factor *= s / smoothed
            }
            let opacity = 1 / (1 + exp(-Double(opacityLogits[i])))
            let bakedOpacity = min(max(opacity * factor, 1e-12), 1 - 1e-12)
            logits[i] = Float(log(bakedOpacity / (1 - bakedOpacity)))
        }
        return GaussianCloud(validated: shDegree, positions: positions, logScales: baked,
                             rotations: rotations, opacityLogits: logits, sh: sh)
    }
}
