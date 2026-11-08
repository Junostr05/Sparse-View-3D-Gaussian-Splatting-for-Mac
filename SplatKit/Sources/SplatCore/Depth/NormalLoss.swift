import Foundation

/// Weights and gates of the surface-normal consistency loss (v1.5 normal prior).
public struct NormalLossWeights: Sendable, Equatable {
    /// λ_n, weight of the mean `1 − n·m`.
    public var weight: Float
    /// A stencil pixel with rendered opacity `A` at or below this is excluded (same gate as the depth loss).
    public var alphaMask: Float
    /// Fewer valid pixels than this skips the term (zero loss and gradient).
    public var minimumPixels: Int

    public init(weight: Float, alphaMask: Float, minimumPixels: Int) {
        self.weight = weight
        self.alphaMask = alphaMask
        self.minimumPixels = minimumPixels
    }
}

/// Output of one normal loss evaluation.
public struct NormalLossResult: Sendable {
    /// `λ_n · mean(1 − n·m)` over the valid pixels, 0 when degenerate.
    public var loss: Double
    /// Mean cosine `n·m` over the valid pixels, 0 when there are none.
    public var meanCosine: Double
    /// Pixels whose stencil and prior normal are valid.
    public var validPixels: Int
    /// True when there were fewer valid pixels than `minimumPixels` (no loss, no gradient).
    public var degenerate: Bool
    /// Per pixel `(∂L/∂Draw, ∂L/∂A)`, zero where the pixel is in no valid stencil.
    public var gradient: [SIMD2<Float>]

    public init(loss: Double, meanCosine: Double, validPixels: Int, degenerate: Bool, gradient: [SIMD2<Float>]) {
        self.loss = loss
        self.meanCosine = meanCosine
        self.validPixels = validPixels
        self.degenerate = degenerate
        self.gradient = gradient
    }
}

/// Pinhole intrinsics of the view the normal loss runs on, in pixels.
public struct NormalLossIntrinsics: Sendable, Equatable {
    public var fx: Float
    public var fy: Float
    public var cx: Float
    public var cy: Float

    public init(fx: Float, fy: Float, cx: Float, cy: Float) {
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    }

    public init(camera: Camera) {
        self.init(fx: camera.fx, fy: camera.fy, cx: camera.cx, cy: camera.cy)
    }
}

/// CPU reference of the v1.5 surface-normal consistency loss. Accumulates in `Double`.
///
/// The rendered depth `D = Draw / A` is back-projected to camera-frame points `P(x, y) = D · r(x, y)` with the
/// pixel-centre ray `r = ((x + 0.5 − cx) / fx, (y + 0.5 − cy) / fy, 1)` (OpenCV frame: x right, y down, z
/// forward). The rendered normal of pixel `p = (x, y)` comes from forward differences with its right neighbour
/// `r = (x + 1, y)` and lower neighbour `d = (x, y + 1)`:
///
///     tx = P_r − P_p,  ty = P_d − P_p,  c = ty × tx,  n = c / |c|
///
/// so `n` points towards the camera (a fronto-parallel plane gives `n = (0, 0, −1)`). The prior normal `m` is a
/// unit vector in the same frame and convention; a zero vector means "no prior here". A pixel is valid when its
/// prior normal is set, the stencil lies inside the image and all three stencil pixels have `A > alphaMask`
/// (and `A > 0`). The loss is `λ_n · (1/N) Σ (1 − n·m)` over the `N` valid pixels.
public enum NormalLoss {
    /// Stencils whose cross product `|c|` is at or below this are degenerate and skipped.
    static let minimumCrossLength: Double = 1e-12

    public static func evaluate(depth: DepthRender, normals: [SIMD3<Float>], width: Int, height: Int,
                                intrinsics: NormalLossIntrinsics, weights: NormalLossWeights) -> NormalLossResult {
        let draw: [Double] = depth.raw.map { Double($0) }
        let alpha: [Double] = depth.alpha.map { Double($0) }
        return evaluateDouble(draw: draw, alpha: alpha, normals: normals, width: width, height: height,
                              intrinsics: intrinsics, weights: weights)
    }

    /// Camera-frame ray of the centre of pixel `(x, y)`.
    static func ray(x: Int, y: Int, intrinsics: NormalLossIntrinsics) -> SIMD3<Double> {
        let u: Double = Double(x) + 0.5
        let v: Double = Double(y) + 0.5
        let rx: Double = (u - Double(intrinsics.cx)) / Double(intrinsics.fx)
        let ry: Double = (v - Double(intrinsics.cy)) / Double(intrinsics.fy)
        return SIMD3<Double>(rx, ry, 1)
    }

    static func cross(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> SIMD3<Double> {
        SIMD3<Double>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
    }

    static func dot(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    /// Unit normals (towards the camera) of a camera-z depth map, by the same forward-difference stencil as the
    /// loss. Pixels whose stencil leaves the image or has a non-positive depth get a zero vector.
    public static func normals(depth: [Float], width: Int, height: Int,
                               intrinsics: NormalLossIntrinsics) -> [SIMD3<Float>] {
        precondition(depth.count == width * height, "depth must match the size")
        var result: [SIMD3<Float>] = [SIMD3<Float>](repeating: .zero, count: depth.count)
        guard width > 1, height > 1 else { return result }
        for y in 0..<(height - 1) {
            for x in 0..<(width - 1) {
                let p: Int = y * width + x
                let dp: Double = Double(depth[p])
                let dr: Double = Double(depth[p + 1])
                let dd: Double = Double(depth[p + width])
                guard dp.isFinite, dr.isFinite, dd.isFinite, dp > 0, dr > 0, dd > 0 else { continue }
                let pointP: SIMD3<Double> = dp * ray(x: x, y: y, intrinsics: intrinsics)
                let pointR: SIMD3<Double> = dr * ray(x: x + 1, y: y, intrinsics: intrinsics)
                let pointD: SIMD3<Double> = dd * ray(x: x, y: y + 1, intrinsics: intrinsics)
                let c: SIMD3<Double> = cross(pointD - pointP, pointR - pointP)
                let length: Double = dot(c, c).squareRoot()
                guard length > minimumCrossLength else { continue }
                let unit: SIMD3<Double> = c / length
                result[p] = SIMD3<Float>(Float(unit.x), Float(unit.y), Float(unit.z))
            }
        }
        return result
    }

    /// Double-precision core, also used by the finite-difference tests.
    static func evaluateDouble(draw: [Double], alpha: [Double], normals: [SIMD3<Float>], width: Int, height: Int,
                               intrinsics: NormalLossIntrinsics, weights: NormalLossWeights) -> NormalLossResult {
        let count: Int = width * height
        precondition(draw.count == count && alpha.count == count && normals.count == count,
                     "draw, alpha and normals must match the size")
        let alphaMask: Double = Double(weights.alphaMask)

        func inMask(_ index: Int) -> Bool {
            let a: Double = alpha[index]
            return a > 0 && a > alphaMask
        }

        // Pass 1: valid stencils, their normals and the loss value.
        struct Stencil {
            var pixel: Int
            var x: Int
            var y: Int
            var cross: SIMD3<Double>
            var tx: SIMD3<Double>
            var ty: SIMD3<Double>
            var normal: SIMD3<Double>
            var prior: SIMD3<Double>
        }
        var stencils: [Stencil] = []
        var cosineSum: Double = 0
        if width > 1 && height > 1 {
            for y in 0..<(height - 1) {
                for x in 0..<(width - 1) {
                    let p: Int = y * width + x
                    let prior: SIMD3<Float> = normals[p]
                    let priorLength: Float = (prior * prior).sum()
                    guard priorLength > 0.25, prior.x.isFinite, prior.y.isFinite, prior.z.isFinite else { continue }
                    let r: Int = p + 1
                    let d: Int = p + width
                    guard inMask(p), inMask(r), inMask(d) else { continue }
                    let pointP: SIMD3<Double> = (draw[p] / alpha[p]) * ray(x: x, y: y, intrinsics: intrinsics)
                    let pointR: SIMD3<Double> = (draw[r] / alpha[r]) * ray(x: x + 1, y: y, intrinsics: intrinsics)
                    let pointD: SIMD3<Double> = (draw[d] / alpha[d]) * ray(x: x, y: y + 1, intrinsics: intrinsics)
                    let tx: SIMD3<Double> = pointR - pointP
                    let ty: SIMD3<Double> = pointD - pointP
                    let c: SIMD3<Double> = cross(ty, tx)
                    let length: Double = dot(c, c).squareRoot()
                    guard length > minimumCrossLength else { continue }
                    let n: SIMD3<Double> = c / length
                    let m: SIMD3<Double> = SIMD3<Double>(Double(prior.x), Double(prior.y), Double(prior.z))
                    cosineSum += dot(n, m)
                    stencils.append(Stencil(pixel: p, x: x, y: y, cross: c, tx: tx, ty: ty, normal: n, prior: m))
                }
            }
        }
        let validCount: Int = stencils.count
        var gradient: [SIMD2<Float>] = [SIMD2<Float>](repeating: .zero, count: count)
        let meanCosine: Double = validCount > 0 ? cosineSum / Double(validCount) : 0
        guard validCount > 0, validCount >= weights.minimumPixels else {
            return NormalLossResult(loss: 0, meanCosine: meanCosine, validPixels: validCount, degenerate: true,
                                    gradient: gradient)
        }
        let lambda: Double = Double(weights.weight)
        let nd: Double = Double(validCount)
        let loss: Double = lambda * (nd - cosineSum) / nd

        // Pass 2: chain rule. dL/dn = −λ m / N; dL/dc = (I − n nᵀ) dL/dn / |c|; c = ty × tx gives
        // dL/dty = tx × g and dL/dtx = g × ty; P = D·ray, D = Draw / A.
        var gradientD: [Double] = [Double](repeating: 0, count: count)
        for stencil in stencils {
            let gN: SIMD3<Double> = (-lambda / nd) * stencil.prior
            let length: Double = dot(stencil.cross, stencil.cross).squareRoot()
            let projected: SIMD3<Double> = gN - dot(stencil.normal, gN) * stencil.normal
            let gC: SIMD3<Double> = projected / length
            let gTy: SIMD3<Double> = cross(stencil.tx, gC)
            let gTx: SIMD3<Double> = cross(gC, stencil.ty)
            let p: Int = stencil.pixel
            let rayP: SIMD3<Double> = ray(x: stencil.x, y: stencil.y, intrinsics: intrinsics)
            let rayR: SIMD3<Double> = ray(x: stencil.x + 1, y: stencil.y, intrinsics: intrinsics)
            let rayD: SIMD3<Double> = ray(x: stencil.x, y: stencil.y + 1, intrinsics: intrinsics)
            gradientD[p] += dot(rayP, -(gTx + gTy))
            gradientD[p + 1] += dot(rayR, gTx)
            gradientD[p + width] += dot(rayD, gTy)
        }
        for index in 0..<count where gradientD[index] != 0 {
            let a: Double = alpha[index]
            let d: Double = draw[index] / a
            let g: Double = gradientD[index]
            gradient[index] = SIMD2<Float>(Float(g / a), Float(-g * d / a))
        }
        return NormalLossResult(loss: loss, meanCosine: meanCosine, validPixels: validCount, degenerate: false,
                                gradient: gradient)
    }
}
