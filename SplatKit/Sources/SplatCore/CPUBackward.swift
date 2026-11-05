import Foundation

// CPU reference backward pass of the tile rasterizer.
//
// Structure:
//   1. forward bookkeeping: project + sort (shared with `render`), replay the pixel loop
//      and record every blended sample (`blendedSamples`);
//   2. pixel backward: accumulate per-Gaussian gradients of colour, opacity, conic, mean2D
//      by walking each pixel's samples back to front;
//   3. per-Gaussian chain rule (`GaussianChain`) in Double down to the stored parameters.
//
// Gradient conventions are those of the Phase 2 plan: gating decisions are constants,
// α = min(0.999, o·exp(−σ)) has zero gradient to o and σ when clamped, colour = max(SH+0.5, 0)
// has zero gradient where clamped, the Jacobian's tx/ty clamp blocks the gradient to x/z.

// MARK: - Forward replay

/// One Gaussian blended into a pixel.
struct BlendedSample {
    /// Index into the depth-sorted projected list.
    var entry: Int
    /// α actually used for blending (after the 0.999 clamp).
    var alpha: Float
    /// Transmittance before this sample was blended.
    var transmittance: Float
    /// True when `o·exp(−σ) > 0.999`, i.e. α is a constant.
    var clamped: Bool
    var sigma: Float
    /// `mean2D − pixelCenter`.
    var dx: Float
    var dy: Float
}

/// Front-to-back samples blended into one pixel. Samples that triggered early termination
/// are not included (they were never blended).
struct PixelSamples {
    var samples: [BlendedSample]
    var finalTransmittance: Float
}

extension CPURasterizer {
    /// Replays the forward pixel loop of `render`, recording the blended samples per pixel
    /// (row-major `width·height`).
    static func blendedSamples(_ projected: [(index: Int, gaussian: ProjectedGaussian)],
                               camera: Camera,
                               settings: RasterSettings) -> [PixelSamples] {
        let width = camera.width
        let height = camera.height
        let tileSize = settings.tileSize
        var result: [PixelSamples] = []
        result.reserveCapacity(width * height)

        for y in 0..<height {
            let ty = Int32(y / tileSize)
            for x in 0..<width {
                let tx = Int32(x / tileSize)
                let center = SIMD2<Float>(Float(x) + 0.5, Float(y) + 0.5)
                var transmittance: Float = 1
                var samples: [BlendedSample] = []

                for (entryIndex, entry) in projected.enumerated() {
                    let g = entry.gaussian
                    guard tx >= g.tileMin.x, tx < g.tileMax.x,
                          ty >= g.tileMin.y, ty < g.tileMax.y else { continue }

                    let d = g.mean2D - center
                    let sigma = 0.5 * (g.conic.x * d.x * d.x + g.conic.z * d.y * d.y)
                        + g.conic.y * d.x * d.y
                    guard sigma >= 0 else { continue }

                    let rawAlpha = g.opacity * exp(-sigma)
                    let alpha = min(0.999, rawAlpha)
                    guard alpha >= 1.0 / 255.0 else { continue }

                    let nextT = transmittance * (1 - alpha)
                    if nextT <= 1e-4 { break }

                    samples.append(BlendedSample(entry: entryIndex, alpha: alpha,
                                                 transmittance: transmittance,
                                                 clamped: rawAlpha > 0.999,
                                                 sigma: sigma, dx: d.x, dy: d.y))
                    transmittance = nextT
                }
                result.append(PixelSamples(samples: samples, finalTransmittance: transmittance))
            }
        }
        return result
    }
}

// MARK: - Backward

/// Gradients w.r.t. the per-Gaussian quantities produced by the pixel stage.
private struct ScreenSpaceGradient {
    var color = SIMD3<Double>.zero
    var opacity = 0.0
    var conic = SIMD3<Double>.zero
    var mean2D = SIMD2<Double>.zero
}

/// Result of a backward pass together with the screen-space gradients that densification
/// statistics are built from.
public struct ScreenSpaceBackward: Sendable {
    /// Gradients with respect to the stored Gaussian parameters.
    public var gradients: GaussianGradients
    /// `dL/dmean2D` in pixels, one entry per Gaussian of the input cloud (zero where not visible).
    public var mean2D: [SIMD2<Float>]
    /// True for Gaussians that were projected and whose tile rectangle is non-empty.
    public var visible: [Bool]
}

extension CPURasterizer {
    /// Gradients of `L = Σ imageGradient · image` with respect to every Gaussian parameter.
    ///
    /// - Parameter imageGradient: `dL/dpixel`, row-major `width·height`.
    public static func backward(_ cloud: GaussianCloud,
                                camera: Camera,
                                settings: RasterSettings,
                                imageGradient: [SIMD3<Float>]) -> GaussianGradients {
        backwardWithScreenSpace(cloud, camera: camera, settings: settings,
                                imageGradient: imageGradient).gradients
    }

    /// Same computation as `backward`, additionally returning `dL/dmean2D` and the visibility
    /// of every Gaussian. `gradients` is bit-identical to `backward`.
    public static func backwardWithScreenSpace(_ cloud: GaussianCloud,
                                               camera: Camera,
                                               settings: RasterSettings,
                                               imageGradient: [SIMD3<Float>]) -> ScreenSpaceBackward {
        precondition(imageGradient.count == camera.width * camera.height,
                     "imageGradient must hold width·height pixels")

        let sorted = projectedAndSorted(cloud, camera: camera, settings: settings)
        let pixels = blendedSamples(sorted, camera: camera, settings: settings)

        // Pixel stage.
        var screen = [ScreenSpaceGradient](repeating: ScreenSpaceGradient(), count: sorted.count)
        let background = SIMD3<Double>(settings.background)
        for (pixelIndex, pixel) in pixels.enumerated() where !pixel.samples.isEmpty {
            let g = SIMD3<Double>(imageGradient[pixelIndex])
            // S = Σ_{j>i} c_j α_j T_j + T_final · background, built back to front.
            var behind = Double(pixel.finalTransmittance) * background
            for sample in pixel.samples.reversed() {
                let gaussian = sorted[sample.entry].gaussian
                let color = SIMD3<Double>(gaussian.color)
                let alpha = Double(sample.alpha)
                let t = Double(sample.transmittance)

                screen[sample.entry].color += (alpha * t) * g

                let dAlpha = ((color * t - behind / (1 - alpha)) * g).sum()
                behind += color * (alpha * t)

                guard !sample.clamped else { continue }
                let e = exp(-Double(sample.sigma))
                let o = Double(gaussian.opacity)
                let dSigma = -o * e * dAlpha
                let dx = Double(sample.dx), dy = Double(sample.dy)
                let conic = SIMD3<Double>(gaussian.conic)

                screen[sample.entry].opacity += e * dAlpha
                screen[sample.entry].conic += dSigma * SIMD3(0.5 * dx * dx, dx * dy, 0.5 * dy * dy)
                screen[sample.entry].mean2D += dSigma * SIMD2(conic.x * dx + conic.y * dy,
                                                              conic.y * dx + conic.z * dy)
            }
        }

        // Per-Gaussian chain rule. Screen-space gradients are also reported per cloud index;
        // culled Gaussians are not visible and keep zero.
        var gradients = GaussianGradients(zeroLike: cloud)
        var mean2D = [SIMD2<Float>](repeating: .zero, count: cloud.count)
        var visible = [Bool](repeating: false, count: cloud.count)
        for (k, entry) in sorted.enumerated() {
            let chain = GaussianChain(cloud: cloud, index: entry.index, camera: camera,
                                      settings: settings)
            chain.propagate(screen[k], into: &gradients)
            mean2D[entry.index] = SIMD2(Float(screen[k].mean2D.x), Float(screen[k].mean2D.y))
            visible[entry.index] = true
        }
        return ScreenSpaceBackward(gradients: gradients, mean2D: mean2D, visible: visible)
    }
}

// MARK: - Small dense Double matrices (row-major, [row][col])

private typealias Mat = [[Double]]

private func matmul(_ a: Mat, _ b: Mat) -> Mat {
    let n = a.count, m = b[0].count, k = b.count
    var out = Mat(repeating: [Double](repeating: 0, count: m), count: n)
    for i in 0..<n { for j in 0..<m { for l in 0..<k { out[i][j] += a[i][l] * b[l][j] } } }
    return out
}

private func transpose(_ a: Mat) -> Mat {
    var out = Mat(repeating: [Double](repeating: 0, count: a.count), count: a[0].count)
    for i in 0..<a.count { for j in 0..<a[0].count { out[j][i] = a[i][j] } }
    return out
}

private func matrix(_ m: Matrix3) -> Mat {
    [[Double(m.c0.x), Double(m.c1.x), Double(m.c2.x)],
     [Double(m.c0.y), Double(m.c1.y), Double(m.c2.y)],
     [Double(m.c0.z), Double(m.c1.z), Double(m.c2.z)]]
}

// MARK: - SH basis with derivatives

/// SH basis values (gsplat convention, degrees 0…3) and their derivatives with respect to
/// the components of the (unit) direction, treating x, y, z as independent polynomial inputs.
private func shBasis(degree: Int, direction d: SIMD3<Double>)
    -> (value: [Double], dx: [Double], dy: [Double], dz: [Double]) {
    let count = (degree + 1) * (degree + 1)
    var v = [Double](repeating: 0, count: 16)
    var dx = v, dy = v, dz = v
    let x = d.x, y = d.y, z = d.z
    let xx = x * x, yy = y * y, zz = z * z

    let c0 = Double(SphericalHarmonics.c0)
    let c1 = Double(SphericalHarmonics.c1)
    let c2 = SphericalHarmonics.c2.map(Double.init)
    let c3 = SphericalHarmonics.c3.map(Double.init)

    v[0] = c0
    if degree >= 1 {
        v[1] = -c1 * y; dy[1] = -c1
        v[2] = c1 * z; dz[2] = c1
        v[3] = -c1 * x; dx[3] = -c1
    }
    if degree >= 2 {
        v[4] = c2[0] * x * y; dx[4] = c2[0] * y; dy[4] = c2[0] * x
        v[5] = c2[1] * y * z; dy[5] = c2[1] * z; dz[5] = c2[1] * y
        v[6] = c2[2] * (2 * zz - xx - yy)
        dx[6] = c2[2] * (-2 * x); dy[6] = c2[2] * (-2 * y); dz[6] = c2[2] * 4 * z
        v[7] = c2[3] * x * z; dx[7] = c2[3] * z; dz[7] = c2[3] * x
        v[8] = c2[4] * (xx - yy); dx[8] = c2[4] * 2 * x; dy[8] = c2[4] * (-2 * y)
    }
    if degree >= 3 {
        v[9] = c3[0] * y * (3 * xx - yy)
        dx[9] = c3[0] * 6 * x * y; dy[9] = c3[0] * (3 * xx - 3 * yy)
        v[10] = c3[1] * x * y * z
        dx[10] = c3[1] * y * z; dy[10] = c3[1] * x * z; dz[10] = c3[1] * x * y
        v[11] = c3[2] * y * (4 * zz - xx - yy)
        dx[11] = c3[2] * (-2 * x * y); dy[11] = c3[2] * (4 * zz - xx - 3 * yy)
        dz[11] = c3[2] * 8 * y * z
        v[12] = c3[3] * z * (2 * zz - 3 * xx - 3 * yy)
        dx[12] = c3[3] * (-6 * x * z); dy[12] = c3[3] * (-6 * y * z)
        dz[12] = c3[3] * (6 * zz - 3 * xx - 3 * yy)
        v[13] = c3[4] * x * (4 * zz - xx - yy)
        dx[13] = c3[4] * (4 * zz - 3 * xx - yy); dy[13] = c3[4] * (-2 * x * y)
        dz[13] = c3[4] * 8 * x * z
        v[14] = c3[5] * z * (xx - yy)
        dx[14] = c3[5] * 2 * x * z; dy[14] = c3[5] * (-2 * y * z); dz[14] = c3[5] * (xx - yy)
        v[15] = c3[6] * x * (xx - 3 * yy)
        dx[15] = c3[6] * (3 * xx - 3 * yy); dy[15] = c3[6] * (-6 * x * y)
    }
    return (Array(v[0..<count]), Array(dx[0..<count]), Array(dy[0..<count]), Array(dz[0..<count]))
}

// MARK: - Per-Gaussian projection with intermediates and its chain rule

/// Re-computes the projection of one Gaussian in Double, keeping every intermediate needed by
/// the chain rule, and propagates screen-space gradients to the stored parameters.
private struct GaussianChain {
    let cloud: GaussianCloud
    let index: Int
    let camera: Camera
    let settings: RasterSettings

    // Parameters.
    let position: SIMD3<Double>
    let rawQuaternion: SIMD4<Double>   // (w, x, y, z)
    let quaternionNorm: Double
    let unitQuaternion: SIMD4<Double>
    let scale: SIMD3<Double>
    let opacity: Double

    // Camera.
    let w: Mat                         // world → camera rotation
    let cameraCenter: SIMD3<Double>

    // Intermediates.
    let pCam: SIMD3<Double>
    let rq: Mat
    let m: Mat                         // R_q · diag(s)
    let sigma3D: Mat
    let jacobian: Mat                  // 2×3
    let xClamp: Double?                // clamp limit applied to x/z, nil if unclamped
    let yClamp: Double?
    let tx: Double
    let ty: Double
    let t: Mat                         // 2×3, J·W
    let conicMatrix: Mat               // Σ2D⁻¹

    init(cloud: GaussianCloud, index: Int, camera: Camera, settings: RasterSettings) {
        self.cloud = cloud
        self.index = index
        self.camera = camera
        self.settings = settings

        position = SIMD3(Double(cloud.positions[3 * index]),
                         Double(cloud.positions[3 * index + 1]),
                         Double(cloud.positions[3 * index + 2]))
        let q = SIMD4(Double(cloud.rotations[4 * index]), Double(cloud.rotations[4 * index + 1]),
                      Double(cloud.rotations[4 * index + 2]), Double(cloud.rotations[4 * index + 3]))
        rawQuaternion = q
        let norm = (q * q).sum().squareRoot()
        quaternionNorm = norm
        unitQuaternion = norm > 0 ? q / norm : SIMD4(1, 0, 0, 0)
        scale = SIMD3(exp(Double(cloud.logScales[3 * index])),
                      exp(Double(cloud.logScales[3 * index + 1])),
                      exp(Double(cloud.logScales[3 * index + 2])))
        opacity = 1 / (1 + exp(-Double(cloud.opacityLogits[index])))

        let wm = matrix(camera.rotation)
        w = wm
        let translation = SIMD3(Double(camera.translation.x), Double(camera.translation.y),
                                Double(camera.translation.z))
        let p = position
        let pc = SIMD3(wm[0][0] * p.x + wm[0][1] * p.y + wm[0][2] * p.z,
                       wm[1][0] * p.x + wm[1][1] * p.y + wm[1][2] * p.z,
                       wm[2][0] * p.x + wm[2][1] * p.y + wm[2][2] * p.z) + translation
        pCam = pc
        // C = −Wᵀ t
        cameraCenter = -SIMD3(wm[0][0] * translation.x + wm[1][0] * translation.y + wm[2][0] * translation.z,
                              wm[0][1] * translation.x + wm[1][1] * translation.y + wm[2][1] * translation.z,
                              wm[0][2] * translation.x + wm[1][2] * translation.y + wm[2][2] * translation.z)

        // Covariance.
        let r = GaussianChain.rotationMatrix(unitQuaternion)
        rq = r
        var mm = r
        for i in 0..<3 { for j in 0..<3 { mm[i][j] = r[i][j] * scale[j] } }
        m = mm
        sigma3D = matmul(mm, transpose(mm))

        // Jacobian with clamped tx, ty.
        let fx = Double(camera.fx), fy = Double(camera.fy)
        let cx = Double(camera.cx), cy = Double(camera.cy)
        let width = Double(camera.width), height = Double(camera.height)
        let tanFovX = 0.5 * width / fx, tanFovY = 0.5 * height / fy
        let limXPos = (width - cx) / fx + 0.3 * tanFovX
        let limXNeg = cx / fx + 0.3 * tanFovX
        let limYPos = (height - cy) / fy + 0.3 * tanFovY
        let limYNeg = cy / fy + 0.3 * tanFovY
        let z = pc.z
        let xz = pc.x / z, yz = pc.y / z
        var xc: Double?
        var yc: Double?
        if xz > limXPos { xc = limXPos } else if xz < -limXNeg { xc = -limXNeg }
        if yz > limYPos { yc = limYPos } else if yz < -limYNeg { yc = -limYNeg }
        xClamp = xc
        yClamp = yc
        let txv = z * (xc ?? xz)
        let tyv = z * (yc ?? yz)
        tx = txv
        ty = tyv
        let j = [[fx / z, 0, -fx * txv / (z * z)],
                 [0, fy / z, -fy * tyv / (z * z)]]
        jacobian = j
        let tm = matmul(j, wm)
        t = tm

        // Σ2D (+eps) and its inverse.
        var s2 = matmul(matmul(tm, sigma3D), transpose(tm))
        s2[0][0] += Double(settings.eps2D)
        s2[1][1] += Double(settings.eps2D)
        let det = s2[0][0] * s2[1][1] - s2[0][1] * s2[0][1]
        conicMatrix = [[s2[1][1] / det, -s2[0][1] / det],
                       [-s2[0][1] / det, s2[0][0] / det]]
    }

    /// Rotation matrix [row][col] of a unit quaternion (w, x, y, z); same entries as
    /// `Matrix3.rotation(quaternion:)`.
    static func rotationMatrix(_ q: SIMD4<Double>) -> Mat {
        let w = q.x, x = q.y, y = q.z, z = q.w
        return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    }

    func propagate(_ g: ScreenSpaceGradient, into out: inout GaussianGradients) {
        let fx = Double(camera.fx), fy = Double(camera.fy)

        // --- opacity: o = sigmoid(logit)
        out.opacityLogits[index] += Float(g.opacity * opacity * (1 - opacity))

        // --- conic → Σ2D:  dΣ = −C·G·C, G symmetric with off-diagonals g_y/2.
        let gm: Mat = [[g.conic.x, 0.5 * g.conic.y], [0.5 * g.conic.y, g.conic.z]]
        var g2 = matmul(matmul(conicMatrix, gm), conicMatrix)
        for i in 0..<2 { for j in 0..<2 { g2[i][j] = -g2[i][j] } }

        // --- Σ2D = T·Σ3D·Tᵀ
        let g3 = matmul(matmul(transpose(t), g2), t)                       // dL/dΣ3D
        var dT = matmul(matmul(g2, t), sigma3D)                            // (G2 + G2ᵀ) T Σ3D / 2 …
        for i in 0..<2 { for c in 0..<3 { dT[i][c] *= 2 } }                // … = 2·G2·T·Σ3D
        let dJ = matmul(dT, transpose(w))                                  // dL/dJ (2×3)

        // --- J and mean2D → camera-space mean
        let x = pCam.x, z = pCam.z
        let y = pCam.y
        var dpc = SIMD3<Double>.zero
        // J entries: J00 = fx/z, J02 = −fx·tx/z², J11 = fy/z, J12 = −fy·ty/z².
        dpc.z += -fx / (z * z) * dJ[0][0] - fy / (z * z) * dJ[1][1]
        dpc.z += 2 * fx * tx / (z * z * z) * dJ[0][2] + 2 * fy * ty / (z * z * z) * dJ[1][2]
        let dtx = -fx / (z * z) * dJ[0][2]
        let dty = -fy / (z * z) * dJ[1][2]
        // tx = z·clamp(x/z): unclamped → tx = x; clamped → tx = z·limit (no gradient to x).
        if let lim = xClamp { dpc.z += dtx * lim } else { dpc.x += dtx }
        if let lim = yClamp { dpc.z += dty * lim } else { dpc.y += dty }
        // mean2D = (fx·x/z + cx, fy·y/z + cy), unclamped.
        dpc.x += fx / z * g.mean2D.x
        dpc.y += fy / z * g.mean2D.y
        dpc.z += -fx * x / (z * z) * g.mean2D.x - fy * y / (z * z) * g.mean2D.y

        // --- camera → world: Wᵀ · dpc
        var dPosition = SIMD3(w[0][0] * dpc.x + w[1][0] * dpc.y + w[2][0] * dpc.z,
                              w[0][1] * dpc.x + w[1][1] * dpc.y + w[2][1] * dpc.z,
                              w[0][2] * dpc.x + w[1][2] * dpc.y + w[2][2] * dpc.z)

        // --- Σ3D = M·Mᵀ,  M = R_q·diag(s)
        var dM = matmul(g3, m)
        for i in 0..<3 { for j in 0..<3 { dM[i][j] *= 2 } }                // (G3 + G3ᵀ)·M
        var dRq = dM
        var dLogScale = SIMD3<Double>.zero
        for j in 0..<3 {
            var ds = 0.0
            for i in 0..<3 {
                ds += rq[i][j] * dM[i][j]
                dRq[i][j] = dM[i][j] * scale[j]
            }
            dLogScale[j] = scale[j] * ds
        }
        for j in 0..<3 { out.logScales[3 * index + j] += Float(dLogScale[j]) }

        // --- R_q(q̂) → q̂ → raw q
        if quaternionNorm > 0 {
            let q = unitQuaternion
            let w_ = q.x, qx = q.y, qy = q.z, qz = q.w
            let r = dRq
            let dqHat = SIMD4(
                2 * (qz * (r[1][0] - r[0][1]) + qy * (r[0][2] - r[2][0]) + qx * (r[2][1] - r[1][2])),
                2 * (qy * (r[1][0] + r[0][1]) + qz * (r[2][0] + r[0][2]) + w_ * (r[2][1] - r[1][2])
                     - 2 * qx * (r[1][1] + r[2][2])),
                2 * (qx * (r[1][0] + r[0][1]) + qz * (r[2][1] + r[1][2]) + w_ * (r[0][2] - r[2][0])
                     - 2 * qy * (r[0][0] + r[2][2])),
                2 * (w_ * (r[1][0] - r[0][1]) + qx * (r[2][0] + r[0][2]) + qy * (r[2][1] + r[1][2])
                     - 2 * qz * (r[0][0] + r[1][1])))
            let radial = (q * dqHat).sum()
            let dq = (dqHat - q * radial) / quaternionNorm
            for c in 0..<4 { out.rotations[4 * index + c] += Float(dq[c]) }
        }

        // --- colour: SH + 0.5, clamped at 0 per channel
        let degree = min(cloud.shDegree, settings.shDegreeLimit)
        let coefficientCount = GaussianCloud.coefficientCount(degree: cloud.shDegree)
        let used = (degree + 1) * (degree + 1)
        let offset = position - cameraCenter
        let length = (offset * offset).sum().squareRoot()
        if length > 0 {
            let direction = offset / length
            let basis = shBasis(degree: degree, direction: direction)
            let base = index * coefficientCount * 3
            var raw = SIMD3<Double>.zero
            for k in 0..<used {
                for c in 0..<3 { raw[c] += basis.value[k] * Double(cloud.sh[base + 3 * k + c]) }
            }
            var dColor = g.color
            for c in 0..<3 where raw[c] + 0.5 <= 0 { dColor[c] = 0 }

            var dDirection = SIMD3<Double>.zero
            for k in 0..<used {
                var coefficient = SIMD3<Double>.zero
                for c in 0..<3 {
                    out.sh[base + 3 * k + c] += Float(basis.value[k] * dColor[c])
                    coefficient[c] = Double(cloud.sh[base + 3 * k + c])
                }
                let projected = (coefficient * dColor).sum()
                dDirection += projected * SIMD3(basis.dx[k], basis.dy[k], basis.dz[k])
            }
            // direction = v/|v|, v = p − C
            dPosition += (dDirection - direction * (direction * dDirection).sum()) / length
        }

        for c in 0..<3 { out.positions[3 * index + c] += Float(dPosition[c]) }
    }
}
