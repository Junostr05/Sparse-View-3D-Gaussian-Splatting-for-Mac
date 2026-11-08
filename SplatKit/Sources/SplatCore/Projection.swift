import Foundation

/// Rasterization parameters shared by the CPU reference and the Metal kernels.
public struct RasterSettings: Sendable, Equatable {
    public var tileSize: Int
    public var nearPlane: Float
    public var farPlane: Float
    public var eps2D: Float
    public var shDegreeLimit: Int
    public var background: SIMD3<Float>
    /// Training-time anti-aliasing (Mip-Splatting). `.classic` (the default) leaves every path unchanged.
    public var mipFilter: MipFilter
    /// Experimental DropGaussian (training only): probability that a Gaussian is left out of this pass; the others
    /// get opacity × 1 / (1 − dropRate). 0 (the default) leaves every path unchanged. Metal path only.
    public var dropRate: Float = 0
    /// Selects the random subset of a pass; the same value must be used for its forward and backward pass.
    public var dropSeed: UInt32 = 0

    public init(tileSize: Int = 16,
                nearPlane: Float = 0.01,
                farPlane: Float = 1e10,
                eps2D: Float = 0.3,
                shDegreeLimit: Int = 3,
                background: SIMD3<Float> = .zero,
                mipFilter: MipFilter = .classic) {
        self.tileSize = tileSize
        self.nearPlane = nearPlane
        self.farPlane = farPlane
        self.eps2D = eps2D
        self.shDegreeLimit = shDegreeLimit
        self.background = background
        self.mipFilter = mipFilter
    }
}

/// One Gaussian after projection to screen space.
public struct ProjectedGaussian: Sendable, Equatable {
    public var mean2D: SIMD2<Float>
    public var depth: Float
    /// Inverse 2D covariance `(a, b, c)` such that `σ = 0.5·(a·dx² + c·dy²) + b·dx·dy`.
    public var conic: SIMD3<Float>
    public var color: SIMD3<Float>
    public var opacity: Float
    public var radius: Int32
    /// Inclusive minimum tile coordinate.
    public var tileMin: SIMD2<Int32>
    /// Exclusive maximum tile coordinate.
    public var tileMax: SIMD2<Int32>

    public init(mean2D: SIMD2<Float>,
                depth: Float,
                conic: SIMD3<Float>,
                color: SIMD3<Float>,
                opacity: Float,
                radius: Int32,
                tileMin: SIMD2<Int32>,
                tileMax: SIMD2<Int32>) {
        self.mean2D = mean2D
        self.depth = depth
        self.conic = conic
        self.color = color
        self.opacity = opacity
        self.radius = radius
        self.tileMin = tileMin
        self.tileMax = tileMax
    }
}

/// Projection of a single Gaussian (gsplat classic, non-antialiased forward pass).
public enum Projection {
    /// Projects Gaussian `index` of `cloud`. Returns `nil` if the Gaussian is culled.
    public static func project(_ cloud: GaussianCloud,
                               index: Int,
                               camera: Camera,
                               settings: RasterSettings) -> ProjectedGaussian? {
        // World → camera.
        let pos = SIMD3<Float>(cloud.positions[3 * index],
                               cloud.positions[3 * index + 1],
                               cloud.positions[3 * index + 2])
        let pCam = camera.rotation * pos + camera.translation
        let x = pCam.x, y = pCam.y, z = pCam.z

        guard z >= settings.nearPlane, z <= settings.farPlane else { return nil }

        // 3D covariance: Σ3D = M·Mᵀ with M = R_q · diag(scale).
        let rq = Matrix3.rotation(quaternion: SIMD4(cloud.rotations[4 * index],
                                                    cloud.rotations[4 * index + 1],
                                                    cloud.rotations[4 * index + 2],
                                                    cloud.rotations[4 * index + 3]))
        let mip = settings.mipFilter.isEnabled
        let scale: SIMD3<Float>
        var opacityFactor: Float = 1
        if mip {
            let rawScale = SIMD3<Double>(exp(Double(cloud.logScales[3 * index])),
                                         exp(Double(cloud.logScales[3 * index + 1])),
                                         exp(Double(cloud.logScales[3 * index + 2])))
            let smoothed = settings.mipFilter.smoothing(index: index, rawScale: rawScale)
            scale = SIMD3<Float>(Float(smoothed.scale.x), Float(smoothed.scale.y), Float(smoothed.scale.z))
            opacityFactor = Float(smoothed.opacityFactor)
        } else {
            scale = SIMD3<Float>(exp(cloud.logScales[3 * index]),
                                 exp(cloud.logScales[3 * index + 1]),
                                 exp(cloud.logScales[3 * index + 2]))
        }
        let m = Matrix3(c0: rq.c0 * scale.x, c1: rq.c1 * scale.y, c2: rq.c2 * scale.z)
        let sigma3D = m * m.transposed

        // Screen-space mean.
        let fx = camera.fx, fy = camera.fy, cx = camera.cx, cy = camera.cy
        let mean2D = SIMD2<Float>(fx * x / z + cx, fy * y / z + cy)

        // Jacobian with clamped tx/ty (limits are 1.3× the field of view).
        let width = Float(camera.width), height = Float(camera.height)
        let tanFovX = 0.5 * width / fx
        let tanFovY = 0.5 * height / fy
        let limXPos = (width - cx) / fx + 0.3 * tanFovX
        let limXNeg = cx / fx + 0.3 * tanFovX
        let limYPos = (height - cy) / fy + 0.3 * tanFovY
        let limYNeg = cy / fy + 0.3 * tanFovY
        let tx = z * min(max(x / z, -limXNeg), limXPos)
        let ty = z * min(max(y / z, -limYNeg), limYPos)
        let j0 = SIMD3<Float>(fx / z, 0, -fx * tx / (z * z))
        let j1 = SIMD3<Float>(0, fy / z, -fy * ty / (z * z))

        // T = J · R, where entry [i][j] = dot(J_i, column j of R).
        let rot = camera.rotation
        let t0 = SIMD3<Float>(simdDot(j0, rot.c0), simdDot(j0, rot.c1), simdDot(j0, rot.c2))
        let t1 = SIMD3<Float>(simdDot(j1, rot.c0), simdDot(j1, rot.c1), simdDot(j1, rot.c2))

        // Σ2D = T · Σ3D · Tᵀ, computed entry by entry.
        let sigmaT0 = sigma3D * t0
        let sigmaT1 = sigma3D * t1
        var a = simdDot(t0, sigmaT0)
        let b = simdDot(t0, sigmaT1)
        var c = simdDot(t1, sigmaT1)
        var compensation: Float = 1
        if mip {
            // Mip-Splatting 2D filter: Σ' = Σ + v2·I, opacity × sqrt(det Σ / det Σ') (det Σ clamped below).
            let v2 = settings.mipFilter.filter2DVariance
            let rawDet = max(a * c - b * b, 1e-12)
            a += v2
            c += v2
            let filteredDet = a * c - b * b
            guard filteredDet > 0 else { return nil }
            compensation = (rawDet / filteredDet).squareRoot()
        } else {
            a += settings.eps2D
            c += settings.eps2D
        }

        let det = a * c - b * b
        guard det > 0 else { return nil }
        let conic = SIMD3<Float>(c / det, -b / det, a / det)

        // Radius from the larger eigenvalue of Σ2D.
        let mid = 0.5 * (a + c)
        let lambda = mid + max(0.1, mid * mid - det).squareRoot()
        let radiusF = (3 * lambda.squareRoot()).rounded(.up)
        guard mean2D.x.isFinite, mean2D.y.isFinite, radiusF.isFinite, radiusF > 0 else {
            return nil
        }
        let radius = Int32(min(radiusF, Float(1 << 30)))

        // Tile rectangle. Truncation toward zero, clamped to the tile grid (in Float so
        // that huge values cannot overflow Int32).
        let tileSize = settings.tileSize
        let ts = Float(tileSize)
        let tilesX = Float((camera.width + tileSize - 1) / tileSize)
        let tilesY = Float((camera.height + tileSize - 1) / tileSize)
        let r = Float(radius)
        let minXf = min(max((mean2D.x - r) / ts, 0), tilesX)
        let maxXf = min(max((mean2D.x + r + (ts - 1)) / ts, 0), tilesX)
        let minYf = min(max((mean2D.y - r) / ts, 0), tilesY)
        let maxYf = min(max((mean2D.y + r + (ts - 1)) / ts, 0), tilesY)
        let tileMin = SIMD2<Int32>(Int32(minXf), Int32(minYf))
        let tileMax = SIMD2<Int32>(Int32(maxXf), Int32(maxYf))
        guard tileMin.x < tileMax.x, tileMin.y < tileMax.y else { return nil }

        // Color from SH, view direction from camera center to Gaussian.
        let center = camera.center
        let offset = pos - center
        let direction = offset / (offset * offset).sum().squareRoot()
        let degree = min(cloud.shDegree, settings.shDegreeLimit)
        let k = GaussianCloud.coefficientCount(degree: cloud.shDegree)
        let coefficients = cloud.sh[(index * k * 3)..<((index + 1) * k * 3)]
        let raw = SphericalHarmonics.evaluate(degree: degree,
                                              coefficients: coefficients,
                                              direction: direction)
        let shifted = raw + SIMD3<Float>(repeating: 0.5)
        let color = SIMD3<Float>(max(shifted.x, 0), max(shifted.y, 0), max(shifted.z, 0))

        var opacity = 1 / (1 + exp(-cloud.opacityLogits[index]))
        if mip {
            opacity *= opacityFactor * compensation
        }

        return ProjectedGaussian(
            mean2D: mean2D,
            depth: z,
            conic: conic,
            color: color,
            opacity: opacity,
            radius: radius,
            tileMin: tileMin,
            tileMax: tileMax
        )
    }
}

/// Dot product for SIMD3 vectors (the stdlib has no free `dot`).
@inline(__always)
func simdDot(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
    (a * b).sum()
}
