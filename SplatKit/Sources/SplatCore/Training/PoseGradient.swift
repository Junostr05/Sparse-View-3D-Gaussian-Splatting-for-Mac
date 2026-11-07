import Foundation

// Joint camera pose refinement (v1.2): CPU reference for the pose gradient, the exponential map, folding and Adam.
//
// A training camera with world→camera pose (R, t), p_cam = R·p_w + t, is corrected by δ = (ω, τ):
//   R' = exp([ω]×)·R,   t' = exp([ω]×)·t + τ,   so   p_cam' = E·p_cam + τ with E = exp([ω]×).
// Gradients are taken at δ = 0. The correction is then folded into the stored camera, so δ is zero whenever a
// gradient is evaluated.
//
// Gradient at δ = 0 (checked against finite differences in PoseGradientTests):
// - Camera-space point: pc' = E·pc + τ, so dpc'/dτ = I and dpc'/dω_k = e_k × pc. With g_pc = dL/dpc (through
//   mean2D, the clamped Jacobian and the depth, all functions of pc):  dL/dτ += g_pc,  dL/dω += pc × g_pc.
// - Covariance: Σ_cam = R·Σ3·Rᵀ and S' = E·S·Eᵀ, so dS = [ω]×·S − S·[ω]×. With G = dL/dS = Jᵀ·g2·J,
//   dL = tr(G·dS) = tr([ω]×·M) with M = S·G − G·S (antisymmetric), i.e. dL/dω += 2·(M12, −M02, M01).
//   Only the covariance path goes through R; the Jacobian path is in g_pc.
// - View direction: the colour depends on dir = normalize(p_w − C) with C = −Rᵀt. The refined camera centre is
//   C' = −R'ᵀt' = C − Rᵀ·Eᵀ·τ, so at δ = 0: dC'/dτ = −Rᵀ and dC'/dω = 0. With u = dL/dp_w through the direction,
//   dL/dτ += R·u and there is no dL/dω term.

/// Gradient of a loss with respect to one camera's pose correction δ = (ω, τ), evaluated at δ = 0.
public struct PoseGradient: Sendable, Equatable {
    /// dL/dω, the axis-angle correction of the rotation (camera space).
    public var rotation: SIMD3<Double>
    /// dL/dτ, the translation correction (camera space).
    public var translation: SIMD3<Double>

    public static let zero = PoseGradient(rotation: .zero, translation: .zero)

    public init(rotation: SIMD3<Double>, translation: SIMD3<Double>) {
        self.rotation = rotation
        self.translation = translation
    }

    public static func += (lhs: inout PoseGradient, rhs: PoseGradient) {
        lhs.rotation += rhs.rotation
        lhs.translation += rhs.translation
    }
}

extension CPURasterizer {
    /// Gradient of `L = Σ imageGradient · image (+ Σ gD·Draw + gA·A)` with respect to the pose correction δ of
    /// `camera`, at δ = 0. The same chain as `backward`; the parameter gradients are not returned.
    public static func poseBackward(_ cloud: GaussianCloud,
                                    camera: Camera,
                                    settings: RasterSettings,
                                    imageGradient: [SIMD3<Float>],
                                    depthGradient: [SIMD2<Float>]? = nil) -> PoseGradient {
        backwardCore(cloud, camera: camera, settings: settings, imageGradient: imageGradient,
                     depthGradient: depthGradient, collectPose: true).pose
    }
}

/// Adam state of one camera pose: first and second moments of the six components (rotation, then translation).
public struct PoseAdamState: Sendable, Equatable {
    var firstMoment = [Double](repeating: 0, count: 6)
    var secondMoment = [Double](repeating: 0, count: 6)
    /// Applied updates so far (the bias-correction step counter).
    public internal(set) var step = 0

    public init() {}
}

/// Pose correction helpers: exponential map, folding into a `Camera`, and the Adam update of δ.
public enum PoseRefinement {
    /// Folds δ = (ω, τ) into `camera`: R ← exp([ω]×)·R and t ← exp([ω]×)·t + τ. The intrinsics and the image size
    /// are unchanged. Exact in double precision, rounded to `Float` for storage.
    public static func fold(_ camera: Camera, rotation omega: SIMD3<Double>, translation tau: SIMD3<Double>) -> Camera {
        let e = exponentialMap(omega)
        let rotation = e * DoubleMatrix3(camera.rotation)
        let t = SIMD3<Double>(Double(camera.translation.x), Double(camera.translation.y),
                              Double(camera.translation.z))
        let translation = e * t + tau
        return Camera(width: camera.width,
                      height: camera.height,
                      fx: camera.fx,
                      fy: camera.fy,
                      cx: camera.cx,
                      cy: camera.cy,
                      rotation: rotation.matrix3,
                      translation: SIMD3<Float>(Float(translation.x), Float(translation.y), Float(translation.z)))
    }

    /// One Adam update of a camera's pose. `gradient` is dL/dδ at δ = 0 (already averaged over the views that
    /// contributed); the returned δ is the correction to fold: `δ = −lr · m̂ / (√v̂ + ε)`, with the rotation
    /// components at `rotationRate` and the translation components at `translationRate`, as in `CPUAdam`.
    public static func adamStep(_ state: inout PoseAdamState,
                                gradient: PoseGradient,
                                rotationRate: Double,
                                translationRate: Double,
                                hyper: AdamHyperparameters) -> (rotation: SIMD3<Double>, translation: SIMD3<Double>) {
        state.step += 1
        let t = Double(state.step)
        let beta1 = Double(hyper.beta1)
        let beta2 = Double(hyper.beta2)
        let epsilon = Double(hyper.epsilon)
        let bias1 = 1 - pow(beta1, t)
        let bias2 = 1 - pow(beta2, t)
        let components: [Double] = [gradient.rotation.x, gradient.rotation.y, gradient.rotation.z,
                                    gradient.translation.x, gradient.translation.y, gradient.translation.z]
        var delta = [Double](repeating: 0, count: 6)
        for i in 0..<6 {
            let rate = i < 3 ? rotationRate : translationRate
            let g = components[i]
            let m = beta1 * state.firstMoment[i] + (1 - beta1) * g
            let v = beta2 * state.secondMoment[i] + (1 - beta2) * g * g
            state.firstMoment[i] = m
            state.secondMoment[i] = v
            let stepSize = rate / bias1
            let denominator = v.squareRoot() / bias2.squareRoot() + epsilon
            delta[i] = -stepSize * m / denominator
        }
        return (SIMD3(delta[0], delta[1], delta[2]), SIMD3(delta[3], delta[4], delta[5]))
    }

    /// exp([ω]×) by Rodrigues' formula, `I + (sin θ/θ)·K + ((1 − cos θ)/θ²)·K²` with θ = |ω| and K = [ω]×.
    static func exponentialMap(_ omega: SIMD3<Double>) -> DoubleMatrix3 {
        let theta2 = (omega * omega).sum()
        let a: Double
        let b: Double
        if theta2 < 1e-8 {
            // Taylor series, exact to the order that matters at this size.
            a = 1 - theta2 / 6
            b = 0.5 - theta2 / 24
        } else {
            let theta = theta2.squareRoot()
            a = sin(theta) / theta
            b = (1 - cos(theta)) / theta2
        }
        // K·v = ω × v; columns of K = [ω]× are (0, ω3, −ω2), (−ω3, 0, ω1), (ω2, −ω1, 0).
        let k = DoubleMatrix3(c0: SIMD3(0, omega.z, -omega.y),
                              c1: SIMD3(-omega.z, 0, omega.x),
                              c2: SIMD3(omega.y, -omega.x, 0))
        return DoubleMatrix3.identity + k * a + (k * k) * b
    }
}

/// Column-major 3×3 matrix of `Double`, used for the pose maths.
struct DoubleMatrix3: Equatable, Sendable {
    var c0: SIMD3<Double>
    var c1: SIMD3<Double>
    var c2: SIMD3<Double>

    static let identity = DoubleMatrix3(c0: SIMD3(1, 0, 0), c1: SIMD3(0, 1, 0), c2: SIMD3(0, 0, 1))

    init(c0: SIMD3<Double>, c1: SIMD3<Double>, c2: SIMD3<Double>) {
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
    }

    init(_ m: Matrix3) {
        c0 = SIMD3(Double(m.c0.x), Double(m.c0.y), Double(m.c0.z))
        c1 = SIMD3(Double(m.c1.x), Double(m.c1.y), Double(m.c1.z))
        c2 = SIMD3(Double(m.c2.x), Double(m.c2.y), Double(m.c2.z))
    }

    /// Rounded to `Float`.
    var matrix3: Matrix3 {
        Matrix3(c0: SIMD3(Float(c0.x), Float(c0.y), Float(c0.z)),
                c1: SIMD3(Float(c1.x), Float(c1.y), Float(c1.z)),
                c2: SIMD3(Float(c2.x), Float(c2.y), Float(c2.z)))
    }

    static func + (lhs: DoubleMatrix3, rhs: DoubleMatrix3) -> DoubleMatrix3 {
        DoubleMatrix3(c0: lhs.c0 + rhs.c0, c1: lhs.c1 + rhs.c1, c2: lhs.c2 + rhs.c2)
    }

    static func * (lhs: DoubleMatrix3, scalar: Double) -> DoubleMatrix3 {
        DoubleMatrix3(c0: lhs.c0 * scalar, c1: lhs.c1 * scalar, c2: lhs.c2 * scalar)
    }

    static func * (lhs: DoubleMatrix3, rhs: DoubleMatrix3) -> DoubleMatrix3 {
        DoubleMatrix3(c0: lhs * rhs.c0, c1: lhs * rhs.c1, c2: lhs * rhs.c2)
    }

    static func * (lhs: DoubleMatrix3, v: SIMD3<Double>) -> SIMD3<Double> {
        lhs.c0 * v.x + lhs.c1 * v.y + lhs.c2 * v.z
    }
}
