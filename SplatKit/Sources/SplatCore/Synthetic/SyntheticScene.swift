import Foundation

/// Synthetic ground-truth scene for the Phase 4 CPU tests (Phase 4 Task 11). Mirrors
/// `Tools/synthetic/make_colmap_scene.py`: a coloured sphere of radius 1 at the origin, a grey
/// ground disc on the plane y = -1.05, and cameras on a 110° elevated arc at radius 3.5 looking at
/// the origin, with fx = fy = 1.3·width.
///
/// Deliberate differences from the Python tool: the scene is a set of Gaussians rendered by
/// `CPURasterizer` (not discs), the background is black, and every Gaussian has the same
/// isotropic scale. Pure Swift and deterministic for a given seed.
public struct SyntheticScene: Sendable {
    /// Sphere Gaussians followed by ground Gaussians, degree 0.
    public var groundTruth: GaussianCloud
    /// Training views rendered from `groundTruth`, without depth.
    public var train: [TrainingView]
    /// Interpolated cameras between the training ones, rendered from `groundTruth`, held out for evaluation.
    public var novel: [TrainingView]
    /// Metric ground-truth depth of each training view (camera z, alpha > 0.5), row-major.
    public var trainDepth: [DepthMap]
    /// Seeded sample of the ground-truth Gaussian means with 0.01 Gaussian noise (SfM stand-in).
    public var points: PointCloud

    /// Isotropic linear scale of every ground-truth Gaussian (matches `SPLAT_K` of the Python tool).
    static let gaussianScale: Float = 0.12
    /// Ground Gaussians: `max(minimumGroundCount, sphere count / groundDivisor)`, as in the Python tool.
    static let groundDivisor = 8
    static let minimumGroundCount = 12
    static let groundY: Float = -1.05
    static let groundRadius: Double = 2.5
    static let groundColour = SIMD3<Float>(110.0 / 255, 104.0 / 255, 98.0 / 255)
    static let cameraDistance: Double = 3.5
    static let cameraElevation: Double = 15 * Double.pi / 180
    static let arcSpan: Double = 110 * Double.pi / 180
    static let focalScale: Float = 1.3
    static let shDCScale: Float = 0.28209479177387814

    /// Training cameras on the arc, novel cameras between them.
    /// - Parameters:
    ///   - viewCount: number of training views (≥ 1).
    ///   - novelCount: number of held-out novel views.
    ///   - gaussians: number of sphere Gaussians (the ground disc adds `max(12, gaussians / 8)`).
    ///   - sparsePoints: number of ground-truth Gaussian means returned as the SfM point cloud.
    public static func make(viewCount: Int, novelCount: Int, width: Int, height: Int,
                            gaussians: Int, sparsePoints: Int, seed: UInt64) -> SyntheticScene {
        var generator = SeededGenerator(seed: seed)
        let sphereCount: Int = max(1, gaussians)
        let groundCount: Int = max(minimumGroundCount, sphereCount / groundDivisor)

        var positions: [SIMD3<Float>] = []
        var colours: [SIMD3<Float>] = []
        positions.reserveCapacity(sphereCount + groundCount)
        colours.reserveCapacity(sphereCount + groundCount)
        for _ in 0..<sphereCount {
            let point: SIMD3<Float> = sphereSurfacePoint(generator: &generator)
            positions.append(point)
            colours.append(sphereColour(point))
        }
        for _ in 0..<groundCount {
            let radius: Double = groundRadius * Double.random(in: 0..<1, using: &generator).squareRoot()
            let angle: Double = Double.random(in: 0..<(2 * Double.pi), using: &generator)
            let position = SIMD3<Float>(Float(radius * cos(angle)), groundY, Float(radius * sin(angle)))
            positions.append(position)
            colours.append(groundColour)
        }
        let groundTruth: GaussianCloud = makeCloud(positions: positions, colours: colours)

        let intrinsics = Intrinsics(width: width, height: height)
        let trainCameras: [Camera] = trainingAzimuths(viewCount).map { camera(azimuth: $0, intrinsics: intrinsics) }
        let novelCameras: [Camera] = novelAzimuths(views: viewCount, count: novelCount)
            .map { camera(azimuth: $0, intrinsics: intrinsics) }

        let settings = RasterSettings(shDegreeLimit: 0, background: .zero)
        var train: [TrainingView] = []
        var trainDepth: [DepthMap] = []
        for (index, cam) in trainCameras.enumerated() {
            let rendered = CPURasterizer.renderWithDepth(groundTruth, camera: cam, settings: settings)
            train.append(TrainingView(name: "view_\(index)", camera: cam, image: rendered.image))
            let metric: [Float] = rendered.depth.normalized(minimumAlpha: 0.5)
            trainDepth.append(DepthMap(width: width, height: height, kind: .metric, values: metric))
        }

        var novel: [TrainingView] = []
        for (index, cam) in novelCameras.enumerated() {
            let image = CPURasterizer.render(groundTruth, camera: cam, settings: settings)
            novel.append(TrainingView(name: "novel_\(index)", camera: cam, image: image))
        }

        // SfM stand-in: seeded subset of the ground-truth means, plus 0.01 noise.
        let chosen: [Int] = GaussianInitializer.subsetIndices(total: positions.count,
                                                              maxCount: sparsePoints, seed: seed)
        var pointPositions: [SIMD3<Float>] = []
        var pointColours: [SIMD3<Float>] = []
        pointPositions.reserveCapacity(chosen.count)
        pointColours.reserveCapacity(chosen.count)
        for index in chosen {
            let noise = SIMD3<Float>(gaussian(&generator), gaussian(&generator), gaussian(&generator)) * 0.01
            pointPositions.append(positions[index] + noise)
            pointColours.append(colours[index])
        }

        return SyntheticScene(groundTruth: groundTruth,
                              train: train,
                              novel: novel,
                              trainDepth: trainDepth,
                              points: PointCloud(positions: pointPositions, colors: pointColours))
    }

    /// Affine relative inverse depth `a / z + b` of a training view; 0 where the ground truth is invalid.
    public func relativeInverseDepth(view: Int, a: Float, b: Float) -> DepthMap {
        let metric: DepthMap = trainDepth[view]
        var values = [Float](repeating: 0, count: metric.values.count)
        for index in metric.values.indices where metric.values[index] > 0 {
            values[index] = a / metric.values[index] + b
        }
        return DepthMap(width: metric.width, height: metric.height, kind: .relativeInverseDepth,
                        values: values, isEstimated: true)
    }

    /// The training views with their camera, image and points, ready for `CPUTrainer`.
    public var dataset: TrainingDataset {
        TrainingDataset(views: train, points: points)
    }

    // MARK: - Helpers

    private struct Intrinsics {
        let width: Int
        let height: Int
        let fx: Float
        let fy: Float
        let cx: Float
        let cy: Float

        init(width: Int, height: Int) {
            self.width = width
            self.height = height
            fx = SyntheticScene.focalScale * Float(width)
            fy = fx
            cx = Float(width) / 2
            cy = Float(height) / 2
        }
    }

    /// Camera on the elevated arc at `azimuth`, looking at the origin with +Y up (`look_at` of the Python tool).
    private static func camera(azimuth: Double, intrinsics: Intrinsics) -> Camera {
        let horizontal: Double = cameraDistance * cos(cameraElevation)
        let centre = SIMD3<Float>(Float(horizontal * sin(azimuth)),
                                  Float(cameraDistance * sin(cameraElevation)),
                                  Float(horizontal * cos(azimuth)))
        let forward: SIMD3<Float> = normalized(-centre)
        let upNegative = SIMD3<Float>(0, -1, 0)
        let downRaw: SIMD3<Float> = upNegative - forward * dot(upNegative, forward)
        let down: SIMD3<Float> = normalized(downRaw)
        let right: SIMD3<Float> = cross(down, forward)
        let rotation = Matrix3(rows: [right, down, forward])
        let translation = SIMD3<Float>(-dot(right, centre), -dot(down, centre), -dot(forward, centre))
        return Camera(width: intrinsics.width, height: intrinsics.height,
                      fx: intrinsics.fx, fy: intrinsics.fy,
                      cx: intrinsics.cx, cy: intrinsics.cy,
                      rotation: rotation, translation: translation)
    }

    /// Azimuths of `count` training cameras spread over the arc (a single camera sits at 0).
    private static func trainingAzimuths(_ count: Int) -> [Double] {
        if count <= 1 {
            return [0]
        }
        return (0..<count).map { index in
            -arcSpan / 2 + Double(index) * arcSpan / Double(count - 1)
        }
    }

    /// Midpoints of consecutive training azimuths first, then quarter points; repeats when `count` is larger.
    private static func novelAzimuths(views: Int, count: Int) -> [Double] {
        let training: [Double] = trainingAzimuths(views)
        var gaps: [(Double, Double)] = []
        for index in 0..<max(0, training.count - 1) {
            gaps.append((training[index], training[index + 1]))
        }
        var candidates: [Double] = gaps.map { ($0.0 + $0.1) / 2 }
        candidates += gaps.map { $0.0 + ($0.1 - $0.0) / 4 }
        candidates += gaps.map { $0.0 + 3 * ($0.1 - $0.0) / 4 }
        guard !candidates.isEmpty else {
            return [Double](repeating: 0, count: max(0, count))
        }
        return (0..<max(0, count)).map { candidates[$0 % candidates.count] }
    }

    /// Point uniform on the unit sphere (normalised Gaussian direction).
    private static func sphereSurfacePoint(generator: inout SeededGenerator) -> SIMD3<Float> {
        while true {
            let x: Double = gaussianDouble(&generator)
            let y: Double = gaussianDouble(&generator)
            let z: Double = gaussianDouble(&generator)
            let length: Double = (x * x + y * y + z * z).squareRoot()
            if length < 1e-6 {
                continue
            }
            return SIMD3<Float>(Float(x / length), Float(y / length), Float(z / length))
        }
    }

    /// Hue follows longitude, brightness follows latitude (as the Python tool).
    private static func sphereColour(_ point: SIMD3<Float>) -> SIMD3<Float> {
        let turn: Double = atan2(Double(point.z), Double(point.x)) / (2 * Double.pi)
        let hue: Double = turn - floor(turn)
        let value: Double = 0.75 + 0.25 * (Double(point.y) + 1) / 2
        return hsvToRGB(hue: hue, saturation: 0.85, value: value)
    }

    private static func hsvToRGB(hue: Double, saturation: Double, value: Double) -> SIMD3<Float> {
        let scaled: Double = hue * 6
        let sectorValue: Double = floor(scaled)
        let fraction: Double = scaled - sectorValue
        let p: Double = value * (1 - saturation)
        let q: Double = value * (1 - saturation * fraction)
        let t: Double = value * (1 - saturation * (1 - fraction))
        let sector: Int = Int(sectorValue) % 6
        var r: Double = value
        var g: Double = p
        var b: Double = q
        switch sector {
        case 0:
            r = value; g = t; b = p
        case 1:
            r = q; g = value; b = p
        case 2:
            r = p; g = value; b = t
        case 3:
            r = p; g = q; b = value
        case 4:
            r = t; g = p; b = value
        default:
            r = value; g = p; b = q
        }
        return SIMD3<Float>(Float(r), Float(g), Float(b))
    }

    /// Degree-0 Gaussians with the colour stored as the DC term, so that `0.5 + c0·dc` is the colour.
    private static func makeCloud(positions: [SIMD3<Float>], colours: [SIMD3<Float>]) -> GaussianCloud {
        let n: Int = positions.count
        var flat: [Float] = []
        var logScales: [Float] = []
        var rotations: [Float] = []
        var logits: [Float] = []
        var sh: [Float] = []
        flat.reserveCapacity(3 * n)
        logScales.reserveCapacity(3 * n)
        rotations.reserveCapacity(4 * n)
        logits.reserveCapacity(n)
        sh.reserveCapacity(3 * n)
        let logScale: Float = log(gaussianScale)
        let opacityLogit: Float = log(9)   // opacity 0.9
        for index in 0..<n {
            let p: SIMD3<Float> = positions[index]
            flat.append(p.x)
            flat.append(p.y)
            flat.append(p.z)
            logScales.append(logScale)
            logScales.append(logScale)
            logScales.append(logScale)
            rotations.append(1)
            rotations.append(0)
            rotations.append(0)
            rotations.append(0)
            logits.append(opacityLogit)
            let c: SIMD3<Float> = colours[index]
            sh.append((c.x - 0.5) / shDCScale)
            sh.append((c.y - 0.5) / shDCScale)
            sh.append((c.z - 0.5) / shDCScale)
        }
        return GaussianCloud(validated: 0, positions: flat, logScales: logScales,
                             rotations: rotations, opacityLogits: logits, sh: sh)
    }

    private static func normalized(_ v: SIMD3<Float>) -> SIMD3<Float> {
        let length: Float = (v.x * v.x + v.y * v.y + v.z * v.z).squareRoot()
        return v / length
    }

    private static func dot(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    private static func cross(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3<Float>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
    }

    /// Standard normal sample, Box-Muller over the seeded generator.
    private static func gaussianDouble(_ generator: inout SeededGenerator) -> Double {
        let u1: Double = max(Double.random(in: 0..<1, using: &generator), 1e-12)
        let u2: Double = Double.random(in: 0..<1, using: &generator)
        return (-2 * log(u1)).squareRoot() * cos(2 * Double.pi * u2)
    }

    private static func gaussian(_ generator: inout SeededGenerator) -> Float {
        Float(gaussianDouble(&generator))
    }
}
