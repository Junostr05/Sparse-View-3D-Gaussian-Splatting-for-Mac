import Foundation

/// Errors of the scene transform and crop box serialisation.
public enum SceneTransformError: Error, Equatable {
    /// The JSON does not decode, or holds values outside their valid range.
    case malformed(String)
    /// The file names a format version this build does not know.
    case unsupportedVersion(Int)
}

/// A similarity transform of a Gaussian scene: `p' = scale · R(rotation) · p + translation`.
///
/// Stored non-destructively in a project. The viewer shows the transformed scene, exports apply it,
/// and training and checkpoints never see it. `rotation` is a unit quaternion `(w, x, y, z)`, the
/// same convention as `GaussianCloud.rotations`.
public struct SceneTransform: Sendable, Equatable {
    /// Rotation as a unit quaternion `(w, x, y, z)`.
    public var rotation: SIMD4<Float>
    public var translation: SIMD3<Float>
    /// Uniform scale factor, strictly positive.
    public var scale: Float
    /// True once the scale came from a known real length (`calibrationScale`), so measurements are metric.
    public var isMetric: Bool

    public static let identity = SceneTransform(rotation: SIMD4<Float>(1, 0, 0, 0),
                                                translation: SIMD3<Float>(0, 0, 0),
                                                scale: 1,
                                                isMetric: false)

    public init(rotation: SIMD4<Float>, translation: SIMD3<Float>, scale: Float, isMetric: Bool = false) {
        self.rotation = rotation
        self.translation = translation
        self.scale = scale
        self.isMetric = isMetric
    }

    /// True when the transform changes nothing.
    public var isIdentity: Bool {
        rotation == SIMD4<Float>(1, 0, 0, 0) && translation == SIMD3<Float>(0, 0, 0) && scale == 1
    }

    /// The rotation as a matrix.
    public var rotationMatrix: Matrix3 {
        Matrix3.rotation(quaternion: rotation)
    }

    /// Maps one point: `scale · R · p + translation`.
    public func transformPoint(_ point: SIMD3<Float>) -> SIMD3<Float> {
        let rotated: SIMD3<Float> = rotationMatrix * point
        return rotated * scale + translation
    }

    /// `self ∘ inner`: applies `inner` first, then `self`.
    public func composing(_ inner: SceneTransform) -> SceneTransform {
        let combinedRotation: SIMD4<Float> = QuaternionMath.normalized(QuaternionMath.multiply(rotation, inner.rotation))
        let moved: SIMD3<Float> = rotationMatrix * inner.translation
        return SceneTransform(rotation: combinedRotation,
                              translation: moved * scale + translation,
                              scale: scale * inner.scale,
                              isMetric: isMetric || inner.isMetric)
    }

    /// The inverse similarity transform: `inverse.transformPoint(transformPoint(p)) == p`.
    public var inverse: SceneTransform {
        let inverseScale: Float = 1 / scale
        let transposed: Matrix3 = rotationMatrix.transposed
        let shift: SIMD3<Float> = transposed * translation * (-inverseScale)
        return SceneTransform(rotation: QuaternionMath.conjugate(rotation),
                              translation: shift,
                              scale: inverseScale,
                              isMetric: isMetric)
    }

    /// The same transform followed by a uniform scale `factor` about the origin.
    public func scaled(by factor: Float) -> SceneTransform {
        SceneTransform(rotation: rotation,
                       translation: translation * factor,
                       scale: scale * factor,
                       isMetric: isMetric)
    }

    /// Scale factor that makes the distance between two picked points `realLength`, or `nil` when the
    /// points coincide or the input is not finite and positive. Multiply the scene by it with `scaled(by:)`.
    public static func calibrationScale(pointA: SIMD3<Float>, pointB: SIMD3<Float>, realLength: Float) -> Float? {
        guard realLength.isFinite, realLength > 0 else { return nil }
        let difference: SIMD3<Float> = pointA - pointB
        let distance: Float = (difference * difference).sum().squareRoot()
        guard distance.isFinite, distance > 1e-12 else { return nil }
        let factor: Float = realLength / distance
        return factor.isFinite && factor > 0 ? factor : nil
    }

    /// Applies the transform to a cloud, returning a new cloud. Positions move, Gaussian rotations are composed
    /// with the rotation, log-scales gain `log(scale)`, and the view-dependent SH coefficients of degree 1 to 3 are
    /// rotated with the real SH rotation matrices, so the colour seen from the moved camera is the colour the
    /// original scene showed from the original camera. Opacity and the DC term are unchanged.
    public func apply(to cloud: GaussianCloud) throws(GaussianCloudError) -> GaussianCloud {
        if isIdentity {
            return cloud
        }
        let count: Int = cloud.count
        let degree: Int = cloud.shDegree
        let coefficients: Int = GaussianCloud.coefficientCount(degree: degree)
        let rotationMatrix: Matrix3 = self.rotationMatrix
        let logScaleShift: Float = log(scale)

        var positions: [Float] = [Float](repeating: 0, count: 3 * count)
        var logScales: [Float] = [Float](repeating: 0, count: 3 * count)
        var rotations: [Float] = [Float](repeating: 0, count: 4 * count)
        for index in 0..<count {
            let base: Int = 3 * index
            let point: SIMD3<Float> = SIMD3<Float>(cloud.positions[base], cloud.positions[base + 1], cloud.positions[base + 2])
            let moved: SIMD3<Float> = rotationMatrix * point * scale + translation
            positions[base] = moved.x
            positions[base + 1] = moved.y
            positions[base + 2] = moved.z
            logScales[base] = cloud.logScales[base] + logScaleShift
            logScales[base + 1] = cloud.logScales[base + 1] + logScaleShift
            logScales[base + 2] = cloud.logScales[base + 2] + logScaleShift
            let gaussianRotation: SIMD4<Float> = SIMD4<Float>(cloud.rotations[4 * index], cloud.rotations[4 * index + 1],
                                                              cloud.rotations[4 * index + 2], cloud.rotations[4 * index + 3])
            let composed: SIMD4<Float> = QuaternionMath.normalized(QuaternionMath.multiply(rotation, gaussianRotation))
            rotations[4 * index] = composed.x
            rotations[4 * index + 1] = composed.y
            rotations[4 * index + 2] = composed.z
            rotations[4 * index + 3] = composed.w
        }

        var sh: [Float] = cloud.sh
        if degree >= 1 {
            var bandMatrices: [[Float]] = []
            for band in 1...degree {
                bandMatrices.append(SphericalHarmonicRotation.bandMatrix(rotation: rotationMatrix, degree: degree, band: band))
            }
            SphericalHarmonicRotation.rotateCoefficients(&sh, count: count, coefficients: coefficients,
                                                         degree: degree, bandMatrices: bandMatrices)
        }

        return try GaussianCloud(shDegree: degree,
                                 positions: positions,
                                 logScales: logScales,
                                 rotations: rotations,
                                 opacityLogits: cloud.opacityLogits,
                                 sh: sh)
    }

    /// Plane fit through picked points on the ground. Returns the transform that moves the plane to height 0 in
    /// `upAxis` and the centroid of the points to the origin, with the plane normal turned to `upAxis.vector`.
    ///
    /// The normal is fitted by weighted least squares (principal axis of the covariance), with Tukey weights
    /// recomputed from the residuals a few times so that a stray pick hardly moves the plane. The normal keeps the
    /// side it had: it points the same way as `upAxis.vector` in the current frame, so an upright scene stays upright.
    /// The returned transform is in the frame of `points` (apply it on top of the current transform).
    ///
    /// - Returns: `nil` for fewer than 3 finite points, or points that do not span a plane.
    public static func fromFloor(points: [SIMD3<Float>], upAxis: UpAxis) -> SceneTransform? {
        guard points.count >= 3 else { return nil }
        for point in points where !(point.x.isFinite && point.y.isFinite && point.z.isFinite) {
            return nil
        }
        let up: SIMD3<Double> = doubleVector(upAxis.vector)
        let doubles: [SIMD3<Double>] = points.map { doubleVector($0) }
        // A robust start (the best three-point plane) keeps a stray pick from tilting the first fit.
        guard let start = bestTriplePlane(doubles) else { return nil }
        var normal: SIMD3<Double> = start.normal
        var anchor: SIMD3<Double> = start.anchor
        // The picks' spread sets a floor for the robust scale: a few clicks on the ground are never exact, and
        // a median of nearly exact residuals would otherwise reject genuine picks at the edge of the set.
        var spreadCentre: SIMD3<Double> = SIMD3<Double>(0, 0, 0)
        for point in doubles {
            spreadCentre += point
        }
        spreadCentre /= Double(doubles.count)
        var squaredSpread: Double = 0
        for point in doubles {
            let offset: SIMD3<Double> = point - spreadCentre
            squaredSpread += dotProduct(offset, offset)
        }
        let spread: Double = (squaredSpread / Double(doubles.count)).squareRoot()
        for _ in 0..<6 {
            // Tukey biweight weights from the residuals to the current plane, then a weighted principal-axis fit.
            let residuals: [Double] = doubles.map { abs(dotProduct(normal, $0 - anchor)) }
            let sorted: [Double] = residuals.sorted()
            let robustScale: Double = max(1.4826 * sorted[sorted.count / 2], 0.05 * spread, 1e-12)
            let cutoff: Double = 4.685 * robustScale
            var weights: [Double] = residuals.map { (residual: Double) -> Double in
                let ratio: Double = residual / cutoff
                return ratio < 1 ? (1 - ratio * ratio) * (1 - ratio * ratio) : 0
            }
            if weights.reduce(0, +) <= 0 {
                weights = [Double](repeating: 1, count: doubles.count)
            }
            // The anchor is the plain mean of the inliers (points with a non-zero weight), so the picks the
            // fit kept are centred on the origin.
            var inlierSum: SIMD3<Double> = SIMD3<Double>(0, 0, 0)
            var inlierCount: Double = 0
            for (index, point) in doubles.enumerated() where weights[index] > 0 {
                inlierSum += point
                inlierCount += 1
            }
            anchor = inlierSum / inlierCount
            let weightSum: Double = weights.reduce(0, +)
            var centroid: SIMD3<Double> = SIMD3<Double>(0, 0, 0)
            for (index, point) in doubles.enumerated() {
                centroid += weights[index] * point
            }
            centroid /= weightSum

            var covariance: [[Double]] = [[Double]](repeating: [Double](repeating: 0, count: 3), count: 3)
            for (index, point) in doubles.enumerated() {
                let offset: SIMD3<Double> = point - centroid
                for row in 0..<3 {
                    for column in 0..<3 {
                        covariance[row][column] += weights[index] * offset[row] * offset[column]
                    }
                }
            }
            guard let fitted = principalNormal(covariance) else { return nil }
            normal = fitted
        }

        // Turn the normal to the side of the current up axis, then rotate it onto the target up axis.
        if dotProduct(normal, up) < 0 {
            normal = -normal
        }
        let target: SIMD3<Float> = upAxis.vector
        let normalFloat: SIMD3<Float> = SIMD3<Float>(Float(normal.x), Float(normal.y), Float(normal.z))
        let rotationQuaternion: SIMD4<Float> = QuaternionMath.rotation(from: normalFloat, to: target)
        let rotationMatrix: Matrix3 = Matrix3.rotation(quaternion: rotationQuaternion)
        let centroidFloat: SIMD3<Float> = SIMD3<Float>(Float(anchor.x), Float(anchor.y), Float(anchor.z))
        let rotatedCentroid: SIMD3<Float> = rotationMatrix * centroidFloat
        return SceneTransform(rotation: rotationQuaternion,
                              translation: -rotatedCentroid,
                              scale: 1,
                              isMetric: false)
    }

    // MARK: Serialisation

    private struct File: Codable {
        var version: Int
        var rotation: [Float]
        var translation: [Float]
        var scale: Float
        var isMetric: Bool
    }

    /// The format version written by `jsonData()`.
    public static let formatVersion: Int = 1

    /// JSON for `edits/transform.json`.
    public func jsonData() throws -> Data {
        let file = File(version: Self.formatVersion,
                        rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
                        translation: [translation.x, translation.y, translation.z],
                        scale: scale,
                        isMetric: isMetric)
        let encoder: JSONEncoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(file)
    }

    /// Decodes `edits/transform.json`. Throws for an unknown version, wrong array sizes, non-finite values,
    /// a non-positive scale or a zero rotation quaternion.
    public static func decode(jsonData data: Data) throws(SceneTransformError) -> SceneTransform {
        let file: File
        do {
            file = try JSONDecoder().decode(File.self, from: data)
        } catch {
            throw .malformed("transform: \(error)")
        }
        guard file.version == formatVersion else {
            throw .unsupportedVersion(file.version)
        }
        guard file.rotation.count == 4, file.translation.count == 3 else {
            throw .malformed("transform: wrong number of components")
        }
        let values: [Float] = file.rotation + file.translation + [file.scale]
        guard values.allSatisfy({ $0.isFinite }) else {
            throw .malformed("transform: non-finite value")
        }
        guard file.scale > 0 else {
            throw .malformed("transform: scale must be positive")
        }
        let quaternion: SIMD4<Float> = SIMD4<Float>(file.rotation[0], file.rotation[1], file.rotation[2], file.rotation[3])
        guard (quaternion * quaternion).sum() > 0 else {
            throw .malformed("transform: zero rotation")
        }
        return SceneTransform(rotation: QuaternionMath.normalized(quaternion),
                              translation: SIMD3<Float>(file.translation[0], file.translation[1], file.translation[2]),
                              scale: file.scale,
                              isMetric: file.isMetric)
    }
}

/// A box in scene space, used for the crop: centre, half extents along its own axes, and orientation.
public struct CropBox: Sendable, Equatable {
    public var center: SIMD3<Float>
    /// Half extents along the box axes. Non-negative.
    public var halfExtents: SIMD3<Float>
    /// Orientation as a unit quaternion `(w, x, y, z)`.
    public var rotation: SIMD4<Float>

    public init(center: SIMD3<Float>, halfExtents: SIMD3<Float>, rotation: SIMD4<Float> = SIMD4<Float>(1, 0, 0, 0)) {
        self.center = center
        self.halfExtents = halfExtents
        self.rotation = QuaternionMath.normalized(rotation)
    }

    /// Rows of the world-to-box rotation: `local = (row0·d, row1·d, row2·d)` for an offset `d` from the centre.
    public var localRows: [SIMD3<Float>] {
        let matrix: Matrix3 = Matrix3.rotation(quaternion: rotation)
        return [matrix.c0, matrix.c1, matrix.c2]
    }

    /// True when `point` lies inside the box (boundary included). Non-finite points are outside.
    public func contains(_ point: SIMD3<Float>) -> Bool {
        contains(point, rows: localRows)
    }

    /// `contains` with the rows of `localRows` precomputed, for loops over many points.
    private func contains(_ point: SIMD3<Float>, rows: [SIMD3<Float>]) -> Bool {
        guard point.x.isFinite, point.y.isFinite, point.z.isFinite else { return false }
        let offset: SIMD3<Float> = point - center
        let local: SIMD3<Float> = SIMD3<Float>((rows[0] * offset).sum(), (rows[1] * offset).sum(), (rows[2] * offset).sum())
        return abs(local.x) <= halfExtents.x && abs(local.y) <= halfExtents.y && abs(local.z) <= halfExtents.z
    }

    /// Gaussians whose centre lies OUTSIDE the box: the ones a crop removes.
    public func mask(for cloud: GaussianCloud) -> SelectionMask {
        let count: Int = cloud.count
        let rows: [SIMD3<Float>] = localRows
        var outside: [Int] = []
        for index in 0..<count {
            let base: Int = 3 * index
            let point: SIMD3<Float> = SIMD3<Float>(cloud.positions[base], cloud.positions[base + 1], cloud.positions[base + 2])
            if !contains(point, rows: rows) {
                outside.append(index)
            }
        }
        return SelectionMask(count: count, indices: outside)
    }

    /// The box after `transform` is applied to the scene it lives in.
    public func transformed(by transform: SceneTransform) -> CropBox {
        CropBox(center: transform.transformPoint(center),
                halfExtents: halfExtents * transform.scale,
                rotation: QuaternionMath.multiply(transform.rotation, rotation))
    }

    // MARK: Serialisation

    private struct File: Codable {
        var version: Int
        var center: [Float]
        var halfExtents: [Float]
        var rotation: [Float]
    }

    /// The format version written by `jsonData()`.
    public static let formatVersion: Int = 1

    /// JSON for `edits/crop.json`.
    public func jsonData() throws -> Data {
        let file = File(version: Self.formatVersion,
                        center: [center.x, center.y, center.z],
                        halfExtents: [halfExtents.x, halfExtents.y, halfExtents.z],
                        rotation: [rotation.x, rotation.y, rotation.z, rotation.w])
        let encoder: JSONEncoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(file)
    }

    /// Decodes `edits/crop.json`. Throws for an unknown version, wrong sizes, non-finite values, negative
    /// half extents or a zero rotation.
    public static func decode(jsonData data: Data) throws(SceneTransformError) -> CropBox {
        let file: File
        do {
            file = try JSONDecoder().decode(File.self, from: data)
        } catch {
            throw .malformed("crop: \(error)")
        }
        guard file.version == formatVersion else {
            throw .unsupportedVersion(file.version)
        }
        guard file.center.count == 3, file.halfExtents.count == 3, file.rotation.count == 4 else {
            throw .malformed("crop: wrong number of components")
        }
        let values: [Float] = file.center + file.halfExtents + file.rotation
        guard values.allSatisfy({ $0.isFinite }) else {
            throw .malformed("crop: non-finite value")
        }
        guard file.halfExtents.allSatisfy({ $0 >= 0 }) else {
            throw .malformed("crop: negative half extent")
        }
        let quaternion: SIMD4<Float> = SIMD4<Float>(file.rotation[0], file.rotation[1], file.rotation[2], file.rotation[3])
        guard (quaternion * quaternion).sum() > 0 else {
            throw .malformed("crop: zero rotation")
        }
        return CropBox(center: SIMD3<Float>(file.center[0], file.center[1], file.center[2]),
                       halfExtents: SIMD3<Float>(file.halfExtents[0], file.halfExtents[1], file.halfExtents[2]),
                       rotation: quaternion)
    }
}

extension Camera {
    /// The world-to-camera pose that sees the transformed scene exactly as this camera sees the original one.
    /// The rotation stays orthonormal; the translation is scaled by the scene scale, so every camera-space point
    /// is multiplied by the same factor and the projection, the depth order and the visibility are unchanged.
    public func transformed(by transform: SceneTransform) -> Camera {
        let combined: Matrix3 = rotation * transform.rotationMatrix.transposed
        let shifted: SIMD3<Float> = translation * transform.scale - combined * transform.translation
        return Camera(width: width, height: height, fx: fx, fy: fy, cx: cx, cy: cy,
                      rotation: combined, translation: shifted)
    }
}

/// Widens a Float vector for the double-precision plane fit.
private func doubleVector(_ v: SIMD3<Float>) -> SIMD3<Double> {
    SIMD3<Double>(Double(v.x), Double(v.y), Double(v.z))
}

private func dotProduct(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double {
    a.x * b.x + a.y * b.y + a.z * b.z
}

/// Unit normal of the plane through the three points that fits the points best (median absolute residual),
/// with one point of that plane as anchor. Every triple for up to 12 points, a fixed sample of triples above that.
private func bestTriplePlane(_ points: [SIMD3<Double>]) -> (normal: SIMD3<Double>, anchor: SIMD3<Double>)? {
    let count: Int = points.count
    guard count >= 3 else { return nil }
    var triples: [(Int, Int, Int)] = []
    if count <= 12 {
        for i in 0..<count {
            for j in (i + 1)..<count {
                for k in (j + 1)..<count {
                    triples.append((i, j, k))
                }
            }
        }
    } else {
        for sample in 0..<300 {
            let i: Int = (sample * 7919) % count
            let j: Int = (sample * 104_729 + 1) % count
            let k: Int = (sample * 1_299_709 + 2) % count
            if i != j, j != k, i != k {
                triples.append((i, j, k))
            }
        }
    }
    var best: (normal: SIMD3<Double>, anchor: SIMD3<Double>)?
    var bestScore: Double = Double.infinity
    for (i, j, k) in triples {
        let edgeA: SIMD3<Double> = points[j] - points[i]
        let edgeB: SIMD3<Double> = points[k] - points[i]
        let cross: SIMD3<Double> = SIMD3<Double>(edgeA.y * edgeB.z - edgeA.z * edgeB.y,
                                                 edgeA.z * edgeB.x - edgeA.x * edgeB.z,
                                                 edgeA.x * edgeB.y - edgeA.y * edgeB.x)
        let length: Double = dotProduct(cross, cross).squareRoot()
        guard length > 1e-18 else { continue }
        let normal: SIMD3<Double> = cross / length
        let residuals: [Double] = points.map { abs(dotProduct(normal, $0 - points[i])) }
        let score: Double = residuals.sorted()[count / 2]
        if score < bestScore {
            bestScore = score
            best = (normal, points[i])
        }
    }
    return best
}

/// Unit eigenvector of the smallest eigenvalue of a covariance matrix. `nil` when the points do not span a plane:
/// the middle eigenvalue must be a real fraction of the largest one.
private func principalNormal(_ covariance: [[Double]]) -> SIMD3<Double>? {
    let eigen: (values: [Double], vectors: [[Double]]) = SymmetricEigen.decompose3(covariance)
    var order: [Int] = [0, 1, 2]
    order.sort { eigen.values[$0] < eigen.values[$1] }
    let middle: Double = eigen.values[order[1]]
    let largest: Double = eigen.values[order[2]]
    guard largest > 0, middle > 1e-9 * largest else { return nil }
    let column: Int = order[0]
    return SIMD3<Double>(eigen.vectors[0][column], eigen.vectors[1][column], eigen.vectors[2][column])
}

/// Quaternion helpers in the `(w, x, y, z)` convention, stored as `SIMD4<Float>(w, x, y, z)`.
enum QuaternionMath {
    static func multiply(_ a: SIMD4<Float>, _ b: SIMD4<Float>) -> SIMD4<Float> {
        let aw: Float = a.x, ax: Float = a.y, ay: Float = a.z, az: Float = a.w
        let bw: Float = b.x, bx: Float = b.y, by: Float = b.z, bz: Float = b.w
        let w: Float = aw * bw - ax * bx - ay * by - az * bz
        let x: Float = aw * bx + ax * bw + ay * bz - az * by
        let y: Float = aw * by - ax * bz + ay * bw + az * bx
        let z: Float = aw * bz + ax * by - ay * bx + az * bw
        return SIMD4<Float>(w, x, y, z)
    }

    static func conjugate(_ q: SIMD4<Float>) -> SIMD4<Float> {
        SIMD4<Float>(q.x, -q.y, -q.z, -q.w)
    }

    /// Unit quaternion; the identity for a zero quaternion.
    static func normalized(_ q: SIMD4<Float>) -> SIMD4<Float> {
        let length: Float = (q * q).sum().squareRoot()
        guard length > 0, length.isFinite else { return SIMD4<Float>(1, 0, 0, 0) }
        return q / length
    }

    /// Shortest rotation taking unit vector `from` to unit vector `to`, for `from · to ≥ 0`.
    static func rotation(from: SIMD3<Float>, to: SIMD3<Float>) -> SIMD4<Float> {
        let dot: Float = (from * to).sum()
        let cross: SIMD3<Float> = SIMD3<Float>(from.y * to.z - from.z * to.y,
                                               from.z * to.x - from.x * to.z,
                                               from.x * to.y - from.y * to.x)
        return normalized(SIMD4<Float>(1 + dot, cross.x, cross.y, cross.z))
    }
}

/// Real spherical-harmonic rotation. The coefficients of band `l` transform as `v' = G_l v`, where `G_l` is found
/// numerically: the basis of `SphericalHarmonics.evaluate` is fitted at sample directions `d` and `R d`, so the
/// matrix is exact for the renderer's own basis and needs no hand-derived Wigner matrices.
enum SphericalHarmonicRotation {
    /// Sample directions on the unit sphere (Fibonacci lattice), enough for bands up to degree 3.
    static let sampleCount: Int = 64

    /// Real SH basis values of all bands up to `degree`, in the order of the coefficient layout.
    static func basis(degree: Int, direction: SIMD3<Float>) -> [Float] {
        let count: Int = (degree + 1) * (degree + 1)
        var values: [Float] = [Float](repeating: 0, count: count)
        var coefficients: [Float] = [Float](repeating: 0, count: 3 * count)
        for index in 0..<count {
            coefficients[3 * index] = 1
            values[index] = SphericalHarmonics.evaluate(degree: degree,
                                                        coefficients: coefficients[0..<(3 * count)],
                                                        direction: direction).x
            coefficients[3 * index] = 0
        }
        return values
    }

    static func sampleDirections() -> [SIMD3<Float>] {
        let golden: Float = Float.pi * (3 - Float(5).squareRoot())
        var directions: [SIMD3<Float>] = []
        directions.reserveCapacity(sampleCount)
        for index in 0..<sampleCount {
            let z: Float = 1 - (2 * Float(index) + 1) / Float(sampleCount)
            let radius: Float = max(1 - z * z, 0).squareRoot()
            let phi: Float = golden * Float(index)
            directions.append(SIMD3<Float>(radius * cos(phi), radius * sin(phi), z))
        }
        return directions
    }

    /// The matrix `G_l` (row-major, `(2l+1)²` values) that maps band-`l` coefficients of one colour channel
    /// to the coefficients of the rotated scene.
    static func bandMatrix(rotation: Matrix3, degree: Int, band: Int) -> [Float] {
        let size: Int = 2 * band + 1
        let first: Int = band * band
        var normalMatrix: [[Double]] = [[Double]](repeating: [Double](repeating: 0, count: size), count: size)
        var crossMatrix: [[Double]] = [[Double]](repeating: [Double](repeating: 0, count: size), count: size)
        for direction in sampleDirections() {
            let original: [Float] = basis(degree: degree, direction: direction)
            let rotatedDirection: SIMD3<Float> = rotation * direction
            let transformed: [Float] = basis(degree: degree, direction: rotatedDirection)
            for row in 0..<size {
                let y: Double = Double(original[first + row])
                for column in 0..<size {
                    normalMatrix[row][column] += y * Double(original[first + column])
                    crossMatrix[row][column] += y * Double(transformed[first + column])
                }
            }
        }
        // Z = Y·Dᵀ in the least-squares sense, so M = Dᵀ = (YᵀY)⁻¹YᵀZ, and the coefficients transform by M⁻¹.
        guard let m: [[Double]] = LinearAlgebra.solve(normalMatrix, crossMatrix) else {
            return identity(size)
        }
        var identityColumns: [[Double]] = [[Double]](repeating: [Double](repeating: 0, count: size), count: size)
        for index in 0..<size {
            identityColumns[index][index] = 1
        }
        guard let inverse: [[Double]] = LinearAlgebra.solve(m, identityColumns) else {
            return identity(size)
        }
        var flat: [Float] = []
        flat.reserveCapacity(size * size)
        for row in 0..<size {
            for column in 0..<size {
                flat.append(Float(inverse[row][column]))
            }
        }
        return flat
    }

    private static func identity(_ size: Int) -> [Float] {
        var result: [Float] = [Float](repeating: 0, count: size * size)
        for index in 0..<size {
            result[index * size + index] = 1
        }
        return result
    }

    /// Rotates the band coefficients of every Gaussian in place. `bandMatrices[l - 1]` belongs to band `l`.
    static func rotateCoefficients(_ sh: inout [Float], count: Int, coefficients: Int, degree: Int,
                                   bandMatrices: [[Float]]) {
        for index in 0..<count {
            let gaussianBase: Int = index * coefficients * 3
            for band in 1...degree {
                let size: Int = 2 * band + 1
                let first: Int = band * band
                let matrix: [Float] = bandMatrices[band - 1]
                for channel in 0..<3 {
                    var input: [Float] = [Float](repeating: 0, count: size)
                    for row in 0..<size {
                        input[row] = sh[gaussianBase + (first + row) * 3 + channel]
                    }
                    for row in 0..<size {
                        var sum: Float = 0
                        for column in 0..<size {
                            sum += matrix[row * size + column] * input[column]
                        }
                        sh[gaussianBase + (first + row) * 3 + channel] = sum
                    }
                }
            }
        }
    }
}

/// Dense helpers for the small matrices of the SH rotation and the plane fit.
enum LinearAlgebra {
    /// Solves `A X = B` by Gauss-Jordan elimination with partial pivoting. `nil` when `A` is singular.
    static func solve(_ a: [[Double]], _ b: [[Double]]) -> [[Double]]? {
        let n: Int = a.count
        guard n > 0, b.count == n, let columns = b.first?.count else { return nil }
        var augmented: [[Double]] = []
        augmented.reserveCapacity(n)
        for row in 0..<n {
            augmented.append(a[row] + b[row])
        }
        let width: Int = n + columns
        for column in 0..<n {
            var pivotRow: Int = column
            for row in column..<n where abs(augmented[row][column]) > abs(augmented[pivotRow][column]) {
                pivotRow = row
            }
            let pivot: Double = augmented[pivotRow][column]
            guard abs(pivot) > 1e-14 else { return nil }
            augmented.swapAt(column, pivotRow)
            for entry in 0..<width {
                augmented[column][entry] /= pivot
            }
            for row in 0..<n where row != column {
                let factor: Double = augmented[row][column]
                if factor != 0 {
                    for entry in 0..<width {
                        augmented[row][entry] -= factor * augmented[column][entry]
                    }
                }
            }
        }
        return (0..<n).map { row -> [Double] in
            Array(augmented[row][n..<width])
        }
    }
}

/// Cyclic Jacobi eigen-decomposition of symmetric 3 × 3 matrices.
enum SymmetricEigen {
    /// Eigenvalues and the matching eigenvectors (columns of `vectors`).
    static func decompose3(_ input: [[Double]]) -> (values: [Double], vectors: [[Double]]) {
        var a: [[Double]] = input
        var v: [[Double]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for _ in 0..<100 {
            var p: Int = 0
            var q: Int = 1
            var largest: Double = 0
            for row in 0..<3 {
                for column in (row + 1)..<3 where abs(a[row][column]) > largest {
                    largest = abs(a[row][column])
                    p = row
                    q = column
                }
            }
            if largest < 1e-300 { break }
            let theta: Double = (a[q][q] - a[p][p]) / (2 * a[p][q])
            let sign: Double = theta < 0 ? -1 : 1
            let t: Double = sign / (abs(theta) + (theta * theta + 1).squareRoot())
            let c: Double = 1 / (t * t + 1).squareRoot()
            let s: Double = t * c
            for k in 0..<3 {
                let akp: Double = a[k][p]
                let akq: Double = a[k][q]
                a[k][p] = c * akp - s * akq
                a[k][q] = s * akp + c * akq
            }
            for k in 0..<3 {
                let apk: Double = a[p][k]
                let aqk: Double = a[q][k]
                a[p][k] = c * apk - s * aqk
                a[q][k] = s * apk + c * aqk
            }
            for k in 0..<3 {
                let vkp: Double = v[k][p]
                let vkq: Double = v[k][q]
                v[k][p] = c * vkp - s * vkq
                v[k][q] = s * vkp + c * vkq
            }
        }
        return ([a[0][0], a[1][1], a[2][2]], v)
    }
}
