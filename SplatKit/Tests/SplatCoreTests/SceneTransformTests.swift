import Foundation
import Testing
import SplatCore

/// Deterministic pseudo-random numbers in 0..<1, so every run sees the same scene.
private struct TestRandom {
    var state: UInt64

    mutating func next() -> Float {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Float(state >> 40) / Float(1 << 24)
    }

    mutating func vector(scale: Float = 1) -> SIMD3<Float> {
        let x: Float = next() * 2 - 1
        let y: Float = next() * 2 - 1
        let z: Float = next() * 2 - 1
        return SIMD3<Float>(x, y, z) * scale
    }
}

/// Quaternion `(w, x, y, z)` for a rotation of `angle` radians about `axis`.
private func axisAngle(_ axis: SIMD3<Float>, _ angle: Float) -> SIMD4<Float> {
    let unit: SIMD3<Float> = axis / (axis * axis).sum().squareRoot()
    let half: Float = angle / 2
    let s: Float = sin(half)
    return SIMD4<Float>(cos(half), unit.x * s, unit.y * s, unit.z * s)
}

private func randomCloud(count: Int, degree: Int, seed: UInt64) throws -> GaussianCloud {
    var random = TestRandom(state: seed)
    var positions: [Float] = []
    var logScales: [Float] = []
    var rotations: [Float] = []
    var opacity: [Float] = []
    var sh: [Float] = []
    let coefficients: Int = GaussianCloud.coefficientCount(degree: degree)
    for _ in 0..<count {
        let p: SIMD3<Float> = random.vector(scale: 3)
        positions += [p.x, p.y, p.z]
        let s: SIMD3<Float> = SIMD3<Float>(repeating: -1) + random.vector(scale: 0.5)
        logScales += [s.x, s.y, s.z]
        let q: SIMD4<Float> = SIMD4<Float>(random.next() + 0.1, random.vector().x, random.vector().y, random.vector().z)
        rotations += [q.x, q.y, q.z, q.w]
        opacity.append(random.next() * 4 - 2)
        for _ in 0..<(coefficients * 3) {
            sh.append(random.next() * 2 - 1)
        }
    }
    return try GaussianCloud(shDegree: degree, positions: positions, logScales: logScales,
                             rotations: rotations, opacityLogits: opacity, sh: sh)
}

private func point(_ cloud: GaussianCloud, _ index: Int) -> SIMD3<Float> {
    SIMD3<Float>(cloud.positions[3 * index], cloud.positions[3 * index + 1], cloud.positions[3 * index + 2])
}

private func sampleTransform() -> SceneTransform {
    SceneTransform(rotation: axisAngle(SIMD3<Float>(0.3, -0.7, 0.5), 0.9),
                   translation: SIMD3<Float>(0.4, -1.2, 2.0),
                   scale: 1.7)
}

@Suite struct SceneTransformTests {
    @Test func identityLeavesCloudUntouched() throws {
        let cloud = try randomCloud(count: 20, degree: 3, seed: 1)
        let moved = try SceneTransform.identity.apply(to: cloud)
        #expect(moved.positions == cloud.positions)
        #expect(moved.sh == cloud.sh)
        #expect(moved.logScales == cloud.logScales)
    }

    @Test func movesPositionsAsSimilarity() throws {
        let cloud = try randomCloud(count: 30, degree: 0, seed: 2)
        let transform = sampleTransform()
        let moved = try transform.apply(to: cloud)
        for index in 0..<cloud.count {
            let expected: SIMD3<Float> = transform.transformPoint(point(cloud, index))
            #expect(approxEqual(point(moved, index), expected, tol: 1e-4))
        }
    }

    @Test func scalesLogScalesByScaleFactor() throws {
        let cloud = try randomCloud(count: 10, degree: 0, seed: 3)
        let transform = SceneTransform(rotation: SIMD4<Float>(1, 0, 0, 0), translation: .zero, scale: 2)
        let moved = try transform.apply(to: cloud)
        for index in 0..<(3 * cloud.count) {
            #expect(approxEqual(moved.logScales[index], cloud.logScales[index] + log(Float(2)), tol: 1e-5))
        }
    }

    @Test func covarianceFollowsRotationAndScale() throws {
        // Σ = R S² Rᵀ with S from the log-scales and R from the quaternion. After the transform Σ' = s² · R_T Σ R_Tᵀ.
        let cloud = try randomCloud(count: 12, degree: 0, seed: 4)
        let transform = sampleTransform()
        let moved = try transform.apply(to: cloud)
        for index in 0..<cloud.count {
            let original: Matrix3 = covariance(cloud, index)
            let expected: Matrix3 = scaleMatrix(transform.rotationMatrix * original * transform.rotationMatrix.transposed,
                                                by: transform.scale * transform.scale)
            let actual: Matrix3 = covariance(moved, index)
            #expect(matrixApproxEqual(actual, expected, tol: 2e-3))
        }
    }

    @Test func compositionMatchesNestedApplication() {
        let first = sampleTransform()
        let second = SceneTransform(rotation: axisAngle(SIMD3<Float>(1, 0, 0), -0.4),
                                    translation: SIMD3<Float>(-0.3, 0.5, 0.1),
                                    scale: 0.6)
        let combined: SceneTransform = second.composing(first)
        var random = TestRandom(state: 5)
        for _ in 0..<10 {
            let p: SIMD3<Float> = random.vector(scale: 4)
            let nested: SIMD3<Float> = second.transformPoint(first.transformPoint(p))
            #expect(approxEqual(combined.transformPoint(p), nested, tol: 1e-4))
        }
    }

    @Test func inverseUndoesTransform() {
        let transform = sampleTransform()
        var random = TestRandom(state: 6)
        for _ in 0..<10 {
            let p: SIMD3<Float> = random.vector(scale: 4)
            let back: SIMD3<Float> = transform.inverse.transformPoint(transform.transformPoint(p))
            #expect(approxEqual(back, p, tol: 1e-4))
        }
    }

    @Test func scaledByMultipliesScaleAndDistances() {
        let transform = sampleTransform()
        let scaled: SceneTransform = transform.scaled(by: 3)
        let a = SIMD3<Float>(0.5, 0.2, -0.1)
        let b = SIMD3<Float>(-1, 0.9, 0.7)
        let before: Float = (transform.transformPoint(a) - transform.transformPoint(b)).length
        let after: Float = (scaled.transformPoint(a) - scaled.transformPoint(b)).length
        #expect(abs(after - 3 * before) < 1e-4)
        #expect(abs(scaled.scale - 3 * transform.scale) < 1e-6)
    }

    @Test func calibrationScaleMakesMeasuredLengthReal() {
        let a = SIMD3<Float>(0, 0, 0)
        let b = SIMD3<Float>(3, 4, 0)
        #expect(SceneTransform.calibrationScale(pointA: a, pointB: b, realLength: 10) == 2)
        #expect(SceneTransform.calibrationScale(pointA: a, pointB: a, realLength: 10) == nil)
        #expect(SceneTransform.calibrationScale(pointA: a, pointB: b, realLength: 0) == nil)
        #expect(SceneTransform.calibrationScale(pointA: a, pointB: b, realLength: .nan) == nil)
    }

    // MARK: Spherical harmonics

    @Test func shColourIsPreservedUnderRotation() throws {
        // Colour seen from the moved camera equals the colour of the original scene from the original camera:
        // evaluate(coef', R·d) == evaluate(coef, d) for every band up to degree 3.
        let cloud = try randomCloud(count: 16, degree: 3, seed: 7)
        let rotation: SIMD4<Float> = axisAngle(SIMD3<Float>(0.2, 0.9, -0.4), 1.3)
        let transform = SceneTransform(rotation: rotation, translation: .zero, scale: 1)
        let moved = try transform.apply(to: cloud)
        let k: Int = GaussianCloud.coefficientCount(degree: 3)
        var random = TestRandom(state: 8)
        for index in 0..<cloud.count {
            for _ in 0..<4 {
                let direction: SIMD3<Float> = normalized(random.vector())
                let original: SIMD3<Float> = SphericalHarmonics.evaluate(
                    degree: 3, coefficients: cloud.sh[(index * k * 3)..<((index + 1) * k * 3)], direction: direction)
                let rotatedDirection: SIMD3<Float> = transform.rotationMatrix * direction
                let rotated: SIMD3<Float> = SphericalHarmonics.evaluate(
                    degree: 3, coefficients: moved.sh[(index * k * 3)..<((index + 1) * k * 3)], direction: rotatedDirection)
                #expect(approxEqual(rotated, original, tol: 2e-4))
            }
        }
    }

    @Test func shRotationForIdentityIsIdentity() throws {
        let cloud = try randomCloud(count: 4, degree: 2, seed: 9)
        let moved = try SceneTransform(rotation: SIMD4<Float>(1, 0, 0, 0), translation: .init(1, 2, 3), scale: 1)
            .apply(to: cloud)
        #expect(moved.sh.count == cloud.sh.count)
        for index in 0..<cloud.sh.count {
            #expect(abs(moved.sh[index] - cloud.sh[index]) < 1e-4)
        }
    }

    // MARK: Floor

    @Test func floorFitPlacesPlaneAtHeightZero() {
        // Points on a tilted plane through (1, 2, -1), normal close to +Y, with a small noise and one stray pick.
        let normal: SIMD3<Float> = normalized(SIMD3<Float>(0.2, 1, 0.1))
        let anchor = SIMD3<Float>(1, 2, -1)
        var random = TestRandom(state: 10)
        var picks: [SIMD3<Float>] = []
        for _ in 0..<5 {
            let base: SIMD3<Float> = random.vector(scale: 2)
            let inPlane: SIMD3<Float> = base - normal * (base * normal).sum()
            picks.append(anchor + inPlane + normal * (random.next() - 0.5) * 0.002)
        }
        picks.append(anchor + normal * 3)
        // The stray pick sits well above the plane, so it should be rejected by the robust weights.
        guard let floor = SceneTransform.fromFloor(points: picks, upAxis: .positiveY) else {
            Issue.record("expected a transform")
            return
        }
        let transformedUp: [Float] = picks.dropLast().map { floor.transformPoint($0).y }
        for height in transformedUp {
            #expect(abs(height) < 5e-3)
        }
        // The centroid of the fitted points sits at the origin.
        var sum: SIMD3<Float> = .zero
        for pick in picks.dropLast() {
            sum += floor.transformPoint(pick)
        }
        #expect(approxEqual(sum / Float(picks.count - 1), .zero, tol: 5e-3))
        // The rotation is proper.
        let matrix: Matrix3 = floor.rotationMatrix
        #expect(matrixApproxEqual(matrix * matrix.transposed, Matrix3.identity, tol: 1e-5))
    }

    @Test func floorFitRejectsDegenerateInput() {
        #expect(SceneTransform.fromFloor(points: [.zero, .init(1, 0, 0)], upAxis: .positiveY) == nil)
        let collinear: [SIMD3<Float>] = (0..<5).map { SIMD3<Float>(Float($0), 0, 0) }
        #expect(SceneTransform.fromFloor(points: collinear, upAxis: .positiveY) == nil)
    }

    // MARK: Crop box

    @Test func cropContainmentRespectsRotation() {
        // Rotated 90° about z: the box's own x axis points along world y.
        let box = CropBox(center: .zero, halfExtents: SIMD3<Float>(1, 1, 1),
                          rotation: axisAngle(SIMD3<Float>(0, 0, 1), .pi / 2))
        #expect(box.contains(SIMD3<Float>(0, 0.9, 0)))
        #expect(!box.contains(SIMD3<Float>(0, 1.5, 0)))
        #expect(!box.contains(SIMD3<Float>(1.5, 0, 0)))
        #expect(!box.contains(SIMD3<Float>(.nan, 0, 0)))
    }

    @Test func cropMaskSelectsOutsideGaussians() throws {
        let cloud = try GaussianCloud(shDegree: 0,
                                      positions: [0, 0, 0, 5, 0, 0, 0.5, 0.5, 0.5],
                                      logScales: [Float](repeating: 0, count: 9),
                                      rotations: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                                      opacityLogits: [0, 0, 0],
                                      sh: [Float](repeating: 0, count: 9))
        let box = CropBox(center: .zero, halfExtents: SIMD3<Float>(1, 1, 1))
        let outside: SelectionMask = box.mask(for: cloud)
        #expect(outside.selectedIndices() == [1])
    }

    @Test func transformedCropFollowsTheScene() {
        let transform = sampleTransform()
        let box = CropBox(center: SIMD3<Float>(0.3, -0.2, 0.1), halfExtents: SIMD3<Float>(0.8, 0.5, 0.9),
                          rotation: axisAngle(SIMD3<Float>(1, 1, 0), 0.6))
        let moved: CropBox = box.transformed(by: transform)
        var random = TestRandom(state: 11)
        for _ in 0..<200 {
            let p: SIMD3<Float> = random.vector(scale: 1.2)
            #expect(box.contains(p) == moved.contains(transform.transformPoint(p)))
        }
    }

    // MARK: Serialisation

    @Test func transformJSONRoundTrips() throws {
        var transform = sampleTransform()
        transform.isMetric = true
        let data = try transform.jsonData()
        let decoded = try SceneTransform.decode(jsonData: data)
        #expect(decoded == transform)
    }

    @Test func cropJSONRoundTrips() throws {
        let box = CropBox(center: SIMD3<Float>(1, 2, 3), halfExtents: SIMD3<Float>(0.5, 1, 2),
                          rotation: axisAngle(SIMD3<Float>(0, 1, 0), 0.3))
        let decoded = try CropBox.decode(jsonData: try box.jsonData())
        #expect(decoded == box)
    }

    @Test func rejectsMalformedFiles() {
        let badScale = Data(#"{"version":1,"rotation":[1,0,0,0],"translation":[0,0,0],"scale":0,"isMetric":false}"#.utf8)
        #expect(throws: SceneTransformError.self) { try SceneTransform.decode(jsonData: badScale) }
        let future = Data(#"{"version":9,"rotation":[1,0,0,0],"translation":[0,0,0],"scale":1,"isMetric":false}"#.utf8)
        #expect(throws: SceneTransformError.unsupportedVersion(9)) { try SceneTransform.decode(jsonData: future) }
        let negativeExtent = Data(#"{"version":1,"center":[0,0,0],"halfExtents":[-1,1,1],"rotation":[1,0,0,0]}"#.utf8)
        #expect(throws: SceneTransformError.self) { try CropBox.decode(jsonData: negativeExtent) }
        #expect(throws: SceneTransformError.self) { try CropBox.decode(jsonData: Data("nope".utf8)) }
    }
}

// MARK: - Test helpers

private func normalized(_ v: SIMD3<Float>) -> SIMD3<Float> {
    v / (v * v).sum().squareRoot()
}

private extension SIMD3<Float> {
    var length: Float { (self * self).sum().squareRoot() }
}

private func covariance(_ cloud: GaussianCloud, _ index: Int) -> Matrix3 {
    let rotation: Matrix3 = Matrix3.rotation(quaternion: SIMD4<Float>(cloud.rotations[4 * index], cloud.rotations[4 * index + 1],
                                                                      cloud.rotations[4 * index + 2], cloud.rotations[4 * index + 3]))
    let s0: Float = exp(cloud.logScales[3 * index])
    let s1: Float = exp(cloud.logScales[3 * index + 1])
    let s2: Float = exp(cloud.logScales[3 * index + 2])
    let scale = Matrix3(c0: SIMD3<Float>(s0, 0, 0), c1: SIMD3<Float>(0, s1, 0), c2: SIMD3<Float>(0, 0, s2))
    let m: Matrix3 = rotation * scale
    return m * m.transposed
}

private func scaleMatrix(_ m: Matrix3, by factor: Float) -> Matrix3 {
    Matrix3(c0: m.c0 * factor, c1: m.c1 * factor, c2: m.c2 * factor)
}

private func matrixApproxEqual(_ a: Matrix3, _ b: Matrix3, tol: Float) -> Bool {
    approxEqual(a.c0, b.c0, tol: tol) && approxEqual(a.c1, b.c1, tol: tol) && approxEqual(a.c2, b.c2, tol: tol)
}
