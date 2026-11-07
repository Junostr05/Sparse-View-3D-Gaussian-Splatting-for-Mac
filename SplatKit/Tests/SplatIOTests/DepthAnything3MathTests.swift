import Foundation
import Testing
@testable import SplatIO

@Suite struct DepthAnything3MathTests {
    /// Reference pose from the PyTorch implementation: [T(3), quaternion x y z w, fovH, fovW].
    private let referencePose: [Double] = [0.1, -0.2, 0.3, 0.1, 0.2, 0.3, 0.9, 0.9, 0.7]

    @Test func worldToCameraMatchesReferenceRotationAndTranslation() {
        let converted = DepthAnything3Math.worldToCamera(pose: referencePose)
        let expectedRotation: [Double] = [
            0.726316, 0.610526, -0.315789,
            -0.526316, 0.789474, 0.315789,
            0.442105, -0.063158, 0.894737,
        ]
        #expect(converted.rotation.count == 9)
        for index in 0..<9 {
            #expect(abs(converted.rotation[index] - expectedRotation[index]) < 1e-5, "rotation element \(index)")
        }
        #expect(abs(converted.translation.x - 0.144211) < 1e-5)
        #expect(abs(converted.translation.y - 0.115789) < 1e-5)
        #expect(abs(converted.translation.z - (-0.325263)) < 1e-5)
        #expect(converted.fovH == 0.9)
        #expect(converted.fovW == 0.7)
    }

    @Test func worldToCameraTransposeIsCameraToWorldRotation() {
        // The reference camera-to-world rows are the transpose of the returned world-to-camera rotation.
        let converted = DepthAnything3Math.worldToCamera(pose: referencePose)
        let expectedCameraToWorld: [Double] = [
            0.726316, -0.526316, 0.442105,
            0.610526, 0.789474, -0.063158,
            -0.315789, 0.315789, 0.894737,
        ]
        for row in 0..<3 {
            for column in 0..<3 {
                let transposed: Double = converted.rotation[column * 3 + row]
                #expect(abs(transposed - expectedCameraToWorld[row * 3 + column]) < 1e-5, "c2w element \(row),\(column)")
            }
        }
        // The stored translation is t_w2c = −R_w2c·T, with T the model's translation (0.1, −0.2, 0.3).
        let r: [Double] = converted.rotation
        let tx: Double = 0.1
        let ty: Double = -0.2
        let tz: Double = 0.3
        let rotatedX: Double = r[0] * tx + r[1] * ty + r[2] * tz
        let rotatedY: Double = r[3] * tx + r[4] * ty + r[5] * tz
        let rotatedZ: Double = r[6] * tx + r[7] * ty + r[8] * tz
        #expect(abs(converted.translation.x + rotatedX) < 1e-12)
        #expect(abs(converted.translation.y + rotatedY) < 1e-12)
        #expect(abs(converted.translation.z + rotatedZ) < 1e-12)
    }

    @Test func worldToCameraIgnoresQuaternionScale() {
        var scaled: [Double] = referencePose
        for index in 3..<7 {
            scaled[index] *= 3
        }
        let original = DepthAnything3Math.worldToCamera(pose: referencePose)
        let rescaled = DepthAnything3Math.worldToCamera(pose: scaled)
        for index in 0..<9 {
            #expect(abs(original.rotation[index] - rescaled.rotation[index]) < 1e-12)
        }
    }

    @Test func intrinsicsMatchReferenceForPortraitView() {
        // Portrait model image: 378 wide, 504 high.
        let intrinsics = DepthAnything3Math.intrinsics(fovH: 0.9, fovW: 0.7, width: 378, height: 504)
        #expect(abs(intrinsics.fx - 517.7678) < 1e-4)
        #expect(abs(intrinsics.fy - 521.6796) < 1e-4)
        #expect(intrinsics.cx == 189)
        #expect(intrinsics.cy == 252)
    }

    @Test func intrinsicsClampDegenerateFieldOfView() {
        let intrinsics = DepthAnything3Math.intrinsics(fovH: 0, fovW: 0, width: 100, height: 80)
        #expect(intrinsics.fx.isFinite)
        #expect(intrinsics.fy.isFinite)
        #expect(intrinsics.fx == 50 / 1e-6)
        #expect(intrinsics.fy == 40 / 1e-6)
    }

    @Test func unprojectFlatDepthWithIdentityPoseGivesDepthTwo() {
        let width = 4
        let height = 3
        let depth = [Float](repeating: 2, count: width * height)
        let confidence = [Float](repeating: 1, count: width * height)
        let fx = 2.0
        let fy = 2.0
        let cx = 1.5
        let cy = 1.0
        let identity: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 1]

        let result = DepthAnything3Math.unproject(depth: depth, confidence: confidence, width: width, height: height,
                                                  fx: fx, fy: fy, cx: cx, cy: cy,
                                                  rotationW2C: identity, translationW2C: SIMD3<Double>(0, 0, 0),
                                                  stride: 1, minConfidence: 0.5)
        #expect(result.points.count == width * height)
        #expect(result.pixelIndices == Array(0..<(width * height)))
        for (point, index) in zip(result.points, result.pixelIndices) {
            let column = Double(index % width)
            let row = Double(index / width)
            #expect(point.z == 2)
            #expect(abs(point.x - (column - cx) / fx * 2) < 1e-12)
            #expect(abs(point.y - (row - cy) / fy * 2) < 1e-12)
        }
    }

    @Test func unprojectSkipsInvalidDepthAndLowConfidence() {
        let depth: [Float] = [1, 0, Float.nan, 3]
        let confidence: [Float] = [0.9, 0.9, 0.9, 0.1]
        let identity: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        let result = DepthAnything3Math.unproject(depth: depth, confidence: confidence, width: 2, height: 2,
                                                  fx: 1, fy: 1, cx: 0, cy: 0,
                                                  rotationW2C: identity, translationW2C: SIMD3<Double>(0, 0, 0),
                                                  stride: 1, minConfidence: 0.5)
        #expect(result.pixelIndices == [0])
    }

    @Test func unprojectAppliesTheWorldToCameraInverse() {
        // A camera point (0, 0, 2) seen from a world origin shifted by t: x_world = Rᵀ·(x_cam − t).
        let converted = DepthAnything3Math.worldToCamera(pose: referencePose)
        let depth: [Float] = [2]
        let confidence: [Float] = [1]
        let result = DepthAnything3Math.unproject(depth: depth, confidence: confidence, width: 1, height: 1,
                                                  fx: 1, fy: 1, cx: 0, cy: 0,
                                                  rotationW2C: converted.rotation, translationW2C: converted.translation,
                                                  stride: 1, minConfidence: 0)
        #expect(result.points.count == 1)
        // Projecting the world point back with the forward transform must give the camera point (0, 0, 2).
        let world: SIMD3<Double> = result.points[0]
        let r = converted.rotation
        let camX: Double = r[0] * world.x + r[1] * world.y + r[2] * world.z + converted.translation.x
        let camY: Double = r[3] * world.x + r[4] * world.y + r[5] * world.z + converted.translation.y
        let camZ: Double = r[6] * world.x + r[7] * world.y + r[8] * world.z + converted.translation.z
        #expect(abs(camX) < 1e-9)
        #expect(abs(camY) < 1e-9)
        #expect(abs(camZ - 2) < 1e-9)
    }

    @Test func functionSelectionTable() {
        let expectations: [(count: Int, name: String, viewCount: Int)] = [
            (2, "v4_504x378", 4),
            (4, "v4_504x378", 4),
            (5, "v6_504x378", 6),
            (6, "v6_504x378", 6),
            (7, "v8_504x378", 8),
            (9, "v12_504x378", 12),
            (13, "v16_504x378", 16),
            (16, "v16_504x378", 16),
        ]
        for expectation in expectations {
            let selected = DepthAnything3Math.function(forFrameCount: expectation.count, portrait: true)
            #expect(selected?.name == expectation.name, "count \(expectation.count)")
            #expect(selected?.viewCount == expectation.viewCount, "count \(expectation.count)")
            #expect(selected?.height == 504)
            #expect(selected?.width == 378)
        }
        let landscape = DepthAnything3Math.function(forFrameCount: 6, portrait: false)
        #expect(landscape?.name == "v6_378x504")
        #expect(landscape?.height == 378)
        #expect(landscape?.width == 504)
        #expect(DepthAnything3Math.function(forFrameCount: 1, portrait: true) == nil)
        #expect(DepthAnything3Math.function(forFrameCount: 0, portrait: true) == nil)
        #expect(DepthAnything3Math.function(forFrameCount: 17, portrait: true) == nil)
    }

    @Test func evenlySpacedIndicesIncludeEndsAndIncrease() {
        #expect(DepthAnything3Math.evenlySpacedIndices(count: 10, keep: 4) == [0, 3, 6, 9])
        #expect(DepthAnything3Math.evenlySpacedIndices(count: 5, keep: 9) == [0, 1, 2, 3, 4])
        #expect(DepthAnything3Math.evenlySpacedIndices(count: 7, keep: 1) == [0])
        #expect(DepthAnything3Math.evenlySpacedIndices(count: 0, keep: 4) == [])

        let subsampled = DepthAnything3Math.evenlySpacedIndices(count: 100, keep: 16)
        #expect(subsampled.count == 16)
        #expect(subsampled.first == 0)
        #expect(subsampled.last == 99)
        for (previous, next) in zip(subsampled, subsampled.dropFirst()) {
            #expect(previous < next)
        }
    }

    /// Camera-to-world rotation of a model pose quaternion, from `worldToCamera` transposed back.
    private func cameraToWorldRotation(xyzw: [Double]) -> [Double] {
        // worldToCamera reads the pose as [T(3), x, y, z, w, fovH, fovW].
        let pose: [Double] = [0, 0, 0, xyzw[0], xyzw[1], xyzw[2], xyzw[3], 0.5, 0.5]
        let converted = DepthAnything3Math.worldToCamera(pose: pose)
        var cameraToWorld: [Double] = [Double](repeating: 0, count: 9)
        for row in 0..<3 {
            for column in 0..<3 {
                cameraToWorld[row * 3 + column] = converted.rotation[column * 3 + row]
            }
        }
        return cameraToWorld
    }

    @Test func quaternionRoundTripsThroughRotation() {
        // Unit quaternions as (w, x, y, z) with w >= 0, including 180° rotations about the axes.
        let cases: [[Double]] = [
            [0.5, 0.5, 0.5, 0.5],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0.9 / 0.95.squareRoot(), 0.1 / 0.95.squareRoot(), 0.2 / 0.95.squareRoot(), 0.3 / 0.95.squareRoot()],
        ]
        for expected in cases {
            // The pose stores (x, y, z, w), so reorder before building the rotation.
            let xyzw: [Double] = [expected[1], expected[2], expected[3], expected[0]]
            let recovered: SIMD4<Double> = DepthAnything3Math.quaternionWXYZ(fromRowMajor: cameraToWorldRotation(xyzw: xyzw))
            let normSquared: Double = recovered[0] * recovered[0] + recovered[1] * recovered[1]
                + recovered[2] * recovered[2] + recovered[3] * recovered[3]
            #expect(abs(normSquared - 1) < 1e-12)
            #expect(recovered[0] >= 0)
            for component in 0..<4 {
                #expect(abs(recovered[component] - expected[component]) < 1e-9, "quaternion \(expected), component \(component)")
            }
        }
    }

    @Test func quaternionFromRotationHasNonNegativeW() {
        // (w, x, y, z) = (-0.5, 0.5, -0.5, 0.5) is the same rotation as (0.5, -0.5, 0.5, -0.5).
        let cameraToWorld: [Double] = cameraToWorldRotation(xyzw: [0.5, -0.5, 0.5, -0.5])
        let recovered: SIMD4<Double> = DepthAnything3Math.quaternionWXYZ(fromRowMajor: cameraToWorld)
        let expected: [Double] = [0.5, -0.5, 0.5, -0.5]
        for component in 0..<4 {
            #expect(abs(recovered[component] - expected[component]) < 1e-9, "component \(component)")
        }
    }
}
