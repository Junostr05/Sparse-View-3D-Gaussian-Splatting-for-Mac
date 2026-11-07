import Foundation
import Testing
import SplatCore

private func vtDot(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
    a.x * b.x + a.y * b.y + a.z * b.z
}

private func vtCross(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> SIMD3<Float> {
    SIMD3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
}

/// Rodrigues rotation, written independently of the library.
private func vtRotate(_ v: SIMD3<Float>, about axis: SIMD3<Float>, by angle: Float) -> SIMD3<Float> {
    let k = axis / vtDot(axis, axis).squareRoot()
    let c = cos(angle)
    let s = sin(angle)
    return v * c + vtCross(k, v) * s + k * (vtDot(k, v) * (1 - c))
}

private func vtIsOrthonormal(_ c: ViewCamera, tol: Float = 1e-4) -> Bool {
    let unitLengths = abs(vtDot(c.right, c.right) - 1) < tol
        && abs(vtDot(c.down, c.down) - 1) < tol
        && abs(vtDot(c.forward, c.forward) - 1) < tol
    let orthogonal = abs(vtDot(c.right, c.down)) < tol
        && abs(vtDot(c.right, c.forward)) < tol
        && abs(vtDot(c.down, c.forward)) < tol
    let handed = approxEqual(vtCross(c.right, c.down), c.forward, tol: tol)
    return unitLengths && orthogonal && handed
}

private func vtExpectCamerasClose(_ a: Camera, _ b: Camera, tol: Float) {
    #expect(approxEqual(a.rotation.c0, b.rotation.c0, tol: tol))
    #expect(approxEqual(a.rotation.c1, b.rotation.c1, tol: tol))
    #expect(approxEqual(a.rotation.c2, b.rotation.c2, tol: tol))
    #expect(approxEqual(a.translation, b.translation, tol: tol))
    #expect(a.width == b.width && a.height == b.height)
    #expect(approxEqual(a.fx, b.fx, tol: tol * max(1, a.fx)))
    #expect(approxEqual(a.fy, b.fy, tol: tol * max(1, a.fy)))
}

/// A roll of `angle` about the forward axis, applied to a copy of the view.
private func vtRolled(_ camera: ViewCamera, by angle: Float) -> ViewCamera {
    var rolled = camera
    rolled.right = vtRotate(camera.right, about: camera.forward, by: angle)
    rolled.down = vtRotate(camera.down, about: camera.forward, by: angle)
    return rolled
}

private let vtBounds = Bounds(min: SIMD3(-1, -2, -3), max: SIMD3(2, 1, 0.5))

@Suite struct ViewCameraTests {
    @Test func framingMatchesOrbitCamera() {
        for upAxis in UpAxis.allCases {
            let orbit = OrbitCamera.framing(vtBounds, upAxis: upAxis)
            let view = ViewCamera.framing(vtBounds, upAxis: upAxis)
            #expect(view.pivot == orbit.target)
            #expect(view.distance == orbit.distance)
            #expect(view.minDistance == orbit.minDistance)
            #expect(view.maxDistance == orbit.maxDistance)
            #expect(view.fovY == orbit.fovY)
            vtExpectCamerasClose(view.camera(width: 640, height: 480),
                                 orbit.camera(width: 640, height: 480), tol: 1e-5)
        }
    }

    @Test func orthonormalityAfterRandomOperations() {
        for upAxis in UpAxis.allCases {
            var camera = ViewCamera.framing(vtBounds, upAxis: upAxis)
            // Start rolled so that the roll-carrying paths are exercised.
            camera = vtRolled(camera, by: 0.5)
            var rng = SeededGenerator(seed: 7)
            for _ in 0..<1000 {
                switch Int.random(in: 0..<5, using: &rng) {
                case 0:
                    camera.orbit(deltaYaw: Float.random(in: -0.6...0.6, using: &rng),
                                 deltaPitch: Float.random(in: -0.6...0.6, using: &rng))
                case 1:
                    camera.zoom(factor: Float.random(in: 0.7...1.4, using: &rng))
                case 2:
                    camera.pan(deltaPixels: SIMD2(Float.random(in: -40...40, using: &rng),
                                                  Float.random(in: -40...40, using: &rng)),
                               viewportHeight: 480)
                case 3:
                    camera.orbit(deltaYaw: Float.random(in: -3...3, using: &rng),
                                 deltaPitch: Float.random(in: -3...3, using: &rng))
                default:
                    camera.removeRoll()
                }
                #expect(vtIsOrthonormal(camera), "\(upAxis) lost orthonormality")
            }
        }
    }

    @Test func pitchLimitIsNeverExceeded() {
        for upAxis in UpAxis.allCases {
            var camera = ViewCamera.framing(vtBounds, upAxis: upAxis)
            var rng = SeededGenerator(seed: 11)
            let up = upAxis.vector
            for _ in 0..<1000 {
                camera.orbit(deltaYaw: Float.random(in: -0.3...0.3, using: &rng),
                             deltaPitch: Float.random(in: -0.8...0.8, using: &rng))
                #expect(abs(vtDot(camera.forward, up)) <= ViewCamera.maxUpAlignment + 1e-5)
            }

            // A large pitch stops exactly at the limit, in both directions.
            var down = ViewCamera.framing(vtBounds, upAxis: upAxis)
            down.orbit(deltaYaw: 0, deltaPitch: 10)
            #expect(abs(abs(vtDot(down.forward, up)) - ViewCamera.maxUpAlignment) < 1e-5)
            var upward = ViewCamera.framing(vtBounds, upAxis: upAxis)
            upward.orbit(deltaYaw: 0, deltaPitch: -10)
            #expect(abs(abs(vtDot(upward.forward, up)) - ViewCamera.maxUpAlignment) < 1e-5)
        }
    }

    @Test func pitchFromTopViewMovesAwayFromThePole() {
        // The view is exactly along the up axis; the limit is relaxed to the current value, so
        // pitching must be able to leave the pole.
        var camera = ViewCamera.aligned(.top, like: ViewCamera.framing(vtBounds, upAxis: .positiveZ))
        camera.orbit(deltaYaw: 0, deltaPitch: 0.5)
        #expect(abs(vtDot(camera.forward, SIMD3(0, 0, 1))) < 1)
        #expect(vtIsOrthonormal(camera))
    }

    @Test func orbitMatchesOrbitCameraDirectionsAndPan() {
        for upAxis in UpAxis.allCases {
            for (yaw, pitch) in [(Float(0), Float(0)), (0.7, 0.3), (-1.2, -0.4)] {
                var orbit = OrbitCamera(target: SIMD3(0.5, -0.25, 2), distance: 7,
                                        yaw: yaw, pitch: pitch, upAxis: upAxis,
                                        minDistance: 0.07, maxDistance: 700)
                var view = ViewCamera(camera: orbit.camera(width: 640, height: 480),
                                      pivotDistance: 7, upAxis: upAxis,
                                      minDistance: 0.07, maxDistance: 700)
                let steps: [(Float, Float)] = [(0.3, 0.2), (-0.5, -0.1), (1.0, 0.4)]
                for (dy, dp) in steps {
                    orbit.orbit(deltaYaw: dy, deltaPitch: dp)
                    view.orbit(deltaYaw: dy, deltaPitch: dp)
                    vtExpectCamerasClose(view.camera(width: 640, height: 480),
                                         orbit.camera(width: 640, height: 480), tol: 1e-4)
                }
                orbit.pan(deltaPixels: SIMD2(12, -7), viewportHeight: 480)
                view.pan(deltaPixels: SIMD2(12, -7), viewportHeight: 480)
                vtExpectCamerasClose(view.camera(width: 640, height: 480),
                                     orbit.camera(width: 640, height: 480), tol: 1e-4)
            }
        }
    }

    @Test func initFromCameraRoundTrips() {
        let orbit = OrbitCamera(target: SIMD3(0.5, -0.25, 2), distance: 7, yaw: 0.8, pitch: 0.4,
                                upAxis: .negativeY, minDistance: 0.07, maxDistance: 700)
        let input = orbit.camera(width: 640, height: 480)
        let view = ViewCamera(camera: input, pivotDistance: 7, upAxis: .negativeY,
                              minDistance: 0.07, maxDistance: 700)
        vtExpectCamerasClose(view.camera(width: 640, height: 480), input, tol: 1e-4)
        #expect(approxEqual(view.eye, input.center, tol: 1e-4))
        #expect(abs(view.distance - 7) < 1e-6)
    }

    @Test func initFromRolledCameraRoundTrips() {
        let orbit = OrbitCamera(target: .zero, distance: 5, yaw: -0.4, pitch: 0.2,
                                upAxis: .positiveZ, minDistance: 0.05, maxDistance: 500)
        let base = orbit.camera(width: 640, height: 480)
        let frame = ViewCamera(camera: base, pivotDistance: 5, upAxis: .positiveZ,
                               minDistance: 0.05, maxDistance: 500)
        let rolled = vtRolled(frame, by: 0.5).camera(width: 640, height: 480)
        let view = ViewCamera(camera: rolled, pivotDistance: 5, upAxis: .positiveZ,
                              minDistance: 0.05, maxDistance: 500)
        vtExpectCamerasClose(view.camera(width: 640, height: 480), rolled, tol: 1e-4)
        #expect(abs(vtDot(view.right, SIMD3(0, 0, 1))) > 0.1, "roll must be present")
    }

    @Test func rollIsPreservedByYawAndRemovedByRemoveRoll() {
        for upAxis in UpAxis.allCases {
            let up = upAxis.vector
            var camera = vtRolled(ViewCamera.framing(vtBounds, upAxis: upAxis), by: Float.pi / 6)
            let rollAngle = asin(vtDot(camera.right, up))
            #expect(abs(rollAngle) > 0.1, "test camera must be rolled")

            for yaw: Float in [0.5, -1.0, 2.0] {
                camera.orbit(deltaYaw: yaw, deltaPitch: 0)
                #expect(abs(asin(vtDot(camera.right, up)) - rollAngle) < 1e-4)
            }

            let forwardBefore = camera.forward
            camera.removeRoll()
            #expect(abs(vtDot(camera.right, up)) < 1e-5)
            #expect(approxEqual(camera.forward, forwardBefore, tol: 1e-5))
            #expect(vtIsOrthonormal(camera))
        }
    }

    @Test func presetsForPositiveZ() {
        let current = ViewCamera.framing(vtBounds, upAxis: .positiveZ)
        let expected: [(ViewPreset, SIMD3<Float>, SIMD3<Float>)] = [
            (.front, SIMD3(0, 1, 0), SIMD3(1, 0, 0)),
            (.back, SIMD3(0, -1, 0), SIMD3(-1, 0, 0)),
            (.right, SIMD3(-1, 0, 0), SIMD3(0, 1, 0)),
            (.left, SIMD3(1, 0, 0), SIMD3(0, -1, 0)),
            (.top, SIMD3(0, 0, -1), SIMD3(1, 0, 0)),
            (.bottom, SIMD3(0, 0, 1), SIMD3(1, 0, 0)),
        ]
        for (preset, forward, right) in expected {
            let view = ViewCamera.aligned(preset, like: current)
            #expect(approxEqual(view.forward, forward, tol: 1e-6), "\(preset) forward")
            #expect(approxEqual(view.right, right, tol: 1e-6), "\(preset) right")
            #expect(vtIsOrthonormal(view), "\(preset) orthonormal")
            #expect(view.pivot == current.pivot)
            #expect(view.distance == current.distance)
            #expect(view.fovY == current.fovY)
            #expect(view.upAxis == current.upAxis)
        }
        // Blender top view: screen up is +Y, so down is -Y.
        let top = ViewCamera.aligned(.top, like: current)
        #expect(approxEqual(top.down, SIMD3(0, -1, 0), tol: 1e-6))
    }

    @Test func presetsAreOrthonormalForAllUpAxes() {
        for upAxis in UpAxis.allCases {
            let current = ViewCamera.framing(vtBounds, upAxis: upAxis)
            let up = upAxis.vector
            for preset in ViewPreset.allCases {
                let view = ViewCamera.aligned(preset, like: current)
                #expect(vtIsOrthonormal(view), "\(upAxis) \(preset)")
                #expect(view.upAxis == upAxis)
            }
            #expect(approxEqual(ViewCamera.aligned(.top, like: current).forward, -up, tol: 1e-6))
            #expect(approxEqual(ViewCamera.aligned(.bottom, like: current).forward, up, tol: 1e-6))
            // Front and back look in opposite horizontal directions.
            let front = ViewCamera.aligned(.front, like: current).forward
            let back = ViewCamera.aligned(.back, like: current).forward
            #expect(approxEqual(front, -back, tol: 1e-6))
            #expect(abs(vtDot(front, up)) < 1e-6)
        }
    }

    @Test func interpolatedEndpointsAndMidpoint() {
        let a = ViewCamera(pivot: SIMD3(0, 0, 0), distance: 4,
                           right: SIMD3(1, 0, 0), down: SIMD3(0, 1, 0), forward: SIMD3(0, 0, 1),
                           fovY: 0.8, upAxis: .negativeY, minDistance: 0.1, maxDistance: 100)
        // 90° about +Y: forward Z -> X, right X -> -Z, down unchanged.
        let b = ViewCamera(pivot: SIMD3(4, 0, 0), distance: 2,
                           right: SIMD3(0, 0, -1), down: SIMD3(0, 1, 0), forward: SIMD3(1, 0, 0),
                           fovY: 1.0, upAxis: .negativeY, minDistance: 0.1, maxDistance: 100)

        let start = ViewCamera.interpolated(a, b, t: 0)
        #expect(approxEqual(start.eye, a.eye, tol: 1e-6))
        #expect(approxEqual(start.right, a.right, tol: 1e-6))
        #expect(approxEqual(start.down, a.down, tol: 1e-6))
        #expect(approxEqual(start.forward, a.forward, tol: 1e-6))
        #expect(abs(start.fovY - a.fovY) < 1e-6)

        let end = ViewCamera.interpolated(a, b, t: 1)
        #expect(approxEqual(end.eye, b.eye, tol: 1e-5))
        #expect(approxEqual(end.right, b.right, tol: 1e-5))
        #expect(approxEqual(end.down, b.down, tol: 1e-5))
        #expect(approxEqual(end.forward, b.forward, tol: 1e-5))
        #expect(abs(end.fovY - b.fovY) < 1e-5)

        let mid = ViewCamera.interpolated(a, b, t: 0.5)
        let s: Float = Float(0.5).squareRoot()
        // Halfway between Z and X: 45° from each.
        #expect(approxEqual(mid.forward, SIMD3(s, 0, s), tol: 1e-5))
        #expect(abs(vtDot(mid.forward, a.forward) - s) < 1e-5)
        #expect(abs(vtDot(mid.forward, b.forward) - s) < 1e-5)
        #expect(vtIsOrthonormal(mid))
        #expect(approxEqual(mid.eye, (a.eye + b.eye) / 2, tol: 1e-5))
        #expect(abs(mid.fovY - 0.9) < 1e-6)
        #expect(abs(mid.distance - 3) < 1e-6)
    }

    @Test func frameKeepsOrientationAndCentersPivot() {
        var camera = ViewCamera.framing(vtBounds, upAxis: .negativeY)
        camera.orbit(deltaYaw: 0.7, deltaPitch: 0.2)
        let forward = camera.forward
        let other = Bounds(min: SIMD3(10, 10, 10), max: SIMD3(12, 14, 11))
        camera.frame(other)
        #expect(camera.pivot == other.center)
        #expect(camera.distance == OrbitCamera.framing(other, upAxis: .negativeY).distance)
        #expect(approxEqual(camera.forward, forward, tol: 1e-6))
    }

    @Test func zoomIsClamped() {
        var camera = ViewCamera.framing(vtBounds, upAxis: .negativeY)
        camera.zoom(factor: 1e9)
        #expect(camera.distance == camera.maxDistance)
        camera.zoom(factor: 1e-12)
        #expect(camera.distance == camera.minDistance)
    }

    @Test func setDepthKeepsEyeAndForwardAndClamps() {
        var camera = ViewCamera.framing(vtBounds, upAxis: .negativeY)
        camera.orbit(deltaYaw: 0.4, deltaPitch: 0.15)
        let eye = camera.eye
        let forward = camera.forward
        camera.setDepth(3.5)
        #expect(approxEqual(camera.eye, eye, tol: 1e-5))
        #expect(approxEqual(camera.forward, forward, tol: 1e-6))
        #expect(abs(camera.distance - 3.5) < 1e-5)
        #expect(approxEqual(camera.pivot, eye + forward * 3.5, tol: 1e-4))

        camera.setDepth(1e9)
        #expect(camera.distance == camera.maxDistance)
        camera.setDepth(-4)
        #expect(camera.distance == camera.minDistance)
        #expect(approxEqual(camera.eye, eye, tol: 1e-4))
    }

    @Test func lookAroundKeepsEyeAndOrthonormality() {
        for upAxis in UpAxis.allCases {
            var camera = vtRolled(ViewCamera.framing(vtBounds, upAxis: upAxis), by: 0.3)
            var rng = SeededGenerator(seed: 23)
            for _ in 0..<500 {
                let eye = camera.eye
                let distance = camera.distance
                camera.lookAround(deltaYaw: Float.random(in: -0.5...0.5, using: &rng),
                                  deltaPitch: Float.random(in: -0.5...0.5, using: &rng))
                #expect(approxEqual(camera.eye, eye, tol: 1e-4), "\(upAxis) eye moved")
                #expect(abs(camera.distance - distance) < 1e-6)
                #expect(vtIsOrthonormal(camera), "\(upAxis) lost orthonormality")
                #expect(abs(vtDot(camera.forward, upAxis.vector)) <= ViewCamera.maxUpAlignment + 1e-5)
            }
        }
    }

    @Test func lookAroundKeepsRoll() {
        let camera = vtRolled(ViewCamera.framing(vtBounds, upAxis: .negativeY), by: Float.pi / 8)
        let up = SIMD3<Float>(0, -1, 0)
        let rollBefore = asin(vtDot(camera.right, up))
        var looking = camera
        looking.lookAround(deltaYaw: 0.6, deltaPitch: 0.2)
        #expect(abs(asin(vtDot(looking.right, up)) - rollBefore) < 1e-4)
    }

    @Test func moveEyeTranslatesByExactAmounts() {
        for upAxis in UpAxis.allCases {
            var camera = vtRolled(ViewCamera.framing(vtBounds, upAxis: upAxis), by: 0.2)
            camera.orbit(deltaYaw: 0.3, deltaPitch: 0.1)
            let eye = camera.eye
            let pivot = camera.pivot
            let right = camera.right
            let forward = camera.forward
            let up = upAxis.vector
            let expected: SIMD3<Float> = right * 1.5 + up * -0.75 + forward * 2.0
            camera.moveEye(right: 1.5, up: -0.75, forward: 2)
            #expect(approxEqual(camera.eye, eye + expected, tol: 1e-4), "\(upAxis) eye")
            #expect(approxEqual(camera.pivot, pivot + expected, tol: 1e-4), "\(upAxis) pivot")
            #expect(approxEqual(camera.right, right, tol: 1e-6))
            #expect(approxEqual(camera.forward, forward, tol: 1e-6))
            #expect(vtIsOrthonormal(camera))
        }
    }

    @Test func orbitAroundPivotEqualsOrbit() {
        for upAxis in UpAxis.allCases {
            var reference = vtRolled(ViewCamera.framing(vtBounds, upAxis: upAxis), by: 0.4)
            var anchored = reference
            var rng = SeededGenerator(seed: 31)
            for _ in 0..<200 {
                let yaw = Float.random(in: -0.6...0.6, using: &rng)
                let pitch = Float.random(in: -0.6...0.6, using: &rng)
                reference.orbit(deltaYaw: yaw, deltaPitch: pitch)
                anchored.orbit(around: anchored.pivot, deltaYaw: yaw, deltaPitch: pitch)
                #expect(approxEqual(anchored.pivot, reference.pivot, tol: 1e-4), "\(upAxis) pivot")
                #expect(approxEqual(anchored.eye, reference.eye, tol: 1e-4), "\(upAxis) eye")
                #expect(approxEqual(anchored.forward, reference.forward, tol: 1e-5))
                #expect(approxEqual(anchored.right, reference.right, tol: 1e-5))
            }
        }
    }

    @Test func orbitAroundPointKeepsPointUnderPixelAndRoll() {
        for upAxis in UpAxis.allCases {
            var camera = vtRolled(ViewCamera.framing(vtBounds, upAxis: upAxis), by: 0.35)
            let point = SIMD3<Float>(0.3, -0.4, 0.2)
            let up = upAxis.vector
            let rollBefore = asin(vtDot(camera.right, up))
            let distance = camera.distance
            var rng = SeededGenerator(seed: 41)
            let startCamera: Camera = camera.camera(width: 640, height: 480)
            let before: SIMD3<Float> = startCamera.rotation * point + startCamera.translation
            for _ in 0..<300 {
                camera.orbit(around: point,
                             deltaYaw: Float.random(in: -0.4...0.4, using: &rng),
                             deltaPitch: Float.random(in: -0.4...0.4, using: &rng))
                let cam = camera.camera(width: 640, height: 480)
                let after: SIMD3<Float> = cam.rotation * point + cam.translation
                #expect(approxEqual(after, before, tol: 1e-3), "\(upAxis) point moved in camera space")
                #expect(vtIsOrthonormal(camera), "\(upAxis) lost orthonormality")
                #expect(abs(vtDot(camera.forward, up)) <= ViewCamera.maxUpAlignment + 1e-5)
                #expect(abs(camera.distance - distance) < 1e-6)
            }
            // Roll is kept by the rotation about the point, as for orbit.
            camera.orbit(around: point, deltaYaw: 0.5, deltaPitch: 0)
            #expect(abs(asin(vtDot(camera.right, up)) - rollBefore) < 1e-4)
        }
    }

    @Test func orbitAroundPointFromAPivotThatIsNotThePoint() {
        // The eye is unchanged by a pure orbit about the eye, but a point off the view axis moves on an arc
        // while staying at the same camera coordinates.
        var camera = ViewCamera.framing(vtBounds, upAxis: .negativeY)
        let point = camera.eye + camera.right * 1.2 + camera.forward * 3
        let before: SIMD3<Float> = camera.camera(width: 640, height: 480).rotation * point
            + camera.camera(width: 640, height: 480).translation
        let eyeBefore = camera.eye
        camera.orbit(around: point, deltaYaw: 0.5, deltaPitch: -0.2)
        let cam = camera.camera(width: 640, height: 480)
        #expect(approxEqual(cam.rotation * point + cam.translation, before, tol: 1e-4))
        #expect(!approxEqual(camera.eye, eyeBefore, tol: 1e-3), "the eye must move on the arc")
    }
}
