import Foundation
import Testing
import SplatCore

private func gradientImage(width: Int, height: Int) -> RGBImage {
    var pixels: [SIMD3<Float>] = []
    pixels.reserveCapacity(width * height)
    for y in 0..<height {
        for x in 0..<width {
            let r = Float(x) / Float(width)
            let g = Float(y) / Float(height)
            let b = Float((x + y) % 3) / 2
            pixels.append(SIMD3(r, g, b))
        }
    }
    return RGBImage(width: width, height: height, pixels: pixels)
}

@Test func zeroDistortionIsIdentity() {
    let image = gradientImage(width: 8, height: 6)
    let model = CameraModel.opencv(fx: 50, fy: 50, cx: 4, cy: 3,
                                   k1: 0, k2: 0, p1: 0, p2: 0)
    #expect(!model.hasDistortion)
    let out = Undistortion.undistort(image, model: model)
    #expect(out.width == image.width)
    #expect(out.height == image.height)
    for i in 0..<image.pixels.count {
        #expect(approxEqual(out.pixels[i], image.pixels[i], tol: 1e-6))
    }
}

@Test func simpleRadialMovesPointsOutward() {
    let model = CameraModel.simpleRadial(f: 1, cx: 0, cy: 0, k: 0.1)
    let d = model.distort(SIMD2(0.5, 0))
    #expect(abs(d.x - 0.5125) < 1e-12)
    #expect(abs(d.y) < 1e-12)
}

@Test func opencvTangentialTerms() {
    // r² = 0.13, uv = -0.06, radial term = 0 (k1 = k2 = 0).
    // du = 2·p1·uv + p2·(r² + 2u²) = 0.005, dv = 2·p2·uv + p1·(r² + 2v²) = -0.0003.
    let model = CameraModel.opencv(fx: 1, fy: 1, cx: 0, cy: 0,
                                   k1: 0, k2: 0, p1: 0.01, p2: 0.02)
    #expect(model.hasDistortion)
    let d = model.distort(SIMD2(0.3, -0.2))
    #expect(abs(d.x - 0.305) < 1e-12)
    #expect(abs(d.y - (-0.2003)) < 1e-12)
}

@Test func undistortMovesEdgeToUndistortedLocation() throws {
    // Distorted source: pixels with index x >= 40 are white, the rest black, so the
    // distorted edge sits at continuous column 40. Row 24 has v = 0 exactly (cy = 24.5),
    // so the edge can be located analytically from the 1-D radial equation u(1 + k·u²) = d.
    let f = 100.0
    let cx = 32.0
    let cy = 24.5
    let k = 0.5
    let model = CameraModel.simpleRadial(f: f, cx: cx, cy: cy, k: k)
    let w = 64
    let h = 48
    var pixels = [SIMD3<Float>](repeating: .zero, count: w * h)
    for y in 0..<h {
        for x in 40..<w {
            pixels[y * w + x] = SIMD3(repeating: 1)
        }
    }
    let source = RGBImage(width: w, height: h, pixels: pixels)
    let out = Undistortion.undistort(source, model: model)

    let target = (40.0 - cx) / f
    var lo = 0.0
    var hi = 1.0
    for _ in 0..<100 {
        let mid = 0.5 * (lo + hi)
        if mid + k * mid * mid * mid < target { lo = mid } else { hi = mid }
    }
    let undistortedEdge = f * lo + cx          // continuous column of the undistorted edge
    let expectedIndex = undistortedEdge - 0.5  // index where the value crosses 0.5

    var crossing: Double? = nil
    for x in 0..<(w - 1) {
        let a = Double(out.pixels[24 * w + x].x)
        let b = Double(out.pixels[24 * w + x + 1].x)
        if a < 0.5 && b >= 0.5 {
            crossing = Double(x) + (0.5 - a) / (b - a)
            break
        }
    }
    let found = try #require(crossing)
    #expect(abs(found - expectedIndex) < 0.5)
    // The undistorted edge must move inward compared with the distorted one (column 39.5).
    #expect(found < 39.5)
}

@Test func downsampleAveragesBlocks() {
    let image = RGBImage(width: 4, height: 2, pixels: [
        SIMD3(repeating: 1), SIMD3(repeating: 2), SIMD3(repeating: 3), SIMD3(repeating: 4),
        SIMD3(repeating: 5), SIMD3(repeating: 6), SIMD3(repeating: 7), SIMD3(repeating: 8),
    ])
    let out = image.downsampled(by: 2)
    #expect(out.width == 2)
    #expect(out.height == 1)
    #expect(approxEqual(out.pixels[0], SIMD3(repeating: 3.5), tol: 1e-6))
    #expect(approxEqual(out.pixels[1], SIMD3(repeating: 5.5), tol: 1e-6))
}

@Test func bilinearSampleClampsToEdge() {
    let image = RGBImage(width: 2, height: 1, pixels: [SIMD3(1, 0, 0), SIMD3(0, 1, 0)])
    #expect(approxEqual(image.bilinearSample(x: -3, y: 0), SIMD3(1, 0, 0), tol: 1e-6))
    #expect(approxEqual(image.bilinearSample(x: 5, y: 0), SIMD3(0, 1, 0), tol: 1e-6))
    #expect(approxEqual(image.bilinearSample(x: 0.5, y: 0), SIMD3(0.5, 0.5, 0), tol: 1e-6))
}

@Test func downscaledCameraProjectsConsistently() {
    let rotation = Matrix3.rotation(quaternion: SIMD4(0.9238795, 0, 0.3826834, 0))
    let translation = SIMD3<Float>(0.2, -0.1, 3)
    let full = Camera(width: 640, height: 480, fx: 500, fy: 510, cx: 320, cy: 240,
                      rotation: rotation, translation: translation)
    let half = full.downscaled(by: 2)
    #expect(half.width == 320)
    #expect(half.height == 240)

    let world = SIMD3<Float>(0.4, -0.3, 1.5)
    let pc: SIMD3<Float> = full.rotation * world + full.translation
    let pFull = SIMD2<Float>(full.fx * pc.x / pc.z + full.cx, full.fy * pc.y / pc.z + full.cy)
    let pc2: SIMD3<Float> = half.rotation * world + half.translation
    let pHalf = SIMD2<Float>(half.fx * pc2.x / pc2.z + half.cx, half.fy * pc2.y / pc2.z + half.cy)
    #expect(approxEqual(pHalf, pFull / 2, tol: 1e-4))
}

@Test func rescaledCameraScalesIntrinsics() {
    let camera = Camera(width: 640, height: 480, fx: 500, fy: 400, cx: 320, cy: 240,
                        rotation: .identity, translation: .zero)
    let out = camera.rescaled(toWidth: 320, height: 240)
    #expect(out.width == 320)
    #expect(out.height == 240)
    #expect(abs(out.fx - 250) < 1e-4)
    #expect(abs(out.cx - 160) < 1e-4)
    #expect(abs(out.fy - 200) < 1e-4)
    #expect(abs(out.cy - 120) < 1e-4)
}

@Test func cameraFromPinholeModelKeepsIntrinsics() {
    let camera = Camera(model: .pinhole(fx: 500, fy: 480, cx: 320, cy: 240),
                        width: 640, height: 480, rotation: .identity, translation: .zero)
    #expect(camera.fx == 500)
    #expect(camera.fy == 480)
    #expect(camera.cx == 320)
    #expect(camera.cy == 240)
    #expect(camera.width == 640)
    #expect(camera.height == 480)

    let simple = CameraModel.simplePinhole(f: 300, cx: 10, cy: 20)
    #expect(simple.focal == SIMD2(300, 300))
    #expect(simple.principalPoint == SIMD2(10, 20))
    #expect(!simple.hasDistortion)
    #expect(CameraModel.simpleRadial(f: 1, cx: 0, cy: 0, k: 0).hasDistortion == false)
    #expect(CameraModel.radial(f: 1, cx: 0, cy: 0, k1: 0, k2: 0.1).hasDistortion)
}

@Test func downsampleFactorForMaxSide() {
    #expect(ImageResampling.downsampleFactor(width: 4000, height: 3000, maxSide: 1600) == 3)
    #expect(ImageResampling.downsampleFactor(width: 1600, height: 900, maxSide: 1600) == 1)
    #expect(ImageResampling.downsampleFactor(width: 800, height: 600, maxSide: 1600) == 1)
    #expect(ImageResampling.downsampleFactor(width: 3200, height: 100, maxSide: 1600) == 2)
}
