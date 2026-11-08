import Foundation
import Testing
import SplatCore

// MARK: - Helpers

/// Metrics with a given sharpness and exposure. The signature is a deterministic pseudo-random pattern per `seed`,
/// so photos with different seeds do not correlate and are never flagged as duplicates by accident.
private func makeMetrics(sharpness: Double,
                         mean: Double = 0.5,
                         shadows: Double = 0.01,
                         highlights: Double = 0.01,
                         seed: UInt64) -> PhotoMetrics {
    var state: UInt64 = seed &* 2654435761 &+ 1
    var signature: [Double] = []
    signature.reserveCapacity(64)
    for _ in 0..<64 {
        state = (state &* 1103515245 &+ 12345) % 2147483648
        let value: Double = Double(state) / 2147483648.0
        signature.append(value)
    }
    return PhotoMetrics(sharpness: sharpness,
                        meanLuminance: mean,
                        shadowFraction: shadows,
                        highlightFraction: highlights,
                        signature: signature)
}

/// A camera at `centre` looking along the rotation's third row. The translation is `-R·C`, so the centre is `C`.
private func makeCamera(centre: SIMD3<Float>, rotation: Matrix3 = Matrix3.identity) -> Camera {
    let rotated: SIMD3<Float> = rotation * centre
    return Camera(width: 4, height: 4, fx: 1, fy: 1, cx: 0, cy: 0,
                  rotation: rotation, translation: -rotated)
}

/// Rotation about the y axis by `degrees`; its viewing direction is `(sin, 0, cos)`.
private func yRotation(degrees: Float) -> Matrix3 {
    let radians: Float = degrees * Float.pi / 180
    let c: Float = cos(radians)
    let s: Float = sin(radians)
    return Matrix3(rows: [SIMD3<Float>(c, 0, -s), SIMD3<Float>(0, 1, 0), SIMD3<Float>(s, 0, c)])
}

/// A grey image from a per-pixel value function (explicit loops keep the type checker happy).
private func makeImage(width: Int, height: Int, value: (Int, Int) -> Float) -> RGBImage {
    var pixels: [SIMD3<Float>] = []
    pixels.reserveCapacity(width * height)
    for y in 0..<height {
        for x in 0..<width {
            let v: Float = value(x, y)
            pixels.append(SIMD3<Float>(v, v, v))
        }
    }
    return RGBImage(width: width, height: height, pixels: pixels)
}

// MARK: - Measurements

@Test func checkerboardIsSharperThanSmoothGradient() {
    let checker: RGBImage = makeImage(width: 64, height: 64) { (x: Int, y: Int) -> Float in
        ((x / 2 + y / 2) % 2 == 0) ? 1 : 0
    }
    let smooth: RGBImage = makeImage(width: 64, height: 64) { (x: Int, _: Int) -> Float in
        Float(x) / 63
    }
    let sharp: PhotoMetrics = PhotoQuality.measure(checker)
    let blurred: PhotoMetrics = PhotoQuality.measure(smooth)
    #expect(sharp.sharpness > 0.05)
    #expect(blurred.sharpness < 1e-6)
    #expect(sharp.sharpness > 100 * blurred.sharpness)
}

@Test func uniformMidGreyKeepsItsLuminanceAfterDownscaling() {
    let image: RGBImage = makeImage(width: 1024, height: 512) { (_: Int, _: Int) -> Float in 0.5 }
    let metrics: PhotoMetrics = PhotoQuality.measure(image)
    #expect(abs(metrics.meanLuminance - 0.5) < 1e-6)
    #expect(metrics.shadowFraction == 0)
    #expect(metrics.highlightFraction == 0)
    #expect(metrics.signature.count == PhotoQuality.signatureSide * PhotoQuality.signatureSide)
}

@Test func blackAndWhiteImagesReportClippedFractions() {
    let black: RGBImage = makeImage(width: 16, height: 16) { (_: Int, _: Int) -> Float in 0 }
    let white: RGBImage = makeImage(width: 16, height: 16) { (_: Int, _: Int) -> Float in 1 }
    #expect(PhotoQuality.measure(black).shadowFraction == 1)
    #expect(PhotoQuality.measure(black).meanLuminance == 0)
    #expect(PhotoQuality.measure(white).highlightFraction == 1)
    #expect(abs(PhotoQuality.measure(white).meanLuminance - 1) < 1e-9)
}

@Test func emptyImageMeasuresAsZeros() {
    let empty: RGBImage = RGBImage(width: 0, height: 0, pixels: [])
    let metrics: PhotoMetrics = PhotoQuality.measure(empty)
    #expect(metrics.sharpness == 0)
    #expect(metrics.meanLuminance == 0)
    #expect(metrics.signature.allSatisfy { $0 == 0 })
}

// MARK: - Exposure

@Test func darkAndBrightPhotosAreFlagged() {
    let inputs: [PhotoInput] = [
        PhotoInput(name: "ok.jpg", metrics: makeMetrics(sharpness: 10, seed: 1)),
        PhotoInput(name: "dark.jpg", metrics: makeMetrics(sharpness: 10, mean: 0.05, shadows: 0.7, seed: 2)),
        PhotoInput(name: "bright.jpg", metrics: makeMetrics(sharpness: 10, mean: 0.95, highlights: 0.8, seed: 3))
    ]
    let verdicts: [PhotoVerdict] = PhotoTriage.assess(images: inputs)
    #expect(verdicts.map(\.kind) == [.ok, .tooDark, .tooBright])
    #expect(verdicts[1].reason.hasPrefix("Too dark"))
    #expect(verdicts[2].reason.hasPrefix("Too bright"))
}

// MARK: - Blur

@Test func blurIsRelativeToTheSet() {
    let inputs: [PhotoInput] = [
        PhotoInput(name: "a.jpg", metrics: makeMetrics(sharpness: 100, seed: 1)),
        PhotoInput(name: "b.jpg", metrics: makeMetrics(sharpness: 105, seed: 2)),
        PhotoInput(name: "c.jpg", metrics: makeMetrics(sharpness: 98, seed: 3)),
        PhotoInput(name: "d.jpg", metrics: makeMetrics(sharpness: 102, seed: 4)),
        PhotoInput(name: "e.jpg", metrics: makeMetrics(sharpness: 4, seed: 5))
    ]
    let verdicts: [PhotoVerdict] = PhotoTriage.assess(images: inputs)
    #expect(verdicts.map(\.kind) == [.ok, .ok, .ok, .ok, .blurry])
}

@Test func uniformlySharpSetFlagsNothingAsBlurry() {
    let inputs: [PhotoInput] = (0..<6).map { (index: Int) -> PhotoInput in
        PhotoInput(name: "p\(index).jpg", metrics: makeMetrics(sharpness: 50 + Double(index) * 0.01,
                                                              seed: UInt64(index + 1)))
    }
    let verdicts: [PhotoVerdict] = PhotoTriage.assess(images: inputs)
    #expect(verdicts.allSatisfy { $0.isOK })
}

@Test func smallSetHasNoBlurVerdict() {
    let inputs: [PhotoInput] = [
        PhotoInput(name: "a.jpg", metrics: makeMetrics(sharpness: 100, seed: 1)),
        PhotoInput(name: "b.jpg", metrics: makeMetrics(sharpness: 100, seed: 2)),
        PhotoInput(name: "c.jpg", metrics: makeMetrics(sharpness: 1, seed: 3))
    ]
    let verdicts: [PhotoVerdict] = PhotoTriage.assess(images: inputs)
    #expect(verdicts.allSatisfy { $0.isOK })
}

// MARK: - Redundancy without poses

@Test func identicalSignaturesAreDuplicatesOfTheEarlierPhoto() {
    let shared: PhotoMetrics = makeMetrics(sharpness: 50, seed: 7)
    let inputs: [PhotoInput] = [
        PhotoInput(name: "first.jpg", metrics: shared),
        PhotoInput(name: "second.jpg", metrics: shared),
        PhotoInput(name: "other.jpg", metrics: makeMetrics(sharpness: 50, seed: 8))
    ]
    let verdicts: [PhotoVerdict] = PhotoTriage.assess(images: inputs)
    #expect(verdicts[0].kind == .ok)
    #expect(verdicts[1].kind == .duplicate(of: "first.jpg"))
    #expect(verdicts[2].kind == .ok)
    #expect(verdicts[1].reason == "Nearly the same view as first.jpg.")
}

@Test func blurryPhotoIsNotUsedAsTheOriginalOfADuplicate() {
    let sharp: PhotoMetrics = makeMetrics(sharpness: 100, seed: 1)
    let blurryCopy: PhotoMetrics = makeMetrics(sharpness: 2, seed: 1)
    let inputs: [PhotoInput] = [
        PhotoInput(name: "blurry.jpg", metrics: blurryCopy),
        PhotoInput(name: "sharp.jpg", metrics: sharp),
        PhotoInput(name: "b.jpg", metrics: makeMetrics(sharpness: 100, seed: 2)),
        PhotoInput(name: "c.jpg", metrics: makeMetrics(sharpness: 100, seed: 3)),
        PhotoInput(name: "d.jpg", metrics: makeMetrics(sharpness: 100, seed: 4))
    ]
    let verdicts: [PhotoVerdict] = PhotoTriage.assess(images: inputs)
    #expect(verdicts[0].kind == .blurry)
    #expect(verdicts[1].kind == .ok)
}

// MARK: - Redundancy with poses

@Test func nearbyCameraWithSameViewIsADuplicate() {
    // Cameras on a 1-unit line; the second photo sits 0.005 units from the first, same view.
    let positions: [SIMD3<Float>] = [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(2, 0, 0), SIMD3(3, 0, 0),
                                     SIMD3(0.005, 0, 0)]
    let names: [String] = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "a-copy.jpg"]
    var inputs: [PhotoInput] = []
    var cameras: [String: Camera] = [:]
    for index in names.indices {
        inputs.append(PhotoInput(name: names[index], metrics: makeMetrics(sharpness: 50, seed: UInt64(index + 1))))
        cameras[names[index]] = makeCamera(centre: positions[index])
    }
    let verdicts: [PhotoVerdict] = PhotoTriage.assess(images: inputs, cameras: cameras)
    #expect(verdicts[4].kind == .duplicate(of: "a.jpg"))
    #expect(verdicts[0...3].allSatisfy { $0.isOK })
}

@Test func sameCentreWithDifferentViewIsNotADuplicate() {
    let positions: [SIMD3<Float>] = [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(2, 0, 0), SIMD3(3, 0, 0)]
    var inputs: [PhotoInput] = []
    var cameras: [String: Camera] = [:]
    for index in positions.indices {
        let name: String = "p\(index).jpg"
        inputs.append(PhotoInput(name: name, metrics: makeMetrics(sharpness: 50, seed: UInt64(index + 1))))
        cameras[name] = makeCamera(centre: positions[index])
    }
    // The same centre as p0, but turned by 10 degrees, which is beyond the 3 degree limit.
    inputs.append(PhotoInput(name: "turned.jpg", metrics: makeMetrics(sharpness: 50, seed: 9)))
    cameras["turned.jpg"] = makeCamera(centre: SIMD3(0, 0, 0), rotation: yRotation(degrees: 10))
    let verdicts: [PhotoVerdict] = PhotoTriage.assess(images: inputs, cameras: cameras)
    #expect(verdicts.allSatisfy { $0.isOK })
}

@Test func photoWithPoseComparedAgainstPhotoWithoutPoseUsesSignatures() {
    // The posed photo and its unposed twin share a signature, so the signature rule flags the twin.
    let shared: PhotoMetrics = makeMetrics(sharpness: 50, seed: 11)
    let inputs: [PhotoInput] = [
        PhotoInput(name: "posed.jpg", metrics: shared),
        PhotoInput(name: "unposed.jpg", metrics: shared),
        PhotoInput(name: "far.jpg", metrics: makeMetrics(sharpness: 50, seed: 12))
    ]
    let cameras: [String: Camera] = [
        "posed.jpg": makeCamera(centre: SIMD3(0, 0, 0)),
        "far.jpg": makeCamera(centre: SIMD3(5, 0, 0))
    ]
    let verdicts: [PhotoVerdict] = PhotoTriage.assess(images: inputs, cameras: cameras)
    #expect(verdicts[1].kind == .duplicate(of: "posed.jpg"))
    #expect(verdicts[2].kind == .ok)
}

// MARK: - Output shape

@Test func verdictsKeepInputOrderAndCount() {
    let inputs: [PhotoInput] = (0..<7).map { (index: Int) -> PhotoInput in
        PhotoInput(name: "img\(index).jpg", metrics: makeMetrics(sharpness: 40, seed: UInt64(index + 20)))
    }
    let verdicts: [PhotoVerdict] = PhotoTriage.assess(images: inputs)
    #expect(verdicts.map(\.name) == inputs.map(\.name))
    #expect(PhotoTriage.assess(images: []).isEmpty)
}
