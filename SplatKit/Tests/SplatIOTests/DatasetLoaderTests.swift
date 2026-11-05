import Foundation
import Testing
import SplatCore
@testable import SplatIO

private let half = Double(0.5).squareRoot()

private let identityTransform: [Double] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

private func solidPPM(width: Int, height: Int, color: SIMD3<Float> = .zero) -> Data {
    let pixels = [SIMD3<Float>](repeating: color, count: width * height)
    return PPMCodec.encode(RGBImage(width: width, height: height, pixels: pixels))
}

private func pinhole(width: UInt64, height: UInt64,
                     fx: Double, fy: Double, cx: Double, cy: Double) -> FixtureCamera {
    FixtureCamera(id: 1, model: 1, width: width, height: height, params: [fx, fy, cx, cy])
}

private func identityImage(_ name: String) -> FixtureImage {
    FixtureImage(id: 1, rotation: SIMD4(1, 0, 0, 0), translation: .zero, cameraID: 1, name: name)
}

private func loadCOLMAP(cameras: [FixtureCamera],
                        images: [FixtureImage],
                        files: [String: Data],
                        maxImageSide: Int = 1600) throws(DatasetError) -> TrainingDataset {
    try TrainingDatasetLoader.loadCOLMAP(cameras: makeCamerasBinary(cameras),
                                         images: makeImagesBinary(images),
                                         points3D: nil,
                                         imageData: { files[$0] },
                                         maxImageSide: maxImageSide,
                                         decode: PPMCodec.decode)
}

private func arkitFrame(name: String = "f.ppm",
                        cameraToWorld: [Double] = identityTransform) -> ARKitCapture.Frame {
    ARKitCapture.Frame(image: name,
                       width: 64,
                       height: 48,
                       intrinsics: [40, 40, 32, 24],
                       cameraToWorld: cameraToWorld)
}

// MARK: - COLMAP

@Test func colmapPoseProjectsKnownPoint() throws {
    // 90° about y (world→camera), translation (0, 0, 3). R = [[0,0,1],[0,1,0],[-1,0,0]].
    // World (1, 0.5, 0) → camera (0, 0.5, 2) → pixel (32, 24 + 40·0.5/2) = (32, 34).
    let image = FixtureImage(id: 1,
                             rotation: SIMD4(half, 0, half, 0),
                             translation: SIMD3(0, 0, 3),
                             cameraID: 1,
                             name: "a.ppm")
    let dataset = try loadCOLMAP(cameras: [pinhole(width: 64, height: 48, fx: 40, fy: 40, cx: 32, cy: 24)],
                                 images: [image],
                                 files: ["a.ppm": solidPPM(width: 64, height: 48)])
    let camera = dataset.views[0].camera
    let world = SIMD3<Float>(1, 0.5, 0)
    let inCamera = camera.rotation * world + camera.translation
    let u = camera.fx * inCamera.x / inCamera.z + camera.cx
    let v = camera.fy * inCamera.y / inCamera.z + camera.cy
    #expect(abs(u - 32) < 1e-3)
    #expect(abs(v - 34) < 1e-3)
}

@Test func scalesIntrinsicsWhenImageFileIsSmaller() throws {
    let dataset = try loadCOLMAP(cameras: [pinhole(width: 64, height: 48, fx: 40, fy: 40, cx: 32, cy: 24)],
                                 images: [identityImage("a.ppm")],
                                 files: ["a.ppm": solidPPM(width: 32, height: 24)])
    let camera = dataset.views[0].camera
    #expect(camera.width == 32)
    #expect(camera.height == 24)
    #expect(camera.fx == 20)
    #expect(camera.fy == 20)
    #expect(camera.cx == 16)
    #expect(camera.cy == 12)
}

@Test func rejectsAspectMismatch() {
    #expect(throws: DatasetError.imageSizeMismatch(name: "a.ppm",
                                                   expected: SIMD2(64, 48),
                                                   actual: SIMD2(48, 48))) {
        try loadCOLMAP(cameras: [pinhole(width: 64, height: 48, fx: 40, fy: 40, cx: 32, cy: 24)],
                       images: [identityImage("a.ppm")],
                       files: ["a.ppm": solidPPM(width: 48, height: 48)])
    }
}

@Test func missingImagesAreReportedByName() {
    let camera = pinhole(width: 64, height: 48, fx: 40, fy: 40, cx: 32, cy: 24)
    let images = [identityImage("a.ppm"), identityImage("b.ppm"), identityImage("c.ppm")]
    let present = solidPPM(width: 64, height: 48)
    #expect(throws: DatasetError.missingImages(["b.ppm", "c.ppm"])) {
        try TrainingDatasetLoader.loadCOLMAP(cameras: makeCamerasBinary([camera]),
                                             images: makeImagesBinary(images),
                                             points3D: nil,
                                             imageData: { $0 == "a.ppm" ? present : nil },
                                             maxImageSide: 1600,
                                             decode: PPMCodec.decode)
    }
}

@Test func distortedCameraIsUndistorted() throws {
    // SIMPLE_RADIAL: f, cx, cy, k.
    let camera = FixtureCamera(id: 1, model: 2, width: 64, height: 48, params: [40, 32, 24, 0.2])
    var pixels: [SIMD3<Float>] = []
    for y in 0..<48 {
        for x in 0..<64 {
            pixels.append(SIMD3(Float(x) / 63, Float(y) / 47, 0.5))
        }
    }
    let raw = PPMCodec.encode(RGBImage(width: 64, height: 48, pixels: pixels))
    let dataset = try loadCOLMAP(cameras: [camera],
                                 images: [identityImage("a.ppm")],
                                 files: ["a.ppm": raw])
    let view = dataset.views[0]
    #expect(view.camera.fx == 40)
    #expect(view.camera.fy == 40)
    #expect(view.camera.cx == 32)
    #expect(view.camera.cy == 24)

    let decoded = try PPMCodec.decode(raw)
    let expected = Undistortion.undistort(decoded, model: .simpleRadial(f: 40, cx: 32, cy: 24, k: 0.2))
    #expect(view.image == expected)
    #expect(view.image != decoded)
}

@Test func largeImagesAreDownsampled() throws {
    let color = SIMD3<Float>(128.0 / 255.0, 64.0 / 255.0, 1)
    let dataset = try loadCOLMAP(cameras: [pinhole(width: 64, height: 32, fx: 40, fy: 40, cx: 32, cy: 16)],
                                 images: [identityImage("a.ppm")],
                                 files: ["a.ppm": solidPPM(width: 64, height: 32, color: color)],
                                 maxImageSide: 16)
    let view = dataset.views[0]
    #expect(view.image.width == 16)
    #expect(view.image.height == 8)
    #expect(view.camera.width == 16)
    #expect(view.camera.height == 8)
    #expect(view.camera.fx == 10)
    #expect(view.camera.fy == 10)
    #expect(view.camera.cx == 8)
    #expect(view.camera.cy == 4)
    for pixel in view.image.pixels {
        #expect(abs(pixel.x - color.x) < 1e-6)
        #expect(abs(pixel.y - color.y) < 1e-6)
        #expect(abs(pixel.z - color.z) < 1e-6)
    }
}

// MARK: - ARKit

@Test func arkitIdentityLooksDownNegativeZ() throws {
    let camera = try ARKitCapture.camera(for: arkitFrame())
    func project(_ point: SIMD3<Float>) -> SIMD2<Float> {
        let inCamera = camera.rotation * point + camera.translation
        return SIMD2(camera.fx * inCamera.x / inCamera.z + camera.cx,
                     camera.fy * inCamera.y / inCamera.z + camera.cy)
    }
    let center = project(SIMD3(0, 0, -2))
    #expect(abs(center.x - 32) < 1e-3)
    #expect(abs(center.y - 24) < 1e-3)
    // Camera up is world +y, so a point above the optical axis has a smaller image row.
    let up = project(SIMD3(0, 1, -2))
    #expect(up.y < center.y)
    #expect(abs(up.y - 4) < 1e-3)
}

@Test func arkitRejectsNonOrthonormalPose() {
    var scaledColumn = identityTransform
    scaledColumn[0] = 2
    #expect(throws: DatasetError.invalidPose(name: "f.ppm")) {
        try ARKitCapture.camera(for: arkitFrame(cameraToWorld: scaledColumn))
    }

    var badLastRow = identityTransform
    badLastRow[15] = 2
    #expect(throws: DatasetError.invalidPose(name: "f.ppm")) {
        try ARKitCapture.camera(for: arkitFrame(cameraToWorld: badLastRow))
    }

    var nonFinite = identityTransform
    nonFinite[12] = .nan
    #expect(throws: DatasetError.invalidPose(name: "f.ppm")) {
        try ARKitCapture.camera(for: arkitFrame(cameraToWorld: nonFinite))
    }
}

@Test func arkitLoaderSortsViewsByName() throws {
    let capture = ARKitCapture(version: 1, frames: [arkitFrame(name: "b.ppm"), arkitFrame(name: "a.ppm")])
    let data = try JSONEncoder().encode(capture)
    let files = ["a.ppm": solidPPM(width: 64, height: 48), "b.ppm": solidPPM(width: 64, height: 48)]
    let dataset = try TrainingDatasetLoader.loadARKit(capture: data,
                                                      imageData: { files[$0] },
                                                      maxImageSide: 1600,
                                                      decode: PPMCodec.decode)
    #expect(dataset.views.map(\.name) == ["a.ppm", "b.ppm"])
    #expect(dataset.points == nil)
}

// MARK: - PPM

@Test func ppmRoundTrip() throws {
    // Explicit steps: Xcode's type checker times out on the one-expression form.
    var pixels: [SIMD3<Float>] = []
    for index in 0..<6 {
        let r: Float = Float(index * 40) / 255
        let g: Float = Float(255 - index * 40) / 255
        let b: Float = Float(index) / 255
        pixels.append(SIMD3<Float>(r, g, b))
    }
    let image = RGBImage(width: 3, height: 2, pixels: pixels)
    let decoded = try PPMCodec.decode(PPMCodec.encode(image))
    #expect(decoded == image)
}

@Test func ppmAcceptsHeaderComments() throws {
    let data = Data("P6\n# made by a test\n1 1\n255\n".utf8) + Data([255, 0, 0])
    let image = try PPMCodec.decode(data)
    #expect(image.width == 1)
    #expect(image.height == 1)
    #expect(image.pixels == [SIMD3(1, 0, 0)])
}

@Test func ppmRejectsMalformedInput() {
    #expect(throws: PPMError.invalidHeader) {
        try PPMCodec.decode(Data("P5\n1 1\n255\n".utf8) + Data([0, 0, 0]))
    }
    #expect(throws: PPMError.unsupportedMaxValue(65535)) {
        try PPMCodec.decode(Data("P6\n1 1\n65535\n".utf8) + Data([0, 0, 0, 0, 0, 0]))
    }
    #expect(throws: PPMError.truncated) {
        try PPMCodec.decode(Data("P6\n2 2\n255\n".utf8) + Data([0, 0, 0]))
    }
}
