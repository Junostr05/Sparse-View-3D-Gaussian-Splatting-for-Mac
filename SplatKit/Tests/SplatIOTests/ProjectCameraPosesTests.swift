import Foundation
import Testing
import SplatCore
@testable import SplatIO

private let identityTransform: [Double] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

/// A PPM whose size is the camera's own size, so the stub decoder (`PPMCodec.decode`) returns an
/// image that needs no rescaling.
private func ppm(width: Int, height: Int) -> Data {
    let pixels = [SIMD3<Float>](repeating: SIMD3(0.5, 0.25, 0.75), count: width * height)
    return PPMCodec.encode(RGBImage(width: width, height: height, pixels: pixels))
}

/// Largest element-wise difference between two rotations, measured on the basis vectors.
private func maxRotationDifference(_ a: Matrix3, _ b: Matrix3) -> Float {
    let basis: [SIMD3<Float>] = [SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1)]
    var largest: Float = 0
    for v in basis {
        let difference: SIMD3<Float> = a * v - b * v
        largest = max(largest, abs(difference.x), abs(difference.y), abs(difference.z))
    }
    return largest
}

/// Checks that a pose-only camera matches the camera a full dataset load produced for the same image.
private func expectSamePose(_ pose: Camera, _ view: Camera, name: String) {
    #expect(maxRotationDifference(pose.rotation, view.rotation) < 1e-5, "rotation of \(name)")
    let translationDifference: SIMD3<Float> = pose.translation - view.translation
    #expect(abs(translationDifference.x) < 1e-5 && abs(translationDifference.y) < 1e-5
            && abs(translationDifference.z) < 1e-5, "translation of \(name)")
    #expect(abs(pose.fx - view.fx) < 1e-5, "fx of \(name)")
    #expect(abs(pose.fy - view.fy) < 1e-5, "fy of \(name)")
    #expect(pose.width == view.width, "width of \(name)")
    #expect(pose.height == view.height, "height of \(name)")
}

/// COLMAP project with two cameras of different size. Image names are not in sorted order.
private func colmapProject() -> ProjectContents {
    let cameras = makeCamerasBinary([
        FixtureCamera(id: 1, model: 1, width: 64, height: 48, params: [40, 40, 32, 24]),
        FixtureCamera(id: 2, model: 1, width: 32, height: 24, params: [20, 22, 16, 12]),
    ])
    // 90° about y for "c.ppm": R = [[0,0,1],[0,1,0],[-1,0,0]].
    let half = Double(0.5).squareRoot()
    let images = makeImagesBinary([
        FixtureImage(id: 1, rotation: SIMD4(half, 0, half, 0), translation: SIMD3(0, 0, 3),
                     cameraID: 1, name: "c.ppm"),
        FixtureImage(id: 2, rotation: SIMD4(1, 0, 0, 0), translation: SIMD3(0, 0, 1),
                     cameraID: 2, name: "a.ppm"),
        FixtureImage(id: 3, rotation: SIMD4(1, 0, 0, 0), translation: SIMD3(0.1, -0.2, 2),
                     cameraID: 1, name: "b.ppm"),
    ])
    let imageFiles: [String: Data] = [
        "a.ppm": ppm(width: 32, height: 24),
        "b.ppm": ppm(width: 64, height: 48),
        "c.ppm": ppm(width: 64, height: 48),
    ]
    let empty = ProjectContents(manifest: ProjectManifest(), files: [:])
    return ProjectBundle.importCOLMAP(cameras: cameras,
                                      images: images,
                                      points3D: nil,
                                      imageFiles: imageFiles,
                                      into: empty)
}

/// ARKit project with two frames of different size, listed out of name order.
private func arkitProject() throws -> ProjectContents {
    // Translation (0.5, 0, 1) in the last column, identity rotation.
    let translated: [Double] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.5, 0, 1, 1]
    let frames = [
        ARKitCapture.Frame(image: "z.ppm", width: 64, height: 48,
                           intrinsics: [40, 41, 32, 24], cameraToWorld: translated),
        ARKitCapture.Frame(image: "m.ppm", width: 32, height: 24,
                           intrinsics: [20, 20, 16, 12], cameraToWorld: identityTransform),
    ]
    let capture = try JSONEncoder().encode(ARKitCapture(version: 1, frames: frames))
    let imageFiles: [String: Data] = [
        "m.ppm": ppm(width: 32, height: 24),
        "z.ppm": ppm(width: 64, height: 48),
    ]
    let empty = ProjectContents(manifest: ProjectManifest(), files: [:])
    return ProjectBundle.importARKit(capture: capture, imageFiles: imageFiles, into: empty)
}

// MARK: - Project pose API

@Test func colmapProjectCameraPosesMatchDatasetViews() throws {
    let contents = colmapProject()
    let poses = try ProjectBundle.cameraPoses(contents)
    let dataset = try ProjectBundle.loadDataset(contents, decode: PPMCodec.decode)

    #expect(poses.map(\.name) == ["a.ppm", "b.ppm", "c.ppm"])
    #expect(poses.count == dataset.views.count)
    for (pose, view) in zip(poses, dataset.views) {
        #expect(pose.name == view.name)
        expectSamePose(pose.camera, view.camera, name: pose.name)
    }
}

@Test func arkitProjectCameraPosesMatchDatasetViews() throws {
    let contents = try arkitProject()
    let poses = try ProjectBundle.cameraPoses(contents)
    let dataset = try ProjectBundle.loadDataset(contents, decode: PPMCodec.decode)

    #expect(poses.map(\.name) == ["m.ppm", "z.ppm"])
    #expect(poses.count == dataset.views.count)
    for (pose, view) in zip(poses, dataset.views) {
        #expect(pose.name == view.name)
        expectSamePose(pose.camera, view.camera, name: pose.name)
    }
}

@Test func projectWithoutSourceHasNoCameraPoses() throws {
    let empty = ProjectContents(manifest: ProjectManifest(), files: [:])
    #expect(try ProjectBundle.cameraPoses(empty).isEmpty)
}

// MARK: - Loader entry points

@Test func colmapCamerasWithoutImagesIsEmptyNotAnError() throws {
    let cameras = makeCamerasBinary([
        FixtureCamera(id: 1, model: 1, width: 64, height: 48, params: [40, 40, 32, 24]),
    ])
    let images = makeImagesBinary([])
    let poses = try TrainingDatasetLoader.cameras(colmapCameras: cameras, colmapImages: images)
    #expect(poses.isEmpty)
}

@Test func colmapCamerasKeepOwnSizeWithoutImageFiles() throws {
    // The image file would be 32x24, but no image is read: the camera keeps the size from cameras.bin.
    let cameras = makeCamerasBinary([
        FixtureCamera(id: 1, model: 1, width: 64, height: 48, params: [40, 40, 32, 24]),
    ])
    let images = makeImagesBinary([
        FixtureImage(id: 1, rotation: SIMD4(1, 0, 0, 0), translation: .zero, cameraID: 1, name: "a.ppm"),
    ])
    let poses = try TrainingDatasetLoader.cameras(colmapCameras: cameras, colmapImages: images)
    #expect(poses.count == 1)
    #expect(poses[0].camera.width == 64)
    #expect(poses[0].camera.height == 48)
    #expect(poses[0].camera.fx == 40)
}
