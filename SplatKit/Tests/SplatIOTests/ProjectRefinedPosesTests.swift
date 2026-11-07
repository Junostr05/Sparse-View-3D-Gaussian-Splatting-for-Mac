import Foundation
import Testing
import SplatCore
@testable import SplatIO

/// A PPM of the camera's own size, so the stub decoder returns an image that needs no rescaling.
private func refinedPPM(width: Int, height: Int) -> Data {
    let pixels = [SIMD3<Float>](repeating: SIMD3(0.5, 0.25, 0.75), count: width * height)
    return PPMCodec.encode(RGBImage(width: width, height: height, pixels: pixels))
}

/// Largest element-wise difference between two rotations, over their three columns.
private func refinedRotationDifference(_ a: Matrix3, _ b: Matrix3) -> Float {
    let columns: [SIMD3<Float>] = [a.c0 - b.c0, a.c1 - b.c1, a.c2 - b.c2]
    var largest: Float = 0
    for column in columns {
        largest = max(largest, abs(column.x), abs(column.y), abs(column.z))
    }
    return largest
}

/// Largest element-wise difference between two translations.
private func refinedTranslationDifference(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
    let difference: SIMD3<Float> = a - b
    return max(abs(difference.x), abs(difference.y), abs(difference.z))
}

private struct RefinedColmapInput {
    let cameras: Data
    let images: Data
    let imageFiles: [String: Data]
}

/// Three 32x24 views with one shared camera. The image records are not in name order.
private func refinedColmapInput() -> RefinedColmapInput {
    let cameras = makeCamerasBinary([
        FixtureCamera(id: 1, model: 1, width: 32, height: 24, params: [20, 20, 16, 12]),
    ])
    let half = Double(0.5).squareRoot()
    let images = makeImagesBinary([
        FixtureImage(id: 1, rotation: SIMD4(1, 0, 0, 0), translation: SIMD3(0, 0, 1),
                     cameraID: 1, name: "b.ppm"),
        FixtureImage(id: 2, rotation: SIMD4(half, 0, half, 0), translation: SIMD3(0.1, -0.2, 3),
                     cameraID: 1, name: "a.ppm"),
        FixtureImage(id: 3, rotation: SIMD4(1, 0, 0, 0), translation: SIMD3(0, 0.5, 2),
                     cameraID: 1, name: "c.ppm"),
    ])
    let imageFiles: [String: Data] = [
        "a.ppm": refinedPPM(width: 32, height: 24),
        "b.ppm": refinedPPM(width: 32, height: 24),
        "c.ppm": refinedPPM(width: 32, height: 24),
    ]
    return RefinedColmapInput(cameras: cameras, images: images, imageFiles: imageFiles)
}

private func refinedColmapProject() -> ProjectContents {
    let input = refinedColmapInput()
    let empty = ProjectContents(manifest: ProjectManifest(), files: [:])
    return ProjectBundle.importCOLMAP(cameras: input.cameras,
                                      images: input.images,
                                      points3D: nil,
                                      imageFiles: input.imageFiles,
                                      into: empty)
}

/// A refined camera for `name`: a rotation about two axes and a translation that differs from the input.
private func refinedCamera(name: String, translation: SIMD3<Float>) -> NamedCamera {
    let quaternion: SIMD4<Float> = SIMD4(0.8, 0, 0.6, 0)
    let camera = Camera(width: 32, height: 24, fx: 20, fy: 20, cx: 16, cy: 12,
                        rotation: Matrix3.rotation(quaternion: quaternion),
                        translation: translation)
    return NamedCamera(name: name, camera: camera)
}

/// Refined poses for "a.ppm" and "b.ppm" only, so "c.ppm" keeps its original pose.
private func refinedPairs() -> [NamedCamera] {
    [
        refinedCamera(name: "a.ppm", translation: SIMD3(0.4, -0.1, 2.5)),
        refinedCamera(name: "b.ppm", translation: SIMD3(-0.3, 0.2, 1.5)),
    ]
}

// MARK: - Storage

@Test func refinedPosesRoundTripThroughThePackage() throws {
    let stored = ProjectBundle.withRefinedPoses(refinedColmapProject(), cameras: refinedPairs())
    #expect(stored.files[ProjectLayout.refinedPoses] != nil)

    let poses = try ProjectBundle.refinedPoses(stored)
    #expect(Set(poses.keys) == ["a.ppm", "b.ppm"])
    for named in refinedPairs() {
        let pose = try #require(poses[named.name])
        #expect(refinedRotationDifference(pose.rotation, named.camera.rotation) < 1e-6, "rotation of \(named.name)")
        #expect(refinedTranslationDifference(pose.translation, named.camera.translation) < 1e-6,
                "translation of \(named.name)")
    }

    // The package encodes and decodes with the file in it.
    let files = try ProjectBundle.encode(stored)
    let decoded = try ProjectBundle.decode(files: files)
    #expect(try ProjectBundle.refinedPoses(decoded).count == 2)
}

@Test func withoutRefinedPosesRemovesTheFile() throws {
    let stored = ProjectBundle.withRefinedPoses(refinedColmapProject(), cameras: refinedPairs())
    let cleared = ProjectBundle.withoutRefinedPoses(stored)
    #expect(cleared.files[ProjectLayout.refinedPoses] == nil)
    #expect(try ProjectBundle.refinedPoses(cleared).isEmpty)
}

@Test func storingNoCamerasRemovesTheFile() throws {
    let stored = ProjectBundle.withRefinedPoses(refinedColmapProject(), cameras: refinedPairs())
    let emptied = ProjectBundle.withRefinedPoses(stored, cameras: [])
    #expect(emptied.files[ProjectLayout.refinedPoses] == nil)
}

@Test func corruptRefinedPosesAreReportedOnlyWhenRead() throws {
    var corrupt = refinedColmapProject()
    corrupt.files[ProjectLayout.refinedPoses] = Data("not json".utf8)

    // Without the override the file is not read, so the dataset and poses load as before.
    #expect(try ProjectBundle.cameraPoses(corrupt).count == 3)
    #expect(try ProjectBundle.loadDataset(corrupt, decode: PPMCodec.decode).views.count == 3)

    #expect(throws: DatasetError.self) {
        try ProjectBundle.cameraPoses(corrupt, useRefinedPoses: true)
    }
    #expect(throws: DatasetError.self) {
        try ProjectBundle.loadDataset(corrupt, decode: PPMCodec.decode, useRefinedPoses: true)
    }
}

// MARK: - Override

@Test func refinedPosesReplaceOnlyWhenAskedAndKeepIntrinsics() throws {
    let contents = ProjectBundle.withRefinedPoses(refinedColmapProject(), cameras: refinedPairs())
    let original = try ProjectBundle.cameraPoses(contents)
    let refinedByName: [String: Camera] = Dictionary(uniqueKeysWithValues: refinedPairs().map { ($0.name, $0.camera) })

    // Default: the original poses, exactly as before refinement existed.
    #expect(try ProjectBundle.cameraPoses(contents, useRefinedPoses: false) == original)

    let overridden = try ProjectBundle.cameraPoses(contents, useRefinedPoses: true)
    #expect(overridden.map(\.name) == ["a.ppm", "b.ppm", "c.ppm"])
    for (index, named) in overridden.enumerated() {
        let before: Camera = original[index].camera
        if let refined = refinedByName[named.name] {
            #expect(refinedRotationDifference(named.camera.rotation, refined.rotation) < 1e-6)
            #expect(refinedTranslationDifference(named.camera.translation, refined.translation) < 1e-6)
        } else {
            // "c.ppm" has no entry and keeps its original pose.
            #expect(named.camera == before, "\(named.name) must keep its pose")
        }
        #expect(named.camera.fx == before.fx && named.camera.fy == before.fy)
        #expect(named.camera.cx == before.cx && named.camera.cy == before.cy)
        #expect(named.camera.width == before.width && named.camera.height == before.height)
    }
}

@Test func refinedPosesReachTheDatasetOnlyWhenAsked() throws {
    let contents = ProjectBundle.withRefinedPoses(refinedColmapProject(), cameras: refinedPairs())
    let plain = try ProjectBundle.loadDataset(contents, decode: PPMCodec.decode)
    let overridden = try ProjectBundle.loadDataset(contents, decode: PPMCodec.decode, useRefinedPoses: true)
    let refinedByName: [String: Camera] = Dictionary(uniqueKeysWithValues: refinedPairs().map { ($0.name, $0.camera) })

    #expect(plain.views.count == 3)
    #expect(overridden.views.map(\.name) == plain.views.map(\.name))
    for (before, after) in zip(plain.views, overridden.views) {
        guard let refined = refinedByName[after.name] else {
            #expect(after.camera == before.camera, "\(after.name) must keep its pose")
            continue
        }
        #expect(refinedRotationDifference(after.camera.rotation, refined.rotation) < 1e-6, "rotation of \(after.name)")
        #expect(refinedTranslationDifference(after.camera.translation, refined.translation) < 1e-6,
                "translation of \(after.name)")
        #expect(after.camera.fx == before.camera.fx && after.camera.cx == before.camera.cx)
        #expect(after.camera.width == before.camera.width && after.camera.height == before.camera.height)
    }
}

// MARK: - Stale removal

@Test func everyImportRemovesStaleRefinedPoses() throws {
    let refined = ProjectBundle.withRefinedPoses(refinedColmapProject(), cameras: refinedPairs())
    #expect(refined.files[ProjectLayout.refinedPoses] != nil)

    // A new COLMAP reconstruction of the same images replaces the poses.
    let input = refinedColmapInput()
    let reimported = ProjectBundle.importCOLMAP(cameras: input.cameras,
                                                images: input.images,
                                                points3D: nil,
                                                imageFiles: input.imageFiles,
                                                into: refined)
    #expect(reimported.files[ProjectLayout.refinedPoses] == nil)
    #expect(try ProjectBundle.refinedPoses(reimported).isEmpty)
    let reimportedDataset = try ProjectBundle.loadDataset(reimported, decode: PPMCodec.decode,
                                                          useRefinedPoses: true)
    #expect(reimportedDataset.views.count == 3)

    // An ARKit capture of other frames replaces them too.
    let frames = [
        ARKitCapture.Frame(image: "a.ppm", width: 32, height: 24,
                           intrinsics: [20, 20, 16, 12],
                           cameraToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    ]
    let capture = try JSONEncoder().encode(ARKitCapture(version: 1, frames: frames))
    let arkit = ProjectBundle.importARKit(capture: capture,
                                          imageFiles: ["a.ppm": refinedPPM(width: 32, height: 24)],
                                          into: refined)
    #expect(arkit.files[ProjectLayout.refinedPoses] == nil)
    #expect(try ProjectBundle.refinedPoses(arkit).isEmpty)
}
