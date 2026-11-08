import Foundation
import Testing
import SplatCore
@testable import SplatIO

/// A 64 x 48 PPM, the size of the one camera used below, so the stub decoder needs no rescaling.
private func ppm() -> Data {
    let pixels = [SIMD3<Float>](repeating: SIMD3(0.5, 0.25, 0.75), count: 64 * 48)
    return PPMCodec.encode(RGBImage(width: 64, height: 48, pixels: pixels))
}

/// A COLMAP project with three views on one camera, `a.ppm`, `b.ppm` and `c.ppm`.
private func threeViewProject() -> ProjectContents {
    let cameras = makeCamerasBinary([
        FixtureCamera(id: 1, model: 1, width: 64, height: 48, params: [40, 40, 32, 24]),
    ])
    let images = makeImagesBinary([
        FixtureImage(id: 1, rotation: SIMD4(1, 0, 0, 0), translation: SIMD3(0, 0, 1), cameraID: 1, name: "a.ppm"),
        FixtureImage(id: 2, rotation: SIMD4(1, 0, 0, 0), translation: SIMD3(0, 0, 2), cameraID: 1, name: "b.ppm"),
        FixtureImage(id: 3, rotation: SIMD4(1, 0, 0, 0), translation: SIMD3(0, 0, 3), cameraID: 1, name: "c.ppm"),
    ])
    let imageFiles: [String: Data] = [
        "a.ppm": ppm(),
        "b.ppm": ppm(),
        "c.ppm": ppm(),
    ]
    let empty = ProjectContents(manifest: ProjectManifest(), files: [:])
    return ProjectBundle.importCOLMAP(cameras: cameras,
                                      images: images,
                                      points3D: nil,
                                      imageFiles: imageFiles,
                                      into: empty)
}

@Test func manifestWithoutExcludedImagesDecodesAsEmpty() throws {
    let data: Data = try ProjectBundle.encodeManifest(ProjectManifest())
    var object: [String: Any] = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
    object.removeValue(forKey: "excludedImages")
    let stripped: Data = try JSONSerialization.data(withJSONObject: object)
    let manifest: ProjectManifest = try JSONDecoder().decode(ProjectManifest.self, from: stripped)
    #expect(manifest.excludedImages.isEmpty)
}

@Test func excludedImagesAreSkippedByDatasetAndCameraPoses() throws {
    var project: ProjectContents = threeViewProject()
    #expect(project.manifest.imageFiles == ["a.ppm", "b.ppm", "c.ppm"])
    project.manifest.excludedImages = ["b.ppm"]

    let dataset: TrainingDataset = try ProjectBundle.loadDataset(project, decode: PPMCodec.decode)
    #expect(dataset.views.map(\.name) == ["a.ppm", "c.ppm"])

    let cameras: [NamedCamera] = try ProjectBundle.cameraPoses(project)
    #expect(cameras.map(\.name) == ["a.ppm", "c.ppm"])
}

@Test func excludingEveryImageLeavesNoViews() {
    var project: ProjectContents = threeViewProject()
    project.manifest.excludedImages = ["a.ppm", "b.ppm", "c.ppm"]
    #expect(throws: DatasetError.noViews) {
        try ProjectBundle.loadDataset(project, decode: PPMCodec.decode)
    }
}
