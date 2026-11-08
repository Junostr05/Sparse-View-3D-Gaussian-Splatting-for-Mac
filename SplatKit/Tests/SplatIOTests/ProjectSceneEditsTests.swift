import Foundation
import Testing
import SplatCore
import SplatIO

private func sceneCloud(count: Int) throws -> GaussianCloud {
    try GaussianCloud(shDegree: 0,
                      positions: (0..<count * 3).map { Float($0) },
                      logScales: [Float](repeating: 0, count: count * 3),
                      rotations: (0..<count).flatMap { _ in [Float(1), 0, 0, 0] },
                      opacityLogits: [Float](repeating: 0, count: count),
                      sh: [Float](repeating: 0, count: count * 3))
}

private func sampleTransform() -> SceneTransform {
    // A unit quaternion, so the decoded value (which is normalised) compares equal.
    SceneTransform(rotation: SIMD4<Float>(0.5, 0.5, 0.5, 0.5), translation: SIMD3<Float>(1, 2, 3), scale: 2.5, isMetric: true)
}

private func sampleCrop() -> CropBox {
    CropBox(center: SIMD3<Float>(0, 1, 0), halfExtents: SIMD3<Float>(2, 3, 4), rotation: SIMD4<Float>(1, 0, 0, 0))
}

@Suite struct ProjectSceneEditsTests {
    @Test func projectWithoutEditsHasNoTransformOrCrop() throws {
        let contents = ProjectContents(manifest: ProjectManifest(), files: [:])
        #expect(try ProjectBundle.sceneTransform(contents) == nil)
        #expect(try ProjectBundle.cropBox(contents) == nil)
    }

    @Test func transformAndCropRoundTripThroughThePackage() throws {
        let base = ProjectBundle.withCheckpoint(
            ProjectContents(manifest: ProjectManifest(), files: [:]),
            cloud: try sceneCloud(count: 4),
            record: CheckpointRecord(iteration: 1, gaussianCount: 4, psnr: nil, date: Date(timeIntervalSince1970: 0)))
        let edited = try ProjectBundle.withCropBox(
            try ProjectBundle.withSceneTransform(base, sampleTransform()), sampleCrop())
        let files = try ProjectBundle.encode(edited)
        let reopened = try ProjectBundle.decode(files: files)
        #expect(try ProjectBundle.sceneTransform(reopened) == sampleTransform())
        #expect(try ProjectBundle.cropBox(reopened) == sampleCrop())
    }

    @Test func identityTransformRemovesTheFile() throws {
        let stored = try ProjectBundle.withSceneTransform(
            ProjectContents(manifest: ProjectManifest(), files: [:]), sampleTransform())
        #expect(stored.files[ProjectLayout.sceneTransform] != nil)
        let cleared = try ProjectBundle.withSceneTransform(stored, .identity)
        #expect(cleared.files[ProjectLayout.sceneTransform] == nil)
        #expect(try ProjectBundle.sceneTransform(cleared) == nil)
    }

    @Test func checkpointKeepsTheSceneTransform() throws {
        // Training writes a new checkpoint; the viewer's transform is a view setting and must survive it.
        let stored = try ProjectBundle.withSceneTransform(
            ProjectContents(manifest: ProjectManifest(), files: [:]), sampleTransform())
        let trained = ProjectBundle.withCheckpoint(
            stored,
            cloud: try sceneCloud(count: 3),
            record: CheckpointRecord(iteration: 2, gaussianCount: 3, psnr: nil, date: Date(timeIntervalSince1970: 0)))
        #expect(try ProjectBundle.sceneTransform(trained) == sampleTransform())
    }

    @Test func corruptTransformIsReported() {
        var files: [String: Data] = [:]
        files[ProjectLayout.sceneTransform] = Data("{".utf8)
        let contents = ProjectContents(manifest: ProjectManifest(), files: files)
        #expect(throws: ProjectError.self) { try ProjectBundle.sceneTransform(contents) }
    }
}
