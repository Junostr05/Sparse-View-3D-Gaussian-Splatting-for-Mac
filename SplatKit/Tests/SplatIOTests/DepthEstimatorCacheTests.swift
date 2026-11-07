import Foundation
import SplatCore
import SplatIO
import Testing

@Suite struct DepthEstimatorCacheTests {

    private func map(_ value: Float) -> DepthMap {
        DepthMap(width: 2, height: 1, kind: .relativeDepth, values: [value, 0], isEstimated: true)
    }

    private func emptyContents() -> ProjectContents {
        ProjectContents(manifest: ProjectManifest(), files: [:])
    }

    @Test func estimatorDirectoriesAreDistinct() {
        #expect(ProjectLayout.estimatedDepthPath(forImage: "a.png", estimator: .moge2Small)
                == "depth/estimated/moge2Small/a.png.sdpt")
        #expect(ProjectLayout.estimatedDepthPath(forImage: "a.png", estimator: .depthAnythingV2Small)
                != ProjectLayout.estimatedDepthPath(forImage: "a.png", estimator: .moge2Small))
    }

    @Test func estimatedMapsAreStoredAndReadPerEstimator() {
        let stored = ProjectBundle.withEstimatedDepthMaps(emptyContents(), ["a.png": map(1)],
                                                          estimator: .depthAnythingV2Small)
        let both = ProjectBundle.withEstimatedDepthMaps(stored, ["a.png": map(2)], estimator: .moge2Small)
        let depthAnything = ProjectBundle.estimatedDepthMaps(both, estimator: .depthAnythingV2Small)
        let moge = ProjectBundle.estimatedDepthMaps(both, estimator: .moge2Small)
        #expect(depthAnything.failures.isEmpty)
        #expect(depthAnything.maps["a.png"]?.values == [1, 0])
        #expect(moge.failures.isEmpty)
        #expect(moge.maps["a.png"]?.values == [2, 0])
    }

    @Test func estimatedMapsAreNotReadAsSuppliedMaps() {
        let stored = ProjectBundle.withEstimatedDepthMaps(emptyContents(), ["a.png": map(1)], estimator: .moge2Small)
        let supplied = ProjectBundle.depthMaps(stored)
        #expect(supplied.maps.isEmpty)
        #expect(supplied.failures.isEmpty)
    }

    @Test func suppliedMapsAreNotReadAsEstimatedMaps() {
        let stored = ProjectBundle.withDepthMaps(emptyContents(), ["b.png": map(3)])
        #expect(ProjectBundle.estimatedDepthMaps(stored, estimator: .moge2Small).maps.isEmpty)
        #expect(ProjectBundle.depthMaps(stored).maps.keys.sorted() == ["b.png"])
    }

    @Test func suppliedAndEstimatedMapsOfOneImageCoexist() {
        let stored = ProjectBundle.withEstimatedDepthMaps(
            ProjectBundle.withDepthMaps(emptyContents(), ["c.png": map(3)]),
            ["c.png": map(1)],
            estimator: .depthAnythingV2Small)
        #expect(ProjectBundle.depthMaps(stored).maps["c.png"]?.values == [3, 0])
        #expect(ProjectBundle.estimatedDepthMaps(stored, estimator: .depthAnythingV2Small).maps["c.png"]?.values == [1, 0])
    }

    @Test func corruptEstimatedMapIsReportedByName() {
        var contents = emptyContents()
        contents.files[ProjectLayout.estimatedDepthPath(forImage: "bad.png", estimator: .moge2Small)] = Data("nope".utf8)
        let loaded = ProjectBundle.estimatedDepthMaps(contents, estimator: .moge2Small)
        #expect(loaded.maps.isEmpty)
        #expect(loaded.failures.keys.sorted() == ["bad.png"])
    }
}
