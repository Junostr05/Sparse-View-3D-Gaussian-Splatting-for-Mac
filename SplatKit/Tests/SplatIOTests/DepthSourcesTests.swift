import Foundation
import SplatCore
import SplatIO
import Testing

@Suite struct DepthSourcesTests {

    private func map(_ value: Float, estimated: Bool) -> DepthMap {
        DepthMap(width: 2, height: 1, kind: .relativeDepth, values: [value, 0], isEstimated: estimated)
    }

    @Test func unflaggedMapsAreSuppliedForEveryEstimator() {
        var contents = ProjectContents(manifest: ProjectManifest(), files: [:])
        contents = ProjectBundle.withDepthMaps(contents, ["lidar.png": map(5, estimated: false)])
        for estimator in DepthEstimatorKind.allCases {
            let sources = ProjectBundle.depthSources(contents, estimator: estimator)
            #expect(sources.supplied["lidar.png"]?.values == [5, 0])
            #expect(sources.cached.isEmpty)
        }
    }

    @Test func legacyEstimatedMapsAreCacheOnlyForDepthAnything() {
        // Written by the app before the per-estimator cache: flagged as estimated, directly under depth/.
        let contents = ProjectBundle.withDepthMaps(ProjectContents(manifest: ProjectManifest(), files: [:]),
                                                   ["legacy.png": map(7, estimated: true)])
        let depthAnything = ProjectBundle.depthSources(contents, estimator: .depthAnythingV2Small)
        #expect(depthAnything.supplied.isEmpty)
        #expect(depthAnything.cached["legacy.png"]?.values == [7, 0])

        let moge = ProjectBundle.depthSources(contents, estimator: .moge2Small)
        #expect(moge.supplied.isEmpty)
        #expect(moge.cached.isEmpty)
    }

    @Test func estimatorCacheWinsOverLegacyMapOfTheSameImage() {
        var contents = ProjectContents(manifest: ProjectManifest(), files: [:])
        contents = ProjectBundle.withDepthMaps(contents, ["view.png": map(7, estimated: true)])
        contents = ProjectBundle.withEstimatedDepthMaps(contents, ["view.png": map(9, estimated: true)],
                                                        estimator: .depthAnythingV2Small)
        let sources = ProjectBundle.depthSources(contents, estimator: .depthAnythingV2Small)
        #expect(sources.cached["view.png"]?.values == [9, 0])
    }

    @Test func mogeCacheIsNotUsedForDepthAnything() {
        var contents = ProjectContents(manifest: ProjectManifest(), files: [:])
        contents = ProjectBundle.withEstimatedDepthMaps(contents, ["view.png": map(4, estimated: true)],
                                                        estimator: .moge2Small)
        #expect(ProjectBundle.depthSources(contents, estimator: .depthAnythingV2Small).cached.isEmpty)
        #expect(ProjectBundle.depthSources(contents, estimator: .moge2Small).cached["view.png"]?.values == [4, 0])
    }

    @Test func corruptFilesAreReportedPerOrigin() {
        var contents = ProjectContents(manifest: ProjectManifest(), files: [:])
        contents.files[ProjectLayout.depthPath(forImage: "bad-supplied.png")] = Data("nope".utf8)
        contents.files[ProjectLayout.estimatedDepthPath(forImage: "bad-cache.png", estimator: .moge2Small)] = Data("nope".utf8)
        let sources = ProjectBundle.depthSources(contents, estimator: .moge2Small)
        #expect(sources.suppliedFailures.keys.sorted() == ["bad-supplied.png"])
        #expect(sources.cachedFailures.keys.sorted() == ["bad-cache.png"])
    }
}
