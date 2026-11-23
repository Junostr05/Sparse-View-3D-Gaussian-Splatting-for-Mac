import Foundation
import Testing
import SplatCore
@testable import SplatIO

@Suite struct PhotoDepthMatchingTests {
    private func map(_ value: Float) -> DepthMap {
        DepthMap(width: 2, height: 2, kind: .metric, values: [value, value, value, value],
                 confidence: nil, isEstimated: true, weights: nil)
    }

    @Test func depthFileNameMapsToPhotoName() {
        #expect(PhotoDepthMatching.photoName(forDepthFile: "IMG_9014.jpg.sdpt") == "IMG_9014.jpg")
        #expect(PhotoDepthMatching.photoName(forDepthFile: "IMG_9014.jpg") == nil)
        #expect(PhotoDepthMatching.photoName(forDepthFile: ".sdpt") == nil)
    }

    @Test func decodedMapsAreKeyedByPhotoNameAndUnreadableFilesAreSkipped() {
        let good = DepthMapCodec.encode(map(2.5))
        let maps = PhotoDepthMatching.decodeMaps([
            (fileName: "IMG_1.jpg.sdpt", data: good),
            (fileName: "IMG_2.jpg.sdpt", data: Data([0, 1, 2])),
            (fileName: "readme.txt", data: good),
        ])
        #expect(Set(maps.keys) == ["IMG_1.jpg"])
        #expect(maps["IMG_1.jpg"]?.values.first == 2.5)
    }

    @Test func framesGetTheDepthOfTheirSourcePhoto() {
        let maps: [String: DepthMap] = ["IMG_1.jpg": map(1), "IMG_3.jpg": map(3)]
        let frames: [(name: String, sourceName: String)] = [
            (name: "photo_0001.png", sourceName: "IMG_1.jpg"),
            (name: "photo_0002.png", sourceName: "IMG_2.jpg"),
            (name: "photo_0003.png", sourceName: "IMG_3.jpg"),
        ]
        let matched = PhotoDepthMatching.depthByFrame(frames: frames, maps: maps)
        #expect(Set(matched.keys) == ["photo_0001.png", "photo_0003.png"])
        #expect(matched["photo_0003.png"]?.values.first == 3)
    }
}
