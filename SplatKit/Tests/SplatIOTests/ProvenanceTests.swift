import Foundation
import Testing
import SplatCore
@testable import SplatIO

private func sampleProvenance() -> [FrameProvenance] {
    [
        FrameProvenance(imageName: "frame_0001.png", sourceName: "clip.mov", timestamp: 0.5),
        FrameProvenance(imageName: "photo_0002.png", sourceName: "IMG_2.jpg", timestamp: nil),
    ]
}

@Test func provenanceRoundTrip() throws {
    var manifest = ProjectManifest()
    manifest.source = .colmap
    manifest.inputKind = .video
    manifest.imageFiles = ["frame_0001.png", "photo_0002.png"]
    manifest.frameProvenance = sampleProvenance()
    let contents = ProjectContents(manifest: manifest, files: [:])

    let files = try ProjectBundle.encode(contents)
    let decoded = try ProjectBundle.decode(files: files)
    #expect(decoded == contents)
    #expect(decoded.manifest.inputKind == .video)
    #expect(decoded.manifest.frameProvenance == sampleProvenance())
}

/// A manifest written by Phase 3 code has no `inputKind` or `frameProvenance` keys.
@Test func phase3ManifestWithoutProvenanceStillLoads() throws {
    let training = try JSONSerialization.jsonObject(with: JSONEncoder().encode(TrainingConfiguration()))
    let legacy: [String: Any] = [
        "formatVersion": 1,
        "source": "colmap",
        "imageFiles": ["a.ppm", "b.ppm"],
        "training": training,
        "checkpoints": [] as [Any],
    ]
    let json = try JSONSerialization.data(withJSONObject: legacy, options: [.sortedKeys])
    let text = String(decoding: json, as: UTF8.self)
    #expect(!text.contains("inputKind"))
    #expect(!text.contains("frameProvenance"))

    let contents = try ProjectBundle.decode(files: [ProjectLayout.manifest: json])
    #expect(contents.manifest.source == .colmap)
    #expect(contents.manifest.imageFiles == ["a.ppm", "b.ppm"])
    #expect(contents.manifest.inputKind == nil)
    #expect(contents.manifest.frameProvenance.isEmpty)
    #expect(contents.manifest.formatVersion == 1)
}

@Test func importCOLMAPSetsInputKindAndClearsProvenance() throws {
    var start = ProjectManifest()
    start.inputKind = .video
    start.frameProvenance = sampleProvenance()
    let contents = ProjectContents(manifest: start, files: [:])

    let imported = ProjectBundle.importCOLMAP(cameras: Data(),
                                              images: Data(),
                                              points3D: nil,
                                              imageFiles: ["a.ppm": Data([1])],
                                              into: contents)
    #expect(imported.manifest.inputKind == .colmap)
    #expect(imported.manifest.frameProvenance.isEmpty)
    #expect(imported.manifest.source == .colmap)
}

@Test func importARKitSetsInputKindAndClearsProvenance() throws {
    var start = ProjectManifest()
    start.inputKind = .photos
    start.frameProvenance = sampleProvenance()
    let contents = ProjectContents(manifest: start, files: [:])

    let imported = ProjectBundle.importARKit(capture: Data("{}".utf8),
                                             imageFiles: ["a.ppm": Data([1])],
                                             into: contents)
    #expect(imported.manifest.inputKind == .arkit)
    #expect(imported.manifest.frameProvenance.isEmpty)
    #expect(imported.manifest.source == .arkit)
}

/// A manifest with provenance entries written before `focalLength35mm` existed still decodes it as nil.
@Test func provenanceWithoutFocalLengthKeyDecodesAsNil() throws {
    let training = try JSONSerialization.jsonObject(with: JSONEncoder().encode(TrainingConfiguration()))
    let legacy: [String: Any] = [
        "formatVersion": 1,
        "source": "colmap",
        "imageFiles": ["photo_0001.png"],
        "training": training,
        "checkpoints": [] as [Any],
        "inputKind": "photos",
        "frameProvenance": [
            ["imageName": "photo_0001.png", "sourceName": "IMG_1.jpg"] as [String: Any],
        ],
    ]
    let json = try JSONSerialization.data(withJSONObject: legacy, options: [.sortedKeys])
    let contents = try ProjectBundle.decode(files: [ProjectLayout.manifest: json])
    #expect(contents.manifest.frameProvenance == [
        FrameProvenance(imageName: "photo_0001.png", sourceName: "IMG_1.jpg", timestamp: nil, focalLength35mm: nil),
    ])
}

/// The focal length survives encode and decode when present.
@Test func provenanceFocalLengthRoundTrips() throws {
    var manifest = ProjectManifest()
    manifest.source = .colmap
    manifest.inputKind = .photos
    manifest.imageFiles = ["photo_0001.png"]
    manifest.frameProvenance = [
        FrameProvenance(imageName: "photo_0001.png", sourceName: "IMG_1.jpg", timestamp: nil, focalLength35mm: 35),
    ]
    let contents = ProjectContents(manifest: manifest, files: [:])

    let files = try ProjectBundle.encode(contents)
    let decoded = try ProjectBundle.decode(files: files)
    #expect(decoded.manifest.frameProvenance.first?.focalLength35mm == 35)
    #expect(decoded == contents)
}
