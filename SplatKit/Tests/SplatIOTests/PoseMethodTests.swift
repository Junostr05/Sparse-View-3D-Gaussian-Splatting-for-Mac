import Foundation
import Testing
import SplatCore
@testable import SplatIO

private func poseFrame(_ name: String) -> CapturedFrame {
    CapturedFrame(name: name, data: Data(name.utf8), sourceName: "source_" + name, timestamp: nil)
}

private func poseResult(names: [String]) -> PoseEstimationResult {
    let camera = EstimatedCamera(modelID: 1, width: 64, height: 48, params: [40, 40, 32, 24])
    let images = names.map { name in
        EstimatedImagePose(name: name,
                           cameraIndex: 0,
                           rotation: SIMD4(0.9, 0.1, -0.2, 0.3),
                           translation: SIMD3(0.5, -1.0, 2.0))
    }
    return PoseEstimationResult(cameras: [camera], images: images, points: [])
}

@Test func poseMethodRoundTripsInManifest() throws {
    var manifest = ProjectManifest()
    manifest.source = .colmap
    manifest.inputKind = .photos
    manifest.imageFiles = ["a.png"]
    manifest.poseMethod = .depthAnything3
    let contents = ProjectContents(manifest: manifest, files: [:])

    let files = try ProjectBundle.encode(contents)
    let decoded = try ProjectBundle.decode(files: files)
    #expect(decoded == contents)
    #expect(decoded.manifest.poseMethod == .depthAnything3)
}

/// A manifest written before `poseMethod` existed decodes it as nil and keeps every other field.
@Test func manifestWithoutPoseMethodDecodesAsNil() throws {
    let training = try JSONSerialization.jsonObject(with: JSONEncoder().encode(TrainingConfiguration()))
    let legacy: [String: Any] = [
        "formatVersion": 1,
        "source": "colmap",
        "imageFiles": ["a.png"],
        "training": training,
        "checkpoints": [] as [Any],
        "inputKind": "photos",
    ]
    let json = try JSONSerialization.data(withJSONObject: legacy, options: [.sortedKeys])
    #expect(!String(decoding: json, as: UTF8.self).contains("poseMethod"))

    let contents = try ProjectBundle.decode(files: [ProjectLayout.manifest: json])
    #expect(contents.manifest.poseMethod == nil)
    #expect(contents.manifest.inputKind == .photos)
    #expect(contents.manifest.formatVersion == 1)
}

@Test func estimatedPosesDefaultToApplePoseMethod() {
    let empty = ProjectContents(manifest: ProjectManifest(), files: [:])
    let imported = ProjectBundle.importEstimatedPoses(frames: [poseFrame("a.png"), poseFrame("b.png")],
                                                      result: poseResult(names: ["a.png", "b.png"]),
                                                      inputKind: .photos,
                                                      into: empty)
    #expect(imported.manifest.poseMethod == .apple)
}

@Test func estimatedPosesRecordDepthAnything3PoseMethod() throws {
    let empty = ProjectContents(manifest: ProjectManifest(), files: [:])
    let imported = ProjectBundle.importEstimatedPoses(frames: [poseFrame("a.png"), poseFrame("b.png")],
                                                      result: poseResult(names: ["a.png", "b.png"]),
                                                      inputKind: .photos,
                                                      method: .depthAnything3,
                                                      into: empty)
    #expect(imported.manifest.poseMethod == .depthAnything3)
    let decoded = try ProjectBundle.decode(files: try ProjectBundle.encode(imported))
    #expect(decoded.manifest.poseMethod == .depthAnything3)
}

@Test func colmapImportResetsPoseMethod() {
    var start = ProjectManifest()
    start.poseMethod = .depthAnything3
    let contents = ProjectContents(manifest: start, files: [:])

    let imported = ProjectBundle.importCOLMAP(cameras: Data(),
                                              images: Data(),
                                              points3D: nil,
                                              imageFiles: ["a.ppm": Data([1])],
                                              into: contents)
    #expect(imported.manifest.poseMethod == nil)
}

@Test func arkitImportResetsPoseMethod() {
    var start = ProjectManifest()
    start.poseMethod = .apple
    let contents = ProjectContents(manifest: start, files: [:])

    let imported = ProjectBundle.importARKit(capture: Data("{}".utf8),
                                             imageFiles: ["a.ppm": Data([1])],
                                             into: contents)
    #expect(imported.manifest.poseMethod == nil)
}

/// A new estimation replaces the method of an earlier one.
@Test func estimatedPosesReplacePoseMethodOfEarlierImport() {
    var start = ProjectManifest()
    start.poseMethod = .depthAnything3
    let contents = ProjectContents(manifest: start, files: [:])

    let imported = ProjectBundle.importEstimatedPoses(frames: [poseFrame("a.png"), poseFrame("b.png")],
                                                      result: poseResult(names: ["a.png", "b.png"]),
                                                      inputKind: .photos,
                                                      method: .apple,
                                                      into: contents)
    #expect(imported.manifest.poseMethod == .apple)
}

@Test func upAxisForProjectFollowsPoseMethodThenInputKind() {
    var colmap = ProjectManifest()
    colmap.inputKind = .colmap
    #expect(UpAxis.forProject(colmap) == .negativeY)

    var untyped = ProjectManifest()
    #expect(UpAxis.forProject(untyped) == .negativeY)

    var photosApple = ProjectManifest()
    photosApple.inputKind = .photos
    photosApple.poseMethod = .apple
    #expect(UpAxis.forProject(photosApple) == .positiveY)

    var videoApple = ProjectManifest()
    videoApple.inputKind = .video
    videoApple.poseMethod = .apple
    #expect(UpAxis.forProject(videoApple) == .positiveY)

    var arkit = ProjectManifest()
    arkit.inputKind = .arkit
    #expect(UpAxis.forProject(arkit) == .positiveY)

    var photosDepth = ProjectManifest()
    photosDepth.inputKind = .photos
    photosDepth.poseMethod = .depthAnything3
    #expect(UpAxis.forProject(photosDepth) == .negativeY)

    var videoDepth = ProjectManifest()
    videoDepth.inputKind = .video
    videoDepth.poseMethod = .depthAnything3
    #expect(UpAxis.forProject(videoDepth) == .negativeY)

    untyped.poseMethod = .depthAnything3
    #expect(UpAxis.forProject(untyped) == .negativeY)
}
