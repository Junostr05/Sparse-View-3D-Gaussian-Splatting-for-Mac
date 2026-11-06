import Foundation
import Synchronization
import Testing
import SplatCore
@testable import SplatIO

// MARK: - Fixtures

private func frame(_ name: String, timestamp: Double? = nil) -> CapturedFrame {
    CapturedFrame(name: name,
                  data: Data(name.utf8),
                  sourceName: "source_" + name,
                  timestamp: timestamp)
}

private func pose(_ name: String, camera: Int = 0) -> EstimatedImagePose {
    EstimatedImagePose(name: name,
                       cameraIndex: camera,
                       rotation: SIMD4(0.9, 0.1, -0.2, 0.3),
                       translation: SIMD3(0.5, -1.0, 2.0))
}

private func result(names: [String]) -> PoseEstimationResult {
    let camera = EstimatedCamera(modelID: 1, width: 64, height: 48, params: [40, 40, 32, 24])
    return PoseEstimationResult(cameras: [camera],
                                images: names.map { pose($0) },
                                points: [])
}

/// Provider that returns a fixed outcome.
private struct StubProvider: PoseProvider {
    let name: String
    let outcome: Result<PoseEstimationResult, PoseEstimationFailure>

    func estimatePoses(for frames: [CapturedFrame]) async throws(PoseEstimationFailure) -> PoseEstimationResult {
        switch outcome {
        case .success(let value):
            return value
        case .failure(let failure):
            throw failure
        }
    }
}

// MARK: - COLMAPWriter round trip

@Test func colmapWriterRoundTripsThroughReader() throws {
    let cameras = [
        EstimatedCamera(modelID: 1, width: 64, height: 48, params: [40, 41, 32, 24]),
        EstimatedCamera(modelID: 2, width: 32, height: 24, params: [20, 16, 12, 0.1]),
    ]
    let images = [
        EstimatedImagePose(name: "c.png", cameraIndex: 1,
                           rotation: SIMD4(1, 0, 0, 0), translation: SIMD3(0, 0, 3)),
        EstimatedImagePose(name: "a.png", cameraIndex: 0,
                           rotation: SIMD4(0.5, 0.5, -0.5, 0.5), translation: SIMD3(1, 2, 3)),
        EstimatedImagePose(name: "b.png", cameraIndex: 0,
                           rotation: SIMD4(0.9, 0.1, 0.2, 0.3), translation: SIMD3(-4, 0.25, 9)),
    ]
    var points: [EstimatedPoint] = []
    for index in 0..<5 {
        let value = Double(index)
        points.append(EstimatedPoint(position: SIMD3(value, value * 0.5, -value),
                                     color: SIMD3(UInt8(index * 10), 128, 255),
                                     error: 0.25 * value))
    }

    let readCameras = try COLMAPReader.readCameras(COLMAPWriter.cameras(cameras))
    #expect(readCameras.map(\.id) == [1, 2])
    #expect(readCameras.map(\.modelID) == [1, 2])
    #expect(readCameras.map(\.width) == [64, 32])
    #expect(readCameras.map(\.height) == [48, 24])
    #expect(readCameras.map(\.params) == cameras.map(\.params))

    let readImages = try COLMAPReader.readImages(COLMAPWriter.images(images))
    #expect(readImages.map(\.name) == ["a.png", "b.png", "c.png"])
    for image in readImages {
        let source = images.first { $0.name == image.name }!
        #expect(image.rotation == source.rotation)
        #expect(image.translation == source.translation)
        #expect(image.cameraID == UInt32(source.cameraIndex + 1))
    }

    let readPoints = try COLMAPReader.readPoints3D(COLMAPWriter.points3D(points))
    #expect(readPoints.count == 5)
    #expect(readPoints.map(\.position) == points.map(\.position))
    #expect(readPoints.map(\.color) == points.map(\.color))
}

// MARK: - Validation

@Test func validatedAcceptsEnoughRegisteredImages() throws {
    let value = result(names: ["a.png", "b.png"])
    #expect(try value.validated(totalFrames: 4) == value)
}

@Test func validatedRejectsTooFewRegisteredImages() {
    #expect(throws: PoseEstimationFailure.registrationFailed(registered: 1, total: 4)) {
        try result(names: ["a.png"]).validated(totalFrames: 4)
    }
    // Five frames need three registered images.
    #expect(throws: PoseEstimationFailure.registrationFailed(registered: 2, total: 5)) {
        try result(names: ["a.png", "b.png"]).validated(totalFrames: 5)
    }
}

@Test func validatedRejectsTooFewFrames() {
    #expect(throws: PoseEstimationFailure.tooFewFrames(have: 1, need: 2)) {
        try result(names: ["a.png"]).validated(totalFrames: 1)
    }
}

@Test func minimumRegisteredPolicy() {
    #expect(PoseEstimationPolicy.minimumRegistered(forTotal: 0) == 2)
    #expect(PoseEstimationPolicy.minimumRegistered(forTotal: 4) == 2)
    #expect(PoseEstimationPolicy.minimumRegistered(forTotal: 5) == 3)
    #expect(PoseEstimationPolicy.minimumRegistered(forTotal: 10) == 5)
}

// MARK: - Chain

@Test func chainFallsThroughFailureAndReportsAttempts() async {
    let log = Mutex<[PoseAttempt]>([])
    let chain = PoseProviderChain(providers: [
        StubProvider(name: "first", outcome: .failure(.unavailable(reason: "no tool"))),
        StubProvider(name: "second", outcome: .success(result(names: ["a.png", "b.png"]))),
        StubProvider(name: "third", outcome: .success(result(names: ["a.png", "b.png"]))),
    ])
    let outcome = await chain.estimate(for: [frame("a.png"), frame("b.png")],
                                       progress: { attempt in log.withLock { $0.append(attempt) } })

    let expectedAttempts = [
        PoseAttempt(providerName: "first", outcome: .failure(.unavailable(reason: "no tool"))),
        PoseAttempt(providerName: "second", outcome: .success),
    ]
    guard case .success(let value, let name, let attempts) = outcome else {
        Issue.record("expected success, got \(outcome)")
        return
    }
    #expect(name == "second")
    #expect(value.registeredImageNames == ["a.png", "b.png"])
    #expect(attempts == expectedAttempts)
    #expect(log.withLock { $0 } == expectedAttempts)
}

@Test func chainFallsThroughInvalidResult() async {
    let chain = PoseProviderChain(providers: [
        StubProvider(name: "sparse", outcome: .success(result(names: ["a.png"]))),
        StubProvider(name: "backup", outcome: .success(result(names: ["a.png", "b.png"]))),
    ])
    let outcome = await chain.estimate(for: [frame("a.png"), frame("b.png")])
    guard case .success(_, let name, let attempts) = outcome else {
        Issue.record("expected success, got \(outcome)")
        return
    }
    #expect(name == "backup")
    #expect(attempts.first?.outcome == .failure(.registrationFailed(registered: 1, total: 2)))
}

@Test func chainStopsOnCancellation() async {
    let chain = PoseProviderChain(providers: [
        StubProvider(name: "first", outcome: .failure(.cancelled)),
        StubProvider(name: "second", outcome: .success(result(names: ["a.png", "b.png"]))),
    ])
    let outcome = await chain.estimate(for: [frame("a.png"), frame("b.png")])
    #expect(outcome == .failure([PoseAttempt(providerName: "first", outcome: .failure(.cancelled))]))
}

@Test func chainFailsWhenEveryProviderFails() async {
    let chain = PoseProviderChain(providers: [
        StubProvider(name: "one", outcome: .failure(.failed(reason: "bad"))),
        StubProvider(name: "two", outcome: .failure(.unavailable(reason: "absent"))),
    ])
    let outcome = await chain.estimate(for: [frame("a.png"), frame("b.png")])
    #expect(outcome == .failure([
        PoseAttempt(providerName: "one", outcome: .failure(.failed(reason: "bad"))),
        PoseAttempt(providerName: "two", outcome: .failure(.unavailable(reason: "absent"))),
    ]))
}

// MARK: - Import

@Test func importEstimatedPosesKeepsRegisteredFramesAndLoads() throws {
    let frames = [
        frame("a.png", timestamp: 0.5),
        frame("b.png", timestamp: 1.0),
        frame("c.png", timestamp: 1.5),
    ]
    var estimated = result(names: ["c.png", "a.png"])
    estimated.points = [
        EstimatedPoint(position: SIMD3(0, 0, 2), color: SIMD3(255, 0, 0), error: 0.1),
        EstimatedPoint(position: SIMD3(1, 0, 2), color: SIMD3(0, 255, 0), error: 0.2),
    ]

    let empty = ProjectContents(manifest: ProjectManifest(), files: [:])
    let imported = ProjectBundle.importEstimatedPoses(frames: frames,
                                                      result: estimated,
                                                      inputKind: .video,
                                                      into: empty)

    #expect(imported.manifest.source == .colmap)
    #expect(imported.manifest.inputKind == .video)
    #expect(imported.manifest.imageFiles == ["a.png", "c.png"])
    #expect(imported.files["images/b.png"] == nil)
    #expect(imported.files["images/a.png"] == Data("a.png".utf8))
    #expect(imported.manifest.frameProvenance == [
        FrameProvenance(imageName: "a.png", sourceName: "source_a.png", timestamp: 0.5),
        FrameProvenance(imageName: "c.png", sourceName: "source_c.png", timestamp: 1.5),
    ])
    #expect(imported.files["poses/colmap/points3D.bin"] != nil)

    let stub: TrainingDatasetLoader.ImageDecoder = { _ in
        RGBImage(width: 64, height: 48, pixels: [SIMD3<Float>](repeating: .zero, count: 64 * 48))
    }
    let dataset = try ProjectBundle.loadDataset(imported, decode: stub)
    #expect(dataset.views.map(\.name) == ["a.png", "c.png"])
    #expect(dataset.points?.positions.count == 2)
}

@Test func importEstimatedPosesCopiesFocalLengthIntoProvenance() {
    let frames = [
        CapturedFrame(name: "a.png", data: Data([1]), sourceName: "a.jpg", timestamp: nil, focalLength35mm: 35),
        CapturedFrame(name: "b.png", data: Data([2]), sourceName: "b.jpg", timestamp: nil),
    ]
    let empty = ProjectContents(manifest: ProjectManifest(), files: [:])
    let imported = ProjectBundle.importEstimatedPoses(frames: frames,
                                                      result: result(names: ["a.png", "b.png"]),
                                                      inputKind: .photos,
                                                      into: empty)
    #expect(imported.manifest.frameProvenance == [
        FrameProvenance(imageName: "a.png", sourceName: "a.jpg", timestamp: nil, focalLength35mm: 35),
        FrameProvenance(imageName: "b.png", sourceName: "b.jpg", timestamp: nil, focalLength35mm: nil),
    ])
}

@Test func importEstimatedPosesWithoutPointsOmitsPointFile() {
    let empty = ProjectContents(manifest: ProjectManifest(), files: [:])
    let imported = ProjectBundle.importEstimatedPoses(frames: [frame("a.png"), frame("b.png")],
                                                      result: result(names: ["a.png", "b.png"]),
                                                      inputKind: .photos,
                                                      into: empty)
    #expect(imported.files["poses/colmap/points3D.bin"] == nil)
    #expect(imported.files["poses/colmap/cameras.bin"] != nil)
    #expect(imported.manifest.inputKind == .photos)
    #expect(imported.manifest.frameProvenance.map(\.imageName) == ["a.png", "b.png"])
}
