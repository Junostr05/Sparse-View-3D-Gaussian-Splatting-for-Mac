import Foundation
import Testing
import SplatCore
@testable import SplatIO

private let identityTransform: [Double] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

private func solidPPM(width: Int, height: Int) -> Data {
    let pixels = [SIMD3<Float>](repeating: SIMD3(0.5, 0.25, 0.75), count: width * height)
    return PPMCodec.encode(RGBImage(width: width, height: height, pixels: pixels))
}

/// Deterministic cloud with arbitrary finite values, for exact round-trip checks.
private func randomCloud(degree: Int, count: Int, seed: UInt64) throws -> GaussianCloud {
    var rng = SeededGenerator(seed: seed)
    let k = GaussianCloud.coefficientCount(degree: degree)
    let positions = (0..<count * 3).map { _ in Float.random(in: -5...5, using: &rng) }
    let logScales = (0..<count * 3).map { _ in Float.random(in: -6...0, using: &rng) }
    let rotations = (0..<count * 4).map { _ in Float.random(in: -1...1, using: &rng) }
    let opacity = (0..<count).map { _ in Float.random(in: -4...4, using: &rng) }
    let sh = (0..<count * k * 3).map { _ in Float.random(in: -2...2, using: &rng) }
    return try GaussianCloud(shDegree: degree,
                             positions: positions,
                             logScales: logScales,
                             rotations: rotations,
                             opacityLogits: opacity,
                             sh: sh)
}

private func assertSameCloud(_ lhs: GaussianCloud, _ rhs: GaussianCloud) {
    #expect(lhs.shDegree == rhs.shDegree)
    #expect(lhs.positions == rhs.positions)
    #expect(lhs.logScales == rhs.logScales)
    #expect(lhs.rotations == rhs.rotations)
    #expect(lhs.opacityLogits == rhs.opacityLogits)
    #expect(lhs.sh == rhs.sh)
}

private func arkitCaptureJSON(image: String) throws -> Data {
    let frame = ARKitCapture.Frame(image: image,
                                   width: 64,
                                   height: 48,
                                   intrinsics: [40, 40, 32, 24],
                                   cameraToWorld: identityTransform)
    return try JSONEncoder().encode(ARKitCapture(version: 1, frames: [frame]))
}

// MARK: - PLY

@Test func plyRoundTripsThroughReader() throws {
    for degree in [0, 1, 3] {
        let cloud = try randomCloud(degree: degree, count: 7, seed: UInt64(degree + 1))
        let read = try PLYReader.read(PLYWriter.write(cloud))
        assertSameCloud(read, cloud)
    }
}

// MARK: - Manifest and versioning

@Test func manifestRoundTrip() throws {
    var manifest = ProjectManifest()
    manifest.source = .colmap
    manifest.imageFiles = ["a.ppm", "sub/b.ppm"]
    manifest.training.iterations = 123
    manifest.checkpoints = [
        CheckpointRecord(iteration: 100,
                         gaussianCount: 42,
                         psnr: 24.5,
                         date: Date(timeIntervalSince1970: 1_700_000_000)),
    ]
    let contents = ProjectContents(manifest: manifest,
                                   files: ["images/a.ppm": solidPPM(width: 2, height: 2)])

    let files = try ProjectBundle.encode(contents)
    #expect(files[ProjectLayout.manifest] != nil)

    let decoded = try ProjectBundle.decode(files: files)
    #expect(decoded == contents)
}

@Test func newerFormatVersionIsRejected() throws {
    var manifest = ProjectManifest()
    manifest.formatVersion = 2
    let files = [ProjectLayout.manifest: try ProjectBundle.encodeManifest(manifest)]
    #expect(throws: ProjectError.unsupportedFormatVersion(2)) {
        try ProjectBundle.decode(files: files)
    }
}

@Test func missingManifestIsRejected() {
    #expect(throws: ProjectError.missingManifest) {
        try ProjectBundle.decode(files: ["images/a.ppm": Data()])
    }
}

@Test func unsafePathsAreRejected() throws {
    let escaping = ProjectContents(manifest: ProjectManifest(),
                                   files: ["../escape.ply": Data([1])])
    #expect(throws: ProjectError.self) {
        try ProjectBundle.encode(escaping)
    }

    let absolute = ProjectContents(manifest: ProjectManifest(),
                                   files: ["/tmp/absolute.ply": Data([1])])
    #expect(throws: ProjectError.self) {
        try ProjectBundle.encode(absolute)
    }

    let manifestData = try ProjectBundle.encodeManifest(ProjectManifest())
    let tampered: [String: Data] = [
        ProjectLayout.manifest: manifestData,
        "images/../../secret.ppm": Data([1]),
    ]
    #expect(throws: ProjectError.self) {
        try ProjectBundle.decode(files: tampered)
    }
}

// MARK: - Import and dataset

@Test func importCOLMAPThenLoadDataset() throws {
    let cameras = makeCamerasBinary([
        FixtureCamera(id: 1, model: 1, width: 64, height: 48, params: [40, 40, 32, 24]),
    ])
    let images = makeImagesBinary([
        FixtureImage(id: 1, rotation: SIMD4(1, 0, 0, 0), translation: .zero, cameraID: 1, name: "b.ppm"),
        FixtureImage(id: 2, rotation: SIMD4(1, 0, 0, 0), translation: SIMD3(0, 0, 1), cameraID: 1, name: "a.ppm"),
    ])
    let points = makePoints3DBinary([
        FixturePoint(id: 1, position: SIMD3(0, 0, 1), color: SIMD3(255, 0, 0), error: 0.5),
    ])
    let imageFiles: [String: Data] = [
        "a.ppm": solidPPM(width: 64, height: 48),
        "b.ppm": solidPPM(width: 64, height: 48),
    ]

    let empty = ProjectContents(manifest: ProjectManifest(), files: [:])
    let imported = ProjectBundle.importCOLMAP(cameras: cameras,
                                              images: images,
                                              points3D: points,
                                              imageFiles: imageFiles,
                                              into: empty)
    #expect(imported.manifest.source == .colmap)
    #expect(imported.manifest.imageFiles == ["a.ppm", "b.ppm"])
    #expect(imported.files["images/a.ppm"] == imageFiles["a.ppm"])
    #expect(imported.files["poses/colmap/cameras.bin"] == cameras)
    #expect(imported.files["poses/colmap/images.bin"] == images)
    #expect(imported.files["poses/colmap/points3D.bin"] == points)

    let dataset = try ProjectBundle.loadDataset(imported, decode: PPMCodec.decode)
    #expect(dataset.views.count == 2)
    #expect(dataset.points?.positions.count == 1)
}

@Test func reimportClearsOldCheckpoint() throws {
    let cameras = makeCamerasBinary([
        FixtureCamera(id: 1, model: 1, width: 64, height: 48, params: [40, 40, 32, 24]),
    ])
    let images = makeImagesBinary([
        FixtureImage(id: 1, rotation: SIMD4(1, 0, 0, 0), translation: .zero, cameraID: 1, name: "a.ppm"),
    ])
    let empty = ProjectContents(manifest: ProjectManifest(), files: [:])
    let withCOLMAP = ProjectBundle.importCOLMAP(cameras: cameras,
                                                images: images,
                                                points3D: nil,
                                                imageFiles: ["a.ppm": solidPPM(width: 64, height: 48)],
                                                into: empty)
    let cloud = try randomCloud(degree: 0, count: 3, seed: 9)
    let checkpointed = ProjectBundle.withCheckpoint(
        withCOLMAP,
        cloud: cloud,
        record: CheckpointRecord(iteration: 10, gaussianCount: 3, psnr: nil,
                                 date: Date(timeIntervalSince1970: 0))
    )

    let reimported = ProjectBundle.importARKit(capture: try arkitCaptureJSON(image: "f.ppm"),
                                               imageFiles: ["f.ppm": solidPPM(width: 64, height: 48)],
                                               into: checkpointed)
    #expect(reimported.manifest.source == .arkit)
    #expect(reimported.manifest.imageFiles == ["f.ppm"])
    #expect(reimported.manifest.checkpoints.isEmpty)
    #expect(reimported.files["images/a.ppm"] == nil)
    #expect(reimported.files["poses/colmap/cameras.bin"] == nil)
    #expect(reimported.files["checkpoints/latest.ply"] == nil)
    #expect(try ProjectBundle.latestCheckpoint(reimported) == nil)
}

// MARK: - Checkpoints

@Test func checkpointRoundTrip() throws {
    let cloud = try randomCloud(degree: 1, count: 11, seed: 3)
    let record = CheckpointRecord(iteration: 500,
                                  gaussianCount: cloud.count,
                                  psnr: 27.25,
                                  date: Date(timeIntervalSince1970: 1_700_000_123))
    let empty = ProjectContents(manifest: ProjectManifest(), files: [:])
    let updated = ProjectBundle.withCheckpoint(empty, cloud: cloud, record: record)

    #expect(updated.manifest.checkpoints == [record])
    let latest = try ProjectBundle.latestCheckpoint(updated)
    #expect(latest != nil)
    if let latest {
        assertSameCloud(latest, cloud)
    }

    // The checkpoint also survives a full encode/decode cycle.
    let decoded = try ProjectBundle.decode(files: try ProjectBundle.encode(updated))
    let reloaded = try ProjectBundle.latestCheckpoint(decoded)
    #expect(decoded.manifest.checkpoints == [record])
    if let reloaded {
        assertSameCloud(reloaded, cloud)
    } else {
        Issue.record("checkpoint missing after encode/decode")
    }
}
