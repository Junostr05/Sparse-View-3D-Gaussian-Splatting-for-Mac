import Foundation
import Testing
@testable import SplatIO

private func makeScratchDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("ModelStoreTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func fakeDescriptor(sha256: String?, sourceURL: URL? = URL(fileURLWithPath: "/nonexistent/model.zip")) -> ModelDescriptor {
    ModelDescriptor(
        id: "fake-model",
        displayName: "Fake Model",
        version: "1.0",
        sourceURL: sourceURL,
        sha256: sha256,
        sizeBytes: nil,
        license: "Apache-2.0",
        fileName: "Fake.mlpackage.zip",
        isArchive: true
    )
}

private let fakePayload = Data("fake model payload for store tests".utf8)

@Test func catalogEntryIsNotConfiguredYet() {
    let descriptor = ModelCatalog.depthAnythingV2Small
    #expect(descriptor.id == "depth-anything-v2-small")
    #expect(descriptor.license == "Apache-2.0")
    #expect(descriptor.sourceURL == nil)
    #expect(descriptor.sha256 == nil)
    #expect(descriptor.isConfigured == false)
    #expect(ModelCatalog.all.contains(descriptor))
}

@Test func descriptorIsConfiguredOnlyWithURLAndLowerHexChecksum() {
    let validHash = SHA256.hexDigest(of: fakePayload)
    #expect(fakeDescriptor(sha256: validHash).isConfigured)
    #expect(fakeDescriptor(sha256: nil).isConfigured == false)
    #expect(fakeDescriptor(sha256: validHash, sourceURL: nil).isConfigured == false)
    #expect(fakeDescriptor(sha256: validHash.uppercased()).isConfigured == false)
    #expect(fakeDescriptor(sha256: String(validHash.dropLast())).isConfigured == false)
    #expect(fakeDescriptor(sha256: String(validHash.dropLast()) + "g").isConfigured == false)
}

@Test func descriptorRoundTripsThroughCodable() throws {
    let descriptor = fakeDescriptor(sha256: SHA256.hexDigest(of: fakePayload))
    let data = try JSONEncoder().encode(descriptor)
    let decoded = try JSONDecoder().decode(ModelDescriptor.self, from: data)
    #expect(decoded == descriptor)
}

@Test func statusIsNotConfiguredForUnpinnedDescriptor() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root)
    #expect(store.status(for: ModelCatalog.depthAnythingV2Small) == .notConfigured)
}

@Test func statusIsMissingWhenNothingIsStored() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root)
    let descriptor = fakeDescriptor(sha256: SHA256.hexDigest(of: fakePayload))
    #expect(store.status(for: descriptor) == .missing)
}

@Test func directoryLayoutIsIDThenVersion() {
    let root = URL(fileURLWithPath: "/models-root", isDirectory: true)
    let store = ModelStore(rootDirectory: root)
    let descriptor = fakeDescriptor(sha256: nil)
    #expect(store.directory(for: descriptor).path == "/models-root/fake-model/1.0")
    #expect(store.fileURL(for: descriptor).lastPathComponent == "Fake.mlpackage.zip")
    #expect(store.fileURL(for: descriptor).deletingLastPathComponent() == store.directory(for: descriptor))
}

@Test func installVerifiesMovesAndReportsReady() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
    let descriptor = fakeDescriptor(sha256: SHA256.hexDigest(of: fakePayload))

    let downloaded = root.appendingPathComponent("download.part")
    try fakePayload.write(to: downloaded)
    try store.install(downloadedFile: downloaded, for: descriptor)

    let expectedURL = store.fileURL(for: descriptor)
    #expect(store.status(for: descriptor) == .ready(expectedURL))
    #expect(try Data(contentsOf: expectedURL) == fakePayload)
    #expect(FileManager.default.fileExists(atPath: downloaded.path) == false)
}

@Test func installReplacesExistingFile() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
    let descriptor = fakeDescriptor(sha256: SHA256.hexDigest(of: fakePayload))

    for attempt in 0..<2 {
        let downloaded = root.appendingPathComponent("download-\(attempt).part")
        try fakePayload.write(to: downloaded)
        try store.install(downloadedFile: downloaded, for: descriptor)
    }
    #expect(store.status(for: descriptor) == .ready(store.fileURL(for: descriptor)))
}

@Test func installRejectsChecksumMismatchAndDeletesDownload() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
    let otherPayload = Data("some other bytes".utf8)
    let descriptor = fakeDescriptor(sha256: SHA256.hexDigest(of: fakePayload))

    let downloaded = root.appendingPathComponent("download.part")
    try otherPayload.write(to: downloaded)

    #expect(throws: ModelStoreError.checksumMismatch(
        expected: SHA256.hexDigest(of: fakePayload),
        actual: SHA256.hexDigest(of: otherPayload)
    )) {
        try store.install(downloadedFile: downloaded, for: descriptor)
    }
    #expect(FileManager.default.fileExists(atPath: downloaded.path) == false)
    #expect(store.status(for: descriptor) == .missing)
}

@Test func installRefusesUnconfiguredDescriptor() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
    let downloaded = root.appendingPathComponent("download.part")
    try fakePayload.write(to: downloaded)

    #expect(throws: ModelStoreError.notConfigured) {
        try store.install(downloadedFile: downloaded, for: ModelCatalog.depthAnythingV2Small)
    }
}

@Test func statusBecomesCorruptWhenStoredFileChanges() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
    let expected = SHA256.hexDigest(of: fakePayload)
    let descriptor = fakeDescriptor(sha256: expected)

    let downloaded = root.appendingPathComponent("download.part")
    try fakePayload.write(to: downloaded)
    try store.install(downloadedFile: downloaded, for: descriptor)

    let otherPayload = Data("tampered".utf8)
    try otherPayload.write(to: store.fileURL(for: descriptor))
    #expect(store.status(for: descriptor) == .corrupt(expected: expected, actual: SHA256.hexDigest(of: otherPayload)))
}

@Test func statusHashesFilesLargerThanOneChunk() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))

    // Larger than the 1 MiB hashing chunk, so the streaming loop runs more than once.
    var large = Data(count: 2_500_000)
    for index in stride(from: 0, to: large.count, by: 4093) {
        large[index] = UInt8(truncatingIfNeeded: index)
    }
    let descriptor = fakeDescriptor(sha256: SHA256.hexDigest(of: large))

    let downloaded = root.appendingPathComponent("download.part")
    try large.write(to: downloaded)
    try store.install(downloadedFile: downloaded, for: descriptor)
    #expect(store.status(for: descriptor) == .ready(store.fileURL(for: descriptor)))
}

@Test func removeDeletesStoredModelAndIsIdempotent() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
    let descriptor = fakeDescriptor(sha256: SHA256.hexDigest(of: fakePayload))

    let downloaded = root.appendingPathComponent("download.part")
    try fakePayload.write(to: downloaded)
    try store.install(downloadedFile: downloaded, for: descriptor)

    try store.remove(descriptor)
    #expect(store.status(for: descriptor) == .missing)
    try store.remove(descriptor)
}

@Test func depthAvailabilityMatrix() {
    let readyURL = URL(fileURLWithPath: "/models/ready.zip")
    let ready = DepthFeatureAvailability.evaluate(status: .ready(readyURL), networkAllowed: false)
    #expect(ready == .ready)
    #expect(DepthFeatureAvailability.evaluate(status: .ready(readyURL), networkAllowed: true) == .ready)

    #expect(DepthFeatureAvailability.evaluate(status: .missing, networkAllowed: true) == .needsDownload)
    #expect(isUnavailable(DepthFeatureAvailability.evaluate(status: .missing, networkAllowed: false)))

    #expect(isUnavailable(DepthFeatureAvailability.evaluate(status: .notConfigured, networkAllowed: true)))
    #expect(isUnavailable(DepthFeatureAvailability.evaluate(status: .notConfigured, networkAllowed: false)))

    let corrupt = ModelStatus.corrupt(expected: "a", actual: "b")
    #expect(DepthFeatureAvailability.evaluate(status: corrupt, networkAllowed: true) == .needsDownload)
    #expect(isUnavailable(DepthFeatureAvailability.evaluate(status: corrupt, networkAllowed: false)))
}

@Test func unavailableReasonsAreUserFacing() {
    let offline = DepthFeatureAvailability.evaluate(status: .missing, networkAllowed: false)
    guard case let .unavailable(reason) = offline else {
        Issue.record("expected unavailable")
        return
    }
    #expect(reason.hasPrefix("Depth features are off"))
}

private func isUnavailable(_ value: DepthFeatureAvailability) -> Bool {
    if case .unavailable = value {
        return true
    }
    return false
}
