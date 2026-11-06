import Foundation
import Testing
@testable import SplatIO

private func makeScratchDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("ModelStoreTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

/// Local stand-in for the remote base URL. These tests never download, so no network is touched.
private let fakeBaseURL = URL(fileURLWithPath: "/nonexistent/fake-model", isDirectory: true)

/// Files of the fake package. One file sits two directories deep, like the real Core ML package.
private let fakePayloads: [(path: String, data: Data)] = [
    ("Manifest.json", Data("fake manifest for store tests".utf8)),
    ("Data/com.apple.CoreML/model.mlmodel", Data("fake model payload for store tests".utf8)),
    ("Data/com.apple.CoreML/weights/weight.bin", Data("fake weight bytes for store tests".utf8)),
]

private func payload(for path: String) -> Data {
    fakePayloads.first { $0.path == path }?.data ?? Data()
}

private func fakeFiles() -> [ModelFile] {
    fakePayloads.map { entry in
        ModelFile(relativePath: entry.path, sha256: SHA256.hexDigest(of: entry.data), sizeBytes: Int64(entry.data.count))
    }
}

private func fakeDescriptor(
    baseURL: URL? = fakeBaseURL,
    packageName: String = "Fake.mlpackage",
    files: [ModelFile]? = nil
) -> ModelDescriptor {
    ModelDescriptor(
        id: "fake-model",
        displayName: "Fake Model",
        version: "1.0",
        license: "Apache-2.0",
        baseURL: baseURL,
        packageName: packageName,
        files: files ?? fakeFiles()
    )
}

/// Writes `entries` below a fresh staging directory at their relative paths and returns that directory.
private func makeStagingDirectory(in root: URL, entries: [(path: String, data: Data)]) throws -> URL {
    let staging = root.appendingPathComponent("staging-\(UUID().uuidString)", isDirectory: true)
    for entry in entries {
        let url = staging.appendingComponents(of: entry.path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try entry.data.write(to: url)
    }
    return staging
}

/// The fake package entries with the payload of `path` replaced by `data`.
private func entries(replacing path: String, with data: Data) -> [(path: String, data: Data)] {
    fakePayloads.map { entry -> (path: String, data: Data) in
        entry.path == path ? (path: entry.path, data: data) : entry
    }
}

private let weightPath = "Data/com.apple.CoreML/weights/weight.bin"
private let modelPath = "Data/com.apple.CoreML/model.mlmodel"

@Test func catalogEntryIsConfiguredWithPinnedPackage() {
    let descriptor = ModelCatalog.depthAnythingV2Small
    #expect(descriptor.id == "depth-anything-v2-small")
    #expect(descriptor.version == "cfef6f6")
    #expect(descriptor.license == "Apache-2.0")
    #expect(descriptor.baseURL == URL(string: "https://huggingface.co/apple/coreml-depth-anything-v2-small/resolve/cfef6f6f2a70783dedc0bfae40cecbc2052285d3"))
    #expect(descriptor.packageName == "DepthAnythingV2SmallF16.mlpackage")
    #expect(descriptor.files.map(\.relativePath) == [
        "Manifest.json",
        "Data/com.apple.CoreML/model.mlmodel",
        "Data/com.apple.CoreML/weights/weight.bin",
    ])
    #expect(descriptor.files.map(\.sizeBytes) == [617, 399_433, 49_419_072])
    let expectedTotal: Int64 = 617 + 399_433 + 49_419_072
    #expect(descriptor.totalSizeBytes == expectedTotal)
    #expect(descriptor.isConfigured)
    #expect(ModelCatalog.all.contains(descriptor))
}

@Test func descriptorIsConfiguredOnlyWithValidBaseURLFilesAndPaths() {
    let validHash = SHA256.hexDigest(of: Data("valid".utf8))
    #expect(fakeDescriptor().isConfigured)
    #expect(fakeDescriptor(baseURL: nil).isConfigured == false)
    #expect(fakeDescriptor(files: []).isConfigured == false)

    let upper = ModelFile(relativePath: "Manifest.json", sha256: validHash.uppercased(), sizeBytes: 1)
    let short = ModelFile(relativePath: "Manifest.json", sha256: String(validHash.dropLast()), sizeBytes: 1)
    let notHex = ModelFile(relativePath: "Manifest.json", sha256: String(validHash.dropLast()) + "g", sizeBytes: 1)
    #expect(fakeDescriptor(files: [upper]).isConfigured == false)
    #expect(fakeDescriptor(files: [short]).isConfigured == false)
    #expect(fakeDescriptor(files: [notHex]).isConfigured == false)

    for unsafePath in ["../evil", "/abs", "a//b", "a/./b", "a/../b", "trailing/", "", "back\\slash"] {
        let file = ModelFile(relativePath: unsafePath, sha256: validHash, sizeBytes: 1)
        #expect(fakeDescriptor(files: [file]).isConfigured == false, "relativePath \(unsafePath) must be rejected")
    }

    for unsafePackage in ["..", ".", "/abs", "a//b", "", "../evil"] {
        #expect(fakeDescriptor(packageName: unsafePackage).isConfigured == false, "packageName \(unsafePackage) must be rejected")
    }
    #expect(fakeDescriptor(packageName: "Nested/Fake.mlpackage").isConfigured)
}

@Test func descriptorRoundTripsThroughCodable() throws {
    let descriptor = fakeDescriptor()
    let data = try JSONEncoder().encode(descriptor)
    let decoded = try JSONDecoder().decode(ModelDescriptor.self, from: data)
    #expect(decoded == descriptor)
}

@Test func statusIsNotConfiguredForDescriptorWithoutBaseURL() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root)
    #expect(store.status(for: fakeDescriptor(baseURL: nil)) == .notConfigured)
}

@Test func statusIsMissingWhenNothingIsStored() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root)
    #expect(store.status(for: fakeDescriptor()) == .missing)
}

@Test func directoryLayoutIsIDVersionPackage() {
    let root = URL(fileURLWithPath: "/models-root", isDirectory: true)
    let store = ModelStore(rootDirectory: root)
    let descriptor = fakeDescriptor()
    #expect(store.directory(for: descriptor).path == "/models-root/fake-model/1.0")
    #expect(store.packageURL(for: descriptor).path == "/models-root/fake-model/1.0/Fake.mlpackage")
    #expect(store.packageURL(for: descriptor).deletingLastPathComponent() == store.directory(for: descriptor))
}

@Test func installVerifiesMovesAndReportsReady() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
    let descriptor = fakeDescriptor()

    let staging = try makeStagingDirectory(in: root, entries: fakePayloads)
    try store.install(stagingDirectory: staging, for: descriptor)

    let package = store.packageURL(for: descriptor)
    #expect(store.status(for: descriptor) == .ready(package))
    for entry in fakePayloads {
        #expect(try Data(contentsOf: package.appendingComponents(of: entry.path)) == entry.data)
    }
    #expect(FileManager.default.fileExists(atPath: staging.path) == false)
}

@Test func installReplacesExistingPackage() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
    let descriptor = fakeDescriptor()

    for _ in 0..<2 {
        let staging = try makeStagingDirectory(in: root, entries: fakePayloads)
        try store.install(stagingDirectory: staging, for: descriptor)
    }
    #expect(store.status(for: descriptor) == .ready(store.packageURL(for: descriptor)))
}

@Test func installRejectsChecksumMismatchAndDeletesStaging() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
    let otherPayload = Data("some other bytes".utf8)
    let descriptor = fakeDescriptor()

    let staging = try makeStagingDirectory(in: root, entries: entries(replacing: modelPath, with: otherPayload))

    #expect(throws: ModelStoreError.checksumMismatch(
        expected: SHA256.hexDigest(of: payload(for: modelPath)),
        actual: SHA256.hexDigest(of: otherPayload)
    )) {
        try store.install(stagingDirectory: staging, for: descriptor)
    }
    #expect(FileManager.default.fileExists(atPath: staging.path) == false)
    #expect(store.status(for: descriptor) == .missing)
}

@Test func installRejectsMissingFileAndDeletesStaging() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
    let descriptor = fakeDescriptor()

    let withoutWeights = fakePayloads.filter { $0.path != weightPath }
    let staging = try makeStagingDirectory(in: root, entries: withoutWeights)

    #expect(throws: ModelStoreError.checksumMismatch(
        expected: SHA256.hexDigest(of: payload(for: weightPath)),
        actual: ""
    )) {
        try store.install(stagingDirectory: staging, for: descriptor)
    }
    #expect(FileManager.default.fileExists(atPath: staging.path) == false)
    #expect(store.status(for: descriptor) == .missing)
}

@Test func failedInstallLeavesExistingPackageIntact() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
    let descriptor = fakeDescriptor()

    let good = try makeStagingDirectory(in: root, entries: fakePayloads)
    try store.install(stagingDirectory: good, for: descriptor)

    let bad = try makeStagingDirectory(in: root, entries: entries(replacing: weightPath, with: Data("bad".utf8)))
    #expect(throws: ModelStoreError.self) {
        try store.install(stagingDirectory: bad, for: descriptor)
    }

    #expect(FileManager.default.fileExists(atPath: bad.path) == false)
    #expect(store.status(for: descriptor) == .ready(store.packageURL(for: descriptor)))
    let installedWeights = store.packageURL(for: descriptor).appendingComponents(of: weightPath)
    #expect(try Data(contentsOf: installedWeights) == payload(for: weightPath))
}

@Test func installRefusesDescriptorWithoutBaseURL() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
    let staging = try makeStagingDirectory(in: root, entries: fakePayloads)

    #expect(throws: ModelStoreError.notConfigured) {
        try store.install(stagingDirectory: staging, for: fakeDescriptor(baseURL: nil))
    }
}

@Test func statusBecomesCorruptWhenOneOfSeveralFilesChanges() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
    let descriptor = fakeDescriptor()

    let staging = try makeStagingDirectory(in: root, entries: fakePayloads)
    try store.install(stagingDirectory: staging, for: descriptor)

    let tampered = Data("tampered".utf8)
    try tampered.write(to: store.packageURL(for: descriptor).appendingComponents(of: modelPath))
    #expect(store.status(for: descriptor) == .corrupt(
        expected: SHA256.hexDigest(of: payload(for: modelPath)),
        actual: SHA256.hexDigest(of: tampered)
    ))
}

@Test func statusIsMissingWhenOneOfSeveralFilesIsRemoved() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
    let descriptor = fakeDescriptor()

    let staging = try makeStagingDirectory(in: root, entries: fakePayloads)
    try store.install(stagingDirectory: staging, for: descriptor)

    try FileManager.default.removeItem(at: store.packageURL(for: descriptor).appendingComponents(of: weightPath))
    #expect(store.status(for: descriptor) == .missing)
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
    let entries: [(path: String, data: Data)] = [
        ("Manifest.json", Data("small".utf8)),
        (weightPath, large),
    ]
    let files = entries.map { entry in
        ModelFile(relativePath: entry.path, sha256: SHA256.hexDigest(of: entry.data), sizeBytes: Int64(entry.data.count))
    }
    let descriptor = fakeDescriptor(files: files)

    let staging = try makeStagingDirectory(in: root, entries: entries)
    try store.install(stagingDirectory: staging, for: descriptor)
    #expect(store.status(for: descriptor) == .ready(store.packageURL(for: descriptor)))
}

@Test func removeDeletesStoredModelAndIsIdempotent() throws {
    let root = try makeScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
    let descriptor = fakeDescriptor()

    let staging = try makeStagingDirectory(in: root, entries: fakePayloads)
    try store.install(stagingDirectory: staging, for: descriptor)

    try store.remove(descriptor)
    #expect(store.status(for: descriptor) == .missing)
    try store.remove(descriptor)
}

@Test func depthAvailabilityMatrix() {
    let readyURL = URL(fileURLWithPath: "/models/ready", isDirectory: true)
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
