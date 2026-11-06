#if canImport(Darwin)
import Foundation
import Testing
@testable import SplatIO

private func makeDownloadScratchDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("ModelDownloaderTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

/// Uses a file:// URL as the source so the test needs no network.
@Test func downloaderInstallsFileURLSource() async throws {
    let root = try makeDownloadScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let payload = Data("small model payload for the downloader test".utf8)
    let source = root.appendingPathComponent("source.bin")
    try payload.write(to: source)

    let descriptor = ModelDescriptor(
        id: "test-download",
        displayName: "Test Download",
        version: "1",
        sourceURL: source,
        sha256: SHA256.hexDigest(of: payload),
        sizeBytes: Int64(payload.count),
        license: "Apache-2.0",
        fileName: "model.bin",
        isArchive: false
    )
    let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))

    try await URLSessionModelDownloader().download(descriptor, to: store, progress: { _ in })

    #expect(store.status(for: descriptor) == .ready(store.fileURL(for: descriptor)))
    #expect(try Data(contentsOf: store.fileURL(for: descriptor)) == payload)
}

@Test func downloaderRefusesUnconfiguredDescriptor() async throws {
    let root = try makeDownloadScratchDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))

    do {
        try await URLSessionModelDownloader().download(ModelCatalog.depthAnythingV2Small, to: store, progress: { _ in })
        Issue.record("download of an unconfigured model must throw")
    } catch let error as ModelDownloadError {
        #expect(error == .notConfigured)
    }
}
#endif
