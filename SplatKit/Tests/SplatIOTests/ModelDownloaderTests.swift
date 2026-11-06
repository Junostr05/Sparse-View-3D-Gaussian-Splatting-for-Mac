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

private let packageName = "Fake.mlpackage"
private let manifestPath = "Manifest.json"
private let modelPath = "Data/com.apple.CoreML/model.mlmodel"
private let weightPath = "Data/com.apple.CoreML/weights/weight.bin"

/// The nested payloads of the fake package.
private let payloads: [(path: String, data: Data)] = [
    (manifestPath, Data("fake manifest for the downloader test".utf8)),
    (modelPath, Data("fake model payload for the downloader test".utf8)),
    (weightPath, Data("fake weight bytes for the downloader test, a little longer".utf8)),
]

/// Larger payloads for the stub tests, so each file is delivered in several chunks.
private let largePayloads: [(path: String, data: Data)] = [
    (manifestPath, Data(repeating: 0x41, count: 40_000)),
    (modelPath, Data(repeating: 0x42, count: 80_000)),
    (weightPath, Data(repeating: 0x43, count: 120_000)),
]

private func payload(for path: String, in entries: [(path: String, data: Data)] = payloads) -> Data {
    entries.first { $0.path == path }?.data ?? Data()
}

/// Writes the fake package below `baseURL/<packageName>/<relativePath>`, the layout the downloader reads from.
private func writeSourceTree(under baseURL: URL) throws {
    let package = baseURL.appendingComponents(of: packageName)
    for entry in payloads {
        let url = package.appendingComponents(of: entry.path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try entry.data.write(to: url)
    }
}

private func fakeFiles(
    entries: [(path: String, data: Data)] = payloads,
    sha256Override: [String: String] = [:]
) -> [ModelFile] {
    entries.map { entry in
        ModelFile(
            relativePath: entry.path,
            sha256: sha256Override[entry.path] ?? SHA256.hexDigest(of: entry.data),
            sizeBytes: Int64(entry.data.count)
        )
    }
}

/// Names of the staging directories currently in the temporary directory.
private func stagingEntries() -> Set<String> {
    let names = (try? FileManager.default.contentsOfDirectory(atPath: FileManager.default.temporaryDirectory.path)) ?? []
    return Set(names.filter { $0.hasPrefix("model-download-") })
}

/// Collects progress values from the `@Sendable` callback.
private final class ProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Double] = []

    func record(_ value: Double) {
        lock.lock()
        values.append(value)
        lock.unlock()
    }

    var snapshot: [Double] {
        lock.lock()
        defer { lock.unlock() }
        return values
    }
}

/// One canned HTTP answer for `StubURLProtocol`.
private struct StubResponse: Sendable {
    let status: Int
    let body: Data
    let chunkSize: Int
    /// When `true`, the stub delivers the first chunk and never finishes, so a download stays in progress.
    let stall: Bool
}

/// Answers requests to `stub.invalid` from `StubRegistry`. No request ever reaches the network.
private final class StubURLProtocol: URLProtocol {
    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host == "stub.invalid"
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard let url = request.url, let stub = StubRegistry.response(for: url),
              let response = HTTPURLResponse(
                  url: url,
                  statusCode: stub.status,
                  httpVersion: "HTTP/1.1",
                  headerFields: ["Content-Length": String(stub.body.count)]
              ) else {
            client?.urlProtocol(self, didFailWithError: URLError(.fileDoesNotExist))
            return
        }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        var offset: Int = 0
        while offset < stub.body.count {
            let end: Int = min(offset + stub.chunkSize, stub.body.count)
            client?.urlProtocol(self, didLoad: stub.body.subdata(in: offset..<end))
            offset = end
            if stub.stall {
                return
            }
        }
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

/// Canned responses for `StubURLProtocol`, keyed by absolute URL.
private enum StubRegistry {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var responses: [String: StubResponse] = [:]

    static func set(_ response: StubResponse, for url: URL) {
        lock.lock()
        defer { lock.unlock() }
        responses[url.absoluteString] = response
    }

    static func response(for url: URL) -> StubResponse? {
        lock.lock()
        defer { lock.unlock() }
        return responses[url.absoluteString]
    }
}

/// Registers every file of the package below `base` with the stub.
private func registerStubPackage(
    base: URL,
    entries: [(path: String, data: Data)],
    status: Int = 200,
    chunkSize: Int = 4096,
    stallOnFirstFile: Bool = false
) {
    let package = base.appendingComponents(of: packageName)
    for (index, entry) in entries.enumerated() {
        let url = package.appendingComponents(of: entry.path)
        StubRegistry.set(
            StubResponse(
                status: status,
                body: entry.data,
                chunkSize: chunkSize,
                stall: stallOnFirstFile && index == 0
            ),
            for: url
        )
    }
}

private func makeStubSession() -> URLSession {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [StubURLProtocol.self]
    return URLSession(configuration: configuration)
}

/// Runs the downloader tests one at a time, because some of them inspect the shared temporary directory
/// and share the stub registry.
@Suite(.serialized)
struct ModelDownloaderTests {
    /// Uses file:// URLs as the source so the tests need no network.
    @Test func downloaderInstallsNestedPackageFromFileURLs() async throws {
        let root = try makeDownloadScratchDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("source", isDirectory: true)
        try writeSourceTree(under: source)

        let descriptor = ModelDescriptor(
            id: "test-download",
            displayName: "Test Download",
            version: "1",
            license: "Apache-2.0",
            baseURL: source,
            packageName: packageName,
            files: fakeFiles()
        )
        let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
        let recorder = ProgressRecorder()

        try await URLSessionModelDownloader().download(descriptor, to: store, progress: { recorder.record($0) })

        let package = store.packageURL(for: descriptor)
        #expect(store.status(for: descriptor) == .ready(package))
        for entry in payloads {
            #expect(try Data(contentsOf: package.appendingComponents(of: entry.path)) == entry.data)
        }

        let values = recorder.snapshot
        #expect(values.last == 1.0)
        #expect(values.allSatisfy { $0 >= 0.0 && $0 <= 1.0 })
        #expect(zip(values, values.dropFirst()).allSatisfy { pair in pair.0 <= pair.1 })
    }

    /// Byte-level progress: the stub delivers each file in several chunks, so the downloader must report
    /// intermediate fractions, not only the final 1.0.
    @Test func downloaderReportsIntermediateProgressOverHTTP() async throws {
        let root = try makeDownloadScratchDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let base = try #require(URL(string: "https://stub.invalid/progress"))
        registerStubPackage(base: base, entries: largePayloads)
        let descriptor = ModelDescriptor(
            id: "test-progress",
            displayName: "Test Progress",
            version: "1",
            license: "Apache-2.0",
            baseURL: base,
            packageName: packageName,
            files: fakeFiles(entries: largePayloads)
        )
        let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
        let recorder = ProgressRecorder()

        try await URLSessionModelDownloader(session: makeStubSession())
            .download(descriptor, to: store, progress: { recorder.record($0) })

        #expect(store.status(for: descriptor) == .ready(store.packageURL(for: descriptor)))
        let values = recorder.snapshot
        #expect(values.contains { $0 > 0.0 && $0 < 1.0 }, "expected progress strictly between 0 and 1, got \(values)")
        #expect(values.allSatisfy { $0 >= 0.0 && $0 <= 1.0 })
        #expect(zip(values, values.dropFirst()).allSatisfy { pair in pair.0 <= pair.1 }, "progress went backwards: \(values)")
        #expect(values.last == 1.0)
    }

    @Test func downloaderRefusesDescriptorWithoutBaseURL() async throws {
        let root = try makeDownloadScratchDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))

        let descriptor = ModelDescriptor(
            id: "test-unconfigured",
            displayName: "Test Unconfigured",
            version: "1",
            license: "Apache-2.0",
            baseURL: nil,
            packageName: packageName,
            files: fakeFiles()
        )

        var thrown: ModelDownloadError?
        do {
            try await URLSessionModelDownloader().download(descriptor, to: store, progress: { _ in })
        } catch let error as ModelDownloadError {
            thrown = error
        }
        #expect(thrown == .notConfigured)
    }

    @Test func downloaderLeavesNoStagingDirectoryAfterChecksumFailure() async throws {
        let root = try makeDownloadScratchDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appendingPathComponent("source", isDirectory: true)
        try writeSourceTree(under: source)

        let wrongHash = SHA256.hexDigest(of: Data("not the model".utf8))
        let descriptor = ModelDescriptor(
            id: "test-checksum",
            displayName: "Test Checksum",
            version: "1",
            license: "Apache-2.0",
            baseURL: source,
            packageName: packageName,
            files: fakeFiles(sha256Override: [modelPath: wrongHash])
        )
        let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
        let before = stagingEntries()

        var thrown: ModelDownloadError?
        do {
            try await URLSessionModelDownloader().download(descriptor, to: store, progress: { _ in })
        } catch let error as ModelDownloadError {
            thrown = error
        }

        #expect(thrown == .store(.checksumMismatch(
            expected: wrongHash,
            actual: SHA256.hexDigest(of: payload(for: modelPath))
        )))
        #expect(stagingEntries().subtracting(before).isEmpty)
        #expect(store.status(for: descriptor) == .missing)
    }

    @Test func downloaderMapsHTTPNotFoundAndLeavesNoStagingDirectory() async throws {
        let root = try makeDownloadScratchDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let base = try #require(URL(string: "https://stub.invalid/not-found"))
        registerStubPackage(base: base, entries: largePayloads, status: 404)
        let descriptor = ModelDescriptor(
            id: "test-404",
            displayName: "Test 404",
            version: "1",
            license: "Apache-2.0",
            baseURL: base,
            packageName: packageName,
            files: fakeFiles(entries: largePayloads)
        )
        let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
        let before = stagingEntries()

        var thrown: ModelDownloadError?
        do {
            try await URLSessionModelDownloader(session: makeStubSession())
                .download(descriptor, to: store, progress: { _ in })
        } catch let error as ModelDownloadError {
            thrown = error
        }

        #expect(thrown == .httpStatus(404))
        #expect(stagingEntries().subtracting(before).isEmpty)
        #expect(store.status(for: descriptor) == .missing)
    }

    @Test func cancellingRunningDownloadMapsToCancelledAndLeavesNoStagingDirectory() async throws {
        let root = try makeDownloadScratchDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let base = try #require(URL(string: "https://stub.invalid/stalled"))
        // The first file stalls after one chunk, so the download is still running when it is cancelled.
        registerStubPackage(base: base, entries: largePayloads, stallOnFirstFile: true)
        let descriptor = ModelDescriptor(
            id: "test-cancel",
            displayName: "Test Cancel",
            version: "1",
            license: "Apache-2.0",
            baseURL: base,
            packageName: packageName,
            files: fakeFiles(entries: largePayloads)
        )
        let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
        let recorder = ProgressRecorder()
        let downloader = URLSessionModelDownloader(session: makeStubSession())
        let before = stagingEntries()

        let task = Task {
            try await downloader.download(descriptor, to: store, progress: { recorder.record($0) })
        }
        // Wait until the download has reported some bytes, then cancel it.
        var waited: Int = 0
        while recorder.snapshot.isEmpty && waited < 500 {
            try await Task.sleep(nanoseconds: 10_000_000)
            waited += 1
        }
        task.cancel()

        var thrown: ModelDownloadError?
        do {
            try await task.value
        } catch let error as ModelDownloadError {
            thrown = error
        }

        #expect(thrown == .cancelled)
        #expect(stagingEntries().subtracting(before).isEmpty)
        #expect(store.status(for: descriptor) == .missing)
    }
}
#endif
