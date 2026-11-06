#if canImport(Darwin)
import Foundation

/// Downloads a model into a `ModelStore`. Abstracted so the UI and tests can substitute it.
public protocol ModelDownloading: Sendable {
    /// Downloads, verifies and installs the model. `progress` receives fractions in 0...1.
    func download(
        _ descriptor: ModelDescriptor,
        to store: ModelStore,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws
}

/// Errors thrown by `ModelDownloading` implementations.
public enum ModelDownloadError: Error, Equatable, LocalizedError {
    case notConfigured
    case httpStatus(Int)
    case offline
    case cancelled
    case store(ModelStoreError)
    /// Any other transport failure, with a description for display.
    case transport(String)

    public var errorDescription: String? {
        switch self {
        case .notConfigured:
            return "The model is not configured in this build."
        case let .httpStatus(code):
            return "The model server answered with HTTP status \(code)."
        case .offline:
            return "The model could not be downloaded because no network connection is available."
        case .cancelled:
            return "The model download was cancelled."
        case let .store(error):
            return error.errorDescription
        case let .transport(message):
            return "The model download failed: \(message)"
        }
    }
}

/// Downloads the model with `URLSession` into a temporary file, then installs it through `ModelStore`,
/// which verifies the SHA-256 checksum.
public struct URLSessionModelDownloader: ModelDownloading {
    private let session: URLSession

    public init(session: URLSession = .shared) {
        self.session = session
    }

    public func download(
        _ descriptor: ModelDescriptor,
        to store: ModelStore,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws {
        guard descriptor.isConfigured, let sourceURL = descriptor.sourceURL else {
            throw ModelDownloadError.notConfigured
        }

        let temporaryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("model-download-\(UUID().uuidString).part", isDirectory: false)
        do {
            try await fetch(from: sourceURL, to: temporaryURL, expectedSize: descriptor.sizeBytes, progress: progress)
            // On success the install moves the file into the store. On failure it may already be deleted.
            try store.install(downloadedFile: temporaryURL, for: descriptor)
            progress(1.0)
        } catch {
            try? FileManager.default.removeItem(at: temporaryURL)
            throw ModelStoreDownloadMapping.map(error)
        }
    }

    private func fetch(
        from url: URL,
        to fileURL: URL,
        expectedSize: Int64?,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws {
        let delegate = DownloadProgressDelegate(expectedSize: expectedSize, progress: progress)
        let (location, response) = try await session.download(from: url, delegate: delegate)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            try? FileManager.default.removeItem(at: location)
            throw ModelDownloadError.httpStatus(http.statusCode)
        }
        do {
            try FileManager.default.moveItem(at: location, to: fileURL)
        } catch {
            throw ModelStoreError.io("could not move the downloaded file: \(error.localizedDescription)")
        }
    }
}

/// Forwards `URLSession` download progress as a fraction in 0...1.
private final class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let expectedSize: Int64?
    private let progress: @Sendable (Double) -> Void

    init(expectedSize: Int64?, progress: @escaping @Sendable (Double) -> Void) {
        self.expectedSize = expectedSize
        self.progress = progress
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        let announced: Int64 = totalBytesExpectedToWrite
        let total: Int64 = announced > 0 ? announced : (expectedSize ?? 0)
        guard total > 0 else {
            return
        }
        let fraction: Double = min(1.0, Double(totalBytesWritten) / Double(total))
        progress(fraction)
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        // The async download API returns the file; nothing to do here.
    }
}

/// Maps arbitrary errors from the download path onto `ModelDownloadError`.
enum ModelStoreDownloadMapping {
    static func map(_ error: Error) -> ModelDownloadError {
        if let downloadError = error as? ModelDownloadError {
            return downloadError
        }
        if let storeError = error as? ModelStoreError {
            return .store(storeError)
        }
        if error is CancellationError {
            return .cancelled
        }
        if let urlError = error as? URLError {
            switch urlError.code {
            case .cancelled:
                return .cancelled
            case .notConnectedToInternet,
                 .networkConnectionLost,
                 .dataNotAllowed,
                 .internationalRoamingOff,
                 .cannotFindHost,
                 .cannotConnectToHost,
                 .timedOut:
                return .offline
            default:
                return .transport(urlError.localizedDescription)
            }
        }
        return .transport(error.localizedDescription)
    }
}
#endif
