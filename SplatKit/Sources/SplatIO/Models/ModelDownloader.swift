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

/// Downloads every file of a model package with `URLSession` into a staging directory, then installs that
/// directory through `ModelStore`, which verifies each SHA-256 checksum.
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
        guard descriptor.isConfigured, let baseURL = descriptor.baseURL else {
            throw ModelDownloadError.notConfigured
        }

        let fileManager = FileManager.default
        let staging = fileManager.temporaryDirectory
            .appendingPathComponent("model-download-\(UUID().uuidString)", isDirectory: true)
        let total: Int64 = descriptor.totalSizeBytes
        do {
            try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
            var finishedBytes: Int64 = 0
            for file in descriptor.files {
                try Task.checkCancellation()
                let source = baseURL
                    .appendingComponents(of: descriptor.packageName)
                    .appendingComponents(of: file.relativePath)
                let destination = staging.appendingComponents(of: file.relativePath)
                try fileManager.createDirectory(
                    at: destination.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                let bytesBefore: Int64 = finishedBytes
                let declaredSize: Int64 = file.sizeBytes
                try await fetch(from: source, to: destination) { writtenInFile in
                    guard total > 0 else {
                        return
                    }
                    // Clamp to the declared size, so a file that is larger than declared cannot push its
                    // fraction past the start of the next file and make the progress jump backwards.
                    let writtenBytes: Int64 = min(writtenInFile, declaredSize)
                    let doneBytes: Int64 = bytesBefore + writtenBytes
                    let fraction: Double = min(1.0, max(0.0, Double(doneBytes) / Double(total)))
                    progress(fraction)
                }
                finishedBytes += declaredSize
            }
            try store.install(stagingDirectory: staging, for: descriptor)
            progress(1.0)
        } catch {
            try? fileManager.removeItem(at: staging)
            throw ModelStoreDownloadMapping.map(error)
        }
    }

    /// Downloads `url` to `fileURL`. `progress` receives the bytes written so far for this file.
    /// Uses a callback download task with a task delegate, because the async `download(from:delegate:)`
    /// API does not deliver per-chunk progress callbacks to the delegate.
    private func fetch(
        from url: URL,
        to fileURL: URL,
        progress: @escaping @Sendable (Int64) -> Void
    ) async throws {
        let delegate = DownloadTaskDelegate(destination: fileURL, progress: progress)
        let task = session.downloadTask(with: url)
        task.delegate = delegate
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                delegate.start(continuation)
                task.resume()
            }
        } onCancel: {
            task.cancel()
        }
    }
}

/// Delegate of one `URLSessionDownloadTask`. It moves the finished file into place, reports bytes written, and
/// resumes the waiting continuation exactly once, from `didCompleteWithError`.
private final class DownloadTaskDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private let destination: URL
    private let progress: @Sendable (Int64) -> Void
    // Guarded by `lock`.
    private var continuation: CheckedContinuation<Void, Error>?
    private var pendingError: Error?
    private var finishedFile: Bool = false

    init(destination: URL, progress: @escaping @Sendable (Int64) -> Void) {
        self.destination = destination
        self.progress = progress
    }

    /// Stores the continuation that `didCompleteWithError` will resume.
    func start(_ continuation: CheckedContinuation<Void, Error>) {
        lock.lock()
        defer { lock.unlock() }
        self.continuation = continuation
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        progress(totalBytesWritten)
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        // `location` is deleted when this method returns, so the file is moved here or the error is kept.
        var failure: Error?
        if let http = downloadTask.response as? HTTPURLResponse, http.statusCode != 200 {
            failure = ModelDownloadError.httpStatus(http.statusCode)
        } else {
            do {
                try FileManager.default.moveItem(at: location, to: destination)
            } catch {
                failure = ModelStoreError.io("could not move the downloaded file: \(error.localizedDescription)")
            }
        }
        lock.lock()
        defer { lock.unlock() }
        if let failure {
            pendingError = failure
        } else {
            finishedFile = true
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        lock.lock()
        let pending: CheckedContinuation<Void, Error>? = continuation
        continuation = nil
        let stored: Error? = pendingError
        let moved: Bool = finishedFile
        lock.unlock()

        guard let pending else {
            return
        }
        if let stored {
            pending.resume(throwing: stored)
        } else if let error {
            pending.resume(throwing: ModelStoreDownloadMapping.map(error))
        } else if moved {
            pending.resume()
        } else {
            pending.resume(throwing: ModelDownloadError.transport("the download finished without a file"))
        }
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
