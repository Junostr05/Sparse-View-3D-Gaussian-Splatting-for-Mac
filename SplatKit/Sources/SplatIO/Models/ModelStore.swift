import Foundation

/// The state of a model in the local store.
public enum ModelStatus: Equatable, Sendable {
    /// The descriptor has no source URL or no valid checksum yet.
    case notConfigured
    /// Nothing is stored for this descriptor.
    case missing
    /// A file is stored, but its checksum does not match. An empty `actual` means the file could not be read.
    case corrupt(expected: String, actual: String)
    /// The file is stored and its checksum matches.
    case ready(URL)
}

/// Errors thrown by `ModelStore`.
public enum ModelStoreError: Error, Equatable, LocalizedError {
    case notConfigured
    case checksumMismatch(expected: String, actual: String)
    case io(String)

    public var errorDescription: String? {
        switch self {
        case .notConfigured:
            return "The model is not configured in this build."
        case let .checksumMismatch(expected, actual):
            return "The downloaded model failed its checksum (expected \(expected), got \(actual))."
        case let .io(message):
            return "The model store could not complete the operation: \(message)"
        }
    }
}

/// Stores downloaded models under `rootDirectory/<id>/<version>/<fileName>`.
public struct ModelStore: Sendable {
    /// Read chunk size used when hashing stored files.
    static let hashChunkSize: Int = 1 << 20

    public let rootDirectory: URL

    public init(rootDirectory: URL) {
        self.rootDirectory = rootDirectory
    }

    public func directory(for descriptor: ModelDescriptor) -> URL {
        rootDirectory
            .appendingPathComponent(descriptor.id, isDirectory: true)
            .appendingPathComponent(descriptor.version, isDirectory: true)
    }

    public func fileURL(for descriptor: ModelDescriptor) -> URL {
        directory(for: descriptor).appendingPathComponent(descriptor.fileName, isDirectory: false)
    }

    public func status(for descriptor: ModelDescriptor) -> ModelStatus {
        guard descriptor.isConfigured, let expected = descriptor.sha256 else {
            return .notConfigured
        }
        let url = fileURL(for: descriptor)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return .missing
        }
        do {
            let actual = try ModelStore.hashFile(at: url)
            if actual == expected {
                return .ready(url)
            }
            return .corrupt(expected: expected, actual: actual)
        } catch {
            return .corrupt(expected: expected, actual: "")
        }
    }

    /// Verifies `downloadedFile` against the descriptor's checksum and moves it into the store.
    /// On a checksum mismatch the downloaded file is deleted and nothing is installed.
    /// On success `downloadedFile` no longer exists at its old location.
    public func install(downloadedFile: URL, for descriptor: ModelDescriptor) throws {
        guard descriptor.isConfigured, let expected = descriptor.sha256 else {
            throw ModelStoreError.notConfigured
        }

        let actual: String
        do {
            actual = try ModelStore.hashFile(at: downloadedFile)
        } catch {
            throw ModelStoreError.io("could not read the downloaded file: \(error.localizedDescription)")
        }
        guard actual == expected else {
            try? FileManager.default.removeItem(at: downloadedFile)
            throw ModelStoreError.checksumMismatch(expected: expected, actual: actual)
        }

        let fileManager = FileManager.default
        let versionDirectory = directory(for: descriptor)
        let destination = fileURL(for: descriptor)
        let temporary = versionDirectory.appendingPathComponent(".\(descriptor.fileName).partial-\(UUID().uuidString)")

        do {
            try fileManager.createDirectory(at: versionDirectory, withIntermediateDirectories: true)
            // Move into the destination directory under a temporary name first, so the final
            // step is a rename within one directory.
            try fileManager.moveItem(at: downloadedFile, to: temporary)
            do {
                if fileManager.fileExists(atPath: destination.path) {
                    #if canImport(Darwin)
                    // Atomic replacement on Apple platforms.
                    _ = try fileManager.replaceItemAt(destination, withItemAt: temporary)
                    #else
                    // swift-corelibs-foundation does not replace reliably; remove, then move.
                    try fileManager.removeItem(at: destination)
                    try fileManager.moveItem(at: temporary, to: destination)
                    #endif
                } else {
                    try fileManager.moveItem(at: temporary, to: destination)
                }
            } catch {
                try? fileManager.removeItem(at: temporary)
                throw error
            }
        } catch {
            throw ModelStoreError.io("could not move the model into place: \(error.localizedDescription)")
        }
    }

    /// Deletes the stored version directory of the descriptor. Does nothing when nothing is stored.
    public func remove(_ descriptor: ModelDescriptor) throws {
        let fileManager = FileManager.default
        let versionDirectory = directory(for: descriptor)
        guard fileManager.fileExists(atPath: versionDirectory.path) else {
            return
        }
        do {
            try fileManager.removeItem(at: versionDirectory)
        } catch {
            throw ModelStoreError.io("could not remove the model: \(error.localizedDescription)")
        }
    }

    /// Streams a file through SHA-256 in 1 MiB chunks and returns the lower-case hex digest.
    static func hashFile(at url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let chunk = try handle.read(upToCount: hashChunkSize) ?? Data()
            if chunk.isEmpty {
                break
            }
            hasher.update(chunk)
        }
        return SHA256.hexString(hasher.finalize())
    }
}
