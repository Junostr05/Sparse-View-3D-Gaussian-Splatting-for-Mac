import Foundation

/// The state of a model in the local store.
public enum ModelStatus: Equatable, Sendable {
    /// The descriptor has no base URL, no files or an invalid checksum or path.
    case notConfigured
    /// The package directory or one of its files is not stored.
    case missing
    /// A file is stored, but its checksum does not match. An empty `actual` means the file could not be read.
    case corrupt(expected: String, actual: String)
    /// Every file is stored and its checksum matches. The payload is the package directory.
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

/// Stores model packages under `rootDirectory/<id>/<version>/<packageName>/<relativePath>`.
public struct ModelStore: Sendable {
    /// Read chunk size used when hashing stored files.
    static let hashChunkSize: Int = 1 << 20

    public let rootDirectory: URL

    public init(rootDirectory: URL) {
        self.rootDirectory = rootDirectory
    }

    /// The version directory of the descriptor.
    public func directory(for descriptor: ModelDescriptor) -> URL {
        rootDirectory
            .appendingPathComponent(descriptor.id, isDirectory: true)
            .appendingPathComponent(descriptor.version, isDirectory: true)
    }

    /// The installed package directory, `<version directory>/<packageName>`.
    public func packageURL(for descriptor: ModelDescriptor) -> URL {
        directory(for: descriptor).appendingComponents(of: descriptor.packageName)
    }

    public func status(for descriptor: ModelDescriptor) -> ModelStatus {
        guard descriptor.isConfigured else {
            return .notConfigured
        }
        let fileManager = FileManager.default
        let package = packageURL(for: descriptor)
        guard fileManager.fileExists(atPath: package.path) else {
            return .missing
        }
        // Check presence of every file before hashing any, so a missing file is reported as missing.
        for file in descriptor.files {
            let url = package.appendingComponents(of: file.relativePath)
            if !fileManager.fileExists(atPath: url.path) {
                return .missing
            }
        }
        for file in descriptor.files {
            let url = package.appendingComponents(of: file.relativePath)
            do {
                let actual = try ModelStore.hashFile(at: url)
                if actual != file.sha256 {
                    return .corrupt(expected: file.sha256, actual: actual)
                }
            } catch {
                return .corrupt(expected: file.sha256, actual: "")
            }
        }
        return .ready(package)
    }

    /// Verifies every file in `stagingDirectory` (stored at its relative path) and moves the directory into place
    /// as the package directory, replacing any previously installed package.
    /// If a file is missing or fails its checksum, the staging directory is deleted, nothing is installed and any
    /// existing package is left untouched. On success `stagingDirectory` no longer exists.
    public func install(stagingDirectory: URL, for descriptor: ModelDescriptor) throws {
        guard descriptor.isConfigured else {
            throw ModelStoreError.notConfigured
        }
        let fileManager = FileManager.default

        do {
            for file in descriptor.files {
                let staged = stagingDirectory.appendingComponents(of: file.relativePath)
                guard fileManager.fileExists(atPath: staged.path) else {
                    throw ModelStoreError.checksumMismatch(expected: file.sha256, actual: "")
                }
                let actual: String
                do {
                    actual = try ModelStore.hashFile(at: staged)
                } catch {
                    throw ModelStoreError.io("could not read \(file.relativePath): \(error.localizedDescription)")
                }
                guard actual == file.sha256 else {
                    throw ModelStoreError.checksumMismatch(expected: file.sha256, actual: actual)
                }
            }
        } catch {
            try? fileManager.removeItem(at: stagingDirectory)
            throw error
        }

        let destination = packageURL(for: descriptor)
        let parent = destination.deletingLastPathComponent()
        // Move into the destination directory under a temporary sibling name first, so the final
        // step is a rename within one directory.
        let temporary = parent.appendingPathComponent(
            ".\(destination.lastPathComponent).partial-\(UUID().uuidString)",
            isDirectory: true
        )

        do {
            try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
            try fileManager.moveItem(at: stagingDirectory, to: temporary)
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
            try? fileManager.removeItem(at: stagingDirectory)
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

    /// Copies a user-chosen `.mlpackage` directory into the store as the package of `descriptor`.
    ///
    /// Only `Manifest.json` is checked, because a user-converted package differs byte-wise from any
    /// published one. An existing copy is replaced atomically: the copy goes to a temporary sibling first.
    /// On failure nothing is left behind and any existing copy is kept.
    public func importPackage(from source: URL, for descriptor: ModelDescriptor) throws {
        guard isSafeRelativePath(descriptor.packageName) else {
            throw ModelStoreError.notConfigured
        }
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: source.appendingPathComponent("Manifest.json").path) else {
            throw ModelStoreError.io("\(source.lastPathComponent) is not a Core ML package (Manifest.json is missing)")
        }

        let destination = packageURL(for: descriptor)
        let parent = destination.deletingLastPathComponent()
        let temporary = parent.appendingPathComponent(
            ".\(destination.lastPathComponent).import-\(UUID().uuidString)",
            isDirectory: true
        )
        do {
            try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
            try fileManager.copyItem(at: source, to: temporary)
        } catch {
            try? fileManager.removeItem(at: temporary)
            throw ModelStoreError.io("could not copy the package: \(error.localizedDescription)")
        }

        do {
            if fileManager.fileExists(atPath: destination.path) {
                #if canImport(Darwin)
                _ = try fileManager.replaceItemAt(destination, withItemAt: temporary)
                #else
                try fileManager.removeItem(at: destination)
                try fileManager.moveItem(at: temporary, to: destination)
                #endif
            } else {
                try fileManager.moveItem(at: temporary, to: destination)
            }
        } catch {
            try? fileManager.removeItem(at: temporary)
            throw ModelStoreError.io("could not move the package into place: \(error.localizedDescription)")
        }
    }

    /// The imported package of `descriptor`, or `nil` when none is stored. Checks only for `Manifest.json`.
    /// Unlike `status(for:)`, this works for descriptors without a download source.
    public func importedPackageURL(for descriptor: ModelDescriptor) -> URL? {
        guard isSafeRelativePath(descriptor.packageName) else {
            return nil
        }
        let package = packageURL(for: descriptor)
        let manifest = package.appendingPathComponent("Manifest.json")
        return FileManager.default.fileExists(atPath: manifest.path) ? package : nil
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
