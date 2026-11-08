import Foundation

/// Reads and writes a `.gsproj` package on disk. The app's document uses the same `ProjectBundle` encoding through
/// `FileWrapper`; this is the path for work without a document (the training queue and the command-line tool).
public enum ProjectPackageFiles {
    public enum PackageError: Error, Equatable, CustomStringConvertible {
        case notADirectory(String)
        case unreadableFile(String)
        case writeFailed(String)

        public var description: String {
            switch self {
            case .notADirectory(let path):
                return "\(path) is not a project package (a folder)"
            case .unreadableFile(let path):
                return "could not read \(path)"
            case .writeFailed(let detail):
                return "could not write the project: \(detail)"
            }
        }
    }

    /// Reads every regular file of the package at `package` and decodes the contents with `ProjectBundle.decode`.
    /// Hidden entries are skipped, as the library does.
    public static func read(package: URL) throws -> ProjectContents {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: package.path(percentEncoded: false), isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw PackageError.notADirectory(package.path(percentEncoded: false))
        }
        var files: [String: Data] = [:]
        // The standardized path of a directory ends with a slash, which the prefix below must not double.
        var root: String = package.standardizedFileURL.path(percentEncoded: false)
        if !root.hasSuffix("/") {
            root += "/"
        }
        let keys: [URLResourceKey] = [.isRegularFileKey]
        guard let enumerator = FileManager.default.enumerator(at: package,
                                                              includingPropertiesForKeys: keys,
                                                              options: [.skipsHiddenFiles]) else {
            throw PackageError.notADirectory(package.path(percentEncoded: false))
        }
        for case let fileURL as URL in enumerator {
            let values: URLResourceValues = try fileURL.resourceValues(forKeys: [.isRegularFileKey])
            guard values.isRegularFile == true else {
                continue
            }
            let full: String = fileURL.standardizedFileURL.path(percentEncoded: false)
            guard full.hasPrefix(root) else {
                continue
            }
            let relative: String = String(full.dropFirst(root.count))
            guard let data = try? Data(contentsOf: fileURL) else {
                throw PackageError.unreadableFile(relative)
            }
            files[relative] = data
        }
        return try ProjectBundle.decode(files: files)
    }

    /// Writes `contents` to `package` atomically: the files go to a hidden sibling first, which then replaces the
    /// package. On failure the existing package is left as it was.
    public static func write(_ contents: ProjectContents, to package: URL) throws {
        let encoded: [String: Data]
        do {
            encoded = try ProjectBundle.encode(contents)
        } catch {
            throw PackageError.writeFailed(String(describing: error))
        }
        let fileManager: FileManager = FileManager.default
        let parent: URL = package.deletingLastPathComponent()
        let staging: URL = parent.appending(path: ".\(package.lastPathComponent).saving-\(UUID().uuidString)",
                                            directoryHint: .isDirectory)
        do {
            try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
            for (path, data) in encoded {
                let file: URL = staging.appending(path: path)
                try fileManager.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
                try data.write(to: file)
            }
            if fileManager.fileExists(atPath: package.path(percentEncoded: false)) {
                #if canImport(Darwin)
                _ = try fileManager.replaceItemAt(package, withItemAt: staging)
                #else
                try fileManager.removeItem(at: package)
                try fileManager.moveItem(at: staging, to: package)
                #endif
            } else {
                try fileManager.moveItem(at: staging, to: package)
            }
        } catch {
            try? fileManager.removeItem(at: staging)
            throw PackageError.writeFailed(error.localizedDescription)
        }
    }
}
