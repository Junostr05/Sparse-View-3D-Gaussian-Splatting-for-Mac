import Foundation
import SplatIO

/// One project package of the library, or a recent project outside it.
struct LibraryItem: Identifiable, Equatable, Sendable, LibrarySortable {
    let url: URL
    /// Package name without the `.gsproj` extension.
    let name: String
    let modificationDate: Date?
    /// When the project was last opened, from the app's own record or the system's recent documents.
    var lastOpened: Date?
    /// `true` for a recent project that is not in the library folder.
    let isOutsideLibrary: Bool
    /// `nil` when the manifest could not be read.
    let metadata: LibraryMetadata?
    /// Sum of the package's files, filled in after the scan.
    var sizeInBytes: Int64?

    var id: URL { url }

    var libraryName: String { name }
    var libraryModificationDate: Date? { modificationDate }
    var libraryLastOpened: Date? { lastOpened }
    var librarySizeInBytes: Int64? { sizeInBytes }
}

/// A file operation the library cannot complete, with a message for the user.
enum LibraryFileError: LocalizedError {
    case invalidName
    case nameTaken(String)
    case notAPackage(String)

    var errorDescription: String? {
        switch self {
        case .invalidName:
            "The name is empty, too long, starts with a dot, or contains a slash, a colon or a control character."
        case .nameTaken(let name):
            "A project named \u{201C}\(name)\u{201D} already exists in the library."
        case .notAPackage(let name):
            "\u{201C}\(name)\u{201D} is not a SparseSplat project."
        }
    }
}

/// File-system work of the library. Everything here runs off the main actor (`@concurrent`), so a large
/// package never blocks the window.
nonisolated enum LibraryFiles {
    private static let packageExtension: String = LibraryNaming.packageExtension

    // MARK: - Scanning

    /// The packages directly inside `folder`. Hidden entries are skipped, and only the manifest of each package
    /// is read. Sizes are not computed here (see `byteSizes`).
    @concurrent
    static func scan(folder: URL) async throws -> [LibraryItem] {
        let keys: [URLResourceKey] = [.contentModificationDateKey, .isDirectoryKey]
        let entries: [URL] = try FileManager.default.contentsOfDirectory(at: folder,
                                                                         includingPropertiesForKeys: keys,
                                                                         options: [.skipsHiddenFiles])
        let packages: [URL] = entries.filter { $0.pathExtension.lowercased() == packageExtension }
        return await readItems(packages, outsideLibrary: false)
    }

    /// Reads the manifest of each package and returns its library item. Packages whose manifest cannot be read
    /// are still listed, without metadata.
    @concurrent
    static func readItems(_ packages: [URL], outsideLibrary: Bool) async -> [LibraryItem] {
        var items: [LibraryItem] = []
        for package in packages {
            let values: URLResourceValues? = try? package.resourceValues(forKeys: [.contentModificationDateKey,
                                                                                    .isDirectoryKey])
            guard values?.isDirectory == true else {
                continue
            }
            let manifestURL: URL = package.appending(path: ProjectLayout.manifest)
            let metadata: LibraryMetadata? = (try? Data(contentsOf: manifestURL))
                .flatMap { LibraryMetadata.read(manifestData: $0) }
            items.append(LibraryItem(url: package,
                                     name: package.deletingPathExtension().lastPathComponent,
                                     modificationDate: values?.contentModificationDate,
                                     lastOpened: nil,
                                     isOutsideLibrary: outsideLibrary,
                                     metadata: metadata,
                                     sizeInBytes: nil))
        }
        return items
    }

    /// The total size of the files of each package, keyed by package URL.
    @concurrent
    static func byteSizes(of packages: [URL]) async -> [URL: Int64] {
        var sizes: [URL: Int64] = [:]
        for package in packages {
            sizes[package] = byteSize(of: package)
        }
        return sizes
    }

    private static func byteSize(of package: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(at: package,
                                                              includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]) else {
            return 0
        }
        var total: Int64 = 0
        while let next = enumerator.nextObject() {
            guard let url = next as? URL,
                  let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
                  values.isRegularFile == true,
                  let size = values.fileSize else {
                continue
            }
            total += Int64(size)
        }
        return total
    }

    // MARK: - Names

    /// The names of the packages in `folder`, without extension.
    @concurrent
    static func packageNames(in folder: URL) async throws -> [String] {
        let entries: [URL] = try FileManager.default.contentsOfDirectory(at: folder,
                                                                         includingPropertiesForKeys: nil,
                                                                         options: [.skipsHiddenFiles])
        return entries
            .filter { $0.pathExtension.lowercased() == packageExtension }
            .map { $0.deletingPathExtension().lastPathComponent }
    }

    // MARK: - Create, rename, duplicate, import, trash

    /// Writes an empty project (the default training preset, no images) as `name.gsproj` in `folder`.
    @concurrent
    static func createProject(named name: String, in folder: URL) async throws -> URL {
        guard LibraryNaming.isValidName(name) else {
            throw LibraryFileError.invalidName
        }
        let package: URL = folder.appending(path: name + "." + packageExtension, directoryHint: .isDirectory)
        guard !FileManager.default.fileExists(atPath: package.path(percentEncoded: false)) else {
            throw LibraryFileError.nameTaken(name)
        }
        let files: [String: Data] = try ProjectBundle.encode(ProjectDocument().contents)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: false)
        try write(files, into: package)
        return package
    }

    /// Renames a package inside its folder. Refuses invalid names and names already taken.
    @concurrent
    static func rename(_ package: URL, to newName: String) async throws -> URL {
        guard LibraryNaming.isValidName(newName) else {
            throw LibraryFileError.invalidName
        }
        let target: URL = package.deletingLastPathComponent()
            .appending(path: newName + "." + packageExtension, directoryHint: .isDirectory)
        if target.standardizedFileURL == package.standardizedFileURL {
            return package
        }
        guard !FileManager.default.fileExists(atPath: target.path(percentEncoded: false)) else {
            throw LibraryFileError.nameTaken(newName)
        }
        try FileManager.default.moveItem(at: package, to: target)
        return target
    }

    /// Copies a package in its folder under `<name> copy`, or `<name> copy 2` and so on.
    @concurrent
    static func duplicate(_ package: URL) async throws -> URL {
        let folder: URL = package.deletingLastPathComponent()
        let name: String = package.deletingPathExtension().lastPathComponent
        let existing: [String] = try await packageNames(in: folder)
        let unique: String = LibraryNaming.uniqueName(base: LibraryNaming.copyBase(for: name), existing: existing)
        let target: URL = folder.appending(path: unique + "." + packageExtension, directoryHint: .isDirectory)
        try FileManager.default.copyItem(at: package, to: target)
        return target
    }

    /// Copies a package from anywhere into `folder`, keeping its name when free and adding a number when not.
    /// A package that already lies in `folder` is returned unchanged.
    @concurrent
    static func importPackage(_ source: URL, into folder: URL) async throws -> URL {
        guard source.pathExtension.lowercased() == packageExtension else {
            throw LibraryFileError.notAPackage(source.lastPathComponent)
        }
        if source.deletingLastPathComponent().standardizedFileURL == folder.standardizedFileURL {
            return source
        }
        let scoped: Bool = source.startAccessingSecurityScopedResource()
        defer {
            if scoped {
                source.stopAccessingSecurityScopedResource()
            }
        }
        let name: String = source.deletingPathExtension().lastPathComponent
        let existing: [String] = try await packageNames(in: folder)
        let unique: String = LibraryNaming.uniqueName(base: name, existing: existing)
        let target: URL = folder.appending(path: unique + "." + packageExtension, directoryHint: .isDirectory)
        try FileManager.default.copyItem(at: source, to: target)
        return target
    }

    /// Moves packages to the Trash. The system keeps them, so the step is recoverable.
    @concurrent
    static func moveToTrash(_ packages: [URL]) async throws {
        for package in packages {
            try FileManager.default.trashItem(at: package, resultingItemURL: nil)
        }
    }

    // MARK: - Helpers

    /// Creates the directories and files of a package tree below `package`.
    private static func write(_ files: [String: Data], into package: URL) throws {
        for path in files.keys.sorted() {
            guard let data = files[path] else {
                continue
            }
            let fileURL: URL = package.appending(path: path)
            try FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try data.write(to: fileURL)
        }
    }
}
