import SwiftUI
import UniformTypeIdentifiers
import SplatCore
import SplatIO

extension UTType {
    nonisolated static let gaussianProject = UTType(exportedAs: "io.github.junostr05.SparseSplat.gsproj",
                                                    conformingTo: .package)
}

/// A `.gsproj` package: images, poses, training settings and checkpoints.
///
/// The document keeps `ProjectContents` as plain data. Reading flattens the package's
/// `FileWrapper` tree into `[path: Data]`; writing rebuilds the directory tree from
/// `ProjectBundle.encode`. `nonisolated` for the same reason as `SplatDocument`.
nonisolated struct ProjectDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.gaussianProject] }
    static var writableContentTypes: [UTType] { [.gaussianProject] }

    var contents: ProjectContents

    /// An empty project with the default manifest.
    init() {
        // New projects start on the default training preset (decision A7), not on the bare defaults.
        var manifest = ProjectManifest()
        manifest.training = TrainingPreset.default.configuration()
        contents = ProjectContents(manifest: manifest, files: [:])
    }

    init(configuration: ReadConfiguration) throws {
        var files: [String: Data] = [:]
        try Self.collectFiles(from: configuration.file, path: "", into: &files)
        let decoded: ProjectContents
        do {
            decoded = try ProjectBundle.decode(files: files)
        } catch {
            throw CocoaError(.fileReadCorruptFile, userInfo: [
                NSLocalizedFailureReasonErrorKey: Self.readableReason(for: error)
            ])
        }
        contents = decoded
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        let encoded = try ProjectBundle.encode(contents)
        let root = FileWrapper(directoryWithFileWrappers: [:])
        for path in encoded.keys.sorted() {
            guard let data = encoded[path] else { continue }
            try Self.insert(data, path: path, into: root)
        }
        return root
    }

    // MARK: - Package flattening

    /// Walks the wrapper tree and stores every regular file under its slash-separated path.
    private static func collectFiles(from wrapper: FileWrapper,
                                     path: String,
                                     into files: inout [String: Data]) throws {
        if wrapper.isDirectory {
            for (name, child) in wrapper.fileWrappers ?? [:] {
                let childPath: String = path.isEmpty ? name : path + "/" + name
                try collectFiles(from: child, path: childPath, into: &files)
            }
        } else if let data = wrapper.regularFileContents {
            files[path] = data
        } else {
            throw CocoaError(.fileReadCorruptFile)
        }
    }

    /// Creates the directories of `path` below `directory` as needed, then adds the file.
    private static func insert(_ data: Data, path: String, into directory: FileWrapper) throws {
        var components: [String] = path.split(separator: "/").map { String($0) }
        guard let name = components.popLast() else {
            throw CocoaError(.fileWriteInvalidFileName)
        }
        var parent: FileWrapper = directory
        for component in components {
            if let existing = parent.fileWrappers?[component], existing.isDirectory {
                parent = existing
            } else {
                let child = FileWrapper(directoryWithFileWrappers: [:])
                child.preferredFilename = component
                _ = parent.addFileWrapper(child)
                parent = child
            }
        }
        let file = FileWrapper(regularFileWithContents: data)
        file.preferredFilename = name
        _ = parent.addFileWrapper(file)
    }

    private static func readableReason(for error: Error) -> String {
        if let projectError = error as? ProjectError {
            switch projectError {
            case .missingManifest:
                return "The project has no manifest.json."
            case .unsupportedFormatVersion(let version):
                return "The project was saved with a newer format version (\(version))."
            case .corruptManifest(let detail):
                return "The project manifest is corrupt: \(detail)."
            case .corruptCheckpoint(let detail):
                return "The saved checkpoint is corrupt: \(detail)."
            }
        }
        return "The project could not be read."
    }
}
