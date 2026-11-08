import Foundation

/// What the project library shows for one `.gsproj` package. Read from `manifest.json` alone, so a
/// library with many projects never decodes their images or checkpoints.
public struct LibraryMetadata: Equatable, Sendable {
    /// Number of photos (stored images) in the package.
    public var photoCount: Int
    /// The kind of capture the images came from, if recorded.
    public var inputKind: InputKind?
    /// Whether the package holds at least one training checkpoint.
    public var hasCheckpoint: Bool
    /// Gaussian count of the latest checkpoint, `nil` before the first run.
    public var gaussianCount: Int?
    /// Iterations of the latest checkpoint, `nil` before the first run.
    public var iterations: Int?
    /// PSNR of the latest checkpoint on the training views, when it was measured.
    public var psnr: Double?
    /// Name of the first stored image under `images/`, used for a cover when the package has none.
    public var firstImageName: String?

    public init(photoCount: Int,
                inputKind: InputKind?,
                hasCheckpoint: Bool,
                gaussianCount: Int?,
                iterations: Int?,
                psnr: Double?,
                firstImageName: String? = nil) {
        self.photoCount = photoCount
        self.inputKind = inputKind
        self.hasCheckpoint = hasCheckpoint
        self.gaussianCount = gaussianCount
        self.iterations = iterations
        self.psnr = psnr
        self.firstImageName = firstImageName
    }

    /// The fields of a decoded manifest. The latest checkpoint is the last record.
    public init(manifest: ProjectManifest) {
        let latest: CheckpointRecord? = manifest.checkpoints.last
        self.init(photoCount: manifest.imageFiles.count,
                  inputKind: manifest.inputKind,
                  hasCheckpoint: latest != nil,
                  gaussianCount: latest?.gaussianCount,
                  iterations: latest?.iteration,
                  psnr: latest?.psnr,
                  firstImageName: manifest.imageFiles.first)
    }

    /// Reads the library fields from the bytes of `manifest.json`. `nil` when the manifest does not decode
    /// or was written by a newer format version.
    public static func read(manifestData: Data) -> LibraryMetadata? {
        guard let contents = try? ProjectBundle.decode(files: [ProjectLayout.manifest: manifestData]) else {
            return nil
        }
        return LibraryMetadata(manifest: contents.manifest)
    }
}

/// Names of projects in the library: validation, copies and collision-free names.
public enum LibraryNaming {
    /// Longest accepted project name, in characters.
    public static let maxNameLength: Int = 120
    /// Extension of a project package.
    public static let packageExtension: String = "gsproj"
    /// Name of a new project before the user changes it.
    public static let defaultProjectName: String = "Untitled Scene"

    /// The name without leading and trailing whitespace.
    public static func trimmed(_ name: String) -> String {
        name.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// `true` for a name that can be used as a package name: not empty, at most `maxNameLength` characters,
    /// already trimmed, not starting with a dot, and free of `/`, `:`, `\` and control characters.
    public static func isValidName(_ name: String) -> Bool {
        guard !name.isEmpty, name.count <= maxNameLength, name == trimmed(name) else {
            return false
        }
        guard !name.hasPrefix(".") else {
            return false
        }
        for scalar in name.unicodeScalars {
            if scalar == "/" || scalar == ":" || scalar == "\\" || CharacterSet.controlCharacters.contains(scalar) {
                return false
            }
        }
        return true
    }

    /// `base` when no existing name equals it, otherwise `base 2`, `base 3`, … The comparison ignores case,
    /// as the file system of a default volume does.
    public static func uniqueName(base: String, existing: [String]) -> String {
        let taken: Set<String> = Set(existing.map { $0.lowercased() })
        if !taken.contains(base.lowercased()) {
            return base
        }
        var index: Int = 2
        while taken.contains("\(base) \(index)".lowercased()) {
            index += 1
        }
        return "\(base) \(index)"
    }

    /// The base name of a copy: `name copy`. Pass it to `uniqueName` for the final name.
    public static func copyBase(for name: String) -> String {
        name + " copy"
    }
}

/// A project as the library sorts it.
public protocol LibrarySortable {
    var libraryName: String { get }
    var libraryModificationDate: Date? { get }
    var libraryLastOpened: Date? { get }
    var librarySizeInBytes: Int64? { get }
}

/// Sort orders of the library. Name sorts A to Z; the other orders put the newest or largest project first,
/// and projects without the value last. Ties are broken by name.
public enum LibrarySortOrder: String, CaseIterable, Sendable {
    case name
    case dateModified
    case lastOpened
    case size

    public static func sorted<Item: LibrarySortable>(_ items: [Item], by order: LibrarySortOrder) -> [Item] {
        items.sorted { (lhs: Item, rhs: Item) -> Bool in
            let byName: ComparisonResult = lhs.libraryName.compare(rhs.libraryName,
                                                                   options: [.caseInsensitive, .numeric])
            switch order {
            case .name:
                return byName == .orderedAscending
            case .dateModified:
                return Self.newestFirst(lhs.libraryModificationDate, rhs.libraryModificationDate, tie: byName)
            case .lastOpened:
                return Self.newestFirst(lhs.libraryLastOpened, rhs.libraryLastOpened, tie: byName)
            case .size:
                return Self.largestFirst(lhs.librarySizeInBytes, rhs.librarySizeInBytes, tie: byName)
            }
        }
    }

    /// Orders by date, the later first. A missing date comes after every date.
    private static func newestFirst(_ lhs: Date?, _ rhs: Date?, tie: ComparisonResult) -> Bool {
        switch (lhs, rhs) {
        case let (left?, right?):
            if left != right {
                return left > right
            }
        case (_?, nil):
            return true
        case (nil, _?):
            return false
        case (nil, nil):
            break
        }
        return tie == .orderedAscending
    }

    /// Orders by size, the larger first. A missing size comes after every size.
    private static func largestFirst(_ lhs: Int64?, _ rhs: Int64?, tie: ComparisonResult) -> Bool {
        switch (lhs, rhs) {
        case let (left?, right?):
            if left != right {
                return left > right
            }
        case (_?, nil):
            return true
        case (nil, _?):
            return false
        case (nil, nil):
            break
        }
        return tie == .orderedAscending
    }
}

/// Search in the library: a case- and diacritic-insensitive substring match on the project name.
public enum LibrarySearch {
    /// `true` when the query is empty or occurs in `name`, ignoring case and diacritics.
    public static func matches(_ name: String, query: String) -> Bool {
        let needle: String = trimmed(query)
        guard !needle.isEmpty else {
            return true
        }
        return fold(name).contains(fold(needle))
    }

    private static func trimmed(_ query: String) -> String {
        query.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func fold(_ text: String) -> String {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: nil)
    }
}
