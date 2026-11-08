import Foundation
import SplatCore

/// Errors of reading an autosave from a package.
public enum AutosaveError: Error, Equatable, Sendable {
    /// The manifest, the PLY or the optimizer file does not decode.
    case corrupt(String)
    /// The three files disagree (for example a crash between two writes). Nothing is recovered from them.
    case inconsistent(String)
}

/// `checkpoints/autosave.json`: what the autosave files hold. It is written last, so it is the commit marker of an
/// autosave: a reader that finds it can trust the other two files only after checking them against it.
public struct AutosaveManifest: Codable, Sendable, Equatable {
    public static let currentFormatVersion = 1

    public var formatVersion: Int
    /// Cumulative iterations of the lineage the autosave belongs to (the project's stored iterations included).
    public var iteration: Int
    /// When the autosave was written.
    public var date: Date
    /// Gaussians of the autosaved cloud.
    public var gaussianCount: Int
    /// Configuration hash of the run, as 16 hex digits.
    public var configurationHash: String
    /// Name of the file generation this manifest commits (`AutosaveLayout`). Each autosave writes new files under a new
    /// generation and names it here last, so the manifest alone decides which files are the autosave.
    public var generation: String

    public init(iteration: Int, date: Date, gaussianCount: Int, configurationHash: String,
                generation: String = UUID().uuidString) {
        self.formatVersion = Self.currentFormatVersion
        self.iteration = iteration
        self.date = date
        self.gaussianCount = gaussianCount
        self.configurationHash = configurationHash
        self.generation = generation
    }

    /// Hex text of an optimizer configuration hash, as stored in the manifest.
    public static func hashText(_ hash: UInt64) -> String {
        let text = String(hash, radix: 16)
        return String(repeating: "0", count: max(0, 16 - text.count)) + text
    }
}

/// A complete autosave, ready to be written or as read back from a package.
public struct AutosaveSnapshot: Sendable {
    public var manifest: AutosaveManifest
    public var cloud: GaussianCloud
    public var optimizer: OptimizerState

    public init(manifest: AutosaveManifest, cloud: GaussianCloud, optimizer: OptimizerState) {
        self.manifest = manifest
        self.cloud = cloud
        self.optimizer = optimizer
    }
}

/// Package paths of an autosave. Inside `checkpoints/`, next to the final checkpoint, which an autosave never
/// overwrites.
public enum AutosaveLayout {
    /// The manifest, the commit point of an autosave. Its name is fixed; the data files are named by generation.
    public static let manifest = "checkpoints/autosave.json"

    public static func cloud(generation: String) -> String {
        "checkpoints/autosave-\(generation).ply"
    }

    public static func optimizer(generation: String) -> String {
        "checkpoints/autosave-\(generation).optimizer.bin"
    }

    /// The files of one generation, manifest last.
    public static func files(generation: String) -> [String] {
        [cloud(generation: generation), optimizer(generation: generation), manifest]
    }

    /// True for every autosave path (any file in `checkpoints/` whose name starts with `autosave`). Used to keep
    /// autosave files out of the in-memory package contents.
    public static func isAutosavePath(_ path: String) -> Bool {
        path.hasPrefix("checkpoints/autosave")
    }

    /// A generation name is a UUID-like token: letters, digits and dashes only, so it cannot leave `checkpoints/`.
    static func isSafeGeneration(_ generation: String) -> Bool {
        !generation.isEmpty && generation.allSatisfy { $0.isLetter || $0.isNumber || $0 == "-" }
    }
}

/// Reads and writes the autosave files of a package folder. Each file is replaced atomically (`Data.write` with
/// `.atomic` writes a temporary file and renames it), and the manifest goes last. A crash between the writes
/// leaves a manifest that disagrees with the other files, which `read` reports instead of recovering.
public enum AutosaveStore {
    /// Writes the three files of `snapshot` into the package folder at `packageURL`. Validates that the parts agree
    /// first, so an inconsistent snapshot is never written.
    public static func write(_ snapshot: AutosaveSnapshot, packageURL: URL) throws {
        let manifest = snapshot.manifest
        guard snapshot.cloud.count == manifest.gaussianCount,
              snapshot.optimizer.gaussianCount == manifest.gaussianCount,
              snapshot.optimizer.configurationHash == hashValue(of: manifest.configurationHash) else {
            throw AutosaveError.inconsistent("the snapshot parts do not agree")
        }
        let directory = packageURL.appending(path: "checkpoints", directoryHint: .isDirectory)
        guard AutosaveLayout.isSafeGeneration(manifest.generation) else {
            throw AutosaveError.inconsistent("unsafe generation name")
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // The data files of this generation are new names, so the previous autosave stays whole until the manifest
        // below names the new one. A crash before that point leaves the previous autosave readable.
        let generation = manifest.generation
        try PLYWriter.write(snapshot.cloud).write(to: packageURL.appending(path: AutosaveLayout.cloud(generation: generation)),
                                                  options: .atomic)
        try snapshot.optimizer.encode().write(to: packageURL.appending(path: AutosaveLayout.optimizer(generation: generation)),
                                              options: .atomic)
        try encodeManifest(manifest).write(to: packageURL.appending(path: AutosaveLayout.manifest), options: .atomic)
        // Only now, after the commit, the earlier generations go. A failure here leaves harmless leftovers.
        try? removeFiles(packageURL: packageURL) { name in name != "autosave.json" && !name.contains(generation) }
    }

    /// The manifest alone, or nil when the package has no autosave. Cheap: the cloud is not read.
    public static func manifest(packageURL: URL) throws -> AutosaveManifest? {
        let url = packageURL.appending(path: AutosaveLayout.manifest)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        return try decodeManifest(Data(contentsOf: url))
    }

    /// The complete autosave, or nil when there is none. Throws `AutosaveError` when a file is corrupt or the files
    /// disagree with the manifest.
    public static func read(packageURL: URL) throws -> AutosaveSnapshot? {
        guard let manifest = try manifest(packageURL: packageURL) else {
            return nil
        }
        guard AutosaveLayout.isSafeGeneration(manifest.generation) else {
            throw AutosaveError.corrupt("autosave generation name")
        }
        let cloud: GaussianCloud
        do {
            cloud = try PLYReader.read(Data(contentsOf: packageURL.appending(path: AutosaveLayout.cloud(generation: manifest.generation))))
        } catch {
            throw AutosaveError.corrupt("autosave cloud: \(error)")
        }
        let optimizer: OptimizerState
        do {
            optimizer = try OptimizerState(decoding: Data(contentsOf: packageURL.appending(path: AutosaveLayout.optimizer(generation: manifest.generation))))
        } catch {
            throw AutosaveError.corrupt("autosave optimizer state: \(error)")
        }
        guard cloud.count == manifest.gaussianCount, optimizer.gaussianCount == manifest.gaussianCount else {
            throw AutosaveError.inconsistent("the autosave files hold \(cloud.count) and \(optimizer.gaussianCount) Gaussians, the manifest \(manifest.gaussianCount)")
        }
        guard optimizer.configurationHash == hashValue(of: manifest.configurationHash) else {
            throw AutosaveError.inconsistent("the autosave optimizer state belongs to another configuration")
        }
        return AutosaveSnapshot(manifest: manifest, cloud: cloud, optimizer: optimizer)
    }

    /// Removes every autosave file of the package, the manifest included. Missing files are not an error.
    public static func remove(packageURL: URL) throws {
        try removeFiles(packageURL: packageURL) { _ in true }
    }

    /// Removes the autosave files of `checkpoints/` whose names pass `shouldRemove`. Only names that start with
    /// `autosave` are candidates, so the final checkpoint and the other project files are never touched.
    private static func removeFiles(packageURL: URL, where shouldRemove: (String) -> Bool) throws {
        let directory = packageURL.appending(path: "checkpoints", directoryHint: .isDirectory)
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        for name in names where name.hasPrefix("autosave") && shouldRemove(name) {
            try FileManager.default.removeItem(at: directory.appending(path: name))
        }
    }

    /// The manifest encoding used on disk: ISO 8601 dates, sorted keys.
    public static func encodeManifest(_ manifest: AutosaveManifest) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(manifest)
    }

    /// Decodes an autosave manifest. Throws `AutosaveError.corrupt` for bad JSON or a newer format.
    public static func decodeManifest(_ data: Data) throws -> AutosaveManifest {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        do {
            let manifest = try decoder.decode(AutosaveManifest.self, from: data)
            guard manifest.formatVersion == AutosaveManifest.currentFormatVersion else {
                throw AutosaveError.corrupt("autosave format version \(manifest.formatVersion)")
            }
            return manifest
        } catch let error as AutosaveError {
            throw error
        } catch {
            throw AutosaveError.corrupt("autosave manifest: \(error)")
        }
    }

    private static func hashValue(of text: String) -> UInt64 {
        UInt64(text, radix: 16) ?? 0
    }
}

/// Which state a project opens with, when it may have an interrupted run.
public enum RecoveryChoice: Sendable, Equatable {
    /// Open the latest checkpoint as it is.
    case checkpoint
    /// Offer the autosave: it is newer than the latest checkpoint.
    case autosave
}

/// Pure rule for the recovery banner. An autosave is offered when it was written after the latest checkpoint, or,
/// for the same date, when it covers more iterations. Without a checkpoint any autosave with progress is offered.
public enum RecoverySelection {
    public static func choose(autosave: AutosaveManifest?, latestCheckpoint: CheckpointRecord?) -> RecoveryChoice {
        guard let autosave, autosave.iteration > 0, autosave.gaussianCount > 0 else {
            return .checkpoint
        }
        guard let latestCheckpoint else {
            return .autosave
        }
        if autosave.date > latestCheckpoint.date {
            return .autosave
        }
        if autosave.date == latestCheckpoint.date && autosave.iteration > latestCheckpoint.iteration {
            return .autosave
        }
        return .checkpoint
    }
}
