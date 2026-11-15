import Foundation

/// A model package found by `ModelStore.availablePackage(for:)`.
public struct AvailablePackage: Equatable, Sendable {
    /// The `.mlpackage` directory.
    public let url: URL
    /// `true` when the package ships inside the app bundle. Such a package is read-only: it is never copied into the
    /// store, never removed and never re-downloaded, and Core ML compiles it into the caches instead.
    public let isBundled: Bool

    public init(url: URL, isBundled: Bool) {
        self.url = url
        self.isBundled = isBundled
    }
}

/// Lookup of model packages shipped inside the app bundle, `Contents/Resources/Models/<id>/<packageName>`.
/// Pure: it only checks the file system, so the app and the command-line tool share it.
public enum BundledModels {
    /// `<baseDirectory>/<descriptor.id>/<descriptor.packageName>` when that directory exists, else `nil`.
    /// `baseDirectory` is `nil` when there is no bundle to look into (the command-line tool).
    public static func packageURL(for descriptor: ModelDescriptor, baseDirectory: URL?) -> URL? {
        guard let baseDirectory, isSafeRelativePath(descriptor.id), isSafeRelativePath(descriptor.packageName) else {
            return nil
        }
        let package: URL = baseDirectory
            .appendingPathComponent(descriptor.id, isDirectory: true)
            .appendingComponents(of: descriptor.packageName)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: package.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return nil
        }
        return package
    }

    /// Name of the compiled-model cache directory of a bundled package: `<id>-<stamp>`, where the stamp is the
    /// whole-second modification time of `Manifest.json` and its size. A rebuilt package changes the stamp, so an
    /// older compiled copy is never reused. `nil` when `Manifest.json` cannot be read.
    public static func cacheDirectoryName(for package: URL, id: String) -> String? {
        let manifest: URL = package.appendingPathComponent("Manifest.json")
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: manifest.path),
              let modified = attributes[.modificationDate] as? Date,
              let size = attributes[.size] as? NSNumber else {
            return nil
        }
        let stamp: Int = Int(modified.timeIntervalSince1970.rounded(.down))
        return "\(id)-\(stamp)-\(size.int64Value)"
    }
}
