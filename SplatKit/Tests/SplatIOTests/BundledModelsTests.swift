import Foundation
import Testing
@testable import SplatIO

/// Tests for the bundled model lookup and the store's preference of imported over bundled packages.
/// Uses temporary directories only.
@Suite struct BundledModelsTests {
    private func makeScratchRoot() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("BundledModelsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Writes a fake `.mlpackage` directory with a manifest, at `<base>/<id>/<packageName>`.
    private func makeBundledPackage(base: URL, descriptor: ModelDescriptor, marker: String) throws -> URL {
        let package = base
            .appendingPathComponent(descriptor.id, isDirectory: true)
            .appendingPathComponent(descriptor.packageName, isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        try Data("manifest \(marker)".utf8).write(to: package.appendingPathComponent("Manifest.json"))
        return package
    }

    /// Writes a fake source package to import into a store.
    private func makeSourcePackage(in root: URL, marker: String) throws -> URL {
        let package = root.appendingPathComponent("converted-\(marker).mlpackage", isDirectory: true)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        try Data("manifest \(marker)".utf8).write(to: package.appendingPathComponent("Manifest.json"))
        return package
    }

    @Test func noBaseDirectoryMeansNoBundledPackage() {
        #expect(BundledModels.packageURL(for: ModelDescriptor.depthAnything3Base, baseDirectory: nil) == nil)
    }

    @Test func bundledPackageIsFoundAtIdAndPackageName() throws {
        let root = try makeScratchRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let descriptor = ModelDescriptor.depthAnything3Base
        let package = try makeBundledPackage(base: root, descriptor: descriptor, marker: "one")

        let found = try #require(BundledModels.packageURL(for: descriptor, baseDirectory: root))
        #expect(found.standardizedFileURL == package.standardizedFileURL)
        #expect(found.lastPathComponent == "DepthAnything3BaseQ8.mlpackage")
        #expect(found.deletingLastPathComponent().lastPathComponent == "depth-anything-3-base")
    }

    @Test func missingOrFileShapedPackageIsNotBundled() throws {
        let root = try makeScratchRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let descriptor = ModelDescriptor.depthAnything3Base
        #expect(BundledModels.packageURL(for: descriptor, baseDirectory: root) == nil)

        // A plain file where the package directory should be is not a package.
        let idDirectory = root.appendingPathComponent(descriptor.id, isDirectory: true)
        try FileManager.default.createDirectory(at: idDirectory, withIntermediateDirectories: true)
        try Data("not a package".utf8).write(to: idDirectory.appendingPathComponent(descriptor.packageName))
        #expect(BundledModels.packageURL(for: descriptor, baseDirectory: root) == nil)
    }

    @Test func unsafePackageNameIsRejected() throws {
        let root = try makeScratchRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let descriptor = ModelDescriptor(id: "depth-anything-3-base", displayName: "x", version: "1", license: "x",
                                         baseURL: nil, packageName: "../escape.mlpackage", files: [])
        #expect(BundledModels.packageURL(for: descriptor, baseDirectory: root) == nil)
    }

    @Test func cacheDirectoryNameChangesWhenTheManifestChanges() throws {
        let root = try makeScratchRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let package = try makeBundledPackage(base: root, descriptor: ModelDescriptor.depthAnything3Base, marker: "one")
        let manifest = package.appendingPathComponent("Manifest.json")

        let first = try #require(BundledModels.cacheDirectoryName(for: package, id: "depth-anything-3-base"))
        #expect(first.hasPrefix("depth-anything-3-base-"))
        // The same manifest gives the same name.
        #expect(BundledModels.cacheDirectoryName(for: package, id: "depth-anything-3-base") == first)

        // A rebuilt package with a different size (and a later modification date) gets a new name.
        try Data("manifest with more bytes".utf8).write(to: manifest)
        try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSinceNow: 3_600)],
                                              ofItemAtPath: manifest.path)
        let second = try #require(BundledModels.cacheDirectoryName(for: package, id: "depth-anything-3-base"))
        #expect(second != first)
    }

    @Test func cacheDirectoryNameIsNilWithoutAManifest() throws {
        let root = try makeScratchRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let empty = root.appendingPathComponent("empty.mlpackage", isDirectory: true)
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        #expect(BundledModels.cacheDirectoryName(for: empty, id: "x") == nil)
    }

    @Test func storeWithoutBundleReportsNoAvailablePackage() throws {
        let root = try makeScratchRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
        #expect(store.availablePackage(for: ModelDescriptor.depthAnything3Base) == nil)
    }

    @Test func bundledPackageIsAvailableWhenNothingIsImported() throws {
        let root = try makeScratchRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let bundle = root.appendingPathComponent("Models", isDirectory: true)
        let bundled = try makeBundledPackage(base: bundle, descriptor: ModelDescriptor.depthAnything3Base, marker: "b")
        let store = ModelStore(rootDirectory: root.appendingPathComponent("store"), bundledDirectory: bundle)

        let available = try #require(store.availablePackage(for: ModelDescriptor.depthAnything3Base))
        #expect(available.isBundled)
        #expect(available.url.standardizedFileURL == bundled.standardizedFileURL)
        // The bundled copy is never copied into the store.
        #expect(store.importedPackageURL(for: ModelDescriptor.depthAnything3Base) == nil)
    }

    @Test func importedPackageWinsOverTheBundledOne() throws {
        let root = try makeScratchRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let bundle = root.appendingPathComponent("Models", isDirectory: true)
        _ = try makeBundledPackage(base: bundle, descriptor: ModelDescriptor.depthAnything3Base, marker: "b")
        let store = ModelStore(rootDirectory: root.appendingPathComponent("store"), bundledDirectory: bundle)
        let descriptor = ModelDescriptor.depthAnything3Base
        let source = try makeSourcePackage(in: root, marker: "user")
        try store.importPackage(from: source, for: descriptor)

        let available = try #require(store.availablePackage(for: descriptor))
        #expect(!available.isBundled)
        #expect(available.url == store.packageURL(for: descriptor))

        // Removing the imported copy falls back to the bundled package.
        try store.remove(descriptor)
        let fallback = try #require(store.availablePackage(for: descriptor))
        #expect(fallback.isBundled)
    }
}
