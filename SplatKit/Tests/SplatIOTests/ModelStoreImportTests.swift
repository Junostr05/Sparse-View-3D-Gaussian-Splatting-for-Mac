import Foundation
import Testing
@testable import SplatIO

/// Tests for importing a user-converted `.mlpackage` into the store. Uses temporary directories only.
@Suite struct ModelStoreImportTests {
    private func makeScratchRoot() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelStoreImportTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Writes a fake `.mlpackage` directory with a manifest and one nested file, and returns its URL.
    private func makeSourcePackage(in root: URL, name: String, marker: String) throws -> URL {
        let package = root.appendingPathComponent(name, isDirectory: true)
        let weights = package.appendingPathComponent("Data/com.apple.CoreML/weights", isDirectory: true)
        try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
        try Data("manifest \(marker)".utf8).write(to: package.appendingPathComponent("Manifest.json"))
        try Data("weights \(marker)".utf8).write(to: weights.appendingPathComponent("weight.bin"))
        return package
    }

    @Test func depthAnything3DescriptorIsNotConfiguredAndNotDownloadable() {
        let descriptor = ModelDescriptor.depthAnything3Base
        #expect(descriptor.id == "depth-anything-3-base")
        #expect(descriptor.version == "1")
        #expect(descriptor.license == "Apache-2.0")
        #expect(descriptor.packageName == "DepthAnything3BaseQ8.mlpackage")
        #expect(descriptor.baseURL == nil)
        // The checksums are recorded, but without a base URL the model is still not configured.
        #expect(descriptor.files.count == 3)
        #expect(descriptor.isConfigured == false)
        #expect(ModelCatalog.all.contains(descriptor) == false)
    }

    @Test func importCopiesPackageAndReportsItsURL() throws {
        let root = try makeScratchRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
        let descriptor = ModelDescriptor.depthAnything3Base
        let source = try makeSourcePackage(in: root, name: "converted.mlpackage", marker: "one")

        #expect(store.importedPackageURL(for: descriptor) == nil)
        try store.importPackage(from: source, for: descriptor)

        let imported = try #require(store.importedPackageURL(for: descriptor))
        #expect(imported == store.packageURL(for: descriptor))
        #expect(imported.lastPathComponent == "DepthAnything3BaseQ8.mlpackage")
        let weight = imported.appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin")
        #expect(try Data(contentsOf: weight) == Data("weights one".utf8))
        // The source is copied, not moved.
        #expect(FileManager.default.fileExists(atPath: source.path))
    }

    @Test func importReplacesExistingCopyAndLeavesNoTemporaryDirectory() throws {
        let root = try makeScratchRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
        let descriptor = ModelDescriptor.depthAnything3Base

        let first = try makeSourcePackage(in: root, name: "first.mlpackage", marker: "first")
        try store.importPackage(from: first, for: descriptor)
        let second = try makeSourcePackage(in: root, name: "second.mlpackage", marker: "second")
        try store.importPackage(from: second, for: descriptor)

        let imported = try #require(store.importedPackageURL(for: descriptor))
        let manifest = try Data(contentsOf: imported.appendingPathComponent("Manifest.json"))
        #expect(manifest == Data("manifest second".utf8))

        let entries = try FileManager.default.contentsOfDirectory(atPath: store.directory(for: descriptor).path)
        #expect(entries == [descriptor.packageName])
    }

    @Test func importRejectsDirectoryWithoutManifest() throws {
        let root = try makeScratchRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
        let descriptor = ModelDescriptor.depthAnything3Base

        let notAPackage = root.appendingPathComponent("notes", isDirectory: true)
        try FileManager.default.createDirectory(at: notAPackage, withIntermediateDirectories: true)
        try Data("hello".utf8).write(to: notAPackage.appendingPathComponent("readme.txt"))

        #expect(throws: ModelStoreError.self) {
            try store.importPackage(from: notAPackage, for: descriptor)
        }
        #expect(store.importedPackageURL(for: descriptor) == nil)
    }

    @Test func importKeepsExistingCopyWhenSourceIsMissing() throws {
        let root = try makeScratchRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ModelStore(rootDirectory: root.appendingPathComponent("store"))
        let descriptor = ModelDescriptor.depthAnything3Base

        let good = try makeSourcePackage(in: root, name: "good.mlpackage", marker: "kept")
        try store.importPackage(from: good, for: descriptor)

        let missing = root.appendingPathComponent("does-not-exist.mlpackage", isDirectory: true)
        #expect(throws: ModelStoreError.self) {
            try store.importPackage(from: missing, for: descriptor)
        }
        let imported = try #require(store.importedPackageURL(for: descriptor))
        #expect(try Data(contentsOf: imported.appendingPathComponent("Manifest.json")) == Data("manifest kept".utf8))
    }
}
