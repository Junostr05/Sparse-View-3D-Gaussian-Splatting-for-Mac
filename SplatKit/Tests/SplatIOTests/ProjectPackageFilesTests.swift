import Foundation
import Testing
import SplatCore
@testable import SplatIO

/// Writing and reading a `.gsproj` package on disk, as the training queue and the command-line tool do.
struct ProjectPackageFilesTests {
    private func scratchDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(path: "ProjectPackageFilesTests-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func sampleContents(imageName: String, payload: [UInt8]) -> ProjectContents {
        var manifest = ProjectManifest()
        manifest.imageFiles = [imageName]
        let files: [String: Data] = [
            "\(ProjectLayout.imagesDirectory)/\(imageName)": Data(payload),
            "preview/cover.jpg": Data([0xFF, 0xD8, 0xFF]),
        ]
        return ProjectContents(manifest: manifest, files: files)
    }

    @Test func writeThenReadRoundTrips() throws {
        let directory = try scratchDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let package = directory.appending(path: "scene.gsproj", directoryHint: .isDirectory)
        let contents = sampleContents(imageName: "a.jpg", payload: [1, 2, 3])

        try ProjectPackageFiles.write(contents, to: package)
        let read = try ProjectPackageFiles.read(package: package)

        #expect(read.manifest.imageFiles == ["a.jpg"])
        #expect(read.files["images/a.jpg"] == Data([1, 2, 3]))
        #expect(read.files["preview/cover.jpg"] == Data([0xFF, 0xD8, 0xFF]))
        // The staging folder is gone: only the package remains in the directory.
        let entries = try FileManager.default.contentsOfDirectory(atPath: directory.path(percentEncoded: false))
        #expect(entries == ["scene.gsproj"])
    }

    @Test func writeReplacesAnExistingPackage() throws {
        let directory = try scratchDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let package = directory.appending(path: "scene.gsproj", directoryHint: .isDirectory)

        try ProjectPackageFiles.write(sampleContents(imageName: "old.jpg", payload: [7]), to: package)
        try ProjectPackageFiles.write(sampleContents(imageName: "new.jpg", payload: [8, 9]), to: package)
        let read = try ProjectPackageFiles.read(package: package)

        #expect(read.manifest.imageFiles == ["new.jpg"])
        #expect(read.files["images/new.jpg"] == Data([8, 9]))
        #expect(read.files["images/old.jpg"] == nil)
    }

    @Test func readSkipsHiddenStagingEntries() throws {
        let directory = try scratchDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let package = directory.appending(path: "scene.gsproj", directoryHint: .isDirectory)
        try ProjectPackageFiles.write(sampleContents(imageName: "a.jpg", payload: [1]), to: package)
        let hidden = package.appending(path: ".DS_Store")
        try Data([0]).write(to: hidden)

        let read = try ProjectPackageFiles.read(package: package)

        #expect(read.files[".DS_Store"] == nil)
        #expect(read.files.count == 2)
    }

    @Test func readRejectsAPathThatIsNotAFolder() throws {
        let directory = try scratchDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appending(path: "plain.txt")
        try Data([1]).write(to: file)

        #expect(throws: ProjectPackageFiles.PackageError.self) {
            _ = try ProjectPackageFiles.read(package: file)
        }
    }

    @Test func readWithoutManifestIsAProjectError() throws {
        let directory = try scratchDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let package = directory.appending(path: "empty.gsproj", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)

        #expect(throws: ProjectError.missingManifest) {
            _ = try ProjectPackageFiles.read(package: package)
        }
    }
}
