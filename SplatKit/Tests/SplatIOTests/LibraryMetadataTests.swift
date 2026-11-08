import Foundation
import Testing
@testable import SplatIO

private struct SortItem: LibrarySortable {
    let libraryName: String
    let libraryModificationDate: Date?
    let libraryLastOpened: Date?
    let librarySizeInBytes: Int64?
}

@Suite("Library metadata, naming, sorting and search")
struct LibraryMetadataTests {
    private func manifestData(images: [String],
                              inputKind: InputKind?,
                              checkpoints: [CheckpointRecord]) throws -> Data {
        var manifest = ProjectManifest()
        manifest.imageFiles = images
        manifest.inputKind = inputKind
        manifest.checkpoints = checkpoints
        return try ProjectBundle.encodeManifest(manifest)
    }

    private func record(iteration: Int, gaussians: Int, psnr: Double?) -> CheckpointRecord {
        CheckpointRecord(iteration: iteration, gaussianCount: gaussians, psnr: psnr, date: Date(timeIntervalSince1970: 1_000))
    }

    // MARK: - Metadata

    @Test("A trained project reports its photos, input and latest checkpoint")
    func trainedProject() throws {
        let data = try manifestData(
            images: ["a.jpg", "b.jpg", "c.jpg"],
            inputKind: .photos,
            checkpoints: [record(iteration: 100, gaussians: 500, psnr: 28.0),
                          record(iteration: 10_000, gaussians: 757_000, psnr: 31.1)]
        )
        let metadata = try #require(LibraryMetadata.read(manifestData: data))
        #expect(metadata.photoCount == 3)
        #expect(metadata.inputKind == .photos)
        #expect(metadata.hasCheckpoint)
        #expect(metadata.gaussianCount == 757_000)
        #expect(metadata.iterations == 10_000)
        #expect(metadata.psnr == 31.1)
        #expect(metadata.firstImageName == "a.jpg")
    }

    @Test("An untrained project has no checkpoint fields")
    func untrainedProject() throws {
        let data = try manifestData(images: ["a.jpg"], inputKind: nil, checkpoints: [])
        let metadata = try #require(LibraryMetadata.read(manifestData: data))
        #expect(metadata.photoCount == 1)
        #expect(metadata.inputKind == nil)
        #expect(!metadata.hasCheckpoint)
        #expect(metadata.gaussianCount == nil)
        #expect(metadata.iterations == nil)
        #expect(metadata.psnr == nil)
    }

    @Test("A checkpoint without PSNR keeps the other fields")
    func checkpointWithoutPSNR() throws {
        let data = try manifestData(images: [], inputKind: .video,
                                    checkpoints: [record(iteration: 5, gaussians: 9, psnr: nil)])
        let metadata = try #require(LibraryMetadata.read(manifestData: data))
        #expect(metadata.psnr == nil)
        #expect(metadata.gaussianCount == 9)
    }

    @Test("Garbage and newer format versions are not readable")
    func unreadableManifests() throws {
        #expect(LibraryMetadata.read(manifestData: Data("not json".utf8)) == nil)
        let newer = Data(#"{"formatVersion": 99, "imageFiles": [], "training": {}, "checkpoints": []}"#.utf8)
        #expect(LibraryMetadata.read(manifestData: newer) == nil)
    }

    // MARK: - Naming

    @Test("Unique names count up from the base")
    func uniqueNames() {
        #expect(LibraryNaming.uniqueName(base: "Untitled Scene", existing: []) == "Untitled Scene")
        #expect(LibraryNaming.uniqueName(base: "Untitled Scene", existing: ["Other"]) == "Untitled Scene")
        #expect(LibraryNaming.uniqueName(base: "Untitled Scene", existing: ["Untitled Scene"]) == "Untitled Scene 2")
        #expect(LibraryNaming.uniqueName(base: "Untitled Scene",
                                         existing: ["Untitled Scene", "Untitled Scene 2"]) == "Untitled Scene 3")
        #expect(LibraryNaming.uniqueName(base: "Untitled Scene", existing: ["untitled scene"]) == "Untitled Scene 2")
    }

    @Test("Copies get the copy suffix and then count up")
    func copyNames() {
        let base = LibraryNaming.copyBase(for: "Table")
        #expect(base == "Table copy")
        #expect(LibraryNaming.uniqueName(base: base, existing: ["Table"]) == "Table copy")
        #expect(LibraryNaming.uniqueName(base: base, existing: ["Table", "Table copy"]) == "Table copy 2")
    }

    @Test("Valid names are trimmed, short and free of path characters")
    func validNames() {
        #expect(LibraryNaming.isValidName("Table scene"))
        #expect(LibraryNaming.isValidName("Scene 2 (final)"))
        #expect(LibraryNaming.isValidName("Café"))
        #expect(LibraryNaming.isValidName(String(repeating: "a", count: 120)))
        #expect(LibraryNaming.trimmed("  Table  ") == "Table")
    }

    @Test("Invalid names are refused")
    func invalidNames() {
        #expect(!LibraryNaming.isValidName(""))
        #expect(!LibraryNaming.isValidName("  "))
        #expect(!LibraryNaming.isValidName(" Table"))
        #expect(!LibraryNaming.isValidName("Table "))
        #expect(!LibraryNaming.isValidName("a/b"))
        #expect(!LibraryNaming.isValidName("a:b"))
        #expect(!LibraryNaming.isValidName("a\\b"))
        #expect(!LibraryNaming.isValidName(".hidden"))
        #expect(!LibraryNaming.isValidName("tab\tname"))
        #expect(!LibraryNaming.isValidName(String(repeating: "a", count: 121)))
    }

    // MARK: - Sorting and search

    private let items: [SortItem] = [
        SortItem(libraryName: "Beta",
                 libraryModificationDate: Date(timeIntervalSince1970: 200),
                 libraryLastOpened: nil,
                 librarySizeInBytes: 10),
        SortItem(libraryName: "alpha",
                 libraryModificationDate: Date(timeIntervalSince1970: 100),
                 libraryLastOpened: Date(timeIntervalSince1970: 500),
                 librarySizeInBytes: nil),
        SortItem(libraryName: "Gamma",
                 libraryModificationDate: Date(timeIntervalSince1970: 300),
                 libraryLastOpened: Date(timeIntervalSince1970: 400),
                 librarySizeInBytes: 99),
    ]

    @Test("Name sort ignores case")
    func sortByName() {
        let names = LibrarySortOrder.sorted(items, by: .name).map(\.libraryName)
        #expect(names == ["alpha", "Beta", "Gamma"])
    }

    @Test("Date sort puts the newest first")
    func sortByDate() {
        let names = LibrarySortOrder.sorted(items, by: .dateModified).map(\.libraryName)
        #expect(names == ["Gamma", "Beta", "alpha"])
    }

    @Test("Last opened sort puts projects never opened last")
    func sortByLastOpened() {
        let names = LibrarySortOrder.sorted(items, by: .lastOpened).map(\.libraryName)
        #expect(names == ["alpha", "Gamma", "Beta"])
    }

    @Test("Size sort puts the largest first and unknown sizes last")
    func sortBySize() {
        let names = LibrarySortOrder.sorted(items, by: .size).map(\.libraryName)
        #expect(names == ["Gamma", "Beta", "alpha"])
    }

    @Test("Search ignores case and diacritics")
    func search() {
        #expect(LibrarySearch.matches("Café table", query: "cafe"))
        #expect(LibrarySearch.matches("Café table", query: "TABLE"))
        #expect(LibrarySearch.matches("Café table", query: "  "))
        #expect(LibrarySearch.matches("Café table", query: ""))
        #expect(!LibrarySearch.matches("Café table", query: "chair"))
    }

    // MARK: - Cover

    @Test("A cover round-trips through encode and decode")
    func coverRoundTrip() throws {
        let jpeg = Data([0xFF, 0xD8, 0xFF, 0xD9])
        let contents = ProjectBundle.withCover(ProjectContents(manifest: ProjectManifest(), files: [:]), jpeg: jpeg)
        #expect(ProjectBundle.cover(contents) == jpeg)

        let encoded = try ProjectBundle.encode(contents)
        #expect(encoded[ProjectLayout.coverImage] == jpeg)
        let decoded = try ProjectBundle.decode(files: encoded)
        #expect(ProjectBundle.cover(decoded) == jpeg)
    }

    @Test("A project without a cover decodes without one")
    func projectWithoutCover() throws {
        let encoded = try ProjectBundle.encode(ProjectContents(manifest: ProjectManifest(), files: [:]))
        let decoded = try ProjectBundle.decode(files: encoded)
        #expect(ProjectBundle.cover(decoded) == nil)
    }

    @Test("An import removes the cover of the replaced project")
    func importRemovesCover() {
        let withCover = ProjectBundle.withCover(ProjectContents(manifest: ProjectManifest(), files: [:]),
                                                jpeg: Data([1, 2, 3]))
        let imported = ProjectBundle.importARKit(capture: Data("{}".utf8), imageFiles: [:], into: withCover)
        #expect(ProjectBundle.cover(imported) == nil)
    }
}
