import Foundation
import Testing
@testable import SplatIO

/// An in-memory file system: paths are registered as files or directories, and each directory lists its entries.
private struct FakeFileSystem: FileTypeProbing {
    private(set) var directories: Set<String> = []
    private(set) var files: Set<String> = []
    private(set) var children: [String: [URL]] = [:]

    mutating func addDirectory(_ path: String) {
        guard !directories.contains(path) else {
            return
        }
        directories.insert(path)
        addToParent(path)
    }

    mutating func addFile(_ path: String) {
        files.insert(path)
        addToParent(path)
    }

    private mutating func addToParent(_ path: String) {
        let url: URL = URL(fileURLWithPath: path)
        let parent: String = url.deletingLastPathComponent().path
        if parent != path {
            addDirectory(parent)
            children[parent, default: []].append(url)
        }
    }

    func isDirectory(_ url: URL) -> Bool {
        directories.contains(url.path)
    }

    func exists(_ url: URL) -> Bool {
        directories.contains(url.path) || files.contains(url.path)
    }

    func contents(of directory: URL) -> [URL] {
        children[directory.path] ?? []
    }
}

private func url(_ path: String) -> URL {
    URL(fileURLWithPath: path)
}

@Suite struct DropClassifierTests {
    // MARK: - Folders

    @Test func colmapFolderNeedsCamerasAndImages() {
        var fs = FakeFileSystem()
        fs.addDirectory("/drop/scene")
        fs.addFile("/drop/scene/sparse/0/cameras.bin")
        fs.addFile("/drop/scene/sparse/0/images.bin")
        let action: DropAction = DropClassifier.classify(urls: [url("/drop/scene")], fileSystem: fs)
        #expect(action == .colmapFolder(url("/drop/scene")))
    }

    @Test func colmapFolderWithoutImagesBinIsNotCOLMAP() {
        var fs = FakeFileSystem()
        fs.addDirectory("/drop/scene")
        fs.addFile("/drop/scene/sparse/0/cameras.bin")
        let action: DropAction = DropClassifier.classify(urls: [url("/drop/scene")], fileSystem: fs)
        #expect(action == .unsupported("scene has no images, no capture.json and no COLMAP sparse/0 folder."))
    }

    @Test func arkitFolderHasCaptureJSON() {
        var fs = FakeFileSystem()
        fs.addDirectory("/drop/capture")
        fs.addFile("/drop/capture/capture.json")
        fs.addFile("/drop/capture/frame_0001.jpg")
        let action: DropAction = DropClassifier.classify(urls: [url("/drop/capture")], fileSystem: fs)
        #expect(action == .arkitFolder(url("/drop/capture")))
    }

    @Test func colmapWinsOverArkitWhenBothPresent() {
        var fs = FakeFileSystem()
        fs.addDirectory("/drop/both")
        fs.addFile("/drop/both/capture.json")
        fs.addFile("/drop/both/sparse/0/cameras.bin")
        fs.addFile("/drop/both/sparse/0/images.bin")
        let action: DropAction = DropClassifier.classify(urls: [url("/drop/both")], fileSystem: fs)
        #expect(action == .colmapFolder(url("/drop/both")))
    }

    @Test func photoFolderIsPhotosWithTheFolder() {
        var fs = FakeFileSystem()
        fs.addDirectory("/drop/photos")
        fs.addFile("/drop/photos/IMG_0001.HEIC")
        fs.addFile("/drop/photos/notes.txt")
        let action: DropAction = DropClassifier.classify(urls: [url("/drop/photos")], fileSystem: fs)
        #expect(action == .photos([url("/drop/photos")]))
    }

    @Test func folderWithoutImagesIsUnsupported() {
        var fs = FakeFileSystem()
        fs.addDirectory("/drop/docs")
        fs.addFile("/drop/docs/readme.txt")
        let action: DropAction = DropClassifier.classify(urls: [url("/drop/docs")], fileSystem: fs)
        guard case .unsupported = action else {
            Issue.record("Expected unsupported, got \(action)")
            return
        }
    }

    @Test func imagesInSubfoldersAreNotPhotos() {
        var fs = FakeFileSystem()
        fs.addDirectory("/drop/outer")
        fs.addFile("/drop/outer/inner/photo.jpg")
        let action: DropAction = DropClassifier.classify(urls: [url("/drop/outer")], fileSystem: fs)
        guard case .unsupported = action else {
            Issue.record("Expected unsupported, got \(action)")
            return
        }
    }

    // MARK: - Single files

    @Test func singleImageIsPhotos() {
        var fs = FakeFileSystem()
        fs.addFile("/drop/shot.PNG")
        let action: DropAction = DropClassifier.classify(urls: [url("/drop/shot.PNG")], fileSystem: fs)
        #expect(action == .photos([url("/drop/shot.PNG")]))
    }

    @Test func everyImageExtensionIsAccepted() {
        for ext in ["jpg", "jpeg", "png", "heic", "heif", "tif", "tiff", "TIFF"] {
            var fs = FakeFileSystem()
            fs.addFile("/drop/shot.\(ext)")
            let action: DropAction = DropClassifier.classify(urls: [url("/drop/shot.\(ext)")], fileSystem: fs)
            #expect(action == .photos([url("/drop/shot.\(ext)")]), "extension \(ext)")
        }
    }

    @Test func singleMovieIsVideo() {
        for ext in ["mov", "mp4", "M4V"] {
            var fs = FakeFileSystem()
            fs.addFile("/drop/clip.\(ext)")
            let action: DropAction = DropClassifier.classify(urls: [url("/drop/clip.\(ext)")], fileSystem: fs)
            #expect(action == .video(url("/drop/clip.\(ext)")), "extension \(ext)")
        }
    }

    @Test func projectPackageIsProjectEvenThoughItIsAFolder() {
        var fs = FakeFileSystem()
        fs.addDirectory("/drop/Scene.gsproj")
        fs.addFile("/drop/Scene.gsproj/manifest.json")
        let action: DropAction = DropClassifier.classify(urls: [url("/drop/Scene.gsproj")], fileSystem: fs)
        #expect(action == .project(url("/drop/Scene.gsproj")))
    }

    @Test func unknownSingleFileIsUnsupported() {
        var fs = FakeFileSystem()
        fs.addFile("/drop/archive.zip")
        let action: DropAction = DropClassifier.classify(urls: [url("/drop/archive.zip")], fileSystem: fs)
        guard case .unsupported = action else {
            Issue.record("Expected unsupported, got \(action)")
            return
        }
    }

    @Test func emptyDropIsUnsupported() {
        let fs = FakeFileSystem()
        let action: DropAction = DropClassifier.classify(urls: [], fileSystem: fs)
        guard case .unsupported = action else {
            Issue.record("Expected unsupported, got \(action)")
            return
        }
    }

    // MARK: - Multi-item drops

    @Test func mixedImagesAndVideoGivePhotosOnly() {
        var fs = FakeFileSystem()
        fs.addFile("/drop/a.jpg")
        fs.addFile("/drop/b.png")
        fs.addFile("/drop/clip.mov")
        let action: DropAction = DropClassifier.classify(
            urls: [url("/drop/a.jpg"), url("/drop/clip.mov"), url("/drop/b.png")],
            fileSystem: fs
        )
        #expect(action == .photos([url("/drop/a.jpg"), url("/drop/b.png")]))
    }

    @Test func imagesWinOverCaptureFolder() {
        var fs = FakeFileSystem()
        fs.addDirectory("/drop/scene")
        fs.addFile("/drop/scene/sparse/0/cameras.bin")
        fs.addFile("/drop/scene/sparse/0/images.bin")
        fs.addFile("/drop/extra.jpg")
        let action: DropAction = DropClassifier.classify(
            urls: [url("/drop/scene"), url("/drop/extra.jpg")],
            fileSystem: fs
        )
        #expect(action == .photos([url("/drop/extra.jpg")]))
    }

    @Test func movieWithOtherNonImageItemsIsVideo() {
        var fs = FakeFileSystem()
        fs.addFile("/drop/clip.mp4")
        fs.addFile("/drop/notes.txt")
        let action: DropAction = DropClassifier.classify(
            urls: [url("/drop/notes.txt"), url("/drop/clip.mp4")],
            fileSystem: fs
        )
        #expect(action == .video(url("/drop/clip.mp4")))
    }

    @Test func imagesWinOverProjectInMixedDrop() {
        var fs = FakeFileSystem()
        fs.addDirectory("/drop/Scene.gsproj")
        fs.addFile("/drop/shot.jpg")
        let action: DropAction = DropClassifier.classify(
            urls: [url("/drop/Scene.gsproj"), url("/drop/shot.jpg")],
            fileSystem: fs
        )
        #expect(action == .photos([url("/drop/shot.jpg")]))
    }

    @Test func singleCaptureFolderAmongNonImagesIsImported() {
        var fs = FakeFileSystem()
        fs.addDirectory("/drop/arkit")
        fs.addFile("/drop/arkit/capture.json")
        fs.addFile("/drop/readme.txt")
        let action: DropAction = DropClassifier.classify(
            urls: [url("/drop/readme.txt"), url("/drop/arkit")],
            fileSystem: fs
        )
        #expect(action == .arkitFolder(url("/drop/arkit")))
    }

    @Test func twoVideosAreRefused() {
        var fs = FakeFileSystem()
        fs.addFile("/drop/a.mov")
        fs.addFile("/drop/b.mov")
        let action: DropAction = DropClassifier.classify(
            urls: [url("/drop/a.mov"), url("/drop/b.mov")],
            fileSystem: fs
        )
        #expect(action == .unsupported("Drop one video at a time."))
    }

    @Test func twoFoldersAreRefused() {
        var fs = FakeFileSystem()
        fs.addDirectory("/drop/one")
        fs.addFile("/drop/one/a.jpg")
        fs.addDirectory("/drop/two")
        fs.addFile("/drop/two/b.jpg")
        let action: DropAction = DropClassifier.classify(
            urls: [url("/drop/one"), url("/drop/two")],
            fileSystem: fs
        )
        #expect(action == .unsupported("Drop one folder at a time."))
    }

    @Test func twoProjectsAreRefused() {
        var fs = FakeFileSystem()
        fs.addDirectory("/drop/A.gsproj")
        fs.addDirectory("/drop/B.gsproj")
        let action: DropAction = DropClassifier.classify(
            urls: [url("/drop/A.gsproj"), url("/drop/B.gsproj")],
            fileSystem: fs
        )
        #expect(action == .unsupported("Drop one project at a time."))
    }

    @Test func nonImageNonVideoMultiDropIsUnsupported() {
        var fs = FakeFileSystem()
        fs.addFile("/drop/a.zip")
        fs.addFile("/drop/b.txt")
        let action: DropAction = DropClassifier.classify(
            urls: [url("/drop/a.zip"), url("/drop/b.txt")],
            fileSystem: fs
        )
        #expect(action == .unsupported("None of the dropped items can be imported."))
    }
}
