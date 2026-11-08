import Foundation

/// Answers the file questions the drop classifier asks. The app passes `LocalFileProbe`; tests pass an in-memory fake.
public protocol FileTypeProbing {
    /// True when `url` names an existing directory.
    func isDirectory(_ url: URL) -> Bool
    /// True when `url` names an existing file or directory.
    func exists(_ url: URL) -> Bool
    /// The entries of a directory, hidden ones left out. Empty when the directory cannot be read.
    func contents(of directory: URL) -> [URL]
}

/// What a drop onto the project window should do.
public enum DropAction: Sendable, Equatable {
    /// A COLMAP reconstruction folder (`sparse/0/cameras.bin` and `sparse/0/images.bin`).
    case colmapFolder(URL)
    /// An ARKit capture folder (`capture.json`).
    case arkitFolder(URL)
    /// Still images: image files, or one folder that holds images. Imported through the photo pipeline.
    case photos([URL])
    /// A movie file, sampled into frames.
    case video(URL)
    /// A `.gsproj` package. Not imported by a drop; the window tells the user to open it instead.
    case project(URL)
    /// Nothing the project can import. The reason is shown to the user.
    case unsupported(String)
}

/// Decides what a drop means. Pure: every file question goes through `FileTypeProbing`.
public enum DropClassifier {
    /// Lower-case image extensions the photo importer reads.
    public static let imageExtensions: Set<String> = ["jpg", "jpeg", "png", "heic", "heif", "tif", "tiff"]
    /// Lower-case movie extensions the video importer reads.
    public static let videoExtensions: Set<String> = ["mov", "mp4", "m4v"]
    /// Extension of the project package.
    public static let projectExtension: String = "gsproj"

    /// Classifies dropped URLs.
    ///
    /// A single item decides by its kind: a folder is a COLMAP folder, an ARKit folder or a folder of photos (in that
    /// order), an image or movie file is that import, a project is a project. A multi-item drop imports its images
    /// only when it has any, ignoring the other items. Otherwise it imports the one video, the one capture folder
    /// or the one photo folder it holds. Several videos, several folders or several projects are refused.
    public static func classify(urls: [URL], fileSystem: some FileTypeProbing) -> DropAction {
        guard !urls.isEmpty else {
            return .unsupported("Nothing that can be imported was dropped.")
        }
        if urls.count == 1 {
            return classifySingle(urls[0], fileSystem: fileSystem)
        }

        let images: [URL] = urls.filter { isImageFile($0, fileSystem: fileSystem) }
        if !images.isEmpty {
            return .photos(images)
        }

        let videos: [URL] = urls.filter { isVideoFile($0) }
        let projects: [URL] = urls.filter { isProjectPackage($0) }
        let folders: [URL] = urls.filter { !isProjectPackage($0) && fileSystem.isDirectory($0) }
        if videos.count > 1 {
            return .unsupported("Drop one video at a time.")
        }
        if let video = videos.first {
            return .video(video)
        }
        if projects.count > 1 {
            return .unsupported("Drop one project at a time.")
        }
        if let project = projects.first {
            return .project(project)
        }
        if folders.count > 1 {
            return .unsupported("Drop one folder at a time.")
        }
        if let folder = folders.first {
            return classifyFolder(folder, fileSystem: fileSystem)
        }
        return .unsupported("None of the dropped items can be imported.")
    }

    // MARK: - Private

    private static func classifySingle(_ url: URL, fileSystem: some FileTypeProbing) -> DropAction {
        if isProjectPackage(url) {
            return .project(url)
        }
        if fileSystem.isDirectory(url) {
            return classifyFolder(url, fileSystem: fileSystem)
        }
        if isImageFile(url, fileSystem: fileSystem) {
            return .photos([url])
        }
        if isVideoFile(url) {
            return .video(url)
        }
        return .unsupported("\(url.lastPathComponent) is not a photo, a video or a capture folder.")
    }

    private static func classifyFolder(_ folder: URL, fileSystem: some FileTypeProbing) -> DropAction {
        let cameras: URL = folder.appending(path: "sparse/0/cameras.bin")
        let images: URL = folder.appending(path: "sparse/0/images.bin")
        if fileSystem.exists(cameras) && fileSystem.exists(images) {
            return .colmapFolder(folder)
        }
        if fileSystem.exists(folder.appending(path: "capture.json")) {
            return .arkitFolder(folder)
        }
        let hasImages: Bool = fileSystem.contents(of: folder).contains { entry in
            isImageFile(entry, fileSystem: fileSystem)
        }
        if hasImages {
            return .photos([folder])
        }
        return .unsupported("\(folder.lastPathComponent) has no images, no capture.json and no COLMAP sparse/0 folder.")
    }

    private static func isImageFile(_ url: URL, fileSystem: some FileTypeProbing) -> Bool {
        guard !fileSystem.isDirectory(url) else {
            return false
        }
        return imageExtensions.contains(url.pathExtension.lowercased())
    }

    private static func isVideoFile(_ url: URL) -> Bool {
        videoExtensions.contains(url.pathExtension.lowercased())
    }

    private static func isProjectPackage(_ url: URL) -> Bool {
        url.pathExtension.lowercased() == projectExtension
    }
}

/// The real file system, for the app. Errors read as "absent" or "empty", so a folder that cannot be read is
/// reported as unsupported rather than failing the drop.
public struct LocalFileProbe: FileTypeProbing {
    public init() {}

    public func isDirectory(_ url: URL) -> Bool {
        let values: URLResourceValues? = try? url.resourceValues(forKeys: [.isDirectoryKey])
        return values?.isDirectory ?? false
    }

    public func exists(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path)
    }

    public func contents(of directory: URL) -> [URL] {
        let entries: [URL]? = try? FileManager.default.contentsOfDirectory(at: directory,
                                                                           includingPropertiesForKeys: [.isDirectoryKey],
                                                                           options: [.skipsHiddenFiles])
        return entries ?? []
    }
}
