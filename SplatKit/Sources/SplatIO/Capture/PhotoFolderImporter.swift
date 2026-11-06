#if canImport(AVFoundation) && canImport(ImageIO) && canImport(CoreGraphics)
import CoreGraphics
import Foundation
import ImageIO
import SplatCore

/// Imports still photos (single files or a folder) as PNG `CapturedFrame`s.
///
/// Each photo is decoded at `maxSide` with its EXIF orientation applied, then re-encoded as PNG
/// without metadata, so GPS and camera EXIF never reach the project. HEIC is read through ImageIO.
///
/// Sandbox: the URLs must be readable by the caller. A caller that got them from a file picker or a
/// dropped folder holds the security scope and releases it after the call. These functions do not
/// start or stop access themselves.
public struct PhotoFolderImporter: Sendable {
    /// Lower-case extensions accepted for import.
    static let supportedExtensions: Set<String> = ["jpg", "jpeg", "png", "heic", "heif", "tif", "tiff"]

    /// Imports the given files in name order (`localizedStandardCompare`). Files with unsupported
    /// extensions throw `unsupportedFile` instead of being skipped.
    /// - Parameters:
    ///   - urls: Image files.
    ///   - maxSide: Longer side of the output images in pixels.
    public static func load(urls: [URL], maxSide: Int = 1600) throws -> [CapturedFrame] {
        guard !urls.isEmpty else {
            throw FrameExtractionError.noInputs
        }
        for url in urls where !isSupported(url) {
            throw FrameExtractionError.unsupportedFile(url.lastPathComponent)
        }
        return try decodeAll(sortedByName(urls), maxSide: maxSide)
    }

    /// Imports every supported image directly inside `folder` (not recursive, hidden files skipped),
    /// in name order. Other files are ignored.
    /// - Parameters:
    ///   - folder: The folder to read.
    ///   - maxSide: Longer side of the output images in pixels.
    public static func load(folder: URL, maxSide: Int = 1600) throws -> [CapturedFrame] {
        let entries: [URL]
        do {
            entries = try FileManager.default.contentsOfDirectory(at: folder,
                                                                  includingPropertiesForKeys: [.isRegularFileKey],
                                                                  options: [.skipsHiddenFiles])
        } catch {
            throw FrameExtractionError.unreadableFolder(folder.lastPathComponent)
        }

        var images: [URL] = []
        for entry in entries where isSupported(entry) && isRegularFile(entry) {
            images.append(entry)
        }
        guard !images.isEmpty else {
            throw FrameExtractionError.emptyFolder(folder.lastPathComponent)
        }
        return try decodeAll(sortedByName(images), maxSide: maxSide)
    }

    // MARK: - Private

    private static func isSupported(_ url: URL) -> Bool {
        return supportedExtensions.contains(url.pathExtension.lowercased())
    }

    private static func isRegularFile(_ url: URL) -> Bool {
        let values: URLResourceValues? = try? url.resourceValues(forKeys: [.isRegularFileKey])
        return values?.isRegularFile ?? false
    }

    private static func sortedByName(_ urls: [URL]) -> [URL] {
        return urls.sorted { (a: URL, b: URL) -> Bool in
            a.lastPathComponent.localizedStandardCompare(b.lastPathComponent) == .orderedAscending
        }
    }

    private static func decodeAll(_ urls: [URL], maxSide: Int) throws -> [CapturedFrame] {
        var frames: [CapturedFrame] = []
        frames.reserveCapacity(urls.count)
        for url in urls {
            let sourceName: String = url.lastPathComponent
            let image: CGImage = try decodeThumbnail(url, sourceName: sourceName, maxSide: maxSide)
            let data: Data = try CaptureImageSupport.pngData(image, sourceName: sourceName)
            let name: String = FrameNaming.name(prefix: "photo", index: frames.count + 1, ext: "png")
            frames.append(CapturedFrame(name: name, data: data, sourceName: sourceName, timestamp: nil))
        }
        return frames
    }

    /// Decodes the first image of the file at `maxSide`, applying the EXIF orientation.
    private static func decodeThumbnail(_ url: URL, sourceName: String, maxSide: Int) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw FrameExtractionError.unreadableImage(sourceName)
        }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceThumbnailMaxPixelSize: NSNumber(value: max(1, maxSide)),
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            throw FrameExtractionError.unreadableImage(sourceName)
        }
        return image
    }
}
#endif
