import AppKit
import CoreGraphics
import Foundation
import ImageIO
import SplatIO
import UniformTypeIdentifiers

/// JPEG and image helpers shared by the cover renderer and the library's thumbnails.
nonisolated enum CoverJPEG {
    /// Longest side of a cover, in pixels.
    static let maxPixelSide: Int = 640
    static let quality: Double = 0.85

    /// A CGImage from BGRA8 pixels with rows packed (`width * 4` bytes per row).
    static func image(bgra: Data, width: Int, height: Int) -> CGImage? {
        guard width > 0, height > 0, bgra.count == width * height * 4,
              let provider = CGDataProvider(data: bgra as CFData),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            return nil
        }
        let bitmapInfo: UInt32 = CGBitmapInfo.byteOrder32Little.rawValue
            | CGImageAlphaInfo.noneSkipFirst.rawValue
        return CGImage(width: width,
                       height: height,
                       bitsPerComponent: 8,
                       bitsPerPixel: 32,
                       bytesPerRow: width * 4,
                       space: colorSpace,
                       bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
                       provider: provider,
                       decode: nil,
                       shouldInterpolate: false,
                       intent: .defaultIntent)
    }

    /// JPEG bytes of `image`, or `nil` when encoding fails.
    static func encode(_ image: CGImage) -> Data? {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil) else {
            return nil
        }
        let options: [CFString: Any] = [kCGImageDestinationLossyCompressionQuality: quality]
        CGImageDestinationAddImage(destination, image, options as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            return nil
        }
        return data as Data
    }

    /// A thumbnail of the image file at `url`, scaled so its longest side is `maxPixelSide`, as JPEG bytes.
    static func thumbnailJPEG(of url: URL) -> Data? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            return nil
        }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixelSide,
        ]
        guard let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            return nil
        }
        return encode(thumbnail)
    }
}

/// Produces the cover of a package: the stored `preview/cover.jpg`, or a thumbnail of its first photo.
nonisolated enum LibraryCoverMaker {
    /// The cover bytes of `package`, read from `cacheFile` when it exists. A new cover is written to `cacheFile`.
    @concurrent
    static func cover(of package: URL, firstImageName: String?, cacheFile: URL?) async -> Data? {
        if let cacheFile, let cached = try? Data(contentsOf: cacheFile) {
            return cached
        }
        var rendered: Data? = try? Data(contentsOf: package.appending(path: ProjectLayout.coverImage))
        if rendered == nil, let firstImageName {
            let imageURL: URL = package.appending(path: ProjectLayout.imagesDirectory)
                .appending(path: firstImageName)
            rendered = CoverJPEG.thumbnailJPEG(of: imageURL)
        }
        if let rendered, let cacheFile {
            try? FileManager.default.createDirectory(at: cacheFile.deletingLastPathComponent(),
                                                     withIntermediateDirectories: true)
            try? rendered.write(to: cacheFile, options: .atomic)
        }
        return rendered
    }

    /// File name in the cover cache for a package at `path` modified at `modificationDate`. FNV-1a hash of the key.
    static func cacheFileName(path: String, modificationDate: Date?) -> String {
        let stamp: Double = modificationDate?.timeIntervalSince1970 ?? 0
        let key: String = path + "|" + String(stamp)
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in key.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x100000001b3
        }
        return String(hash, radix: 16) + ".jpg"
    }
}

/// Covers of the library items: an in-memory cache over the cover cache in the app's Caches directory.
@MainActor
final class LibraryCovers {
    private let memory: NSCache<NSString, NSImage> = NSCache()
    private let cacheDirectory: URL?

    init() {
        cacheDirectory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first?
            .appending(path: "SparseSplatLibraryCovers", directoryHint: .isDirectory)
    }

    /// The cover of `item`, or `nil` when the package has neither a cover nor a readable photo.
    func cover(for item: LibraryItem) async -> NSImage? {
        let fileName: String = LibraryCoverMaker.cacheFileName(path: item.url.path(percentEncoded: false),
                                                               modificationDate: item.modificationDate)
        let memoryKey: NSString = fileName as NSString
        if let cached = memory.object(forKey: memoryKey) {
            return cached
        }
        let cacheFile: URL? = cacheDirectory?.appending(path: fileName)
        guard let data = await LibraryCoverMaker.cover(of: item.url,
                                                       firstImageName: item.metadata?.firstImageName,
                                                       cacheFile: cacheFile),
              let image = NSImage(data: data) else {
            return nil
        }
        memory.setObject(image, forKey: memoryKey)
        return image
    }
}
