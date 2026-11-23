import Foundation
import SplatCore

/// Depth maps that sit next to photos in a folder: `<photo file name>.sdpt`, for example `IMG_9014.jpg.sdpt`.
///
/// A photo folder import renames each photo, but keeps the original file name as its source name. The depth
/// file of a photo is therefore matched by the source name, not by the frame name.
public enum PhotoDepthMatching {
    /// The suffix of a depth file in a photo folder.
    public static let suffix: String = "." + DepthMapCodec.fileExtension

    /// The photo file name a depth file belongs to (`IMG_9014.jpg.sdpt` -> `IMG_9014.jpg`), or `nil` for other names.
    public static func photoName(forDepthFile fileName: String) -> String? {
        guard fileName.hasSuffix(suffix), fileName.count > suffix.count else {
            return nil
        }
        return String(fileName.dropLast(suffix.count))
    }

    /// Decodes the depth files of a folder, keyed by photo file name. Files that are not depth maps, or that do
    /// not decode, are left out.
    public static func decodeMaps(_ files: [(fileName: String, data: Data)]) -> [String: DepthMap] {
        var maps: [String: DepthMap] = [:]
        for file in files {
            guard let photo = photoName(forDepthFile: file.fileName),
                  let map = try? DepthMapCodec.decode(file.data) else {
                continue
            }
            maps[photo] = map
        }
        return maps
    }

    /// The depth map of each frame whose source photo has one, keyed by frame name.
    public static func depthByFrame(frames: [(name: String, sourceName: String)],
                                    maps: [String: DepthMap]) -> [String: DepthMap] {
        var result: [String: DepthMap] = [:]
        for frame in frames {
            if let map = maps[frame.sourceName] {
                result[frame.name] = map
            }
        }
        return result
    }
}
