import Foundation

/// One encoded image taken from a capture (a video frame or a photo), ready to be stored in a project.
public struct CapturedFrame: Sendable, Equatable {
    /// File name stored under `images/`, for example `frame_0003.png`.
    public var name: String
    /// Encoded image bytes.
    public var data: Data
    /// Original file name the frame came from.
    public var sourceName: String
    /// Position in the video in seconds, or `nil` for a photo.
    public var timestamp: Double?
    /// 35 mm-equivalent focal length in millimetres from the photo's EXIF (`FocalLenIn35mmFilm`), or
    /// `nil` when unknown (video frames, photos without EXIF). Image import strips the EXIF block, so
    /// this is the only place the focal length survives for pose providers that return no intrinsics.
    public var focalLength35mm: Double?

    public init(name: String, data: Data, sourceName: String, timestamp: Double?,
                focalLength35mm: Double? = nil) {
        self.name = name
        self.data = data
        self.sourceName = sourceName
        self.timestamp = timestamp
        self.focalLength35mm = focalLength35mm
    }
}
