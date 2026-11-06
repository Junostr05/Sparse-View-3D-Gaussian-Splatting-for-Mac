import Foundation

/// Errors from importing frames out of videos and photo folders. Associated strings are file names
/// (never full paths) so messages are safe to show.
public enum FrameExtractionError: Error, Sendable, Equatable, LocalizedError {
    case noInputs
    case unsupportedFile(String)
    case unreadableFolder(String)
    case emptyFolder(String)
    case unreadableImage(String)
    case unreadableVideo(String)
    case noVideoTrack(String)
    case zeroDuration(String)
    case noDecodableFrames(String)
    case processingFailed(String)
    case encodingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .noInputs:
            return "No files were selected."
        case .unsupportedFile(let name):
            return "\"\(name)\" is not a supported image (JPEG, PNG, HEIC, TIFF)."
        case .unreadableFolder(let name):
            return "The folder \"\(name)\" could not be read."
        case .emptyFolder(let name):
            return "The folder \"\(name)\" contains no supported images."
        case .unreadableImage(let name):
            return "The image \"\(name)\" could not be decoded."
        case .unreadableVideo(let name):
            return "The video \"\(name)\" could not be read."
        case .noVideoTrack(let name):
            return "\"\(name)\" has no video track."
        case .zeroDuration(let name):
            return "\"\(name)\" has no duration."
        case .noDecodableFrames(let name):
            return "No frame could be decoded from \"\(name)\"."
        case .processingFailed(let name):
            return "Image processing failed for \"\(name)\"."
        case .encodingFailed(let name):
            return "The frame from \"\(name)\" could not be encoded as PNG."
        }
    }
}
