import Foundation
import SplatCore

/// The file formats a cloud can be exported to.
public enum ExportFormat: String, CaseIterable, Sendable {
    /// Standard 3DGS PLY, with full spherical harmonics.
    case ply
    /// antimatter15 `.splat`, DC colour only.
    case splat
    /// SPZ version 2, compressed, DC colour plus higher-order SH.
    case spz

    public var fileExtension: String { rawValue }

    public var displayName: String {
        switch self {
        case .ply: ".ply (standard 3DGS)"
        case .splat: ".splat (web viewers)"
        case .spz: ".spz (compressed)"
        }
    }
}

/// Encodes a cloud in any `ExportFormat`. Dispatches to `PLYWriter`, `SplatWriter` and `SPZWriter`.
public enum CloudExporter {
    /// Gaussians with a NaN or infinite value are dropped first, in every format (a stored checkpoint can
    /// contain a few; the viewer window removes them on load, the project window keeps them so that its
    /// deletion indices match the file). Throws `.empty` when nothing is left to export.
    public static func export(_ cloud: GaussianCloud, as format: ExportFormat) throws(SplatIOError) -> Data {
        let clean: GaussianCloud = cloud.removingNonFinite().cloud
        guard clean.count > 0 else {
            throw .empty
        }
        switch format {
        case .ply:
            return PLYWriter.write(clean)
        case .splat:
            return SplatWriter.write(clean)
        case .spz:
            return try SPZWriter.write(clean)
        }
    }
}
