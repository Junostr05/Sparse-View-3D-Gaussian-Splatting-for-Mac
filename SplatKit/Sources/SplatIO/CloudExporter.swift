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
    /// Throws only for SPZ, see `SPZWriter.write`. The other two formats write any cloud.
    public static func export(_ cloud: GaussianCloud, as format: ExportFormat) throws(SplatIOError) -> Data {
        switch format {
        case .ply:
            return PLYWriter.write(cloud)
        case .splat:
            return SplatWriter.write(cloud)
        case .spz:
            return try SPZWriter.write(cloud)
        }
    }
}
