import SwiftUI
import UniformTypeIdentifiers
import SplatCore
import SplatIO

extension UTType {
    nonisolated static let gaussianPLY = UTType(importedAs: "io.github.junostr05.SparseSplat.gaussian-ply")
    nonisolated static let antimatterSplat = UTType(importedAs: "io.github.junostr05.SparseSplat.splat")
    nonisolated static let compressedSplat = UTType(importedAs: "io.github.junostr05.SparseSplat.spz")
}

/// Read-only document holding a parsed Gaussian scene.
///
/// The type is `nonisolated` so the `FileDocument` requirements (and the parsing in
/// `init(configuration:)`) do not run under the app's default `@MainActor` isolation.
nonisolated struct SplatDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.gaussianPLY, .antimatterSplat, .compressedSplat] }
    static var writableContentTypes: [UTType] { [] }

    let scene: LoadedScene

    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents else {
            throw CocoaError(.fileReadCorruptFile, userInfo: [
                NSLocalizedFailureReasonErrorKey: "The file could not be read."
            ])
        }
        var fileExtension = ((configuration.file.filename ?? "") as NSString).pathExtension.lowercased()
        if fileExtension.isEmpty {
            fileExtension = configuration.contentType.preferredFilenameExtension ?? ""
        }
        do throws(SplatIOError) {
            scene = try SceneLoader.load(data, fileExtension: fileExtension)
        } catch {
            throw CocoaError(.fileReadCorruptFile, userInfo: [
                NSLocalizedFailureReasonErrorKey: Self.readableReason(for: error)
            ])
        }
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        throw CocoaError(.featureUnsupported)
    }

    private static func readableReason(for error: SplatIOError) -> String {
        switch error {
        case .notPLY:
            "The file is not a valid PLY file."
        case .unsupportedFormat(let detail):
            "Unsupported file format: \(detail)."
        case .missingProperty(let name):
            "The file does not contain the required property \"\(name)\"."
        case .unsupportedSHCount(let count):
            "The file has an unsupported number of spherical harmonics coefficients (\(count))."
        case .truncated:
            "The file is incomplete."
        case .invalidSize:
            "The file header declares an invalid size."
        case .empty:
            "The file does not contain any Gaussians."
        case .unsupportedCompression:
            "The file is compressed in a way this Mac cannot decode."
        case .corrupt(let detail):
            "The file is corrupt: \(detail)."
        case .nonFinite:
            "The file contains a value that is not a finite number."
        case .outOfRange(let detail):
            "The file contains a value out of range: \(detail)."
        }
    }
}
