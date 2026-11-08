import AppIntents
import Foundation
import SplatIO
import UniformTypeIdentifiers

/// The export formats offered to Shortcuts. Only the formats the app exports today.
enum SplatExportFormatOption: String, AppEnum {
    case ply
    case splat
    case spz

    static let typeDisplayRepresentation: TypeDisplayRepresentation = "Export Format"
    static let caseDisplayRepresentations: [SplatExportFormatOption: DisplayRepresentation] = [
        .ply: "PLY (standard 3DGS)",
        .splat: "SPLAT (web viewers)",
        .spz: "SPZ (compressed)",
    ]

    var format: ExportFormat {
        ExportFormat(rawValue: rawValue) ?? .ply
    }
}

/// Exports the latest trained model of a project. The file is returned to Shortcuts, which saves it where the next
/// action puts it (for example a Save File action with a destination folder). The app does not pick a folder itself.
struct ExportSplatIntent: AppIntent {
    static let title: LocalizedStringResource = "Export Splat"
    static let description = IntentDescription("Exports the latest trained model of a project as a PLY, SPLAT or SPZ file. Shortcuts receives the file; a Save File action chooses the destination folder.")
    static let openAppWhenRun: Bool = false

    @Parameter(title: "Project")
    var project: ProjectEntity

    @Parameter(title: "Format", default: .spz)
    var format: SplatExportFormatOption

    @MainActor
    func perform() async throws -> some ReturnsValue<IntentFile> & ProvidesDialog {
        let cloud = try await IntentWork.checkpointCloud(of: project.url)
        let data: Data
        do {
            data = try CloudExporter.export(cloud, as: format.format)
        } catch {
            throw SparseSplatIntentError("The model could not be exported: \(error.localizedDescription)")
        }
        let extensionName: String = format.format.fileExtension
        let file: IntentFile = IntentFile(data: data,
                                          filename: "\(project.name).\(extensionName)",
                                          type: UTType(filenameExtension: extensionName))
        return .result(value: file,
                       dialog: IntentDialog(stringLiteral: "Exported \(project.name) as \(extensionName.uppercased())."))
    }
}
