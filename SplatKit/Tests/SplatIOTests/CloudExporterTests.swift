import Foundation
import Testing
import SplatCore
import SplatIO

private func exporterCloud(count: Int, degree: Int) throws -> GaussianCloud {
    let k = GaussianCloud.coefficientCount(degree: degree)
    var positions: [Float] = []
    var logScales: [Float] = []
    var rotations: [Float] = []
    var opacity: [Float] = []
    var sh: [Float] = []
    for i in 0..<count {
        let t = Float(i)
        positions += [t * 0.25 - 3, 1 - t * 0.1, 0.5 * t]
        logScales += [-2, -1.5, -3]
        rotations += [1, 0, 0, 0]
        opacity.append(0.5)
        for coefficient in 0..<k {
            for c in 0..<3 {
                sh.append(coefficient == 0 ? 0.2 * Float(c) - 0.1 : 0.05)
            }
        }
    }
    return try GaussianCloud(shDegree: degree, positions: positions, logScales: logScales,
                             rotations: rotations, opacityLogits: opacity, sh: sh)
}

@Suite struct CloudExporterTests {
    @Test func formatMetadata() {
        #expect(ExportFormat.allCases == [.ply, .splat, .spz])
        #expect(ExportFormat.ply.fileExtension == "ply")
        #expect(ExportFormat.splat.fileExtension == "splat")
        #expect(ExportFormat.spz.fileExtension == "spz")
        #expect(ExportFormat.ply.displayName == ".ply (standard 3DGS)")
        #expect(ExportFormat.splat.displayName == ".splat (web viewers)")
        #expect(ExportFormat.spz.displayName == ".spz (compressed)")
    }

    @Test func everyFormatReadsBackWithSameCount() throws {
        let cloud = try exporterCloud(count: 40, degree: 1)

        let ply = try CloudExporter.export(cloud, as: .ply)
        #expect(try PLYReader.read(ply).count == 40)

        let splat = try CloudExporter.export(cloud, as: .splat)
        #expect(try AntimatterSplatReader.read(splat).count == 40)

        let spz = try CloudExporter.export(cloud, as: .spz)
        #expect(try SPZReader.read(spz).count == 40)
    }

    @Test func everyFormatIsReadableBySceneLoader() throws {
        let cloud = try exporterCloud(count: 9, degree: 0)
        for format in ExportFormat.allCases {
            let data = try CloudExporter.export(cloud, as: format)
            let scene = try SceneLoader.load(data, fileExtension: format.fileExtension)
            #expect(scene.cloud.count == 9, "format \(format.rawValue)")
        }
    }

    @Test func spzExportOfEmptyCloudThrows() throws {
        let empty = try GaussianCloud(shDegree: 0, positions: [], logScales: [], rotations: [],
                                      opacityLogits: [], sh: [])
        #expect(throws: SplatIOError.empty) {
            try CloudExporter.export(empty, as: .spz)
        }
    }
}
