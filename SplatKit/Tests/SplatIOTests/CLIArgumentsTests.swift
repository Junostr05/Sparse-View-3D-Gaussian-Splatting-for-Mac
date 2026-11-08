import Foundation
import Testing
import SplatCore
@testable import SplatIO

/// Parsing, help text and report formatting of the `sparsesplat` command line. Pure functions, so they run on Linux.
struct CLIArgumentsTests {
    // MARK: - Commands

    @Test func noArgumentsIsAUsageError() {
        #expect(throws: CLIError.noCommand) { try CLIParser.parse([]) }
    }

    @Test func unknownCommandIsNamed() {
        #expect(throws: CLIError.unknownCommand("frobnicate")) { try CLIParser.parse(["frobnicate"]) }
    }

    @Test func topLevelHelpAndVersion() throws {
        #expect(try CLIParser.parse(["--help"]) == .help(command: nil))
        #expect(try CLIParser.parse(["-h"]) == .help(command: nil))
        #expect(try CLIParser.parse(["help", "train"]) == .help(command: "train"))
        #expect(try CLIParser.parse(["--version"]) == .version)
    }

    @Test func subcommandHelpFlag() throws {
        #expect(try CLIParser.parse(["train", "--help"]) == .help(command: "train"))
        #expect(try CLIParser.parse(["export", "x.gsproj", "-h"]) == .help(command: "export"))
        #expect(try CLIParser.parse(["import-colmap", "--help"]) == .help(command: "import-colmap"))
    }

    // MARK: - train

    @Test func trainWithDefaults() throws {
        let command = try CLIParser.parse(["train", "scene.gsproj"])
        #expect(command == .train(TrainOptions(project: "scene.gsproj")))
    }

    @Test func trainWithEveryOption() throws {
        let command = try CLIParser.parse([
            "train", "scene.gsproj", "--preset", "quality", "--iterations", "200",
            "--continue", "--depth-model", "/models/depth.mlpackage", "--json",
        ])
        #expect(command == .train(TrainOptions(project: "scene.gsproj",
                                               preset: .quality,
                                               iterations: 200,
                                               continueTraining: true,
                                               depthModel: "/models/depth.mlpackage",
                                               json: true)))
    }

    @Test func equalsFormAndCaseOfPreset() throws {
        let command = try CLIParser.parse(["train", "--iterations=50", "--preset=FAST", "scene.gsproj"])
        #expect(command == .train(TrainOptions(project: "scene.gsproj", preset: .fast, iterations: 50)))
    }

    @Test func trainRejectsBadIterations() {
        #expect(throws: CLIError.invalidValue(option: "--iterations", value: "0",
                                              expected: "a whole number of at least 1")) {
            try CLIParser.parse(["train", "scene.gsproj", "--iterations", "0"])
        }
        #expect(throws: CLIError.invalidValue(option: "--iterations", value: "many",
                                              expected: "a whole number of at least 1")) {
            try CLIParser.parse(["train", "scene.gsproj", "--iterations", "many"])
        }
    }

    @Test func trainRejectsUnknownPreset() {
        #expect(throws: CLIError.invalidValue(option: "--preset", value: "ultra",
                                              expected: "fast, standard or quality")) {
            try CLIParser.parse(["train", "scene.gsproj", "--preset", "ultra"])
        }
    }

    @Test func trainNeedsOneProject() {
        #expect(throws: CLIError.missingArgument("project")) {
            try CLIParser.parse(["train"])
        }
        #expect(throws: CLIError.unexpectedArgument("b.gsproj")) {
            try CLIParser.parse(["train", "a.gsproj", "b.gsproj"])
        }
    }

    @Test func valueOptionAsLastTokenIsMissingItsValue() {
        #expect(throws: CLIError.missingValue("--iterations")) {
            try CLIParser.parse(["train", "scene.gsproj", "--iterations"])
        }
    }

    @Test func unknownOptionIsRejectedPerCommand() {
        #expect(throws: CLIError.unknownOption("--format")) {
            try CLIParser.parse(["train", "scene.gsproj", "--format", "ply"])
        }
        #expect(throws: CLIError.unknownOption("--bogus")) {
            try CLIParser.parse(["info", "scene.gsproj", "--bogus"])
        }
    }

    @Test func flagWithInlineValueIsAnError() {
        #expect(throws: CLIError.invalidValue(option: "--json", value: "yes", expected: "no value")) {
            try CLIParser.parse(["info", "scene.gsproj", "--json=yes"])
        }
    }

    // MARK: - export

    @Test func exportInfersFormatFromOutputExtension() throws {
        let command = try CLIParser.parse(["export", "scene.gsproj", "--output", "/tmp/out.spz"])
        #expect(command == .export(ExportOptions(source: "scene.gsproj", format: .spz, output: "/tmp/out.spz")))
    }

    @Test func exportExplicitFormatWinsAndShortOutputFlag() throws {
        let command = try CLIParser.parse(["export", "checkpoint.ply", "-o", "out.bin", "--format", "splat", "--json"])
        #expect(command == .export(ExportOptions(source: "checkpoint.ply", format: .splat,
                                                 output: "out.bin", json: true)))
    }

    @Test func exportNeedsOutput() {
        #expect(throws: CLIError.missingOption("--output")) {
            try CLIParser.parse(["export", "scene.gsproj", "--format", "ply"])
        }
    }

    @Test func exportWithoutKnownExtensionNeedsFormat() {
        #expect(throws: CLIError.unknownFormat("out.obj")) {
            try CLIParser.parse(["export", "scene.gsproj", "--output", "out.obj"])
        }
    }

    @Test func exportRejectsUnknownFormat() {
        #expect(throws: CLIError.invalidValue(option: "--format", value: "html",
                                              expected: "ply, splat or spz")) {
            try CLIParser.parse(["export", "scene.gsproj", "--format", "html", "-o", "a.html"])
        }
    }

    // MARK: - info and import-colmap

    @Test func infoTakesOneProject() throws {
        #expect(try CLIParser.parse(["info", "scene.gsproj"]) == .info(InfoOptions(project: "scene.gsproj")))
        #expect(try CLIParser.parse(["info", "scene.gsproj", "--json"])
                == .info(InfoOptions(project: "scene.gsproj", json: true)))
        #expect(throws: CLIError.missingArgument("project")) { try CLIParser.parse(["info"]) }
    }

    @Test func importColmapNeedsFolderAndOutput() throws {
        #expect(try CLIParser.parse(["import-colmap", "capture", "-o", "new.gsproj"])
                == .importColmap(ImportColmapOptions(folder: "capture", output: "new.gsproj")))
        #expect(throws: CLIError.missingOption("--output")) {
            try CLIParser.parse(["import-colmap", "capture"])
        }
        #expect(throws: CLIError.missingArgument("folder")) {
            try CLIParser.parse(["import-colmap", "--output", "new.gsproj"])
        }
    }

    // MARK: - Help text

    @Test func helpMentionsEveryCommandAndOption() {
        let overview = CLIHelp.text(for: nil)
        for name in CLIParser.commandNames {
            #expect(overview.contains(name))
        }
        #expect(CLIHelp.text(for: "train").contains("--continue"))
        #expect(CLIHelp.text(for: "train").contains("--depth-model"))
        #expect(CLIHelp.text(for: "export").contains("--format ply|splat|spz"))
        #expect(CLIHelp.text(for: "import-colmap").contains("sparse/0"))
        #expect(CLIHelp.text(for: "unknown").contains("Usage:") == false || CLIHelp.text(for: "unknown") == CLIHelp.overviewText)
    }

    // MARK: - Reports

    @Test func progressLineHasStableLayout() {
        let line = CLIReport.progressLine(iteration: 1200, total: 5000, loss: 0.031,
                                          gaussians: 412_000, iterationsPerSecond: 9.2)
        #expect(line == "progress iteration=1200/5000 loss=0.031 gaussians=412000 ips=9.2")
    }

    @Test func progressLineWithoutLossPrintsDash() {
        let line = CLIReport.progressLine(iteration: 0, total: 10, loss: nil, gaussians: 5, iterationsPerSecond: 0)
        #expect(line == "progress iteration=0/10 loss=- gaussians=5 ips=0.0")
    }

    @Test func jsonProgressRecordHasSortedKeysOnOneLine() throws {
        let record = CLIProgressRecord(iteration: 3, total: 9, loss: nil, gaussians: 7, iterationsPerSecond: 1.5)
        let text = try CLIReport.jsonLine(record)
        #expect(!text.contains("\n"))
        #expect(text.hasPrefix("{\"event\":\"progress\""))
        #expect(text.contains("\"gaussians\":7"))
    }

    @Test func projectSummaryDescribesPackage() {
        var manifest = ProjectManifest()
        manifest.imageFiles = ["a.jpg", "b.jpg", "c.jpg"]
        manifest.checkpoints = [
            CheckpointRecord(iteration: 200, gaussianCount: 1_000, psnr: 21.5, date: Date(timeIntervalSince1970: 0)),
        ]
        let summary = ProjectSummary(contents: ProjectContents(manifest: manifest, files: [:]))
        #expect(summary.photos == 3)
        #expect(summary.checkpointCount == 1)
        #expect(summary.checkpoint?.iteration == 200)
        #expect(summary.deletedGaussians == nil)
        let text = summary.text()
        #expect(text.contains("Photos: 3"))
        #expect(text.contains("Latest checkpoint: iteration 200, 1000 Gaussians"))
        #expect(text.contains("PSNR (training views): 21.50 dB"))
    }

    @Test func projectSummaryWithoutCheckpoint() {
        let summary = ProjectSummary(contents: ProjectContents(manifest: ProjectManifest(), files: [:]))
        #expect(summary.checkpoint == nil)
        #expect(summary.text().contains("Latest checkpoint: none"))
    }
}
