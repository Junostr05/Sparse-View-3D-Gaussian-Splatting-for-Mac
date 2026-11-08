import Foundation
import SplatCore
import SplatIO
import SplatTraining

/// `sparsesplat`: headless training and export of `.gsproj` packages. Not sandboxed, so it reads and writes any path.
/// Argument parsing and reports live in SplatIO (tested there); training is `HeadlessTraining`, the same core the
/// app's training queue runs.
@main
enum SparseSplatCLI {
    static let version: String = "1.4.0"

    static func main() async {
        let status: Int32
        do {
            let command: CLICommand = try CLIParser.parse(Array(CommandLine.arguments.dropFirst()))
            status = await execute(command)
        } catch {
            errorLine("sparsesplat: \(error)")
            errorLine("")
            errorLine(CLIHelp.overviewText)
            status = CLIExitCode.usage
        }
        exit(status)
    }

    private static func execute(_ command: CLICommand) async -> Int32 {
        switch command {
        case .help(let topic):
            outputLine(CLIHelp.text(for: topic))
            return CLIExitCode.success
        case .version:
            outputLine("sparsesplat \(version)")
            return CLIExitCode.success
        case .info(let options):
            return info(options)
        case .export(let options):
            return export(options)
        case .importColmap(let options):
            return importColmap(options)
        case .train(let options):
            return await train(options)
        }
    }

    // MARK: - info

    private static func info(_ options: InfoOptions) -> Int32 {
        do {
            let contents: ProjectContents = try ProjectPackageFiles.read(package: URL(fileURLWithPath: options.project))
            let summary: ProjectSummary = ProjectSummary(contents: contents)
            if options.json {
                outputLine(try CLIReport.jsonLine(summary))
            } else {
                outputLine(summary.text())
            }
            return CLIExitCode.success
        } catch {
            return fail(command: "info", message: describe(error), json: options.json)
        }
    }

    // MARK: - export

    private static func export(_ options: ExportOptions) -> Int32 {
        do {
            let source: URL = URL(fileURLWithPath: options.source)
            let cloud: GaussianCloud
            if source.pathExtension.lowercased() == "ply" {
                cloud = try PLYReader.read(try Data(contentsOf: source))
            } else {
                let contents: ProjectContents = try ProjectPackageFiles.read(package: source)
                guard let stored: GaussianCloud = try ProjectBundle.checkpointWithoutDeletions(contents) else {
                    return fail(command: "export", message: "the package has no checkpoint to export", json: options.json)
                }
                cloud = stored
            }
            let data: Data = try CloudExporter.export(cloud, as: options.format)
            try data.write(to: URL(fileURLWithPath: options.output), options: .atomic)
            let message: String = "Exported \(cloud.count) Gaussians as \(options.format.rawValue) to \(options.output)."
            if options.json {
                outputLine(try CLIReport.jsonLine(CLIResultRecord(command: "export", ok: true, message: message,
                                                                  gaussians: cloud.count, output: options.output)))
            } else {
                outputLine(message)
            }
            return CLIExitCode.success
        } catch {
            return fail(command: "export", message: describe(error), json: options.json)
        }
    }

    // MARK: - import-colmap

    private static func importColmap(_ options: ImportColmapOptions) -> Int32 {
        do {
            var manifest: ProjectManifest = ProjectManifest()
            // The same starting point as a new project in the app: the default training preset.
            manifest.training = TrainingPreset.default.configuration()
            let base: ProjectContents = ProjectContents(manifest: manifest, files: [:])
            let contents: ProjectContents = try ProjectFolderImport.colmap(folder: URL(fileURLWithPath: options.folder,
                                                                                      isDirectory: true),
                                                                          into: base)
            try ProjectPackageFiles.write(contents, to: URL(fileURLWithPath: options.output, isDirectory: true))
            let message: String = "Imported \(contents.manifest.imageFiles.count) photos into \(options.output)."
            if options.json {
                outputLine(try CLIReport.jsonLine(CLIResultRecord(command: "import-colmap", ok: true, message: message,
                                                                  checkpoints: contents.manifest.checkpoints.count,
                                                                  output: options.output)))
            } else {
                outputLine(message)
            }
            return CLIExitCode.success
        } catch {
            return fail(command: "import-colmap", message: describe(error), json: options.json)
        }
    }

    // MARK: - train

    private static func train(_ options: TrainOptions) async -> Int32 {
        let packageURL: URL = URL(fileURLWithPath: options.project, isDirectory: true)
        do {
            let contents: ProjectContents = try ProjectPackageFiles.read(package: packageURL)
            let store: ModelStore? = defaultModelStore()
            if let modelPath = options.depthModel {
                guard let store else {
                    return fail(command: "train", message: "the model store could not be opened", json: options.json)
                }
                let estimator: DepthEstimatorKind = contents.manifest.training.depthPrior.estimator
                let source: URL = URL(fileURLWithPath: modelPath, isDirectory: true)
                try store.importPackage(from: source, for: estimator.descriptor)
                noteLine("Imported depth model from \(modelPath) for \(estimator.rawValue).", json: options.json)
            }
            let headless: HeadlessTrainingOptions = HeadlessTrainingOptions(
                mode: options.continueTraining ? .continueTraining : .fromScratch,
                preset: options.preset,
                iterations: options.iterations,
                modelStore: store
            )
            let json: Bool = options.json
            installInterruptHandler()
            let job: Task<HeadlessTrainingResult, Error> = Task {
                try await HeadlessTraining.run(contents: contents, options: headless) { event in
                    if case .running = event.state {
                        report(event, json: json)
                    }
                }
            }
            let watcher: Task<Void, Never> = Task {
                while !Task.isCancelled {
                    if interruptRequestCount() > 0 {
                        job.cancel()
                        return
                    }
                    try? await Task.sleep(for: .milliseconds(200))
                }
            }
            defer {
                watcher.cancel()
            }
            let result: HeadlessTrainingResult = try await job.value
            for warning in result.warnings {
                // Warnings go to standard error in every mode, so JSON output on standard output stays parseable.
                errorLine("sparsesplat train: warning: \(warning)")
            }
            // Written only after a successful run: a failed or unfinished run leaves the package as it was.
            try ProjectPackageFiles.write(result.contents, to: packageURL)
            let interrupted: Bool = interruptRequestCount() > 0
            return finish(result, interrupted: interrupted, json: options.json)
        } catch {
            return fail(command: "train", message: describe(error), json: options.json)
        }
    }

    /// The closing report of a run: a result object in JSON, or a summary line in text.
    private static func finish(_ result: HeadlessTrainingResult, interrupted: Bool, json: Bool) -> Int32 {
        let stopped: Bool = result.state == .cancelled
        let psnr: String = result.psnr.map { String(format: "%.2f dB", $0) } ?? "not measured"
        let message: String
        if stopped {
            message = "Stopped early. The last completed model is stored: iteration \(result.iteration), \(result.gaussianCount) Gaussians, PSNR \(psnr), \(result.checkpointCount) checkpoints in the package."
        } else {
            message = "Checkpoint stored: iteration \(result.iteration), \(result.gaussianCount) Gaussians, PSNR \(psnr), \(result.checkpointCount) checkpoints in the package."
        }
        if json {
            let record: CLIResultRecord = CLIResultRecord(command: "train",
                                                          ok: true,
                                                          message: message,
                                                          state: stopped ? "cancelled" : "finished",
                                                          iteration: result.iteration,
                                                          gaussians: result.gaussianCount,
                                                          psnr: result.psnr,
                                                          checkpoints: result.checkpointCount)
            if let text = try? CLIReport.jsonLine(record) {
                outputLine(text)
            }
        } else {
            outputLine(message)
        }
        return interrupted ? CLIExitCode.interrupted : CLIExitCode.success
    }

    private static func report(_ event: TrainingProgress, json: Bool) {
        if json {
            let record: CLIProgressRecord = CLIProgressRecord(iteration: event.iteration,
                                                              total: event.totalIterations,
                                                              loss: event.loss,
                                                              gaussians: event.gaussianCount,
                                                              iterationsPerSecond: event.iterationsPerSecond)
            if let text = try? CLIReport.jsonLine(record) {
                outputLine(text)
            }
        } else {
            outputLine(CLIReport.progressLine(iteration: event.iteration,
                                              total: event.totalIterations,
                                              loss: event.loss,
                                              gaussians: event.gaussianCount,
                                              iterationsPerSecond: event.iterationsPerSecond))
        }
    }

    /// The model store of the command-line tool: the same folder layout as the app's store, in the user's
    /// Application Support (the sandboxed app keeps its own copy inside its container).
    private static func defaultModelStore() -> ModelStore? {
        guard let support: URL = try? FileManager.default.url(for: .applicationSupportDirectory,
                                                              in: .userDomainMask,
                                                              appropriateFor: nil,
                                                              create: true) else {
            return nil
        }
        return ModelStore(rootDirectory: support.appending(path: "SparseSplat/Models", directoryHint: .isDirectory))
    }

    // MARK: - Output

    private static func fail(command: String, message: String, json: Bool) -> Int32 {
        if json {
            if let text = try? CLIReport.jsonLine(CLIResultRecord(command: command, ok: false, message: message)) {
                outputLine(text)
            }
        } else {
            errorLine("sparsesplat \(command): \(message)")
        }
        return CLIExitCode.failure
    }

    private static func noteLine(_ text: String, json: Bool) {
        if !json {
            outputLine(text)
        }
    }

    private static func outputLine(_ text: String) {
        print(text)
        fflush(stdout)
    }

    private static func errorLine(_ text: String) {
        FileHandle.standardError.write(Data((text + "\n").utf8))
    }

    /// The text of an error for the user. Library errors without a description are printed as their case.
    private static func describe(_ error: Error) -> String {
        if let headless = error as? HeadlessTrainingError {
            return headless.description
        }
        if let packageError = error as? ProjectPackageFiles.PackageError {
            return packageError.description
        }
        if let projectError = error as? ProjectError {
            switch projectError {
            case .missingManifest:
                return "the package has no manifest.json"
            case .unsupportedFormatVersion(let version):
                return "the package was saved with a newer format version (\(version))"
            case .corruptManifest(let detail):
                return "the manifest is corrupt: \(detail)"
            case .corruptCheckpoint(let detail):
                return "the checkpoint is corrupt: \(detail)"
            case .corruptRefinedPoses(let detail):
                return "the refined poses are corrupt: \(detail)"
            }
        }
        if let datasetError = error as? DatasetError {
            return "the images could not be read: \(datasetError)"
        }
        return String(describing: error)
    }
}
