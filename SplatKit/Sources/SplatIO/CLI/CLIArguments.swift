import Foundation
import SplatCore

/// One parsed `sparsesplat` invocation. Parsing is pure: no file or GPU work happens here.
public enum CLICommand: Equatable, Sendable {
    /// `--help` at the top level (`command == nil`) or for one subcommand.
    case help(command: String?)
    case version
    case train(TrainOptions)
    case export(ExportOptions)
    case info(InfoOptions)
    case importColmap(ImportColmapOptions)
}

public struct TrainOptions: Equatable, Sendable {
    /// Path of the `.gsproj` package. Changed in place: the checkpoint and settings are written back.
    public var project: String
    /// Replaces the stored training preset. `nil` keeps the package's settings.
    public var preset: TrainingPreset?
    /// Iterations of the run from scratch, or of the single round of `--continue`.
    public var iterations: Int?
    /// Continue from the stored checkpoint instead of training from scratch.
    public var continueTraining: Bool
    /// A Core ML `.mlpackage` for the monocular depth model. It is copied into the model store first.
    public var depthModel: String?
    public var json: Bool

    public init(project: String, preset: TrainingPreset? = nil, iterations: Int? = nil,
                continueTraining: Bool = false, depthModel: String? = nil, json: Bool = false) {
        self.project = project
        self.preset = preset
        self.iterations = iterations
        self.continueTraining = continueTraining
        self.depthModel = depthModel
        self.json = json
    }
}

public struct ExportOptions: Equatable, Sendable {
    /// A `.gsproj` package (its latest checkpoint minus the stored deletions) or a `.ply` checkpoint file.
    public var source: String
    public var format: ExportFormat
    public var output: String
    public var json: Bool

    public init(source: String, format: ExportFormat, output: String, json: Bool = false) {
        self.source = source
        self.format = format
        self.output = output
        self.json = json
    }
}

public struct InfoOptions: Equatable, Sendable {
    public var project: String
    public var json: Bool

    public init(project: String, json: Bool = false) {
        self.project = project
        self.json = json
    }
}

public struct ImportColmapOptions: Equatable, Sendable {
    /// A COLMAP folder with `sparse/0/` and `images/`.
    public var folder: String
    /// The `.gsproj` package to create. An existing package at this path is replaced.
    public var output: String
    public var json: Bool

    public init(folder: String, output: String, json: Bool = false) {
        self.folder = folder
        self.output = output
        self.json = json
    }
}

/// Usage errors. The command-line tool prints the message and the usage text, and exits with `CLIExitCode.usage`.
public enum CLIError: Error, Equatable, Sendable, CustomStringConvertible {
    case noCommand
    case unknownCommand(String)
    case unknownOption(String)
    /// An option that takes a value was the last token.
    case missingValue(String)
    /// A required option was not given. The payload is the option name.
    case missingOption(String)
    /// A required positional argument was not given. The payload names it.
    case missingArgument(String)
    case unexpectedArgument(String)
    case invalidValue(option: String, value: String, expected: String)
    /// `export` without `--format` where the output extension names no format.
    case unknownFormat(String)

    public var description: String {
        switch self {
        case .noCommand:
            return "no command given"
        case .unknownCommand(let name):
            return "unknown command \u{201C}\(name)\u{201D}"
        case .unknownOption(let name):
            return "unknown option \u{201C}\(name)\u{201D}"
        case .missingValue(let name):
            return "option \(name) needs a value"
        case .missingOption(let name):
            return "missing required option \(name)"
        case .missingArgument(let name):
            return "missing argument <\(name)>"
        case .unexpectedArgument(let value):
            return "unexpected argument \u{201C}\(value)\u{201D}"
        case .invalidValue(let option, let value, let expected):
            return "invalid value \u{201C}\(value)\u{201D} for \(option): expected \(expected)"
        case .unknownFormat(let output):
            return "cannot tell the format from \u{201C}\(output)\u{201D}: use --format ply, splat or spz"
        }
    }
}

/// Exit codes of the command-line tool.
public enum CLIExitCode {
    public static let success: Int32 = 0
    /// Training, reading, writing or export failed.
    public static let failure: Int32 = 1
    /// The command line could not be parsed (sysexits.h `EX_USAGE`).
    public static let usage: Int32 = 64
    /// Training was stopped with Ctrl-C. The last completed model was still stored.
    public static let interrupted: Int32 = 130
}

/// Parses the arguments after the program name. Hand-written: the tool has no third-party argument library.
public enum CLIParser {
    public static let commandNames: [String] = ["train", "export", "info", "import-colmap"]

    public static func parse(_ arguments: [String]) throws(CLIError) -> CLICommand {
        guard let first = arguments.first else {
            throw .noCommand
        }
        let rest: [String] = Array(arguments.dropFirst())
        switch first {
        case "-h", "--help", "help":
            if let topic = rest.first, commandNames.contains(topic) {
                return .help(command: topic)
            }
            return .help(command: nil)
        case "--version", "version":
            return .version
        case "train":
            return try parseTrain(rest)
        case "export":
            return try parseExport(rest)
        case "info":
            return try parseInfo(rest)
        case "import-colmap":
            return try parseImportColmap(rest)
        default:
            throw .unknownCommand(first)
        }
    }

    // MARK: - Subcommands

    private static func parseTrain(_ tokens: [String]) throws(CLIError) -> CLICommand {
        let scan: ArgumentScan = try scanTokens(tokens,
                                          values: ["preset", "iterations", "depth-model"],
                                          flags: ["continue", "json"],
                                          aliases: [:])
        if scan.wantsHelp {
            return .help(command: "train")
        }
        let project: String = try single(scan, name: "project", command: "train")
        let preset: TrainingPreset? = try presetValue(scan)
        let iterations: Int? = try iterationsValue(scan)
        return .train(TrainOptions(project: project,
                                   preset: preset,
                                   iterations: iterations,
                                   continueTraining: scan.flags.contains("continue"),
                                   depthModel: scan.values["depth-model"],
                                   json: scan.flags.contains("json")))
    }

    private static func parseExport(_ tokens: [String]) throws(CLIError) -> CLICommand {
        let scan: ArgumentScan = try scanTokens(tokens,
                                          values: ["format", "output"],
                                          flags: ["json"],
                                          aliases: ["-o": "output"])
        if scan.wantsHelp {
            return .help(command: "export")
        }
        let source: String = try single(scan, name: "source", command: "export")
        guard let output: String = scan.values["output"] else {
            throw .missingOption("--output")
        }
        let format: ExportFormat
        if let raw: String = scan.values["format"] {
            guard let parsed = ExportFormat(rawValue: raw.lowercased()) else {
                throw .invalidValue(option: "--format", value: raw, expected: "ply, splat or spz")
            }
            format = parsed
        } else {
            let extensionName: String = URL(fileURLWithPath: output).pathExtension.lowercased()
            guard let inferred = ExportFormat(rawValue: extensionName) else {
                throw .unknownFormat(output)
            }
            format = inferred
        }
        return .export(ExportOptions(source: source,
                                     format: format,
                                     output: output,
                                     json: scan.flags.contains("json")))
    }

    private static func parseInfo(_ tokens: [String]) throws(CLIError) -> CLICommand {
        let scan: ArgumentScan = try scanTokens(tokens, values: [], flags: ["json"], aliases: [:])
        if scan.wantsHelp {
            return .help(command: "info")
        }
        let project: String = try single(scan, name: "project", command: "info")
        return .info(InfoOptions(project: project, json: scan.flags.contains("json")))
    }

    private static func parseImportColmap(_ tokens: [String]) throws(CLIError) -> CLICommand {
        let scan: ArgumentScan = try scanTokens(tokens,
                                          values: ["output"],
                                          flags: ["json"],
                                          aliases: ["-o": "output"])
        if scan.wantsHelp {
            return .help(command: "import-colmap")
        }
        let folder: String = try single(scan, name: "folder", command: "import-colmap")
        guard let output: String = scan.values["output"] else {
            throw .missingOption("--output")
        }
        return .importColmap(ImportColmapOptions(folder: folder,
                                                 output: output,
                                                 json: scan.flags.contains("json")))
    }

    // MARK: - Values

    private static func presetValue(_ scan: ArgumentScan) throws(CLIError) -> TrainingPreset? {
        guard let raw: String = scan.values["preset"] else {
            return nil
        }
        guard let preset = TrainingPreset(rawValue: raw.lowercased()) else {
            throw .invalidValue(option: "--preset", value: raw, expected: "fast, standard or quality")
        }
        return preset
    }

    private static func iterationsValue(_ scan: ArgumentScan) throws(CLIError) -> Int? {
        guard let raw: String = scan.values["iterations"] else {
            return nil
        }
        guard let count = Int(raw), count >= 1 else {
            throw .invalidValue(option: "--iterations", value: raw, expected: "a whole number of at least 1")
        }
        return count
    }

    /// The one positional argument of a subcommand. Extra arguments are rejected.
    private static func single(_ scan: ArgumentScan, name: String, command: String) throws(CLIError) -> String {
        guard let first: String = scan.positionals.first else {
            throw .missingArgument(name)
        }
        if scan.positionals.count > 1 {
            throw .unexpectedArgument(scan.positionals[1])
        }
        return first
    }

    // MARK: - Scanning

    private struct ArgumentScan {
        var positionals: [String] = []
        /// Option values by long name (without dashes). The last occurrence wins.
        var values: [String: String] = [:]
        var flags: Set<String> = []
        var wantsHelp: Bool = false
    }

    /// Splits `tokens` into positionals, values and flags. `--name value` and `--name=value` are both accepted.
    /// Only the names in `values` and `flags` are known; anything else starting with a dash is an error.
    private static func scanTokens(_ tokens: [String],
                             values: Set<String>,
                             flags: Set<String>,
                             aliases: [String: String]) throws(CLIError) -> ArgumentScan {
        var result = ArgumentScan()
        var index: Int = 0
        while index < tokens.count {
            let token: String = tokens[index]
            index += 1
            if token == "-h" || token == "--help" {
                result.wantsHelp = true
                continue
            }
            if let alias: String = aliases[token] {
                guard index < tokens.count else {
                    throw .missingValue(token)
                }
                result.values[alias] = tokens[index]
                index += 1
                continue
            }
            guard token.hasPrefix("--") else {
                if token.hasPrefix("-") && token.count > 1 {
                    throw .unknownOption(token)
                }
                result.positionals.append(token)
                continue
            }
            var name: String = String(token.dropFirst(2))
            var inlineValue: String?
            if let equals = name.firstIndex(of: "=") {
                inlineValue = String(name[name.index(after: equals)...])
                name = String(name[..<equals])
            }
            if flags.contains(name) {
                if inlineValue != nil {
                    throw .invalidValue(option: "--\(name)", value: inlineValue ?? "", expected: "no value")
                }
                result.flags.insert(name)
            } else if values.contains(name) {
                if let inlineValue {
                    result.values[name] = inlineValue
                } else {
                    guard index < tokens.count else {
                        throw .missingValue("--\(name)")
                    }
                    result.values[name] = tokens[index]
                    index += 1
                }
            } else {
                throw .unknownOption("--\(name)")
            }
        }
        return result
    }
}

/// The usage text of the command-line tool. `CLIHelp.text(for:)` returns the top-level text or one subcommand's.
public enum CLIHelp {
    public static func text(for command: String?) -> String {
        switch command {
        case "train":
            return trainText
        case "export":
            return exportText
        case "info":
            return infoText
        case "import-colmap":
            return importColmapText
        default:
            return overviewText
        }
    }

    public static let overviewText: String = """
    sparsesplat: train and export SparseSplat projects from the command line.

    Usage:
      sparsesplat <command> [options]

    Commands:
      train <project.gsproj>          Train a project (or continue it) and store the checkpoint in the package.
      export <source> --output <file> Export a project's latest checkpoint, or a .ply file, as ply, splat or spz.
      info <project.gsproj>           Show photos, checkpoint, PSNR and iterations of a project.
      import-colmap <folder>          Create a project package from a COLMAP folder (sparse/0 and images/).

    Options:
      -h, --help     Show this help, or the help of a command (sparsesplat train --help).
      --version      Print the version.

    Exit codes: 0 success, 1 failure, 64 usage error.
    Progress lines go to standard output. Add --json to any command for machine-readable output.
    """

    public static let trainText: String = """
    Usage: sparsesplat train <project.gsproj> [options]

    Trains the project headless and stores the new checkpoint and settings in the package.
    Without --continue the run starts from the original poses; with --continue it runs one more
    round from the stored checkpoint (the stored refined poses are kept).

    Options:
      --preset fast|standard|quality   Replace the stored preset (image size, iterations, schedules).
      --iterations N                   Iterations of the run, or of the one round of --continue.
      --continue                       Continue from the stored checkpoint.
      --depth-model <file.mlpackage>   Copy this Core ML depth model into the model store, then use it.
      --json                           Print one JSON object per progress event and a final result object.
      -h, --help                       Show this help.

    The package is replaced when the run finishes, or when Ctrl-C stops it (the last completed model is
    stored). A run that fails leaves the package unchanged.
    """

    public static let exportText: String = """
    Usage: sparsesplat export <source> --output <file> [options]

    Exports the latest checkpoint of a .gsproj package (without its stored deletions), or a .ply file.

    Options:
      -o, --output <file>              Destination file.
      --format ply|splat|spz           Format. Inferred from the output extension when omitted.
      --json                           Print a JSON result object.
      -h, --help                       Show this help.
    """

    public static let infoText: String = """
    Usage: sparsesplat info <project.gsproj> [options]

    Prints the photo count, input kind, stored checkpoints, the latest checkpoint's PSNR and iterations,
    and the stored deletions.

    Options:
      --json                           Print a JSON object.
      -h, --help                       Show this help.
    """

    public static let importColmapText: String = """
    Usage: sparsesplat import-colmap <folder> --output <project.gsproj> [options]

    Creates a project package from a COLMAP folder that contains sparse/0/ (cameras.bin, images.bin,
    points3D.bin) and images/. An existing package at the output path is replaced.

    Options:
      -o, --output <project.gsproj>    Package to create.
      --json                           Print a JSON result object.
      -h, --help                       Show this help.
    """
}
