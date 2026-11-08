import Foundation
import SplatCore

/// Text and JSON reports of the command-line tool. Pure formatting, so it is tested on Linux.
public enum CLIReport {
    /// One progress line with a stable layout, for scripts that parse standard output:
    /// `progress iteration=1200/5000 loss=0.031 gaussians=412000 ips=9.2`. A missing loss prints as `-`.
    public static func progressLine(iteration: Int,
                                    total: Int,
                                    loss: Double?,
                                    gaussians: Int,
                                    iterationsPerSecond: Double) -> String {
        let lossText: String = loss.map { String(format: "%.4g", $0) } ?? "-"
        let rate: String = String(format: "%.1f", iterationsPerSecond)
        return "progress iteration=\(iteration)/\(total) loss=\(lossText) gaussians=\(gaussians) ips=\(rate)"
    }

    /// One JSON object on a single line (JSON Lines), with sorted keys.
    public static func jsonLine<T: Encodable>(_ value: T) throws -> String {
        let encoder: JSONEncoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data: Data = try encoder.encode(value)
        return String(decoding: data, as: UTF8.self)
    }
}

/// A `progress` event in JSON output.
public struct CLIProgressRecord: Codable, Equatable, Sendable {
    public let event: String
    public let iteration: Int
    public let total: Int
    public let loss: Double?
    public let gaussians: Int
    public let iterationsPerSecond: Double

    public init(iteration: Int, total: Int, loss: Double?, gaussians: Int, iterationsPerSecond: Double) {
        self.event = "progress"
        self.iteration = iteration
        self.total = total
        self.loss = loss
        self.gaussians = gaussians
        self.iterationsPerSecond = iterationsPerSecond
    }
}

/// The final object of a command in JSON output. `ok` is false on failure, which also exits with `CLIExitCode.failure`.
public struct CLIResultRecord: Codable, Equatable, Sendable {
    public let event: String
    public let command: String
    public let ok: Bool
    public let message: String?
    public let state: String?
    public let iteration: Int?
    public let gaussians: Int?
    public let psnr: Double?
    public let checkpoints: Int?
    public let output: String?

    public init(command: String,
                ok: Bool,
                message: String? = nil,
                state: String? = nil,
                iteration: Int? = nil,
                gaussians: Int? = nil,
                psnr: Double? = nil,
                checkpoints: Int? = nil,
                output: String? = nil) {
        self.event = "result"
        self.command = command
        self.ok = ok
        self.message = message
        self.state = state
        self.iteration = iteration
        self.gaussians = gaussians
        self.psnr = psnr
        self.checkpoints = checkpoints
        self.output = output
    }
}

/// What `sparsesplat info` reports about a package. Built from the package contents only.
public struct ProjectSummary: Codable, Equatable, Sendable {
    public struct Checkpoint: Codable, Equatable, Sendable {
        public var iteration: Int
        public var gaussians: Int
        public var psnr: Double?
        public var heldOutView: String?
        public var heldOutPSNR: Double?
        public var date: Date
    }

    public var photos: Int
    public var inputKind: String?
    public var poseSource: String?
    public var presetIterations: Int
    public var maxImageSide: Int
    public var checkpointCount: Int
    /// The latest checkpoint, if the package has one.
    public var checkpoint: Checkpoint?
    /// Gaussians removed from the latest checkpoint by stored deletions. `nil` when there are none.
    public var deletedGaussians: Int?
    public var hasRefinedPoses: Bool

    public init(contents: ProjectContents) {
        photos = contents.manifest.imageFiles.count
        inputKind = contents.manifest.inputKind?.rawValue
        poseSource = contents.manifest.source?.rawValue
        presetIterations = contents.manifest.training.iterations
        maxImageSide = contents.manifest.training.maxImageSide
        checkpointCount = contents.manifest.checkpoints.count
        if let latest = contents.manifest.checkpoints.last {
            checkpoint = Checkpoint(iteration: latest.iteration,
                                    gaussians: latest.gaussianCount,
                                    psnr: latest.psnr,
                                    heldOutView: latest.heldOutView,
                                    heldOutPSNR: latest.heldOutPSNR,
                                    date: latest.date)
        } else {
            checkpoint = nil
        }
        if let stored = try? ProjectBundle.deletions(contents) {
            deletedGaussians = stored.deleted.count
        } else {
            deletedGaussians = nil
        }
        hasRefinedPoses = contents.files[ProjectLayout.refinedPoses] != nil
    }

    /// The human-readable report: one `label: value` line per fact.
    public func text() -> String {
        var lines: [String] = []
        lines.append("Photos: \(photos)")
        if let inputKind {
            lines.append("Input: \(inputKind)")
        }
        if let poseSource {
            lines.append("Poses: \(poseSource)")
        }
        lines.append("Training preset: \(presetIterations) iterations, longer side \(maxImageSide) px")
        lines.append("Checkpoints: \(checkpointCount)")
        if let checkpoint {
            lines.append("Latest checkpoint: iteration \(checkpoint.iteration), \(checkpoint.gaussians) Gaussians")
            if let psnr = checkpoint.psnr {
                lines.append("PSNR (training views): \(String(format: "%.2f", psnr)) dB")
            } else {
                lines.append("PSNR (training views): not measured")
            }
            if let view = checkpoint.heldOutView, let heldOut = checkpoint.heldOutPSNR {
                lines.append("Held-out view \(view): \(String(format: "%.2f", heldOut)) dB")
            }
        } else {
            lines.append("Latest checkpoint: none")
        }
        if let deletedGaussians {
            lines.append("Deleted Gaussians: \(deletedGaussians)")
        }
        lines.append("Refined poses: \(hasRefinedPoses ? "yes" : "no")")
        return lines.joined(separator: "\n")
    }
}
