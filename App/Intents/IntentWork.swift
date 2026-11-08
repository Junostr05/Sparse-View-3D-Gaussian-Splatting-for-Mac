import AppIntents
import Foundation
import SplatCore
import SplatIO

/// A failure the App Intents report to Shortcuts. The message is shown to the user as it is.
struct SparseSplatIntentError: LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? {
        message
    }
}

/// Work shared by the App Intents. They run in the app's process, without a window, and use the same library
/// folder, queue and pose chain as the app. Disk and decode work is `@concurrent`.
enum IntentWork {
    /// The library folder the user chose. Fails with a message when none is set.
    static func libraryFolder() throws -> URL {
        // A separate `LibraryFolder` starts the same security-scoped access the app's own instance holds.
        guard let url = LibraryFolder().readyURL else {
            throw SparseSplatIntentError("Choose a library folder in SparseSplat first.")
        }
        return url
    }

    /// Every project package in the library folder, by name.
    @concurrent
    static func projects(in folder: URL) async throws -> [LibraryItem] {
        try await LibraryFiles.scan(folder: folder)
    }

    /// A local file for an intent input. Inputs that arrive as bytes only are written to the caches folder first.
    static func localURL(of file: IntentFile) throws -> URL {
        if let url = file.fileURL {
            return url
        }
        let caches: URL = try FileManager.default.url(for: .cachesDirectory, in: .userDomainMask,
                                                      appropriateFor: nil, create: true)
        let target: URL = caches.appending(path: "SparseSplatIntents", directoryHint: .isDirectory)
            .appending(path: UUID().uuidString + "-" + file.filename)
        try FileManager.default.createDirectory(at: target.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try file.data.write(to: target)
        return target
    }

    /// Frames of a folder of photos, or evenly spaced frames of a video (at most 60). The input's security scope is held.
    @concurrent
    static func frames(from source: URL) async throws -> (frames: [CapturedFrame], inputKind: InputKind) {
        let scoped: Bool = source.startAccessingSecurityScopedResource()
        defer {
            if scoped {
                source.stopAccessingSecurityScopedResource()
            }
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: source.path(percentEncoded: false), isDirectory: &isDirectory) else {
            throw SparseSplatIntentError("The input could not be found.")
        }
        if isDirectory.boolValue {
            let frames: [CapturedFrame] = try PhotoFolderImporter.load(folder: source)
            guard !frames.isEmpty else {
                throw SparseSplatIntentError("The folder has no photos that can be read.")
            }
            return (frames, .photos)
        }
        let candidates: [VideoCandidate] = try await VideoFrameExtractor.scoreCandidates(from: source)
        guard !candidates.isEmpty else {
            throw SparseSplatIntentError("The video has no frames that can be read.")
        }
        let step: Int = max(1, candidates.count / 60)
        var times: [Double] = []
        var index: Int = 0
        while index < candidates.count {
            times.append(candidates[index].requestedTime)
            index += step
        }
        let frames: [CapturedFrame] = try await VideoFrameExtractor.frames(from: source, atRequestedTimes: times)
        guard !frames.isEmpty else {
            throw SparseSplatIntentError("The video frames could not be decoded.")
        }
        return (frames, .video)
    }

    /// Estimates the poses of `frames` with Apple Photogrammetry (the default provider) and writes the package with
    /// the frames, poses and the training preset. Returns the pose summary for the dialog.
    @concurrent
    static func estimatePoses(package: URL,
                              frames: [CapturedFrame],
                              inputKind: InputKind,
                              preset: TrainingPreset) async throws -> String {
        var base: ProjectContents = ProjectContents(manifest: ProjectManifest(), files: [:])
        base.manifest.training = preset.configuration()
        let run: PoseRun = await PipelineWork.estimatePoses(frames: frames,
                                                            base: base,
                                                            inputKind: inputKind,
                                                            preference: .apple,
                                                            da3Package: nil)
        switch run {
        case .imported(let contents, let summary):
            try ProjectPackageFiles.write(contents, to: package)
            return summary
        case .failed(let message):
            throw SparseSplatIntentError(message)
        case .cancelled:
            throw SparseSplatIntentError("Pose estimation was cancelled.")
        }
    }

    /// The latest checkpoint of a project without its stored deletions, as an export writes it.
    @concurrent
    static func checkpointCloud(of package: URL) async throws -> GaussianCloud {
        let contents: ProjectContents = try ProjectPackageFiles.read(package: package)
        guard let cloud = try ProjectBundle.checkpointWithoutDeletions(contents) else {
            throw SparseSplatIntentError("\(package.deletingPathExtension().lastPathComponent) has no trained model yet. Train it first.")
        }
        return cloud
    }
}
