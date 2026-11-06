import CoreGraphics
import Foundation
import ImageIO
import Observation
import SwiftUI
import SplatCore
import SplatIO

/// Blocking work of the input pipeline. Every function runs off the main actor (`@concurrent`) and
/// passes only Sendable values back.
nonisolated enum PipelineWork {
    /// Longer side of the photos and of the full-size video frames that enter the project.
    static let frameMaxSide: Int = 1600
    /// Most candidate times scored for one video.
    static let videoCandidateLimit: Int = 240
    /// Longer side of the PNG previews made while scoring a video.
    static let videoPreviewSide: Int = 320
    /// Longer side of the thumbnails drawn in the frame strip.
    static let thumbnailSide: Int = 280

    /// Reads the supported images of a photo folder. The folder's security scope is held for the read only.
    @concurrent
    static func scanPhotos(folder: URL) async throws -> [CapturedFrame] {
        try ProjectImport.withSecurityScope(folder) { () throws -> [CapturedFrame] in
            try PhotoFolderImporter.load(folder: folder, maxSide: frameMaxSide)
        }
    }

    /// Scores the candidate frames of a video. The caller holds the security scope of `url`.
    @concurrent
    static func scoreVideo(url: URL, sampleInterval: Double) async throws -> [VideoCandidate] {
        try await VideoFrameExtractor.scoreCandidates(from: url,
                                                      sampleInterval: sampleInterval,
                                                      maxCandidates: videoCandidateLimit,
                                                      thumbnailSide: videoPreviewSide)
    }

    /// Decodes the selected video frames at full size, at the times the candidates asked for.
    @concurrent
    static func decodeVideoFrames(url: URL, times: [Double]) async throws -> [CapturedFrame] {
        try await VideoFrameExtractor.frames(from: url, atRequestedTimes: times, maxSide: frameMaxSide)
    }

    /// Decodes a thumbnail of an encoded image. `nil` when the data is not an image.
    @concurrent
    static func thumbnail(data: Data, maxSide: Int) async -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            return nil
        }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxSide
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }

    /// Runs Apple Photogrammetry on the frames and, on success, writes the poses into a copy of `base`.
    @concurrent
    static func estimatePoses(frames: [CapturedFrame], base: ProjectContents, inputKind: InputKind) async -> PoseRun {
        let chain = PoseProviderChain(providers: [ApplePhotogrammetryPoseProvider()])
        let outcome: PoseChainOutcome = await chain.estimate(for: frames)
        if Task.isCancelled {
            return .cancelled
        }
        switch outcome {
        case .success(let result, let providerName, _):
            let contents: ProjectContents = ProjectBundle.importEstimatedPoses(frames: frames,
                                                                               result: result,
                                                                               inputKind: inputKind,
                                                                               into: base)
            let summary: String = "\(result.registeredImageNames.count) of \(frames.count) images registered (\(providerName))"
            return .imported(contents, summary: summary)
        case .failure(let attempts):
            if case .failure(.cancelled)? = attempts.last?.outcome {
                return .cancelled
            }
            return .failed(estimationFailureMessage(attempts))
        }
    }

    /// Checks the depth model in the store. Hashing the stored files is slow, hence off the main actor.
    @concurrent
    static func depthStatus(store: ModelStore, descriptor: ModelDescriptor) async -> ModelStatus {
        store.status(for: descriptor)
    }

    /// Downloads, verifies and installs the depth model.
    @concurrent
    static func downloadDepth(store: ModelStore,
                              descriptor: ModelDescriptor,
                              progress: @escaping @Sendable (Double) -> Void) async throws {
        try await URLSessionModelDownloader().download(descriptor, to: store, progress: progress)
    }

    /// Readable sentence for a pose failure.
    static func describe(_ failure: PoseEstimationFailure) -> String {
        switch failure {
        case .unavailable(let reason):
            return "Pose estimation is not available on this Mac: \(reason)."
        case .tooFewFrames(let have, let need):
            return "Pose estimation needs at least \(need) frames, but \(have) are selected."
        case .registrationFailed(let registered, let total):
            return "Only \(registered) of \(total) frames could be matched into one scene."
        case .cancelled:
            return "Pose estimation was cancelled."
        case .failed(let reason):
            return "Pose estimation failed: \(reason)."
        }
    }

    /// Message for a chain without a usable result: the reason of the last attempt, plus the COLMAP fallback.
    static func estimationFailureMessage(_ attempts: [PoseAttempt]) -> String {
        var reason: String = "Pose estimation failed."
        if let last = attempts.last, case .failure(let failure) = last.outcome {
            reason = describe(failure)
        }
        return reason + " Import a COLMAP folder instead."
    }
}

/// Result of the pose stage, built off the main actor.
nonisolated enum PoseRun: Sendable {
    /// Poses were found and written into the project contents. `summary` is shown afterwards.
    case imported(ProjectContents, summary: String)
    case failed(String)
    case cancelled
}

/// One frame in the frame strip. `id` is the position in its scan and matches `FrameSelectionPlan` ids.
struct PipelineCandidate: Identifiable {
    let id: Int
    /// Seconds in the video, `nil` for photos.
    let time: Double?
    /// Sharpness from video scoring, `nil` for photos.
    let sharpness: Double?
    /// Encoded image (PNG) the thumbnail is decoded from.
    let imageData: Data
    /// Text under the thumbnail: the time for video, the original file name for photos.
    let caption: String
}

/// State of the input pipeline: the pending frames of a photo folder or a video, their selection, the pose
/// estimation and the optional depth model. Nothing here is saved. Blocking work runs in `PipelineWork`.
///
/// A video keeps its security scope from the moment it is picked until the pipeline is discarded, finished
/// or closed, because the frames are decoded later at "Estimate Poses".
@Observable
final class PipelineModel {
    enum Source: Equatable {
        case photos(folderName: String)
        case video(fileName: String)
    }

    static let sampleIntervals: [Double] = [0.1, 0.25, 0.5, 1.0]
    static let defaultSampleInterval: Double = 0.25
    static let depthDescriptor: ModelDescriptor = ModelCatalog.depthAnythingV2Small

    private(set) var source: Source?
    private(set) var candidates: [PipelineCandidate] = []
    private(set) var plan: FrameSelectionPlan?
    private(set) var sampleInterval: Double = PipelineModel.defaultSampleInterval
    private(set) var usesAllFrames: Bool = false
    private(set) var bestCount: Int = FrameSelectionMode.defaultCount
    /// Decoded thumbnails by candidate id. Filled lazily as the strip shows its cells.
    private(set) var thumbnails: [Int: CGImage] = [:]
    private(set) var isScanning: Bool = false
    private(set) var scanFailed: Bool = false
    private(set) var isEstimating: Bool = false
    private(set) var estimationStatus: String?
    private(set) var estimationFailed: Bool = false
    /// Summary of the last successful pose estimation, shown while the project holds its images.
    private(set) var poseSummary: String?
    var errorMessage: String?

    private(set) var depthStatus: ModelStatus?
    /// Download fraction in 0...1 while the depth model downloads, `nil` otherwise.
    private(set) var depthProgress: Double?
    private(set) var depthError: String?

    @ObservationIgnored private var photoFrames: [CapturedFrame] = []
    @ObservationIgnored private var videoCandidates: [VideoCandidate] = []
    @ObservationIgnored private var videoURL: URL?
    /// The video URL when the system granted a security scope. Released by `discard`, or on deinit.
    @ObservationIgnored private var scopedURL: URL?
    @ObservationIgnored private var work: Task<Void, Never>?
    @ObservationIgnored private var depthDownload: Task<Void, Never>?
    /// Incremented whenever work starts or is cancelled. Results of older work are dropped.
    @ObservationIgnored private var generation: Int = 0
    private let modelStore: ModelStore?

    init() {
        modelStore = Self.makeModelStore()
    }

    deinit {
        scopedURL?.stopAccessingSecurityScopedResource()
    }

    // MARK: - Derived state

    /// Selection mode the plan uses: every frame, or the best `bestCount`.
    var mode: FrameSelectionMode {
        usesAllFrames ? .all : .count(bestCount)
    }

    var hasCandidates: Bool {
        !candidates.isEmpty
    }

    var canEstimatePoses: Bool {
        guard source != nil, !isScanning, !isEstimating, let plan else {
            return false
        }
        return plan.selectedCount > 0
    }

    var depthIsReady: Bool {
        if case .ready? = depthStatus {
            return true
        }
        return false
    }

    /// `true` when the model is missing or damaged and a download can fix it.
    var depthCanDownload: Bool {
        switch depthStatus {
        case .missing?, .corrupt?:
            return true
        default:
            return false
        }
    }

    func isSelected(_ id: Int) -> Bool {
        plan?.selectedIDs.contains(id) ?? false
    }

    // MARK: - Starting a pipeline

    /// Reads every supported image of `folder`. The folder is read in full here, so no access is kept.
    func importPhotos(folder: URL) {
        discard()
        let gen: Int = beginGeneration()
        source = .photos(folderName: folder.lastPathComponent)
        poseSummary = nil
        usesAllFrames = true
        isScanning = true
        work = Task {
            do {
                let frames: [CapturedFrame] = try await PipelineWork.scanPhotos(folder: folder)
                guard gen == self.generation else { return }
                self.photoFrames = frames
                self.adoptCandidates(Self.photoItems(frames))
                self.isScanning = false
            } catch {
                guard gen == self.generation else { return }
                self.failScan(error)
            }
        }
    }

    /// Keeps the video's security scope and scores its candidate frames at the default interval.
    func importVideo(url: URL) {
        discard()
        if url.startAccessingSecurityScopedResource() {
            scopedURL = url
        }
        videoURL = url
        source = .video(fileName: url.lastPathComponent)
        poseSummary = nil
        usesAllFrames = false
        scanVideo()
    }

    /// Changes the sampling interval of the video and scores it again. Manual changes are dropped.
    func setSampleInterval(_ interval: Double) {
        guard case .video = source, interval != sampleInterval, !isScanning, !isEstimating else {
            return
        }
        sampleInterval = interval
        scanVideo()
    }

    /// Switches between every frame and the best `bestCount`. Manual changes are dropped.
    func setUsesAllFrames(_ value: Bool) {
        usesAllFrames = value
        plan?.setMode(mode)
    }

    /// Sets the number of best frames. Manual changes are dropped when the mode is "best N".
    func setBestCount(_ count: Int) {
        bestCount = max(1, count)
        if !usesAllFrames {
            plan?.setMode(mode)
        }
    }

    /// Adds or removes one frame from the selection. Manual changes stay until the mode or interval changes.
    func toggle(_ id: Int) {
        plan?.toggle(id)
    }

    /// Drops the pending frames, cancels running work and releases the video's security scope.
    /// Keeps the depth model state and the summary of the last pose estimation.
    func discard() {
        beginGeneration()
        scopedURL?.stopAccessingSecurityScopedResource()
        scopedURL = nil
        videoURL = nil
        photoFrames = []
        videoCandidates = []
        source = nil
        candidates = []
        plan = nil
        thumbnails = [:]
        usesAllFrames = false
        bestCount = FrameSelectionMode.defaultCount
        sampleInterval = Self.defaultSampleInterval
        isScanning = false
        scanFailed = false
        isEstimating = false
        estimationStatus = nil
        estimationFailed = false
        errorMessage = nil
    }

    /// Called when the project window closes. Stops all work and the depth download.
    func close() {
        discard()
        depthDownload?.cancel()
        depthDownload = nil
    }

    // MARK: - Thumbnails

    /// Decodes the thumbnail of one candidate off the main actor, unless it is already cached.
    func loadThumbnail(id: Int) async {
        guard thumbnails[id] == nil,
              let candidate = candidates.first(where: { $0.id == id }) else {
            return
        }
        let gen: Int = generation
        let image: CGImage? = await PipelineWork.thumbnail(data: candidate.imageData, maxSide: PipelineWork.thumbnailSide)
        guard gen == generation, let image else {
            return
        }
        thumbnails[id] = image
    }

    // MARK: - Pose estimation

    /// Decodes the selected frames (video frames at full size) and runs the pose stage on them.
    /// On success the project adopts the poses and this pipeline is discarded.
    func estimatePoses(document: Binding<ProjectDocument>, project: ProjectModel) {
        guard canEstimatePoses, let plan, let source else {
            return
        }
        let selected: [Int] = plan.selectedInOrder
        let base: ProjectContents = document.wrappedValue.contents
        let gen: Int = beginGeneration()
        estimationFailed = false
        errorMessage = nil
        poseSummary = nil
        isEstimating = true
        estimationStatus = "Reading the selected frames\u{2026}"
        work = Task {
            do {
                let inputKind: InputKind
                let frames: [CapturedFrame]
                switch source {
                case .photos:
                    inputKind = .photos
                    var picked: [CapturedFrame] = []
                    for index in selected {
                        picked.append(self.photoFrames[index])
                    }
                    frames = picked
                case .video:
                    guard let url = self.videoURL else {
                        throw CocoaError(.fileNoSuchFile)
                    }
                    inputKind = .video
                    var times: [Double] = []
                    for index in selected {
                        times.append(self.videoCandidates[index].requestedTime)
                    }
                    frames = try await PipelineWork.decodeVideoFrames(url: url, times: times)
                }
                guard gen == self.generation else { return }
                self.estimationStatus = "Estimating poses with Apple Photogrammetry. This can take several minutes."
                let run: PoseRun = await PipelineWork.estimatePoses(frames: frames, base: base, inputKind: inputKind)
                guard gen == self.generation else { return }
                self.finishEstimate(run, document: document, project: project)
            } catch {
                guard gen == self.generation else { return }
                self.isEstimating = false
                self.estimationStatus = nil
                self.estimationFailed = true
                self.errorMessage = Self.errorText(error)
            }
        }
    }

    private func finishEstimate(_ run: PoseRun, document: Binding<ProjectDocument>, project: ProjectModel) {
        switch run {
        case .imported(let contents, let summary):
            project.adopt(contents, into: document)
            discard()
            poseSummary = summary
        case .failed(let message):
            isEstimating = false
            estimationStatus = nil
            estimationFailed = true
            errorMessage = message
        case .cancelled:
            isEstimating = false
            estimationStatus = nil
        }
    }

    // MARK: - Depth model

    /// Reads the depth model state from the store. Runs hashing off the main actor.
    func refreshDepthStatus() async {
        guard let store = modelStore else {
            depthError = "The model folder is not available."
            return
        }
        depthStatus = await PipelineWork.depthStatus(store: store, descriptor: Self.depthDescriptor)
    }

    /// Downloads the depth model into the app container. One download at a time.
    func downloadDepthModel() {
        guard depthDownload == nil, let store = modelStore else {
            return
        }
        depthError = nil
        depthProgress = 0
        let descriptor: ModelDescriptor = Self.depthDescriptor
        depthDownload = Task {
            do {
                try await PipelineWork.downloadDepth(store: store, descriptor: descriptor) { fraction in
                    Task { @MainActor in
                        guard self.depthDownload != nil else { return }
                        self.depthProgress = fraction
                    }
                }
            } catch {
                self.depthError = Self.errorText(error)
            }
            self.depthProgress = nil
            self.depthDownload = nil
            await self.refreshDepthStatus()
        }
    }

    func cancelDepthDownload() {
        depthDownload?.cancel()
    }

    // MARK: - Messages

    /// Plain sentence for a selection warning.
    static func warningText(_ warning: FrameSelectionPlan.Warning) -> String {
        switch warning {
        case .noneSelected:
            return "No frame is selected. Select at least \(FrameSelectionPlan.minimumRecommendedViews) frames."
        case .tooFew(let selected):
            return "Only \(selected) frames are selected. At least \(FrameSelectionPlan.minimumRecommendedViews) are recommended."
        case .manyViews(let selected):
            return "\(selected) frames are selected. More than \(FrameSelectionPlan.slowTrainingViewCount) views make training slow and memory-heavy."
        }
    }

    /// Plain sentence for an error of the frame or model stage.
    static func errorText(_ error: Error) -> String {
        if let failure = error as? PoseEstimationFailure {
            return PipelineWork.describe(failure)
        }
        return error.localizedDescription
    }

    /// Accessibility label of a frame cell, for example "Frame 3 at 1.25 s, selected".
    static func frameLabel(_ candidate: PipelineCandidate, selected: Bool) -> String {
        let base: String
        if let time = candidate.time {
            base = "Frame \(candidate.id + 1) at \(timeText(time))"
        } else {
            base = "Frame \(candidate.id + 1), \(candidate.caption)"
        }
        return selected ? "\(base), selected" : "\(base), not selected"
    }

    static func timeText(_ seconds: Double) -> String {
        String(format: "%.2f s", seconds)
    }

    // MARK: - Private

    /// Cancels the running work and returns the generation that new work must still match.
    @discardableResult
    private func beginGeneration() -> Int {
        generation += 1
        work?.cancel()
        work = nil
        return generation
    }

    /// Starts a video scan at the current interval. Clears the previous candidates and selection.
    private func scanVideo() {
        guard let url = videoURL else {
            return
        }
        let gen: Int = beginGeneration()
        candidates = []
        plan = nil
        thumbnails = [:]
        videoCandidates = []
        errorMessage = nil
        scanFailed = false
        isScanning = true
        let interval: Double = sampleInterval
        work = Task {
            do {
                let found: [VideoCandidate] = try await PipelineWork.scoreVideo(url: url, sampleInterval: interval)
                guard gen == self.generation else { return }
                self.videoCandidates = found
                self.adoptCandidates(Self.videoItems(found))
                self.isScanning = false
            } catch {
                guard gen == self.generation else { return }
                self.failScan(error)
            }
        }
    }

    /// Replaces the candidates and computes the automatic selection for the current mode.
    private func adoptCandidates(_ items: [PipelineCandidate]) {
        candidates = items
        thumbnails = [:]
        var planItems: [FrameSelectionPlan.Item] = []
        planItems.reserveCapacity(items.count)
        for item in items {
            planItems.append(FrameSelectionPlan.Item(id: item.id, time: item.time, sharpness: item.sharpness))
        }
        plan = FrameSelectionPlan(items: planItems, mode: mode)
    }

    /// Drops the pipeline and shows why the scan failed.
    private func failScan(_ error: Error) {
        let message: String = Self.errorText(error)
        discard()
        errorMessage = message
        scanFailed = true
    }

    private static func photoItems(_ frames: [CapturedFrame]) -> [PipelineCandidate] {
        var items: [PipelineCandidate] = []
        items.reserveCapacity(frames.count)
        for (offset, frame) in frames.enumerated() {
            items.append(PipelineCandidate(id: offset,
                                           time: nil,
                                           sharpness: nil,
                                           imageData: frame.data,
                                           caption: frame.sourceName))
        }
        return items
    }

    private static func videoItems(_ found: [VideoCandidate]) -> [PipelineCandidate] {
        var items: [PipelineCandidate] = []
        items.reserveCapacity(found.count)
        for (offset, candidate) in found.enumerated() {
            items.append(PipelineCandidate(id: offset,
                                           time: candidate.time,
                                           sharpness: candidate.sharpness,
                                           imageData: candidate.thumbnail,
                                           caption: timeText(candidate.time)))
        }
        return items
    }

    /// `Application Support/SparseSplat/Models` inside the app container.
    private static func makeModelStore() -> ModelStore? {
        guard let support = try? FileManager.default.url(for: .applicationSupportDirectory,
                                                         in: .userDomainMask,
                                                         appropriateFor: nil,
                                                         create: true) else {
            return nil
        }
        let root: URL = support
            .appending(path: "SparseSplat", directoryHint: .isDirectory)
            .appending(path: "Models", directoryHint: .isDirectory)
        return ModelStore(rootDirectory: root)
    }
}
