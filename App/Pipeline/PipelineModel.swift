import CoreGraphics
import Foundation
import ImageIO
import Observation
import SwiftUI
import SplatCore
import SplatDepth
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

    /// Reads the given image files, as dropped onto the window. Each file's security scope is held for the read only.
    @concurrent
    static func scanPhotos(urls: [URL]) async throws -> [CapturedFrame] {
        var scoped: [URL] = []
        for url in urls where url.startAccessingSecurityScopedResource() {
            scoped.append(url)
        }
        defer {
            for url in scoped {
                url.stopAccessingSecurityScopedResource()
            }
        }
        return try PhotoFolderImporter.load(urls: urls, maxSide: frameMaxSide)
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

    /// Runs the pose stage on the frames and, on success, writes the poses into a copy of `base`.
    ///
    /// The chain follows `preference`: automatic is Apple Photogrammetry, then Depth Anything 3 when its package
    /// is installed; the other choices use one provider only. The provider that succeeded decides the pose method.
    @concurrent
    static func estimatePoses(frames: [CapturedFrame],
                              base: ProjectContents,
                              inputKind: InputKind,
                              preference: PoseMethodPreference,
                              da3Package: URL?) async -> PoseRun {
        let apple: ApplePhotogrammetryPoseProvider = ApplePhotogrammetryPoseProvider()
        let depthAnything: DepthAnything3PoseProvider? = da3Package.map { DepthAnything3PoseProvider(packageURL: $0) }
        var providers: [any PoseProvider] = []
        switch preference {
        case .automatic:
            providers.append(apple)
            if let depthAnything {
                providers.append(depthAnything)
            }
        case .apple:
            providers.append(apple)
        case .depthAnything3:
            guard let depthAnything else {
                return .failed("The Depth Anything 3 model is not installed. Import it in the Pose-Free Model section.")
            }
            providers.append(depthAnything)
        }

        let chain = PoseProviderChain(providers: providers)
        let outcome: PoseChainOutcome = await chain.estimate(for: frames)
        if Task.isCancelled {
            return .cancelled
        }
        switch outcome {
        case .success(let result, let providerName, let attempts):
            let method: PoseMethod = providerName == depthAnything?.name ? .depthAnything3 : .apple
            let contents: ProjectContents = ProjectBundle.importEstimatedPoses(frames: frames,
                                                                               result: result,
                                                                               inputKind: inputKind,
                                                                               method: method,
                                                                               into: base)
            var summary: String = "\(result.registeredImageNames.count) of \(frames.count) images registered (\(providerName))"
            // Depth Anything 3 only runs after Apple Photogrammetry in the automatic chain, so a second attempt is a fallback.
            if method == .depthAnything3, attempts.count > 1 {
                summary += " after Apple Photogrammetry failed"
            }
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

    /// The imported package of `descriptor`, if one is stored. Checks files, hence off the main actor.
    @concurrent
    static func importedPackage(store: ModelStore, descriptor: ModelDescriptor) async -> URL? {
        store.importedPackageURL(for: descriptor)
    }

    /// Copies a user-chosen package into the store. The security scope of `url` is held for the copy.
    @concurrent
    static func importPackage(from url: URL, store: ModelStore, descriptor: ModelDescriptor) async throws {
        try ProjectImport.withSecurityScope(url) { () throws in
            try store.importPackage(from: url, for: descriptor)
        }
    }

    /// Deletes the stored version directory of `descriptor`.
    @concurrent
    static func removePackage(store: ModelStore, descriptor: ModelDescriptor) async throws {
        try store.remove(descriptor)
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

/// The state of one downloadable depth model, as the Depth Model section shows it.
struct DepthModelState: Equatable {
    /// Result of the last check of the store. `nil` until the model has been checked.
    var status: ModelStatus?
    /// Download fraction in 0...1 while this model downloads, `nil` otherwise.
    var progress: Double?
    /// Message of the last failed download or removal.
    var error: String?
}

/// The pose method picked in the Poses section. `automatic` tries Apple Photogrammetry first and Depth Anything 3
/// after it, when its package is installed.
nonisolated enum PoseMethodPreference: String, CaseIterable, Sendable {
    case automatic
    case apple
    case depthAnything3

    var title: String {
        switch self {
        case .automatic: "Automatic"
        case .apple: "Apple Photogrammetry"
        case .depthAnything3: "Depth Anything 3"
        }
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
    /// The downloadable depth models, one per estimator, in the order the Depth Model section lists them.
    static let depthDescriptors: [ModelDescriptor] = ModelCatalog.all.filter { descriptor in
        DepthEstimatorKind.allCases.contains { $0.descriptor.id == descriptor.id }
    }
    static let da3Descriptor: ModelDescriptor = ModelDescriptor.depthAnything3Base
    /// `UserDefaults` key of the pose method preference.
    static let poseMethodDefaultsKey: String = "SparseSplat.poseMethodPreference"

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

    /// State of each depth model, by descriptor id. A model without an entry has not been checked yet.
    private(set) var depthStates: [String: DepthModelState] = [:]

    /// The method the user picked. Kept even while its model is missing.
    private(set) var poseMethodPreference: PoseMethodPreference
    /// The installed Depth Anything 3 package, or `nil` when none is imported.
    private(set) var da3PackageURL: URL?
    private(set) var isImportingDA3: Bool = false
    private(set) var da3Error: String?

    @ObservationIgnored private var photoFrames: [CapturedFrame] = []
    @ObservationIgnored private var videoCandidates: [VideoCandidate] = []
    @ObservationIgnored private var videoURL: URL?
    /// The video URL when the system granted a security scope. Released by `discard`, or on deinit.
    @ObservationIgnored private var scopedURL: URL?
    @ObservationIgnored private var work: Task<Void, Never>?
    /// The running depth model download by descriptor id. One download runs at a time.
    @ObservationIgnored private var depthDownloads: [String: Task<Void, Never>] = [:]
    /// Incremented whenever work starts or is cancelled. Results of older work are dropped.
    @ObservationIgnored private var generation: Int = 0
    private let modelStore: ModelStore?

    init() {
        modelStore = Self.makeModelStore()
        poseMethodPreference = Self.storedPoseMethodPreference()
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

    /// The state of one depth model. Unchecked models report no status and no progress.
    func depthState(_ descriptor: ModelDescriptor) -> DepthModelState {
        depthStates[descriptor.id] ?? DepthModelState()
    }

    /// `true` when the depth model of `kind` is installed and verified. Depth Anything 3 is ready when its
    /// pose-free package is imported (it is not downloaded from the Depth Model section).
    func isDepthReady(_ kind: DepthEstimatorKind) -> Bool {
        if kind.needsImportedPackage {
            return da3PackageURL != nil
        }
        if case .ready? = depthState(kind.descriptor).status {
            return true
        }
        return false
    }

    /// `true` when any depth model is installed. The pipeline step list reads it.
    var anyDepthModelReady: Bool {
        DepthEstimatorKind.allCases.contains { isDepthReady($0) }
    }

    /// `true` while any depth model downloads.
    var isDepthDownloading: Bool {
        !depthDownloads.isEmpty
    }

    /// The first download error of any depth model, for the pipeline step list.
    var depthErrorText: String? {
        for descriptor in Self.depthDescriptors {
            if let error = depthState(descriptor).error {
                return error
            }
        }
        return nil
    }

    /// `true` when the model is missing or damaged and a download can fix it.
    func depthCanDownload(_ descriptor: ModelDescriptor) -> Bool {
        switch depthState(descriptor).status {
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

    /// Reads the given image files, as a drop of files does. Like `importPhotos(folder:)`, no access is kept.
    func importPhotos(urls: [URL]) {
        discard()
        let gen: Int = beginGeneration()
        source = .photos(folderName: "Dropped photos")
        poseSummary = nil
        usesAllFrames = true
        isScanning = true
        work = Task {
            do {
                let frames: [CapturedFrame] = try await PipelineWork.scanPhotos(urls: urls)
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
        plan?.setMode(mode, minimumSpacing: minimumSpacing)
    }

    /// Sets the number of best frames. Manual changes are dropped when the mode is "best N".
    func setBestCount(_ count: Int) {
        bestCount = max(1, count)
        if !usesAllFrames {
            plan?.setMode(mode, minimumSpacing: minimumSpacing)
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

    /// Called when the project window closes. Stops all work and the depth downloads.
    func close() {
        discard()
        for task in depthDownloads.values {
            task.cancel()
        }
        depthDownloads = [:]
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
        let preference: PoseMethodPreference = poseMethodPreference
        let da3Package: URL? = da3PackageURL
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
                self.estimationStatus = "Estimating poses. This can take several minutes."
                let run: PoseRun = await PipelineWork.estimatePoses(frames: frames,
                                                                    base: base,
                                                                    inputKind: inputKind,
                                                                    preference: preference,
                                                                    da3Package: da3Package)
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

    /// Sets the state of one depth model, creating its entry when needed.
    private func updateDepthState(_ id: String, _ change: (inout DepthModelState) -> Void) {
        var state: DepthModelState = depthStates[id] ?? DepthModelState()
        change(&state)
        depthStates[id] = state
    }

    /// Reads the state of every depth model from the store. Runs hashing off the main actor.
    func refreshDepthStatus() async {
        guard let store = modelStore else {
            for descriptor in Self.depthDescriptors {
                updateDepthState(descriptor.id) { $0.error = "The model folder is not available." }
            }
            return
        }
        for descriptor in Self.depthDescriptors {
            let status: ModelStatus = await PipelineWork.depthStatus(store: store, descriptor: descriptor)
            updateDepthState(descriptor.id) { $0.status = status }
        }
    }

    /// Downloads one depth model into the app container. One download at a time, across all depth models.
    func downloadDepthModel(_ descriptor: ModelDescriptor) {
        guard depthDownloads.isEmpty, let store = modelStore else {
            return
        }
        let id: String = descriptor.id
        updateDepthState(id) { state in
            state.error = nil
            state.progress = 0
        }
        depthDownloads[id] = Task {
            do {
                try await PipelineWork.downloadDepth(store: store, descriptor: descriptor) { fraction in
                    Task { @MainActor in
                        guard self.depthDownloads[id] != nil else { return }
                        self.updateDepthState(id) { $0.progress = fraction }
                    }
                }
            } catch {
                self.updateDepthState(id) { $0.error = Self.errorText(error) }
            }
            self.updateDepthState(id) { $0.progress = nil }
            self.depthDownloads[id] = nil
            await self.refreshDepthStatus()
        }
    }

    func cancelDepthDownload(_ descriptor: ModelDescriptor) {
        depthDownloads[descriptor.id]?.cancel()
    }

    /// Deletes one installed depth model. Not allowed while a depth model downloads.
    func removeDepthModel(_ descriptor: ModelDescriptor) {
        guard depthDownloads.isEmpty, let store = modelStore else {
            return
        }
        updateDepthState(descriptor.id) { $0.error = nil }
        Task {
            do {
                try await PipelineWork.removePackage(store: store, descriptor: descriptor)
            } catch {
                self.updateDepthState(descriptor.id) { $0.error = Self.errorText(error) }
            }
            await self.refreshDepthStatus()
        }
    }

    // MARK: - Pose method and Depth Anything 3 model

    /// Stores the pose method preference in `UserDefaults`.
    func setPoseMethodPreference(_ preference: PoseMethodPreference) {
        poseMethodPreference = preference
        UserDefaults.standard.set(preference.rawValue, forKey: Self.poseMethodDefaultsKey)
    }

    /// Reads the Depth Anything 3 package state from the store.
    func refreshDA3Package() async {
        guard let store = modelStore else {
            da3PackageURL = nil
            return
        }
        da3PackageURL = await PipelineWork.importedPackage(store: store, descriptor: Self.da3Descriptor)
    }

    /// Copies a converted Depth Anything 3 package into the store, replacing any earlier one.
    /// Failures are reported in `da3Error`.
    func importDA3Package(from url: URL) async {
        guard !isImportingDA3, !isEstimating else {
            return
        }
        guard let store = modelStore else {
            da3Error = "The model folder is not available."
            return
        }
        isImportingDA3 = true
        da3Error = nil
        do {
            try await PipelineWork.importPackage(from: url, store: store, descriptor: Self.da3Descriptor)
        } catch {
            da3Error = Self.errorText(error)
        }
        isImportingDA3 = false
        await refreshDA3Package()
    }

    /// Deletes the imported Depth Anything 3 package. Not allowed while poses are being estimated.
    func removeDA3Model() {
        guard !isImportingDA3, !isEstimating, let store = modelStore else {
            return
        }
        da3Error = nil
        Task {
            do {
                try await PipelineWork.removePackage(store: store, descriptor: Self.da3Descriptor)
            } catch {
                self.da3Error = Self.errorText(error)
            }
            await self.refreshDA3Package()
        }
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
        plan = FrameSelectionPlan(items: planItems, mode: mode, minimumSpacing: minimumSpacing)
    }

    /// Smallest time between two automatically chosen video frames: half the even spacing of the chosen
    /// count. Nearly identical neighbours (a paused camera) make poor and sometimes unusable views. Photos and
    /// "use all frames" have no spacing.
    private var minimumSpacing: Double {
        guard !usesAllFrames, case .video = source else {
            return 0
        }
        var first: Double = .greatestFiniteMagnitude
        var last: Double = -.greatestFiniteMagnitude
        for item in candidates {
            if let time = item.time {
                first = min(first, time)
                last = max(last, time)
            }
        }
        guard last > first else {
            return 0
        }
        return (last - first) / Double(2 * max(1, bestCount))
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

    /// The preference stored in `UserDefaults`, or `automatic` when none is stored or it is unknown.
    static func storedPoseMethodPreference() -> PoseMethodPreference {
        guard let raw = UserDefaults.standard.string(forKey: poseMethodDefaultsKey),
              let preference = PoseMethodPreference(rawValue: raw) else {
            return .automatic
        }
        return preference
    }

    /// `Application Support/SparseSplat/Models` inside the app container.
    static func makeModelStore() -> ModelStore? {
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
