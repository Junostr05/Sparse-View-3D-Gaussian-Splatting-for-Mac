import Observation
import SwiftUI
import SplatCore
import SplatIO
import SplatTraining

/// Updated package contents plus an optional, non-fatal message for the inspector.
nonisolated struct ImportResult: Sendable {
    let contents: ProjectContents
    let message: String?
}

/// Blocking file, decode and GPU-budget work. Each function runs off the main actor (`@concurrent`).
nonisolated enum ProjectImport {
    /// Reads `sparse/0` and `images/` from a COLMAP folder and returns the updated package contents.
    @concurrent
    static func colmapContents(folder: URL, base: ProjectContents) async throws -> ProjectContents {
        try withSecurityScope(folder) { () throws -> ProjectContents in
            let sparse: URL = folder.appending(path: "sparse/0", directoryHint: .isDirectory)
            let cameras: Data = try Data(contentsOf: sparse.appending(path: "cameras.bin"))
            let images: Data = try Data(contentsOf: sparse.appending(path: "images.bin"))
            let points: Data? = try? Data(contentsOf: sparse.appending(path: "points3D.bin"))

            let records = try COLMAPReader.readImages(images)
            let imageRoot: URL = folder.appending(path: "images", directoryHint: .isDirectory)
            var imageFiles: [String: Data] = [:]
            var missing: [String] = []
            for record in records {
                if let data = try? Data(contentsOf: imageRoot.appending(path: record.name)) {
                    imageFiles[record.name] = data
                } else {
                    missing.append(record.name)
                }
            }
            guard missing.isEmpty else {
                throw DatasetError.missingImages(missing)
            }
            return ProjectBundle.importCOLMAP(cameras: cameras,
                                              images: images,
                                              points3D: points,
                                              imageFiles: imageFiles,
                                              into: base)
        }
    }

    /// Reads `capture.json`, the images it names and each frame's depth file, if it lists one.
    /// Missing or unreadable depth files do not fail the import; they are named in the message.
    @concurrent
    static func arkitContents(folder: URL, base: ProjectContents) async throws -> ImportResult {
        try withSecurityScope(folder) { () throws -> ImportResult in
            let captureData: Data = try Data(contentsOf: folder.appending(path: "capture.json"))
            let capture: ARKitCapture
            do {
                capture = try JSONDecoder().decode(ARKitCapture.self, from: captureData)
            } catch {
                throw DatasetError.invalidARKitCapture(String(describing: error))
            }
            var imageFiles: [String: Data] = [:]
            var missing: [String] = []
            for frame in capture.frames {
                if let data = try? Data(contentsOf: folder.appending(path: frame.image)) {
                    imageFiles[frame.image] = data
                } else {
                    missing.append(frame.image)
                }
            }
            guard missing.isEmpty else {
                throw DatasetError.missingImages(missing)
            }

            var depthFiles: [String: Data] = [:]
            var missingDepth: [String] = []
            var unreadableDepth: [String] = []
            for frame in capture.frames {
                guard let depthName = frame.depth else {
                    continue
                }
                guard let data = try? Data(contentsOf: folder.appending(path: depthName)) else {
                    missingDepth.append(frame.image)
                    continue
                }
                do {
                    _ = try DepthMapCodec.decode(data)
                    depthFiles[frame.image] = data
                } catch {
                    unreadableDepth.append(frame.image)
                }
            }

            let contents: ProjectContents = ProjectBundle.importARKit(capture: captureData,
                                                                      imageFiles: imageFiles,
                                                                      depthFiles: depthFiles,
                                                                      into: base)
            return ImportResult(contents: contents,
                                message: depthMessage(missing: missingDepth, unreadable: unreadableDepth))
        }
    }

    /// Reads every `<image name>.sdpt` in a folder and stores the ones that match an image. Files that match
    /// no image, or do not decode, are named in the message.
    @concurrent
    static func depthContents(folder: URL, base: ProjectContents) async throws -> ImportResult {
        try withSecurityScope(folder) { () throws -> ImportResult in
            let names: Set<String> = Set(base.manifest.imageFiles)
            let suffix: String = "." + DepthMapCodec.fileExtension
            let entries: [URL] = try FileManager.default.contentsOfDirectory(at: folder,
                                                                            includingPropertiesForKeys: nil)
            var maps: [String: DepthMap] = [:]
            var unmatched: [String] = []
            var unreadable: [String] = []
            for entry in entries where entry.lastPathComponent.hasSuffix(suffix) {
                let fileName: String = entry.lastPathComponent
                let imageName: String = String(fileName.dropLast(suffix.count))
                guard names.contains(imageName) else {
                    unmatched.append(fileName)
                    continue
                }
                do {
                    maps[imageName] = try DepthMapCodec.decode(Data(contentsOf: entry))
                } catch {
                    unreadable.append(fileName)
                }
            }

            let contents: ProjectContents = maps.isEmpty ? base : ProjectBundle.withDepthMaps(base, maps)
            var parts: [String] = []
            if maps.count + unmatched.count + unreadable.count == 0 {
                parts.append("No .sdpt depth maps were found in the folder.")
            } else {
                parts.append("Imported \(maps.count) depth map\(maps.count == 1 ? "" : "s").")
            }
            if !unmatched.isEmpty {
                parts.append("No image matches: \(Self.listing(unmatched)).")
            }
            if !unreadable.isEmpty {
                parts.append("Could not be read: \(Self.listing(unreadable)).")
            }
            return ImportResult(contents: contents, message: parts.joined(separator: " "))
        }
    }

    /// Summary of the depth files an ARKit import could not use. `nil` when every listed file was found.
    static func depthMessage(missing: [String], unreadable: [String]) -> String? {
        var parts: [String] = []
        if !missing.isEmpty {
            parts.append("Depth files are missing for \(listing(missing)).")
        }
        if !unreadable.isEmpty {
            parts.append("Depth files could not be read for \(listing(unreadable)).")
        }
        return parts.isEmpty ? nil : parts.joined(separator: " ")
    }

    /// Up to five names, then the count of the rest.
    static func listing(_ names: [String]) -> String {
        let shown: String = names.prefix(5).joined(separator: ", ")
        let rest: Int = names.count - min(names.count, 5)
        return rest > 0 ? "\(shown) and \(rest) more" : shown
    }

    /// Parses the stored checkpoint PLY, if the package has one.
    @concurrent
    static func loadCheckpoint(_ contents: ProjectContents) async throws -> GaussianCloud? {
        try ProjectBundle.latestCheckpoint(contents)
    }

    /// The cloud a continuation starts from: the stored latest checkpoint without its stored deletions.
    /// The deletions only apply while they belong to that checkpoint, as in `restoreCheckpoint`.
    @concurrent
    static func continuationStart(_ contents: ProjectContents) async throws -> GaussianCloud {
        guard let checkpoint = try ProjectBundle.latestCheckpoint(contents) else {
            throw ProjectError.corruptCheckpoint("the latest checkpoint is missing")
        }
        guard let stored = try ProjectBundle.deletions(contents),
              stored.baseGaussianCount == checkpoint.count else {
            return checkpoint
        }
        // Deleted Gaussians stay deleted: the continuation trains on the survivors only.
        let indices: [Int] = stored.deleted.map { Int($0) }
        let mask: SelectionMask = SelectionMask(count: checkpoint.count, indices: indices)
        return try checkpoint.removing(mask)
    }

    /// Runs `body` inside the folder's security-scoped access, when the system grants one.
    static func withSecurityScope<T>(_ folder: URL, _ body: () throws -> T) rethrows -> T {
        let scoped: Bool = folder.startAccessingSecurityScopedResource()
        defer {
            if scoped {
                folder.stopAccessingSecurityScopedResource()
            }
        }
        return try body()
    }
}

/// State and actions of the project window: import, training and checkpoints.
@Observable
final class ProjectModel {
    /// Latest progress without its preview. `nil` before the first run.
    private(set) var progress: TrainingProgress?
    private(set) var isTraining = false
    /// Options of "Continue Training". Edited in the inspector while no run is active.
    var continuation = ContinuationSettings()
    /// True while a continuation of a stored model runs (as opposed to a run from scratch).
    private(set) var isContinuing = false
    /// Iterations of the stored model plus all completed rounds of the current continuation.
    private(set) var totalIterations = 0
    /// 1-based number of the round in progress, 0 when no continuation runs.
    private(set) var round = 0
    var errorMessage: String?
    /// Non-fatal notes from the last import or run, such as depth files that were missing or unreadable.
    var warningMessage: String?
    /// Depth outcome per image from the last preparation. `nil` until a run has prepared depth.
    private(set) var depthStatuses: [DepthStatusRow]?
    /// The viewport model. Created from the stored checkpoint, or from the initial cloud of a run.
    private(set) var viewer: ViewerModel?

    @ObservationIgnored private var session: TrainingSession?
    @ObservationIgnored private var runTask: Task<Void, Never>?
    /// The held-out view of the current leave-one-out run. Named only in the `.selection` phase, so the
    /// final-phase events and the checkpoint take it from here.
    @ObservationIgnored private var heldOutViewName: String?
    /// True while the viewer's base cloud is the stored latest checkpoint, so its edits can be saved.
    /// False while a training preview is the base: a preview is not stored, and its deletions would
    /// refer to a cloud the package does not hold.
    @ObservationIgnored private var editsTargetCheckpoint = false
    /// Iterations already trained before the current session's round: the stored checkpoint's count, plus
    /// the rounds that finished. Zero for a run from scratch. Checkpoint records and progress are cumulative.
    @ObservationIgnored private var iterationOffset = 0
    /// Set by `stop()`. Ends an endless continuation between rounds, when no session is running to cancel.
    @ObservationIgnored private var stopRequested = false

    var isPaused: Bool {
        progress?.state == .paused
    }

    // MARK: - Import

    func importCOLMAP(folder: URL, into document: Binding<ProjectDocument>) {
        guard !isTraining else { return }
        let base: ProjectContents = document.wrappedValue.contents
        Task {
            do {
                let updated = try await ProjectImport.colmapContents(folder: folder, base: base)
                self.adopt(updated, into: document)
            } catch {
                self.errorMessage = Self.describe(error)
            }
        }
    }

    func importARKit(folder: URL, into document: Binding<ProjectDocument>) {
        guard !isTraining else { return }
        let base: ProjectContents = document.wrappedValue.contents
        Task {
            do {
                let updated: ImportResult = try await ProjectImport.arkitContents(folder: folder, base: base)
                self.adopt(updated.contents, into: document)
                self.warningMessage = updated.message
            } catch {
                self.errorMessage = Self.describe(error)
            }
        }
    }

    /// Stores the `.sdpt` depth maps of a folder whose names match the images of the project.
    func importDepthMaps(folder: URL, into document: Binding<ProjectDocument>) {
        guard !isTraining else { return }
        let base: ProjectContents = document.wrappedValue.contents
        Task {
            do {
                let updated: ImportResult = try await ProjectImport.depthContents(folder: folder, base: base)
                // The viewer and the progress stay: depth maps do not change the reconstruction shown.
                document.wrappedValue.contents = updated.contents
                self.errorMessage = nil
                self.warningMessage = updated.message
            } catch {
                self.errorMessage = Self.describe(error)
            }
        }
    }

    /// Replaces the package contents after an import. Any viewer of the old model is dropped.
    func adopt(_ contents: ProjectContents, into document: Binding<ProjectDocument>) {
        document.wrappedValue.contents = contents
        viewer = nil
        progress = nil
        errorMessage = nil
        warningMessage = nil
        depthStatuses = nil
    }

    // MARK: - Viewer

    /// Shows the stored checkpoint in the viewport, if the package has one and no viewer exists yet.
    /// The stored deletions of that checkpoint are applied, and later edits are written into `document`.
    func restoreCheckpoint(_ contents: ProjectContents, document: Binding<ProjectDocument>) async {
        guard viewer == nil, !isTraining else { return }
        do {
            guard let cloud = try await ProjectImport.loadCheckpoint(contents) else { return }
            guard viewer == nil else { return }
            let restored: ViewerModel = makeViewer(cloud: cloud, contents: contents,
                                                   document: document, baseIsCheckpoint: true)
            do {
                // The deletions belong to the latest checkpoint only: a different count means they are stale.
                if let stored = try ProjectBundle.deletions(contents), stored.baseGaussianCount == cloud.count {
                    restored.restoreDeletions(stored.deleted)
                }
            } catch {
                warningMessage = "The stored edits could not be read and were not applied: \(Self.describe(error))"
            }
            viewer = restored
        } catch {
            errorMessage = Self.describe(error)
        }
    }

    /// Creates the viewport model for `cloud`. Its deletions are written into `document` through
    /// `ProjectBundle.withDeletions`, but only while the base cloud is the stored checkpoint.
    private func makeViewer(cloud: GaussianCloud,
                            contents: ProjectContents,
                            document: Binding<ProjectDocument>,
                            baseIsCheckpoint: Bool) -> ViewerModel {
        editsTargetCheckpoint = baseIsCheckpoint
        let model = ViewerModel(scene: LoadedScene(cloud: cloud, removedNonFinite: 0),
                                upAxis: UpAxis.forProject(contents.manifest),
                                cameras: (try? ProjectBundle.cameraPoses(contents)) ?? [])
        // Editing is locked while a run is in progress; `runTraining` unlocks it at the end.
        model.isEditingAllowed = !isTraining
        model.onDeletionsChanged = { [weak self] baseCount, deleted in
            guard self?.editsTargetCheckpoint == true else { return }
            document.wrappedValue.contents = ProjectBundle.withDeletions(document.wrappedValue.contents,
                                                                         baseGaussianCount: baseCount,
                                                                         deleted: deleted)
        }
        return model
    }

    // MARK: - Training

    func startTraining(document: Binding<ProjectDocument>) {
        guard !isTraining else { return }
        let contents: ProjectContents = document.wrappedValue.contents
        guard !contents.manifest.imageFiles.isEmpty else {
            errorMessage = "Import a capture before training."
            return
        }
        errorMessage = nil
        progress = nil
        heldOutViewName = nil
        iterationOffset = 0
        totalIterations = 0
        round = 0
        isContinuing = false
        stopRequested = false
        isTraining = true
        viewer?.isEditingAllowed = false
        Task {
            await self.runTraining(contents: contents, document: document)
        }
    }

    /// True when the project has images and a stored checkpoint to continue from, and no run is active.
    func canContinue(_ contents: ProjectContents) -> Bool {
        !isTraining
            && !contents.manifest.imageFiles.isEmpty
            && contents.files[ProjectLayout.latestCheckpoint] != nil
    }

    /// Continues optimizing the stored checkpoint: rounds of `continuation.roundIterations` each, repeated
    /// until Stop when `continuation.isEndless` is set. Every round saves its result as the checkpoint.
    func continueTraining(document: Binding<ProjectDocument>) {
        let contents: ProjectContents = document.wrappedValue.contents
        guard !isTraining, canContinue(contents) else { return }
        errorMessage = nil
        progress = nil
        heldOutViewName = nil
        iterationOffset = 0
        totalIterations = 0
        round = 0
        isContinuing = true
        stopRequested = false
        isTraining = true
        viewer?.isEditingAllowed = false
        Task {
            await self.runContinuation(contents: contents, document: document)
        }
    }

    private func runTraining(contents: ProjectContents, document: Binding<ProjectDocument>) async {
        do {
            let depth: DepthPreparedRun = try await prepareDepth(contents: contents, document: document)
            if viewer == nil {
                let cloud: GaussianCloud = await DepthPipeline.previewCloud(depth)
                if viewer == nil {
                    viewer = makeViewer(cloud: cloud, contents: contents, document: document, baseIsCheckpoint: false)
                }
            }
            // Each run builds its own initial cloud from its own dataset (see `DepthPreparedRun.initializer`).
            let trainingSession = TrainingSession(dataset: depth.dataset,
                                                  configuration: depth.configuration,
                                                  initializer: depth.initializer)
            _ = await consume(trainingSession, document: document)
        } catch {
            errorMessage = Self.describe(error)
        }
        session = nil
        runTask = nil
        isTraining = false
        viewer?.isEditingAllowed = true
    }

    /// Continuation: loads the stored checkpoint (minus its deletions), then trains rounds from it. Each round
    /// is a new session that starts from the cloud the previous round stored. Endless mode repeats rounds
    /// while the last one finished and no stop was requested.
    private func runContinuation(contents: ProjectContents, document: Binding<ProjectDocument>) async {
        do {
            var cloud: GaussianCloud = try await ProjectImport.continuationStart(contents)
            // Depth is prepared once for all rounds, as in `runTraining`.
            let depth: DepthPreparedRun = try await prepareDepth(contents: contents, document: document)
            iterationOffset = contents.manifest.checkpoints.last?.iteration ?? 0
            totalIterations = iterationOffset
            round = 0
            while true {
                round += 1
                // Each round takes its own copy of the settings, so edits in the inspector apply from the next round.
                let settings: ContinuationSettings = continuation
                let roundConfiguration: TrainingConfiguration = depth.configuration.continued(
                    from: cloud.count,
                    shDegree: cloud.shDegree,
                    round: round - 1,
                    settings: settings
                )
                let roundSession = TrainingSession(dataset: depth.dataset,
                                                   initialCloud: cloud,
                                                   configuration: roundConfiguration)
                // `consume` has already stored this round's checkpoint when it returns.
                guard let terminal = await consume(roundSession, document: document),
                      terminal.state == .finished,
                      let finalCloud = terminal.preview else {
                    break
                }
                iterationOffset += terminal.iteration
                totalIterations = iterationOffset
                cloud = finalCloud
                guard continuation.isEndless, !stopRequested, !Task.isCancelled else {
                    break
                }
            }
        } catch {
            errorMessage = Self.describe(error)
        }
        session = nil
        runTask = nil
        isTraining = false
        isContinuing = false
        round = 0
        viewer?.isEditingAllowed = true
    }

    /// Depth preparation shared by a run and a continuation. Depth problems never fail it: they only appear in
    /// the status and the warning. Maps the monocular model produced are cached in the project.
    private func prepareDepth(contents: ProjectContents,
                              document: Binding<ProjectDocument>) async throws -> DepthPreparedRun {
        let depth: DepthPreparedRun = try await DepthPipeline.prepare(contents: contents,
                                                                      store: PipelineModel.makeModelStore())
        depthStatuses = depth.statuses
        warningMessage = depth.warnings.isEmpty ? nil : depth.warnings.joined(separator: " ")
        if !depth.estimatedMaps.isEmpty {
            // Cache the inferred maps in the project, so the next run skips the model.
            document.wrappedValue.contents = ProjectBundle.withDepthMaps(document.wrappedValue.contents,
                                                                         depth.estimatedMaps)
        }
        return depth
    }

    /// Runs one session as a separate task, so `stop()` and `togglePause()` can reach it, and applies its
    /// progress. Returns the session's last terminal event (`finished`, `cancelled` or `failed`), if any.
    private func consume(_ trainingSession: TrainingSession,
                         document: Binding<ProjectDocument>) async -> TrainingProgress? {
        session = trainingSession
        // The run is a separate task so that `stop()` can cancel it. The stream ends when it returns.
        runTask = Task {
            do {
                _ = try await trainingSession.run()
            } catch {
                // Reported as a `.failed` progress event, which `apply` shows.
            }
        }
        // A stop requested before this task existed (depth preparation, the gap between rounds) must not be lost.
        if stopRequested {
            runTask?.cancel()
        }
        var terminal: TrainingProgress?
        for await event in trainingSession.progress {
            apply(event, document: document)
            switch event.state {
            case .finished, .cancelled, .failed:
                terminal = event
            case .running, .paused:
                break
            }
        }
        session = nil
        runTask = nil
        return terminal
    }

    /// Shows one progress event. Previews update the viewer; the terminal states store a checkpoint.
    private func apply(_ event: TrainingProgress, document: Binding<ProjectDocument>) {
        // Cumulative over the stored checkpoint and the finished rounds; equal to `event.iteration` for a run from scratch.
        let iteration: Int = iterationOffset + event.iteration
        totalIterations = iteration
        if case .failed(let message) = event.state {
            errorMessage = message
        }
        if case .selection(let name) = event.phase {
            heldOutViewName = name
        }
        var stored: TrainingProgress = event
        stored.preview = nil
        progress = stored

        if let cloud = event.preview {
            viewer?.replaceCloud(cloud)
            editsTargetCheckpoint = false
        }

        switch event.state {
        case .finished, .cancelled:
            if let cloud = event.preview {
                // The checkpoint stored below is this cloud, so the viewer's base is the stored checkpoint again.
                editsTargetCheckpoint = true
                // A single run has no held-out fields; leave-one-out carries them on the terminal event.
                let record = CheckpointRecord(iteration: iteration,
                                              gaussianCount: cloud.count,
                                              psnr: event.psnr,
                                              date: Date(),
                                              heldOutView: heldOutView(for: event.phase),
                                              heldOutPSNR: event.heldOut?.psnr,
                                              heldOutSSIM: event.heldOut?.ssim,
                                              bestIteration: event.bestIteration)
                document.wrappedValue.contents = ProjectBundle.withCheckpoint(document.wrappedValue.contents,
                                                                              cloud: cloud,
                                                                              record: record)
            }
        case .running, .paused, .failed:
            break
        }
    }

    /// The held-out view of a phase: named in `.selection`, remembered from it in `.final`, nil for `.single`.
    private func heldOutView(for phase: TrainingPhase) -> String? {
        switch phase {
        case .single:
            return nil
        case .selection(let name):
            return name
        case .final:
            return heldOutViewName
        }
    }

    func togglePause() {
        guard isTraining, let trainingSession = session else { return }
        let resume = isPaused
        Task {
            if resume {
                await trainingSession.resume()
            } else {
                await trainingSession.pause()
            }
        }
    }

    /// Cancels the run. The run stores its last completed model as a checkpoint.
    func stop() {
        stopRequested = true
        runTask?.cancel()
    }

    // MARK: - Messages

    static func describe(_ error: Error) -> String {
        if let datasetError = error as? DatasetError {
            switch datasetError {
            case .missingImages(let names):
                return "Images named by the capture are missing: \(names.prefix(5).joined(separator: ", "))."
            case .noViews:
                return "The capture has no views."
            default:
                return "The capture could not be read: \(String(describing: datasetError))."
            }
        }
        if let trainingError = error as? TrainingError {
            return "Training failed: \(String(describing: trainingError))."
        }
        return "The operation failed: \(String(describing: error))."
    }
}
