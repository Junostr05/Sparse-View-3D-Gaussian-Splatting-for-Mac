import Metal
import Observation
import SwiftUI
import SplatCore
import SplatIO
import SplatTraining

/// Everything `TrainingSession` needs, built off the main actor.
nonisolated struct PreparedRun: Sendable {
    let dataset: TrainingDataset
    let initialCloud: GaussianCloud
    let configuration: TrainingConfiguration
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

    /// Reads `capture.json` and the images it names from an ARKit folder.
    @concurrent
    static func arkitContents(folder: URL, base: ProjectContents) async throws -> ProjectContents {
        try withSecurityScope(folder) { () throws -> ProjectContents in
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
            return ProjectBundle.importARKit(capture: captureData, imageFiles: imageFiles, into: base)
        }
    }

    /// Decodes the dataset and sizes the initial cloud to the memory budget of this Mac.
    @concurrent
    static func prepare(_ contents: ProjectContents) async throws -> PreparedRun {
        let dataset: TrainingDataset = try ProjectBundle.loadDataset(contents, decode: ImageIODecoder.decode)
        let configuration: TrainingConfiguration = contents.manifest.training
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TrainingError.metal("no Metal device")
        }

        var viewSizes: [SIMD2<Int>] = []
        viewSizes.reserveCapacity(dataset.views.count)
        for view in dataset.views {
            viewSizes.append(SIMD2<Int>(view.image.width, view.image.height))
        }
        let reserved: Int = TrainingMemoryBudget.reservedImageBytes(viewSizes: viewSizes)
        let cap: Int = TrainingMemoryBudget.maxGaussians(
            workingSetBytes: device.recommendedMaxWorkingSetSize,
            shDegree: configuration.shDegree,
            reservedBytes: reserved,
            configuredMax: configuration.maxGaussians
        )
        let cloud: GaussianCloud = GaussianInitializer.initialize(dataset,
                                                                  configuration: configuration,
                                                                  maxCount: cap)
        return PreparedRun(dataset: dataset, initialCloud: cloud, configuration: configuration)
    }

    /// Parses the stored checkpoint PLY, if the package has one.
    @concurrent
    static func loadCheckpoint(_ contents: ProjectContents) async throws -> GaussianCloud? {
        try ProjectBundle.latestCheckpoint(contents)
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
    var errorMessage: String?
    /// The viewport model. Created from the stored checkpoint, or from the initial cloud of a run.
    private(set) var viewer: ViewerModel?

    @ObservationIgnored private var session: TrainingSession?
    @ObservationIgnored private var runTask: Task<Void, Never>?

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
                let updated = try await ProjectImport.arkitContents(folder: folder, base: base)
                self.adopt(updated, into: document)
            } catch {
                self.errorMessage = Self.describe(error)
            }
        }
    }

    /// Replaces the package contents after an import. Any viewer of the old model is dropped.
    private func adopt(_ contents: ProjectContents, into document: Binding<ProjectDocument>) {
        document.wrappedValue.contents = contents
        viewer = nil
        progress = nil
        errorMessage = nil
    }

    // MARK: - Viewer

    /// Shows the stored checkpoint in the viewport, if the package has one and no viewer exists yet.
    func restoreCheckpoint(_ contents: ProjectContents) async {
        guard viewer == nil, !isTraining else { return }
        do {
            guard let cloud = try await ProjectImport.loadCheckpoint(contents) else { return }
            guard viewer == nil else { return }
            viewer = ViewerModel(scene: LoadedScene(cloud: cloud, removedNonFinite: 0))
        } catch {
            errorMessage = Self.describe(error)
        }
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
        isTraining = true
        Task {
            await self.runTraining(contents: contents, document: document)
        }
    }

    private func runTraining(contents: ProjectContents, document: Binding<ProjectDocument>) async {
        do {
            let prepared: PreparedRun = try await ProjectImport.prepare(contents)
            let trainingSession = TrainingSession(dataset: prepared.dataset,
                                                  initialCloud: prepared.initialCloud,
                                                  configuration: prepared.configuration)
            session = trainingSession
            if viewer == nil {
                viewer = ViewerModel(scene: LoadedScene(cloud: prepared.initialCloud, removedNonFinite: 0))
            }
            // The run is a separate task so that `stop()` can cancel it. The stream ends when it returns.
            runTask = Task {
                do {
                    _ = try await trainingSession.run()
                } catch {
                    // Reported as a `.failed` progress event, which `apply` shows.
                }
            }
            for await event in trainingSession.progress {
                apply(event, document: document)
            }
        } catch {
            errorMessage = Self.describe(error)
        }
        session = nil
        runTask = nil
        isTraining = false
    }

    /// Shows one progress event. Previews update the viewer; the terminal states store a checkpoint.
    private func apply(_ event: TrainingProgress, document: Binding<ProjectDocument>) {
        if case .failed(let message) = event.state {
            errorMessage = message
        }
        var stored: TrainingProgress = event
        stored.preview = nil
        progress = stored

        if let cloud = event.preview {
            viewer?.replaceCloud(cloud)
        }

        switch event.state {
        case .finished, .cancelled:
            if let cloud = event.preview {
                let record = CheckpointRecord(iteration: event.iteration,
                                              gaussianCount: cloud.count,
                                              psnr: event.psnr,
                                              date: Date())
                document.wrappedValue.contents = ProjectBundle.withCheckpoint(document.wrappedValue.contents,
                                                                              cloud: cloud,
                                                                              record: record)
            }
        case .running, .paused, .failed:
            break
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
