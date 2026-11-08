import Foundation
import Observation
import SplatCore
import SplatIO
import SplatTraining

/// What a queued job does with its project.
nonisolated enum QueueJobMode: String, Codable, CaseIterable, Sendable {
    case fromScratch
    case continueTraining

    var title: String {
        switch self {
        case .fromScratch: "Train from scratch"
        case .continueTraining: "Continue training"
        }
    }

    var headless: HeadlessTrainingMode {
        switch self {
        case .fromScratch: .fromScratch
        case .continueTraining: .continueTraining
        }
    }
}

/// Where a queued job stands.
nonisolated enum QueueJobState: String, Codable, Sendable {
    case waiting
    case running
    case done
    case failed

    var title: String {
        switch self {
        case .waiting: "Waiting"
        case .running: "Running"
        case .done: "Done"
        case .failed: "Failed"
        }
    }
}

/// One project in the training queue. The project is identified by its package path, inside the library folder.
nonisolated struct QueueJob: Codable, Identifiable, Equatable, Sendable {
    let id: UUID
    /// Path of the `.gsproj` package.
    var path: String
    /// Package name without the extension, for display.
    var name: String
    var mode: QueueJobMode
    /// Replaces the project's stored preset for this run. `nil` keeps the stored settings.
    var preset: TrainingPreset?
    var state: QueueJobState
    /// The outcome or the reason of a failure, for display.
    var message: String?
    var addedAt: Date
    var startedAt: Date?
    var endedAt: Date?
    /// Progress of the running job, in iterations of this run. Not stored.
    var iteration: Int?
    var totalIterations: Int?

    var url: URL {
        URL(fileURLWithPath: path, isDirectory: true)
    }

    private enum CodingKeys: String, CodingKey {
        case id, path, name, mode, preset, state, message, addedAt, startedAt, endedAt
    }

    init(id: UUID = UUID(), url: URL, mode: QueueJobMode, preset: TrainingPreset? = nil, addedAt: Date = Date()) {
        self.id = id
        self.path = url.path(percentEncoded: false)
        self.name = url.deletingPathExtension().lastPathComponent
        self.mode = mode
        self.preset = preset
        self.state = .waiting
        self.message = nil
        self.addedAt = addedAt
        self.startedAt = nil
        self.endedAt = nil
        self.iteration = nil
        self.totalIterations = nil
    }

    /// Decodes a stored job. A job that was running when the app quit is reported as failed by the queue on load.
    init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        id = try values.decode(UUID.self, forKey: .id)
        path = try values.decode(String.self, forKey: .path)
        name = try values.decode(String.self, forKey: .name)
        mode = try values.decode(QueueJobMode.self, forKey: .mode)
        preset = try values.decodeIfPresent(TrainingPreset.self, forKey: .preset)
        state = try values.decode(QueueJobState.self, forKey: .state)
        message = try values.decodeIfPresent(String.self, forKey: .message)
        addedAt = try values.decode(Date.self, forKey: .addedAt)
        startedAt = try values.decodeIfPresent(Date.self, forKey: .startedAt)
        endedAt = try values.decodeIfPresent(Date.self, forKey: .endedAt)
        iteration = nil
        totalIterations = nil
    }
}

/// Projects that have a training run open in their own window. The queue skips them while such a run is active.
@MainActor
enum ActiveProjectTraining {
    private static var paths: Set<String> = []

    static func set(_ url: URL, training: Bool) {
        let key: String = url.standardizedFileURL.path(percentEncoded: false)
        if training {
            paths.insert(key)
        } else {
            paths.remove(key)
        }
    }

    static func isTraining(_ url: URL) -> Bool {
        paths.contains(url.standardizedFileURL.path(percentEncoded: false))
    }
}

/// The batch queue: projects of the library trained one after another, headless. Survives closing the library
/// and project windows, keeps the Mac awake while a job runs, and notifies when jobs and the queue finish.
/// The jobs are stored in UserDefaults as JSON.
@Observable
@MainActor
final class TrainingQueue {
    /// Jobs in run order.
    private(set) var jobs: [QueueJob] = []
    /// True from the start of `runQueue()` until the last job ends or the queue is stopped.
    private(set) var isRunning: Bool = false
    /// The last refusal or notice, for an alert in the view.
    var message: String?

    static let storageKey: String = "trainingQueueJobs"
    /// The app's one queue. The library, the project windows and the App Intents share it.
    static let shared: TrainingQueue = TrainingQueue()

    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private var runTask: Task<Void, Never>?
    @ObservationIgnored private var jobTask: Task<HeadlessTrainingResult, Error>?
    @ObservationIgnored private var stopRequested: Bool = false
    @ObservationIgnored private var sleepActivity: NSObjectProtocol?

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        jobs = Self.load(from: defaults)
        // A job still marked running belongs to a previous launch that ended before it finished.
        var recovered: Bool = false
        for index in jobs.indices where jobs[index].state == .running {
            jobs[index].state = .failed
            jobs[index].message = "Interrupted when SparseSplat quit."
            jobs[index].endedAt = Date()
            recovered = true
        }
        if recovered {
            save()
        }
    }

    var hasWaitingJobs: Bool {
        jobs.contains { $0.state == .waiting }
    }

    // MARK: - Editing

    /// Appends the package at `url`. The mode defaults to continuing when the project has a checkpoint.
    /// Returns `false` when the project is already waiting in the queue.
    @discardableResult
    func add(package url: URL, mode: QueueJobMode? = nil, preset: TrainingPreset? = nil) -> Bool {
        let path: String = url.path(percentEncoded: false)
        guard !jobs.contains(where: { $0.path == path && $0.state == .waiting }) else {
            message = "\(url.deletingPathExtension().lastPathComponent) is already waiting in the queue."
            return false
        }
        let chosen: QueueJobMode = mode ?? Self.defaultMode(for: url)
        jobs.append(QueueJob(url: url, mode: chosen, preset: preset))
        QueueNotifications.requestPermissionOnce()
        save()
        return true
    }

    /// Removes waiting, done and failed jobs. A running job cannot be removed; stop the queue first.
    func remove(_ ids: Set<UUID>) {
        let before: Int = jobs.count
        jobs.removeAll { ids.contains($0.id) && $0.state != .running }
        if jobs.count != before {
            save()
        }
    }

    func move(fromOffsets source: IndexSet, toOffset destination: Int) {
        jobs.move(fromOffsets: source, toOffset: destination)
        save()
    }

    /// Changes the mode of a waiting job.
    func setMode(_ id: UUID, to mode: QueueJobMode) {
        updateWaiting(id) { $0.mode = mode }
    }

    func setPreset(_ id: UUID, to preset: TrainingPreset?) {
        updateWaiting(id) { $0.preset = preset }
    }

    /// Removes the finished and failed jobs, keeping the waiting ones.
    func clearFinished() {
        remove(Set(jobs.filter { $0.state == .done || $0.state == .failed }.map(\.id)))
    }

    private func updateWaiting(_ id: UUID, _ change: (inout QueueJob) -> Void) {
        guard let index = jobs.firstIndex(where: { $0.id == id }), jobs[index].state == .waiting else {
            return
        }
        change(&jobs[index])
        save()
    }

    // MARK: - Running

    /// Runs the waiting jobs in order. Jobs whose project is training in a window are skipped and marked failed.
    func runQueue(modelStore: ModelStore?) {
        guard !isRunning, hasWaitingJobs else {
            return
        }
        isRunning = true
        stopRequested = false
        beginKeepingAwake()
        runTask = Task {
            await runLoop(modelStore: modelStore)
        }
    }

    /// Stops the queue. The running job is cancelled; its last completed model is still stored.
    func stop() {
        stopRequested = true
        jobTask?.cancel()
    }

    private func runLoop(modelStore: ModelStore?) async {
        var finished: Int = 0
        var failed: Int = 0
        let order: [UUID] = jobs.filter { $0.state == .waiting }.map(\.id)
        for id in order {
            guard !stopRequested else {
                break
            }
            guard let job: QueueJob = jobs.first(where: { $0.id == id }), job.state == .waiting else {
                continue
            }
            if ActiveProjectTraining.isTraining(job.url) {
                fail(id, message: "Skipped: the project is training in its own window.")
                failed += 1
                continue
            }
            guard FileManager.default.fileExists(atPath: job.path) else {
                fail(id, message: "The project is no longer in the library folder.")
                failed += 1
                continue
            }
            start(id)
            let progress: @Sendable (TrainingProgress) -> Void = { [weak self] event in
                Task { @MainActor in
                    self?.progress(id, event)
                }
            }
            let task: Task<HeadlessTrainingResult, Error> = Task {
                try await ProjectTrainingRunner.run(package: job.url,
                                                    mode: job.mode.headless,
                                                    preset: job.preset,
                                                    modelStore: modelStore,
                                                    onProgress: progress)
            }
            jobTask = task
            do {
                let result: HeadlessTrainingResult = try await task.value
                jobTask = nil
                if result.state == .cancelled || stopRequested {
                    fail(id, message: "Stopped. The last completed model was stored: iteration \(result.iteration).")
                    failed += 1
                    QueueNotifications.post(title: "Training stopped", body: "\(job.name) was stopped early.")
                } else {
                    complete(id, message: Self.summary(of: result))
                    finished += 1
                    QueueNotifications.post(title: "Training finished", body: "\(job.name): \(Self.summary(of: result))")
                }
            } catch {
                jobTask = nil
                let text: String = Self.describe(error)
                fail(id, message: text)
                failed += 1
                QueueNotifications.post(title: "Training failed", body: "\(job.name): \(text)")
            }
        }
        if finished + failed > 0 {
            let body: String = stopRequested
                ? "The queue was stopped. \(finished) finished, \(failed) failed or stopped."
                : "\(finished) finished, \(failed) failed."
            QueueNotifications.post(title: "Training queue done", body: body)
        }
        isRunning = false
        runTask = nil
        stopRequested = false
        endKeepingAwake()
    }

    private func start(_ id: UUID) {
        guard let index = jobs.firstIndex(where: { $0.id == id }) else {
            return
        }
        jobs[index].state = .running
        jobs[index].startedAt = Date()
        jobs[index].endedAt = nil
        jobs[index].message = nil
        jobs[index].iteration = 0
        jobs[index].totalIterations = nil
        save()
    }

    private func progress(_ id: UUID, _ event: TrainingProgress) {
        guard let index = jobs.firstIndex(where: { $0.id == id }), jobs[index].state == .running else {
            return
        }
        if case .running = event.state {
            jobs[index].iteration = event.iteration
            jobs[index].totalIterations = event.totalIterations
        }
    }

    private func complete(_ id: UUID, message: String) {
        finish(id, state: .done, message: message)
    }

    private func fail(_ id: UUID, message: String) {
        finish(id, state: .failed, message: message)
    }

    private func finish(_ id: UUID, state: QueueJobState, message: String) {
        guard let index = jobs.firstIndex(where: { $0.id == id }) else {
            return
        }
        jobs[index].state = state
        jobs[index].message = message
        jobs[index].endedAt = Date()
        jobs[index].iteration = nil
        jobs[index].totalIterations = nil
        save()
    }

    // MARK: - Sleep

    /// Keeps the Mac from sleeping while a job runs. Released when the queue stops.
    private func beginKeepingAwake() {
        sleepActivity = ProcessInfo.processInfo.beginActivity(options: .idleSystemSleepDisabled,
                                                              reason: "Training projects in the SparseSplat queue")
    }

    private func endKeepingAwake() {
        if let sleepActivity {
            ProcessInfo.processInfo.endActivity(sleepActivity)
        }
        sleepActivity = nil
    }

    // MARK: - Persistence

    private static func load(from defaults: UserDefaults) -> [QueueJob] {
        guard let data: Data = defaults.data(forKey: storageKey) else {
            return []
        }
        return (try? JSONDecoder().decode([QueueJob].self, from: data)) ?? []
    }

    private func save() {
        guard let data = try? JSONEncoder().encode(jobs) else {
            return
        }
        defaults.set(data, forKey: Self.storageKey)
    }

    // MARK: - Messages

    /// Chooses continuation for projects with a stored checkpoint, and training from scratch otherwise.
    static func defaultMode(for url: URL) -> QueueJobMode {
        let manifestURL: URL = url.appending(path: ProjectLayout.manifest)
        guard let data = try? Data(contentsOf: manifestURL) else {
            return .fromScratch
        }
        let decoder: JSONDecoder = JSONDecoder()
        // Manifests store their dates as ISO 8601, as `ProjectBundle.encodeManifest` writes them.
        decoder.dateDecodingStrategy = .iso8601
        guard let manifest = try? decoder.decode(ProjectManifest.self, from: data) else {
            return .fromScratch
        }
        return manifest.checkpoints.isEmpty ? .fromScratch : .continueTraining
    }

    static func summary(of result: HeadlessTrainingResult) -> String {
        let psnr: String = result.psnr.map { String(format: "%.1f dB", $0) } ?? "PSNR not measured"
        return "iteration \(result.iteration), \(result.gaussianCount) Gaussians, \(psnr)"
    }

    static func describe(_ error: Error) -> String {
        if let headless = error as? HeadlessTrainingError {
            return headless.description
        }
        if let packageError = error as? ProjectPackageFiles.PackageError {
            return packageError.description
        }
        if error is CancellationError {
            return "Cancelled."
        }
        return "The operation failed: \(String(describing: error))"
    }
}
