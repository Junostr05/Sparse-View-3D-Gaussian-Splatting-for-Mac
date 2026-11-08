import Metal
import Synchronization
import SplatCore
import SplatIO

/// One image's depth outcome from the last preparation, in dataset order.
public nonisolated struct DepthStatusRow: Identifiable, Sendable {
    public let name: String
    public let status: ViewDepthStatus

    public var id: String { name }

    public init(name: String, status: ViewDepthStatus) {
        self.name = name
        self.status = status
    }
}

/// Failures of depth preparation that stop it (as opposed to the warnings it collects).
public enum DepthPipelineError: Error, Equatable {
    /// The default Metal device is missing, so the memory budget cannot be computed.
    case noMetalDevice
}

/// Everything a run needs from depth preparation. Depth problems are listed here and never fail the run.
public nonisolated struct DepthPreparedRun: Sendable {
    /// The full dataset with its depth priors set. Leave-one-out takes views out of it inside the session.
    public let dataset: TrainingDataset
    public let configuration: TrainingConfiguration
    /// Builds the dense initial cloud of a run from that run's dataset, with the memory cap of the full dataset.
    /// Each run calls it, so a held-out view never reaches the cloud of a run that must not see it.
    public let initializer: @Sendable (TrainingDataset) -> GaussianCloud
    public let report: DepthPreparationReport
    /// Statuses in dataset order, for the "Depth status" disclosure.
    public let statuses: [DepthStatusRow]
    /// Maps the monocular model produced in this run, keyed by image name. The project stores them under
    /// `estimator`, so the next run with the same model skips inference.
    public let estimatedMaps: [String: DepthMap]
    /// The estimator that produced `estimatedMaps`.
    public let estimator: DepthEstimatorKind
    /// Non-fatal problems: unreadable depth files, a depth model that would not load or is not installed.
    public let warnings: [String]
}

/// Wraps the monocular estimator and keeps every map it produces, keyed by view name, so the project can store them.
final class EstimatedDepthRecorder: DepthProvider, Sendable {
    private let estimator: any DepthProvider
    private let produced: Mutex<[String: DepthMap]> = Mutex([:])

    init(estimator: any DepthProvider) {
        self.estimator = estimator
    }

    func depthMap(for view: TrainingView) throws -> DepthMap? {
        guard let map = try estimator.depthMap(for: view) else {
            return nil
        }
        produced.withLock { $0[view.name] = map }
        return map
    }

    var maps: [String: DepthMap] {
        produced.withLock { $0 }
    }
}

/// Depth preparation for a training run: supplied maps first, then the depth model when enabled, then
/// alignment, then the dense initial cloud. Blocking work runs off the main actor (`@concurrent`).
public nonisolated enum DepthPipeline {
    /// `useRefinedPoses` starts the run from the stored refined poses (a continuation); `false` uses the original poses.
    @concurrent
    public static func prepare(contents: ProjectContents,
                        store: ModelStore?,
                        useRefinedPoses: Bool = false) async throws -> DepthPreparedRun {
        let configuration: TrainingConfiguration = contents.manifest.training
        let dataset: TrainingDataset = try ProjectBundle.loadDataset(contents,
                                                                     decode: ImageIODecoder.decode,
                                                                     useRefinedPoses: useRefinedPoses)

        var warnings: [String] = []
        // Supplied maps and the cache of the chosen estimator only; another estimator's maps are never read.
        let estimator: DepthEstimatorKind = configuration.depthPrior.estimator
        let sources: DepthSources = ProjectBundle.depthSources(contents, estimator: estimator)
        if !sources.suppliedFailures.isEmpty {
            let names: String = sources.suppliedFailures.keys.sorted().joined(separator: ", ")
            warnings.append("Depth files could not be read: \(names).")
        }
        if !sources.cachedFailures.isEmpty {
            let names: String = sources.cachedFailures.keys.sorted().joined(separator: ", ")
            warnings.append("Cached depth maps could not be read and are estimated again: \(names).")
        }

        // A supplied map wins over inference. The cache of the selected estimator comes next. The model runs only when the depth prior is on.
        var providers: [any DepthProvider] = []
        if !sources.supplied.isEmpty {
            providers.append(SuppliedDepthProvider(maps: sources.supplied))
        }
        let cachedMaps: [String: DepthMap] = sources.cached
        if !cachedMaps.isEmpty {
            providers.append(SuppliedDepthProvider(maps: cachedMaps))
        }
        var recorder: EstimatedDepthRecorder?
        if configuration.isEnabled(.depthPearson), let store {
            do {
                if let provider = try DepthEstimatorFactory.installed(estimator, in: store) {
                    let created = EstimatedDepthRecorder(estimator: provider)
                    recorder = created
                    providers.append(created)
                } else if cachedMaps.isEmpty {
                    warnings.append("\(estimator.descriptor.displayName) is not installed. Download it in the Depth Model section.")
                }
            } catch {
                warnings.append("The depth model could not be loaded: \(String(describing: error)).")
            }
        }
        let provider: (any DepthProvider)? = providers.isEmpty ? nil : FirstAvailableDepthProvider(providers)

        let prepared = DepthPreparation.prepare(dataset,
                                                provider: provider,
                                                trustMetric: contents.manifest.source == .arkit,
                                                settings: configuration.depthPrior,
                                                seed: configuration.seed)

        // Memory cap as in Phase 3, plus the depth buffers (and pseudo-view priors) of this run.
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw DepthPipelineError.noMetalDevice
        }
        var viewSizes: [SIMD2<Int>] = []
        viewSizes.reserveCapacity(prepared.dataset.views.count)
        for view in prepared.dataset.views {
            let size: SIMD2<Int> = SIMD2<Int>(view.image.width, view.image.height)
            viewSizes.append(size)
        }
        let pseudoSizes: [SIMD2<Int>] = pseudoViewSizes(prepared.dataset, configuration: configuration)
        let reserved: Int = TrainingMemoryBudget.reservedImageBytes(viewSizes: viewSizes)
            + TrainingMemoryBudget.reservedDepthBytes(viewSizes: viewSizes, pseudoViewSizes: pseudoSizes)
        let cap: Int = TrainingMemoryBudget.maxGaussians(
            workingSetBytes: device.recommendedMaxWorkingSetSize,
            shDegree: configuration.shDegree,
            reservedBytes: reserved,
            configuredMax: configuration.maxGaussians
        )
        let initializer: @Sendable (TrainingDataset) -> GaussianCloud = { runDataset in
            DenseInitializer.initialize(runDataset, configuration: configuration, maxCount: cap)
        }

        var statuses: [DepthStatusRow] = []
        statuses.reserveCapacity(prepared.dataset.views.count)
        for (index, view) in prepared.dataset.views.enumerated() where index < prepared.report.views.count {
            statuses.append(DepthStatusRow(name: view.name, status: prepared.report.views[index]))
        }

        return DepthPreparedRun(
            dataset: prepared.dataset,
            configuration: configuration,
            initializer: initializer,
            report: prepared.report,
            statuses: statuses,
            estimatedMaps: recorder?.maps ?? [:],
            estimator: estimator,
            warnings: warnings
        )
    }

    /// The first cloud the viewport shows, before any run has a preview: the initializer applied to the full
    /// dataset. Only drawn, never trained on, so it does not leak a held-out view.
    @concurrent
    public static func previewCloud(_ depth: DepthPreparedRun) async -> GaussianCloud {
        depth.initializer(depth.dataset)
    }

    /// Upper bound of the pseudo-view priors: every nearest-neighbour pair yields `perPair` views at
    /// a reduced resolution. Empty when pseudo-views are off.
    static func pseudoViewSizes(_ dataset: TrainingDataset, configuration: TrainingConfiguration) -> [SIMD2<Int>] {
        guard configuration.isEnabled(.pseudoViews) else {
            return []
        }
        let settings: PseudoViewSettings = configuration.pseudoViews
        let divisor: Int = max(1, settings.resolutionDivisor)
        var sizes: [SIMD2<Int>] = []
        for pair in PseudoViews.pairs(dataset.views) {
            let view: TrainingView = dataset.views[pair.x]
            let size: SIMD2<Int> = SIMD2<Int>(view.image.width / divisor, view.image.height / divisor)
            for _ in 0..<max(0, settings.perPair) {
                sizes.append(size)
            }
        }
        return sizes
    }
}
