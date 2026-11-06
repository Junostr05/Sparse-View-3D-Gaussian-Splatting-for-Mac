import Metal
import Synchronization
import SplatCore
import SplatDepth
import SplatIO
import SplatTraining

/// One image's depth outcome from the last preparation, in dataset order.
nonisolated struct DepthStatusRow: Identifiable, Sendable {
    let name: String
    let status: ViewDepthStatus

    var id: String { name }
}

/// Everything a run needs from depth preparation. Depth problems are listed here and never fail the run.
nonisolated struct DepthPreparedRun: Sendable {
    let run: PreparedRun
    let report: DepthPreparationReport
    /// Statuses in dataset order, for the "Depth status" disclosure.
    let statuses: [DepthStatusRow]
    /// Maps the monocular model produced in this run, keyed by image name. The project stores them,
    /// so the next run skips inference.
    let estimatedMaps: [String: DepthMap]
    /// Non-fatal problems: unreadable depth files, a depth model that would not load.
    let warnings: [String]
}

/// Wraps the monocular estimator and keeps every map it produces, keyed by view name, so the project can store them.
nonisolated final class EstimatedDepthRecorder: DepthProvider, Sendable {
    private let estimator: MonocularDepthEstimator
    private let produced: Mutex<[String: DepthMap]> = Mutex([:])

    init(estimator: MonocularDepthEstimator) {
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
nonisolated enum DepthPipeline {
    @concurrent
    static func prepare(contents: ProjectContents, store: ModelStore?) async throws -> DepthPreparedRun {
        let configuration: TrainingConfiguration = contents.manifest.training
        let dataset: TrainingDataset = try ProjectBundle.loadDataset(contents, decode: ImageIODecoder.decode)

        var warnings: [String] = []
        let supplied = ProjectBundle.depthMaps(contents)
        if !supplied.failures.isEmpty {
            let names: String = supplied.failures.keys.sorted().joined(separator: ", ")
            warnings.append("Depth files could not be read: \(names).")
        }

        // A supplied map wins over inference. The model runs only when the depth prior is on.
        var providers: [any DepthProvider] = []
        if !supplied.maps.isEmpty {
            providers.append(SuppliedDepthProvider(maps: supplied.maps))
        }
        var recorder: EstimatedDepthRecorder?
        if configuration.isEnabled(.depthPearson), let store {
            do {
                if let estimator = try MonocularDepthEstimator.installed(in: store,
                                                                         descriptor: ModelCatalog.depthAnythingV2Small) {
                    let created = EstimatedDepthRecorder(estimator: estimator)
                    recorder = created
                    providers.append(created)
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
            throw TrainingError.metal("no Metal device")
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
        let cloud: GaussianCloud = DenseInitializer.initialize(prepared.dataset,
                                                               configuration: configuration,
                                                               maxCount: cap)

        var statuses: [DepthStatusRow] = []
        statuses.reserveCapacity(prepared.dataset.views.count)
        for (index, view) in prepared.dataset.views.enumerated() where index < prepared.report.views.count {
            statuses.append(DepthStatusRow(name: view.name, status: prepared.report.views[index]))
        }

        return DepthPreparedRun(
            run: PreparedRun(dataset: prepared.dataset, initialCloud: cloud, configuration: configuration),
            report: prepared.report,
            statuses: statuses,
            estimatedMaps: recorder?.maps ?? [:],
            warnings: warnings
        )
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
