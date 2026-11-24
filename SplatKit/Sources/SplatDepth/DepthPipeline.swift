import Metal
import Synchronization
import SplatCore
import SplatIO

/// Where a view's depth map came from in the last preparation.
public nonisolated enum DepthMapSource: Sendable, Equatable {
    /// A map that came with the project: a depth file next to the photos, or one imported by hand.
    case supplied
    /// A map the depth model produced, either in this run or in an earlier run (its cache).
    case estimated
    /// No depth map for this view.
    case none
}

/// One image's depth outcome from the last preparation, in dataset order.
public nonisolated struct DepthStatusRow: Identifiable, Sendable {
    public let name: String
    public let status: ViewDepthStatus
    public let source: DepthMapSource

    public var id: String { name }

    public init(name: String, status: ViewDepthStatus, source: DepthMapSource = .none) {
        self.name = name
        self.status = status
        self.source = source
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
    /// The estimator that produced `estimatedMaps`, after `automatic` was resolved.
    public let estimator: DepthEstimatorKind
    /// Which estimator the run uses and why, for example "Depth: Depth Anything 3 (automatic)". Shown with the depth status.
    public let estimatorNote: String
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
        // Automatic becomes Depth Anything 3 when its package is available (imported or bundled), else V2 Small.
        // Everything below uses the resolved kind, so caches and warnings name the estimator that actually runs.
        let chosen: DepthEstimatorKind = configuration.depthPrior.estimator
        let depthAnything3Package: AvailablePackage? = store?.availablePackage(for: ModelDescriptor.depthAnything3Base)
        let estimator: DepthEstimatorKind = chosen.resolved(isDepthAnything3Available: depthAnything3Package != nil)
        let estimatorNote: String = Self.estimatorNote(chosen: chosen, resolved: estimator, package: depthAnything3Package)
        // Supplied maps and the cache of the chosen estimator only; another estimator's maps are never read.
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
        // Depth from an iPhone companion, when the project asks for it and one answers on the network.
        var companionMaps: [String: DepthMap] = [:]
        #if os(macOS)
        if configuration.depthPrior.companionDevice == true {
            companionMaps = await RemoteDepthWorker.depthMaps(for: dataset.views, model: "depth-anything-3-base")
            if companionMaps.isEmpty {
                warnings.append("No iPhone companion answered on the network. Depth is computed on this Mac.")
            } else {
                providers.append(SuppliedDepthProvider(maps: companionMaps))
            }
        }
        #endif
        let cachedMaps: [String: DepthMap] = sources.cached
        if !cachedMaps.isEmpty {
            providers.append(SuppliedDepthProvider(maps: cachedMaps))
        }
        var recorder: EstimatedDepthRecorder?
        if configuration.isEnabled(.depthPearson), let store {
            do {
                // Multi-view estimators (Depth Anything 3) run on all views of the dataset together.
                if let provider = try DepthEstimatorFactory.installed(estimator, in: store, views: dataset.views) {
                    // `installed` already picks the imported or the bundled Depth Anything 3 package.
                    let created = EstimatedDepthRecorder(estimator: provider)
                    recorder = created
                    providers.append(created)
                } else if cachedMaps.isEmpty {
                    if estimator.needsImportedPackage {
                        warnings.append("\(estimator.descriptor.displayName) is not installed. Import the pose-free model package first.")
                    } else {
                        warnings.append("\(estimator.descriptor.displayName) is not installed. Download it in the Depth Model section.")
                    }
                }
            } catch {
                warnings.append("The depth model could not be loaded: \(String(describing: error)).")
            }
        }
        let provider: (any DepthProvider)? = providers.isEmpty ? nil : FirstAvailableDepthProvider(providers)

        var prepared = DepthPreparation.prepare(dataset,
                                                provider: provider,
                                                trustMetric: contents.manifest.source == .arkit,
                                                settings: configuration.depthPrior,
                                                seed: configuration.seed)

        // v1.5 normal prior: MoGe-2 normals on the views that have a depth prior. Not cached; MoGe runs per run.
        if configuration.normalPrior.isEnabled && configuration.isEnabled(.depthPearson) {
            var normalEstimator: MoGeDepthEstimator?
            if let store {
                do {
                    normalEstimator = try MoGeDepthEstimator.installed(in: store, descriptor: ModelCatalog.moge2SmallInt8)
                } catch {
                    warnings.append("MoGe-2 could not be loaded for the normal prior: \(String(describing: error)).")
                }
            }
            if let normalEstimator {
                let withNormals = NormalPreparation.attach(prepared.dataset, provider: normalEstimator)
                prepared.dataset = withNormals.dataset
                if withNormals.failed > 0 {
                    warnings.append("No normal prior for \(withNormals.failed) view(s): MoGe-2 gave no usable normals.")
                }
            } else {
                warnings.append("The normal prior needs MoGe-2. Download it in the Depth Model section.")
            }
        }

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

        // Where each view's map came from: a supplied map wins over the model, as in the provider order above.
        let suppliedNames: Set<String> = Set(sources.supplied.keys)
        var estimatedNames: Set<String> = Set(cachedMaps.keys)
        estimatedNames.formUnion(companionMaps.keys)
        if let recorder {
            estimatedNames.formUnion(recorder.maps.keys)
        }
        var statuses: [DepthStatusRow] = []
        statuses.reserveCapacity(prepared.dataset.views.count)
        for (index, view) in prepared.dataset.views.enumerated() where index < prepared.report.views.count {
            let source: DepthMapSource
            if suppliedNames.contains(view.name) {
                source = .supplied
            } else if estimatedNames.contains(view.name) {
                source = .estimated
            } else {
                source = .none
            }
            statuses.append(DepthStatusRow(name: view.name, status: prepared.report.views[index], source: source))
        }

        return DepthPreparedRun(
            dataset: prepared.dataset,
            configuration: configuration,
            initializer: initializer,
            report: prepared.report,
            statuses: statuses,
            estimatedMaps: recorder?.maps ?? [:],
            estimator: estimator,
            estimatorNote: estimatorNote,
            warnings: warnings
        )
    }

    /// "Depth: Depth Anything 3 (automatic)", naming the resolved estimator. Depth Anything 3 says where its package comes from.
    static func estimatorNote(chosen: DepthEstimatorKind,
                              resolved: DepthEstimatorKind,
                              package: AvailablePackage?) -> String {
        let name: String
        switch resolved {
        case .automatic, .depthAnythingV2Small:
            name = "Depth Anything V2 Small"
        case .moge2Small:
            name = "MoGe-2 Small"
        case .depthAnything3:
            if let package {
                name = package.isBundled ? "Depth Anything 3, bundled" : "Depth Anything 3, imported"
            } else {
                name = "Depth Anything 3, not installed"
            }
        }
        return chosen == .automatic ? "Depth: \(name) (automatic)" : "Depth: \(name)"
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
