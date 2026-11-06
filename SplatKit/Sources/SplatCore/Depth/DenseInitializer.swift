import Foundation

/// Counts from one dense initialization run.
public struct DenseInitReport: Sendable, Equatable {
    /// Dense candidates that passed the edge test and were back-projected (kept + inconsistent).
    public var backProjected: Int
    /// Grid pixels skipped because they lie on a depth edge.
    public var rejectedEdges: Int
    /// Candidates removed because another view sees them in front of its surface.
    public var rejectedInconsistent: Int
    /// Points left after voxel deduplication (SfM points included).
    public var afterDedup: Int
    /// Points left after the seeded cap.
    public var final: Int

    public init(backProjected: Int, rejectedEdges: Int, rejectedInconsistent: Int,
                afterDedup: Int, final: Int) {
        self.backProjected = backProjected
        self.rejectedEdges = rejectedEdges
        self.rejectedInconsistent = rejectedInconsistent
        self.afterDedup = afterDedup
        self.final = final
    }
}

/// Builds the initial Gaussian cloud from metric depth of aligned or measured views (Phase 4 P8).
public enum DenseInitializer {
    /// Returns nil when no view has `depth?.metricDepth`.
    public static func pointCloud(_ dataset: TrainingDataset,
                                  settings: DenseInitSettings,
                                  maxCount: Int,
                                  seed: UInt64) -> (PointCloud, DenseInitReport)? {
        let views = metricViews(dataset)
        guard !views.isEmpty else { return nil }
        let cap = min(settings.maxPoints, maxCount / 2)
        var report = DenseInitReport(backProjected: 0, rejectedEdges: 0,
                                     rejectedInconsistent: 0, afterDedup: 0, final: 0)

        var candidates: [DenseCandidate] = []
        if cap > 0 {
            for viewIndex in views.indices {
                collectCandidates(viewIndex: viewIndex, views: views, settings: settings, cap: cap,
                                  report: &report, candidates: &candidates)
            }
        }

        // Voxel size: the median pixel footprint of the kept dense points. Without dense points
        // there is no size, and SfM points are kept as they are.
        let voxelSize: Float = DepthPrior.median(of: candidates.map { $0.footprint })
        var positions: [SIMD3<Float>] = []
        var colors: [SIMD3<Float>] = []
        var occupied = Set<SIMD3<Int32>>()

        // SfM points are inserted first, so they win over dense points in the same voxel.
        if settings.includeSfMPoints, let points = dataset.points {
            let count = min(points.positions.count, points.colors.count)
            for index in 0..<count {
                let position = points.positions[index]
                guard position.x.isFinite, position.y.isFinite, position.z.isFinite else { continue }
                if voxelSize > 0 {
                    guard occupied.insert(voxelKey(position, size: voxelSize)).inserted else { continue }
                }
                positions.append(position)
                colors.append(points.colors[index])
            }
        }
        // Views in index order, pixels row-major (candidates are already in that order).
        for candidate in candidates {
            if voxelSize > 0 {
                guard occupied.insert(voxelKey(candidate.position, size: voxelSize)).inserted else { continue }
            }
            positions.append(candidate.position)
            colors.append(candidate.color)
        }
        report.afterDedup = positions.count

        var finalPositions = positions
        var finalColors = colors
        if positions.count > cap {
            let selected = GaussianInitializer.subsetIndices(total: positions.count, maxCount: cap, seed: seed)
            finalPositions = selected.map { positions[$0] }
            finalColors = selected.map { colors[$0] }
        }
        report.final = finalPositions.count
        return (PointCloud(positions: finalPositions, colors: finalColors), report)
    }

    /// Dense cloud when possible, else `GaussianInitializer.initialize` (Phase 3). maxCount = budget cap.
    public static func initialize(_ dataset: TrainingDataset,
                                  configuration: TrainingConfiguration,
                                  maxCount: Int) -> GaussianCloud {
        let settings = configuration.denseInit
        if settings.isEnabled,
           let result = pointCloud(dataset, settings: settings, maxCount: maxCount, seed: configuration.seed),
           !result.0.positions.isEmpty {
            return GaussianInitializer.fromPoints(result.0,
                                                  shDegree: configuration.shDegree,
                                                  initialOpacity: configuration.initialOpacity,
                                                  maxCount: maxCount,
                                                  seed: configuration.seed)
        }
        return GaussianInitializer.initialize(dataset, configuration: configuration, maxCount: maxCount)
    }

    // MARK: - Helpers

    /// A view whose metric depth matches its camera and image size.
    private struct MetricView {
        let camera: Camera
        let colors: [SIMD3<Float>]
        let depth: [Float]
    }

    /// One back-projected point that passed the edge and free-space checks.
    private struct DenseCandidate {
        let position: SIMD3<Float>
        let color: SIMD3<Float>
        /// Pixel footprint at this point: `z · stride / fx`.
        let footprint: Float
    }

    /// Views with usable metric depth, in dataset order. Views whose depth or image does not
    /// match the camera size are left out rather than thrown on.
    private static func metricViews(_ dataset: TrainingDataset) -> [MetricView] {
        var views: [MetricView] = []
        for view in dataset.views {
            guard let prior = view.depth, let metric = prior.metricDepth else { continue }
            let width = view.camera.width
            let height = view.camera.height
            guard prior.width == width, prior.height == height, metric.count == width * height,
                  view.image.width == width, view.image.height == height,
                  view.image.pixels.count == width * height else { continue }
            views.append(MetricView(camera: view.camera, colors: view.image.pixels, depth: metric))
        }
        return views
    }

    private static func isPositiveDepth(_ value: Float) -> Bool {
        value.isFinite && value > 0
    }

    private static func collectCandidates(viewIndex: Int,
                                          views: [MetricView],
                                          settings: DenseInitSettings,
                                          cap: Int,
                                          report: inout DenseInitReport,
                                          candidates: inout [DenseCandidate]) {
        let view = views[viewIndex]
        let camera = view.camera
        let width = camera.width
        let height = camera.height

        var validCount = 0
        for value in view.depth where isPositiveDepth(value) {
            validCount += 1
        }
        // Stride g = ceil(sqrt(valid · m / cap)) spreads about `cap / m` points over each view.
        let ratio: Double = Double(validCount) * Double(views.count) / Double(cap)
        let step: Int = max(1, Int(ratio.squareRoot().rounded(.up)))

        for y in stride(from: 0, to: height, by: step) {
            for x in stride(from: 0, to: width, by: step) {
                let index = y * width + x
                let z: Float = view.depth[index]
                guard isPositiveDepth(z) else { continue }
                if DepthGeometry.isDepthEdge(view.depth, width: width, height: height,
                                             x: x, y: y, ratio: settings.edgeRatio) {
                    report.rejectedEdges += 1
                    continue
                }
                report.backProjected += 1
                let position = DepthGeometry.backProject(x: x, y: y, depth: z, camera: camera)
                let votes = consistencyVotes(of: position, ownViewIndex: viewIndex, views: views,
                                             tolerance: settings.consistencyTolerance)
                if votes.violations >= 1, votes.violations >= votes.supports {
                    report.rejectedInconsistent += 1
                    continue
                }
                let footprint: Float = z * Float(step) / camera.fx
                candidates.append(DenseCandidate(position: position,
                                                 color: view.colors[index],
                                                 footprint: footprint))
            }
        }
    }

    /// Counts the other views that see `position` in front of their surface (violation) or on
    /// it within `tolerance` (support). Pixels are looked up at `floor(u), floor(v)`.
    private static func consistencyVotes(of position: SIMD3<Float>,
                                         ownViewIndex: Int,
                                         views: [MetricView],
                                         tolerance: Float) -> (violations: Int, supports: Int) {
        var violations = 0
        var supports = 0
        for otherIndex in views.indices where otherIndex != ownViewIndex {
            let other = views[otherIndex]
            guard let projected = DepthGeometry.project(position, camera: other.camera, near: 0) else {
                continue
            }
            let width = other.camera.width
            let height = other.camera.height
            let column: Float = projected.u.rounded(.down)
            let row: Float = projected.v.rounded(.down)
            guard column >= 0, column < Float(width), row >= 0, row < Float(height) else { continue }
            let surface: Float = other.depth[Int(row) * width + Int(column)]
            guard isPositiveDepth(surface) else { continue }
            let nearLimit: Float = surface * (1 - tolerance)
            let difference: Float = abs(projected.z - surface)
            let band: Float = tolerance * surface
            if projected.z < nearLimit {
                violations += 1
            } else if difference <= band {
                supports += 1
            }
        }
        return (violations, supports)
    }

    /// Voxel key `floor(p / size)`, clamped to the `Int32` range.
    private static func voxelKey(_ position: SIMD3<Float>, size: Float) -> SIMD3<Int32> {
        let scaledX: Float = (position.x / size).rounded(.down)
        let scaledY: Float = (position.y / size).rounded(.down)
        let scaledZ: Float = (position.z / size).rounded(.down)
        return SIMD3<Int32>(clampedInt32(scaledX), clampedInt32(scaledY), clampedInt32(scaledZ))
    }

    private static func clampedInt32(_ value: Float) -> Int32 {
        let limit: Float = 1_000_000_000
        let clamped: Float = min(max(value, -limit), limit)
        return Int32(clamped)
    }
}
