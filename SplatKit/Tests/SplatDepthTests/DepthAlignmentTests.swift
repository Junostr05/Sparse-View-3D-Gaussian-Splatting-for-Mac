import Foundation
import SplatCore
import SplatDepth
import Testing

// MARK: - Synthetic scene

private let imageWidth = 64
private let imageHeight = 48

private func testCamera() -> Camera {
    Camera(width: imageWidth, height: imageHeight, fx: 60, fy: 60, cx: 32, cy: 24,
           rotation: Matrix3.identity, translation: SIMD3<Float>(0, 0, 0))
}

/// Ground-truth camera-z depth of a slanted plane, between 3 and 4.9 scene units.
private func planeDepth(_ x: Int, _ y: Int) -> Float {
    let slopeX: Float = 0.02 * Float(x)
    let slopeY: Float = 0.01 * Float(y)
    return 3 + slopeX + slopeY
}

/// Ground-truth inverse-affine prior. The aligned relation is `0.7 · p + 0.2 = 1/z` (P5 direction:
/// `s·p + t ≈ 1/z`), so the fit must recover scale 0.7 and shift 0.2. Solving for p gives
/// `p = (1/z − 0.2) / 0.7`, a Depth Anything style affine-in-disparity output.
private func inverseAffinePrior(_ z: Float) -> Float {
    (1 / z - 0.2) / 0.7
}

/// Pixel centres on a regular grid.
private func gridPixels(stepX: Int, stepY: Int) -> [(x: Int, y: Int)] {
    var pixels: [(x: Int, y: Int)] = []
    var y = 0
    while y < imageHeight {
        var x = 0
        while x < imageWidth {
            pixels.append((x: x, y: y))
            x += stepX
        }
        y += stepY
    }
    return pixels
}

/// Back-projects the centre of each pixel to the depth given by `depthAt`.
private func scenePoints(at pixels: [(x: Int, y: Int)], depthAt: (Int, Int) -> Float) -> [SIMD3<Float>] {
    let camera = testCamera()
    var result: [SIMD3<Float>] = []
    result.reserveCapacity(pixels.count)
    for pixel in pixels {
        let z: Float = depthAt(pixel.x, pixel.y)
        result.append(DepthGeometry.backProject(x: pixel.x, y: pixel.y, depth: z, camera: camera))
    }
    return result
}

/// A full-size depth map whose value at pixel (x, y) is `valueAt(x, y)`. All pixels are valid.
private func makeMap(kind: DepthKind, valueAt: (Int, Int) -> Float) -> DepthMap {
    var values: [Float] = []
    values.reserveCapacity(imageWidth * imageHeight)
    for y in 0..<imageHeight {
        for x in 0..<imageWidth {
            values.append(valueAt(x, y))
        }
    }
    return DepthMap(width: imageWidth, height: imageHeight, kind: kind, values: values)
}

private func makeView(name: String) -> TrainingView {
    let pixels = [SIMD3<Float>](repeating: SIMD3<Float>(0, 0, 0), count: imageWidth * imageHeight)
    return TrainingView(name: name, camera: testCamera(),
                        image: RGBImage(width: imageWidth, height: imageHeight, pixels: pixels))
}

private func singleViewDataset(points: [SIMD3<Float>]?) -> TrainingDataset {
    var cloud: PointCloud? = nil
    if let points = points {
        let colors = [SIMD3<Float>](repeating: SIMD3<Float>(0.5, 0.5, 0.5), count: points.count)
        cloud = PointCloud(positions: points, colors: colors)
    }
    return TrainingDataset(views: [makeView(name: "view0")], points: cloud)
}

/// Samples of a slanted plane under an arbitrary prior function, depth `2 + 0.05·i`.
private func syntheticSamples(count: Int, prior: (Double) -> Double) -> [(prior: Double, z: Double)] {
    var result: [(prior: Double, z: Double)] = []
    result.reserveCapacity(count)
    for index in 0..<count {
        let z: Double = 2.0 + 0.05 * Double(index)
        result.append((prior: prior(z), z: z))
    }
    return result
}

private func relativeError(_ value: Double, _ expected: Double) -> Double {
    abs(value - expected) / abs(expected)
}

// MARK: - Tests

@Suite struct DepthAlignmentTests {

    @Test func recoversAffineInverseDepth() throws {
        let pixels = gridPixels(stepX: 3, stepY: 4)
        let points = scenePoints(at: pixels, depthAt: planeDepth)
        let prior = makeMap(kind: .relativeInverseDepth) { x, y in
            inverseAffinePrior(planeDepth(x, y))
        }
        let samples = DepthAlignment.samples(points: points, camera: testCamera(), map: prior)
        #expect(samples.count == points.count)

        let result = DepthAlignment.fit(samples, kind: .relativeInverseDepth, scaleOnly: false,
                                        minimumPoints: 8, minimumInlierFraction: 0.3, seed: 1)
        let alignment = try #require(try? result.get())
        #expect(relativeError(alignment.scale, 0.7) < 1e-6)
        #expect(abs(alignment.shift - 0.2) < 1e-6)
        #expect(alignment.inliers >= points.count * 9 / 10)
        #expect(alignment.kindCorrected == false)
        #expect(alignment.fittedKind == .relativeInverseDepth)

        // Metric depth recovered from the aligned map matches the ground truth.
        let zMax: Double = 100 * 4.0
        let metric = DepthAlignment.metricDepth(prior, alignment: alignment, zMax: zMax)
        let index = 10 * imageWidth + 10
        #expect(relativeError(Double(metric[index]), Double(planeDepth(10, 10))) < 1e-4)
    }

    @Test func recoversFromAllPairsForSmallSets() throws {
        // 40 points: at most 64 samples, so every pair is a candidate.
        let pixels = gridPixels(stepX: 8, stepY: 6)
        let trimmed = Array(pixels.prefix(40))
        let points = scenePoints(at: trimmed, depthAt: planeDepth)
        let prior = makeMap(kind: .relativeInverseDepth) { x, y in
            inverseAffinePrior(planeDepth(x, y))
        }
        let samples = DepthAlignment.samples(points: points, camera: testCamera(), map: prior)
        #expect(samples.count == 40)

        let result = DepthAlignment.fit(samples, kind: .relativeInverseDepth, scaleOnly: false,
                                        minimumPoints: 8, minimumInlierFraction: 0.3, seed: 2)
        let alignment = try #require(try? result.get())
        #expect(relativeError(alignment.scale, 0.7) < 1e-6)
        #expect(abs(alignment.shift - 0.2) < 1e-6)
    }

    @Test func recoversWithThirtyPercentOutliers() throws {
        let pixels = gridPixels(stepX: 3, stepY: 4)
        let points = scenePoints(at: pixels, depthAt: planeDepth)

        // About 30 % of the pixels carry a random value instead of the true prior.
        var values: [Float] = []
        values.reserveCapacity(imageWidth * imageHeight)
        var outlierCount = 0
        for y in 0..<imageHeight {
            for x in 0..<imageWidth {
                let index = y * imageWidth + x
                let isOutlier: Bool = SplatRandom.uniform(seed: 3, index: UInt32(index), stream: 0) < 0.3
                if isOutlier {
                    let randomValue: Float = SplatRandom.uniform(seed: 9, index: UInt32(index), stream: 1)
                    values.append(randomValue * 2)
                    if pixels.contains(where: { $0.x == x && $0.y == y }) {
                        outlierCount += 1
                    }
                } else {
                    values.append(inverseAffinePrior(planeDepth(x, y)))
                }
            }
        }
        let prior = DepthMap(width: imageWidth, height: imageHeight, kind: .relativeInverseDepth,
                             values: values)
        let samples = DepthAlignment.samples(points: points, camera: testCamera(), map: prior)
        #expect(outlierCount > 0)
        #expect(outlierCount < samples.count / 2)

        let result = DepthAlignment.fit(samples, kind: .relativeInverseDepth, scaleOnly: false,
                                        minimumPoints: 8, minimumInlierFraction: 0.3, seed: 4)
        let alignment = try #require(try? result.get())
        #expect(relativeError(alignment.scale, 0.7) < 0.01)
        #expect(alignment.inliers >= samples.count / 2)
    }

    @Test func occludedPointsAreRejectedAsOutliers() throws {
        // A front surface at depth 2 covers the left half (x < 32). A back wall at depth 4 is
        // behind it: its points over the left half are occluded and must be pre-filtered.
        // (The wall stays below depth 5, where the inverse-affine prior would reach 0.)
        let allPixels = gridPixels(stepX: 3, stepY: 4)
        let frontPixels = allPixels.filter { $0.x < 32 }
        let frontPoints = scenePoints(at: frontPixels, depthAt: { _, _ in 2 })
        let wallPoints = scenePoints(at: allPixels, depthAt: { _, _ in 4 })
        let points = frontPoints + wallPoints

        let prior = makeMap(kind: .relativeInverseDepth) { x, _ in
            let visibleDepth: Float = x < 32 ? 2 : 4
            return inverseAffinePrior(visibleDepth)
        }
        let samples = DepthAlignment.samples(points: points, camera: testCamera(), map: prior)
        // Every front point survives; of the wall points only the right-half ones survive.
        #expect(samples.count == frontPixels.count + (allPixels.count - frontPixels.count))

        let result = DepthAlignment.fit(samples, kind: .relativeInverseDepth, scaleOnly: false,
                                        minimumPoints: 8, minimumInlierFraction: 0.3, seed: 6)
        let alignment = try #require(try? result.get())
        #expect(relativeError(alignment.scale, 0.7) < 0.02)
        #expect(abs(alignment.shift - 0.2) < 0.02)
    }

    @Test func tooFewPointsLeavesViewUnaligned() throws {
        let empty: [(prior: Double, z: Double)] = []
        #expect(DepthAlignment.fit(empty, kind: .relativeInverseDepth, scaleOnly: false,
                                   minimumPoints: 8, minimumInlierFraction: 0.3, seed: 1)
            == .failure(.tooFewPoints(0)))

        let seven = syntheticSamples(count: 7) { z in 0.7 / z + 0.2 }
        #expect(DepthAlignment.fit(seven, kind: .relativeInverseDepth, scaleOnly: false,
                                   minimumPoints: 8, minimumInlierFraction: 0.3, seed: 1)
            == .failure(.tooFewPoints(7)))

        // Through the preparation path: the view keeps a Pearson prior but no metric depth.
        let pixels = Array(gridPixels(stepX: 3, stepY: 4).prefix(7))
        let points = scenePoints(at: pixels, depthAt: planeDepth)
        let map = makeMap(kind: .relativeInverseDepth) { x, y in
            inverseAffinePrior(planeDepth(x, y))
        }
        let dataset = singleViewDataset(points: points)
        let prepared = DepthPreparation.prepare(dataset,
                                                provider: SuppliedDepthProvider(maps: ["view0": map]),
                                                trustMetric: false, settings: DepthPriorSettings(),
                                                seed: 11)
        #expect(prepared.report.views == [.unaligned(.tooFewPoints(7))])
        let prior = try #require(prepared.dataset.views[0].depth)
        #expect(prior.metricDepth == nil)
        #expect(prior.domain == .inverseDepth)
        #expect(prior.correlationTarget.count == imageWidth * imageHeight)
        #expect(prepared.report.hasAnyPrior)
        #expect(prepared.report.alignedOrMetricCount == 0)
    }

    @Test func rejectsNegativeScale() {
        // Prior falls as inverse depth rises, so every candidate slope is negative.
        let samples = syntheticSamples(count: 60) { z in -0.5 / z + 3 }
        let result = DepthAlignment.fit(samples, kind: .relativeInverseDepth, scaleOnly: false,
                                        minimumPoints: 8, minimumInlierFraction: 0.3, seed: 1)
        #expect(result == .failure(.noPositiveFit))
    }

    @Test func detectsDepthDeclaredAsInverse() throws {
        // Values 2z + 1 declared as inverse depth: the declared fit has a negative slope and
        // fails, so the other relative kind is fitted and kept.
        let pixels = gridPixels(stepX: 3, stepY: 4)
        let points = scenePoints(at: pixels, depthAt: planeDepth)
        let map = makeMap(kind: .relativeInverseDepth) { x, y in
            2 * planeDepth(x, y) + 1
        }
        let prepared = DepthPreparation.prepare(singleViewDataset(points: points),
                                                provider: SuppliedDepthProvider(maps: ["view0": map]),
                                                trustMetric: false, settings: DepthPriorSettings(),
                                                seed: 12)
        let status = try #require(prepared.report.views.first)
        guard case .aligned(let alignment) = status else {
            Issue.record("expected an aligned view, got \(status)")
            return
        }
        #expect(alignment.kindCorrected)
        #expect(alignment.fittedKind == .relativeDepth)
        #expect(relativeError(alignment.scale, 0.5) < 1e-5)
        #expect(abs(alignment.shift + 0.5) < 1e-4)
        let prior = try #require(prepared.dataset.views[0].depth)
        #expect(prior.domain == .depth)
        #expect(prior.isMeasuredMetric == false)
    }

    @Test func scaleOnlyForMetricOnCOLMAP() throws {
        // COLMAP scale is arbitrary: metric depth 3·z is fitted with scale only, z = s·p.
        let pixels = gridPixels(stepX: 3, stepY: 4)
        let points = scenePoints(at: pixels, depthAt: planeDepth)
        let map = makeMap(kind: .metric) { x, y in
            3 * planeDepth(x, y)
        }
        let prepared = DepthPreparation.prepare(singleViewDataset(points: points),
                                                provider: SuppliedDepthProvider(maps: ["view0": map]),
                                                trustMetric: false, settings: DepthPriorSettings(),
                                                seed: 13)
        let status = try #require(prepared.report.views.first)
        guard case .aligned(let alignment) = status else {
            Issue.record("expected an aligned view, got \(status)")
            return
        }
        #expect(relativeError(alignment.scale, 1.0 / 3.0) < 1e-6)
        #expect(alignment.shift == 0)
        #expect(alignment.fittedKind == .metric)
        let prior = try #require(prepared.dataset.views[0].depth)
        #expect(prior.isMeasuredMetric == false)
        let metric = try #require(prior.metricDepth)
        let index = 20 * imageWidth + 30
        #expect(relativeError(Double(metric[index]), Double(planeDepth(30, 20))) < 1e-5)
    }

    @Test func trustedMetricIsMeasured() throws {
        let map = makeMap(kind: .metric) { x, y in
            planeDepth(x, y)
        }
        let prepared = DepthPreparation.prepare(singleViewDataset(points: nil),
                                                provider: SuppliedDepthProvider(maps: ["view0": map]),
                                                trustMetric: true, settings: DepthPriorSettings(),
                                                seed: 14)
        #expect(prepared.report.views == [.measuredMetric])
        #expect(prepared.report.alignedOrMetricCount == 1)
        let prior = try #require(prepared.dataset.views[0].depth)
        #expect(prior.isMeasuredMetric)
        #expect(prior.domain == .depth)
        #expect(prior.l1Normalizer > 0)
        let metric = try #require(prior.metricDepth)
        #expect(abs(metric[0] - planeDepth(0, 0)) < 1e-6)
    }

    @Test func measuredMetricIgnoresInvalidPixels() throws {
        // Confidence 0 pixels (ARKit) are invalid and must not enter the metric depth.
        var values: [Float] = []
        var confidence: [UInt8] = []
        for y in 0..<imageHeight {
            for x in 0..<imageWidth {
                values.append(planeDepth(x, y))
                confidence.append(x == 0 ? 0 : 2)
            }
        }
        let map = DepthMap(width: imageWidth, height: imageHeight, kind: .metric, values: values,
                           confidence: confidence)
        let prepared = DepthPreparation.prepare(singleViewDataset(points: nil),
                                                provider: SuppliedDepthProvider(maps: ["view0": map]),
                                                trustMetric: true, settings: DepthPriorSettings(),
                                                seed: 15)
        let prior = try #require(prepared.dataset.views[0].depth)
        let metric = try #require(prior.metricDepth)
        #expect(metric[0] == 0)
        #expect(metric[1] > 0)
    }

    @Test func prepareWithoutProviderLeavesViewsUntouched() {
        var dataset = TrainingDataset(views: [makeView(name: "a"), makeView(name: "b")], points: nil)
        let existing = DepthPrior(width: imageWidth, height: imageHeight,
                                  correlationTarget: [Float](repeating: 1, count: imageWidth * imageHeight),
                                  domain: .depth)
        dataset.views[0].depth = existing
        let prepared = DepthPreparation.prepare(dataset, provider: nil, trustMetric: false,
                                                settings: DepthPriorSettings(), seed: 16)
        #expect(prepared.dataset.views.map(\.depth) == [existing, nil])
        #expect(prepared.report.views == [.none, .none])
        #expect(prepared.report.hasAnyPrior == false)
        #expect(prepared.report.alignedOrMetricCount == 0)
    }

    @Test func mismatchedDepthIsReportedNotThrown() throws {
        // A 16:9 depth map for a 4:3 view is rejected with a reason, not an error.
        let map = DepthMap(width: 64, height: 36, kind: .relativeInverseDepth,
                           values: [Float](repeating: 1, count: 64 * 36))
        let prepared = DepthPreparation.prepare(singleViewDataset(points: nil),
                                                provider: SuppliedDepthProvider(maps: ["view0": map]),
                                                trustMetric: false, settings: DepthPriorSettings(),
                                                seed: 17)
        let status = try #require(prepared.report.views.first)
        guard case .rejected(let reason) = status else {
            Issue.record("expected a rejected view, got \(status)")
            return
        }
        #expect(!reason.isEmpty)
        #expect(prepared.dataset.views[0].depth == nil)
        #expect(prepared.report.hasAnyPrior == false)
    }

    @Test func providerErrorIsReportedNotThrown() throws {
        let prepared = DepthPreparation.prepare(singleViewDataset(points: nil),
                                                provider: FailingDepthProvider(), trustMetric: false,
                                                settings: DepthPriorSettings(), seed: 18)
        let status = try #require(prepared.report.views.first)
        guard case .rejected = status else {
            Issue.record("expected a rejected view, got \(status)")
            return
        }
        #expect(prepared.dataset.views[0].depth == nil)
    }

    @Test func firstAvailableSkipsThrowingProvider() throws {
        let map = DepthMap(width: imageWidth, height: imageHeight, kind: .metric,
                           values: [Float](repeating: 2, count: imageWidth * imageHeight))
        let chain = FirstAvailableDepthProvider([
            FailingDepthProvider(),
            SuppliedDepthProvider(maps: ["view0": map]),
        ])
        let found = try chain.depthMap(for: makeView(name: "view0"))
        #expect(found == map)
        let missing = try chain.depthMap(for: makeView(name: "unknown"))
        #expect(missing == nil)
    }

    @Test func fitIsDeterministic() throws {
        // n > 64 takes the seeded pair path; 30 % outliers make the candidate choice matter.
        var samples: [(prior: Double, z: Double)] = []
        for index in 0..<200 {
            let z: Double = 2.0 + 0.02 * Double(index)
            let isOutlier: Bool = SplatRandom.uniform(seed: 21, index: UInt32(index), stream: 0) < 0.3
            let prior: Double = isOutlier
                ? Double(SplatRandom.uniform(seed: 22, index: UInt32(index), stream: 1))
                : (1 / z - 0.2) / 0.7
            samples.append((prior: prior, z: z))
        }
        let first = DepthAlignment.fit(samples, kind: .relativeInverseDepth, scaleOnly: false,
                                       minimumPoints: 8, minimumInlierFraction: 0.3, seed: 5)
        let second = DepthAlignment.fit(samples, kind: .relativeInverseDepth, scaleOnly: false,
                                        minimumPoints: 8, minimumInlierFraction: 0.3, seed: 5)
        #expect(first == second)
        #expect(try #require(try? first.get()).inliers >= 60)
    }

    @Test func recoversWithSixtyPercentOccludedPoints() throws {
        // n ≤ 64 takes every pair, n > 64 the seeded sampled pairs. 40 % are consistent.
        for count in [40, 200] {
            let consistentCount: Int = count * 2 / 5

            let metricSamples = occludedSamples(count: count, kind: .metric, scale: 0.8, shift: 0)
            let metricFit = DepthAlignment.fit(metricSamples, kind: .metric, scaleOnly: true,
                                               minimumPoints: 8, minimumInlierFraction: 0.3, seed: 31)
            let metricAlignment = try #require(try? metricFit.get())
            #expect(relativeError(metricAlignment.scale, 0.8) < 0.02)
            #expect(metricAlignment.shift == 0)
            #expect(metricAlignment.inliers >= consistentCount)

            let inverseSamples = occludedSamples(count: count, kind: .relativeInverseDepth, scale: 0.7, shift: 0.2)
            let inverseFit = DepthAlignment.fit(inverseSamples, kind: .relativeInverseDepth, scaleOnly: false,
                                                minimumPoints: 8, minimumInlierFraction: 0.3, seed: 32)
            let inverseAlignment = try #require(try? inverseFit.get())
            #expect(relativeError(inverseAlignment.scale, 0.7) < 0.02)
            #expect(abs(inverseAlignment.shift - 0.2) < 0.02)
            #expect(inverseAlignment.inliers >= consistentCount)
        }
    }

    @Test func syntheticSceneBackProjectionStaysOnTheSurface() throws {
        let scene = SyntheticScene.make(viewCount: 3, novelCount: 4, width: 160, height: 120,
                                        gaussians: 1200, sparsePoints: 200, seed: 5)
        let maps: [String: DepthMap] = Dictionary(uniqueKeysWithValues: zip(scene.train.map(\.name), scene.trainDepth))
        let prepared = DepthPreparation.prepare(scene.dataset, provider: SuppliedDepthProvider(maps: maps),
                                                trustMetric: false, settings: DepthPriorSettings(), seed: 0)
        #expect(prepared.report.views.count == scene.train.count)

        for viewIndex in 0..<scene.train.count {
            guard case .aligned(let alignment) = prepared.report.views[viewIndex] else {
                Issue.record("view \(viewIndex) expected to be aligned, got \(prepared.report.views[viewIndex])")
                continue
            }
            #expect(alignment.scale >= 0.9 && alignment.scale <= 1.1)
            #expect(alignment.shift == 0)

            let view = prepared.dataset.views[viewIndex]
            let raw: DepthMap = scene.trainDepth[viewIndex]
            let metric: [Float] = try #require(view.depth?.metricDepth)
            let width: Int = view.camera.width
            var validCount: Int = 0
            var onSurfaceCount: Int = 0
            for y in stride(from: 0, to: view.camera.height, by: 3) {
                for x in stride(from: 0, to: width, by: 3) {
                    let pixelIndex: Int = y * width + x
                    guard raw.isValid(pixelIndex) else {
                        continue
                    }
                    validCount += 1
                    let depth: Float = metric[pixelIndex]
                    guard depth > 0 else {
                        continue
                    }
                    let point: SIMD3<Float> = DepthGeometry.backProject(x: x, y: y, depth: depth, camera: view.camera)
                    let squared: Float = point.x * point.x + point.y * point.y + point.z * point.z
                    let radialError: Float = abs(squared.squareRoot() - 1)
                    let groundError: Float = abs(point.y + 1.05)
                    if radialError < 0.25 || groundError < 0.25 {
                        onSurfaceCount += 1
                    }
                }
            }
            #expect(validCount > 0)
            #expect(Double(onSurfaceCount) >= 0.85 * Double(validCount))
        }
    }
}

/// Samples of a known affine relation `target = scale·prior + shift`, where the target is the
/// camera-z depth for `.metric` and the inverse depth for `.relativeInverseDepth`. A deterministic
/// 40 % of the samples are consistent. The other 60 % are occluded: their camera z is `factor`
/// times the visible surface, with `factor` in 1.3...2.5, so their target is pushed in one direction.
private func occludedSamples(count: Int, kind: DepthKind, scale: Double, shift: Double) -> [(prior: Double, z: Double)] {
    var result: [(prior: Double, z: Double)] = []
    result.reserveCapacity(count)
    for index in 0..<count {
        let fraction: Double = Double(index) * 0.6180339887 - (Double(index) * 0.6180339887).rounded(.down)
        let prior: Double = 1.0 + 0.9 * fraction
        let target: Double = scale * prior + shift
        let surfaceDepth: Double = kind == .relativeInverseDepth ? 1.0 / target : target
        let factor: Double = 1.3 + 1.2 * (Double(index) * 0.7548776662 - (Double(index) * 0.7548776662).rounded(.down))
        let consistent: Bool = index % 5 < 2
        let depth: Double = consistent ? surfaceDepth : surfaceDepth * factor
        result.append((prior: prior, z: depth))
    }
    return result
}

/// A provider whose lookup always fails.
private struct FailingDepthProvider: DepthProvider {
    struct Failure: Error {}

    func depthMap(for view: TrainingView) throws -> DepthMap? {
        throw Failure()
    }
}
