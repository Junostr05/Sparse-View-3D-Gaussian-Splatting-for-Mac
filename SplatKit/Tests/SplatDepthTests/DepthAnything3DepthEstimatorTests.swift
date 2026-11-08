import Foundation
import SplatCore
@testable import SplatDepth
import SplatIO
import Testing

/// Converted Depth Anything 3 package for the opt-in Core ML smoke test. Unset means the test is skipped.
private let depthAnything3Package: URL? = ProcessInfo.processInfo.environment["DA3_PACKAGE"]
    .map { (path: String) -> URL in URL(fileURLWithPath: path, isDirectory: true) }

private func gradientView(name: String, width: Int, height: Int, shift: Int) -> TrainingView {
    var pixels: [SIMD3<Float>] = []
    pixels.reserveCapacity(width * height)
    for row in 0..<height {
        for column in 0..<width {
            let red: Float = Float(column) / Float(max(1, width - 1))
            let green: Float = Float(row) / Float(max(1, height - 1))
            let blue: Float = Float((column + row + 37 * shift) % 64) / 63
            pixels.append(SIMD3<Float>(red, green, blue))
        }
    }
    let camera = Camera(width: width, height: height, fx: Float(width), fy: Float(width),
                        cx: Float(width) / 2, cy: Float(height) / 2,
                        rotation: Matrix3.identity, translation: SIMD3<Float>(Float(shift) * 0.1, 0, 0))
    return TrainingView(name: name, camera: camera, image: RGBImage(width: width, height: height, pixels: pixels))
}

private func temporaryStoreRoot(_ label: String) throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("da3-depth-\(label)-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

@Suite struct DepthAnything3DepthEstimatorTests {

    @Test func planesBecomeARelativeDepthMapWithConfidenceWeights() {
        let depth: [Float] = [1, 2, .nan, -1, 0, 3]
        let confidence: [Float] = [1, 6.5, 2, .infinity, -2, 11]
        let map = DepthAnything3DepthEstimator.depthMap(depth: depth, confidence: confidence, width: 3, height: 2)
        #expect(map.kind == .relativeDepth)
        #expect(map.isEstimated)
        #expect(map.values == [1, 2, 0, 0, 0, 3])
        #expect(map.weights == [1, 6.5, 2, 0, 0, 11])
    }

    @Test func mapsKeepModelResolutionAtTheViewAspectAndAreResampledOtherwise() throws {
        let values: [Float] = [Float](repeating: 2, count: 12)
        let map = DepthMap(width: 3, height: 4, kind: .relativeDepth, values: values, isEstimated: true,
                           weights: [Float](repeating: 5, count: 12))
        // 3:4 view (450 × 600): kept at model resolution, alignment resamples later.
        let kept = try DepthAnything3DepthEstimator.fitted(map, toWidth: 450, height: 600)
        #expect(kept == map)
        // Square view: stretched to the view size, weights carried along.
        let stretched = try DepthAnything3DepthEstimator.fitted(map, toWidth: 8, height: 8)
        #expect(stretched.width == 8 && stretched.height == 8)
        #expect(stretched.weights?.count == 64)
        #expect(stretched.values.allSatisfy { $0 == 2 })
    }

    @Test func resizedImageHasTheModelSizeAndKeepsColours() throws {
        let pixels: [SIMD3<Float>] = [SIMD3<Float>](repeating: SIMD3<Float>(1, 0, 0), count: 16)
        let rgba: [UInt8] = try DepthAnything3DepthEstimator.resizedRGBA(RGBImage(width: 4, height: 4, pixels: pixels),
                                                                         width: 6, height: 8)
        #expect(rgba.count == 6 * 8 * 4)
        #expect(rgba[0] == 255 && rgba[1] == 0 && rgba[2] == 0)
    }

    @Test func missingPackageFailsPerViewWithoutCrashing() {
        let views = [gradientView(name: "a", width: 12, height: 16, shift: 0),
                     gradientView(name: "b", width: 12, height: 16, shift: 1)]
        let estimator = DepthAnything3DepthEstimator(packageURL: URL(fileURLWithPath: "/nonexistent/DA3.mlpackage"),
                                                     views: views)
        #expect(throws: MonocularDepthError.self) {
            _ = try estimator.depthMap(for: views[0])
        }
        // Through the preparation path the failure is a per-view rejection, never thrown.
        let prepared = DepthPreparation.prepare(TrainingDataset(views: views), provider: estimator, trustMetric: false,
                                                settings: DepthPriorSettings(), seed: 1)
        for status in prepared.report.views {
            guard case .rejected = status else {
                Issue.record("expected a rejected view, got \(status)")
                continue
            }
        }
    }

    @Test func factoryReturnsNilWithoutAnImportedPackage() throws {
        let root: URL = try temporaryStoreRoot("absent")
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ModelStore(rootDirectory: root)
        let provider: (any DepthProvider)? = try DepthEstimatorFactory.installed(.depthAnything3, in: store, views: [])
        #expect(provider == nil)
        #expect(DepthEstimatorKind.depthAnything3.descriptor == ModelDescriptor.depthAnything3Base)
        #expect(DepthEstimatorKind.depthAnything3.needsImportedPackage)
        #expect(!DepthEstimatorKind.depthAnythingV2Small.needsImportedPackage)
    }

    /// The map's weights reach the prior only when confidence weighting is on; off is the unweighted loss.
    @Test func preparationUsesWeightsOnlyWhenConfidenceWeightingIsOn() throws {
        let view = gradientView(name: "v", width: 8, height: 6, shift: 0)
        var values: [Float] = []
        var weights: [Float] = []
        for index in 0..<48 {
            values.append(1 + Float(index) * 0.01)
            weights.append(index % 2 == 0 ? 1 : 3)
        }
        let map = DepthMap(width: 8, height: 6, kind: .relativeDepth, values: values, isEstimated: true,
                           weights: weights)
        let provider = SuppliedDepthProvider(maps: ["v": map])
        let dataset = TrainingDataset(views: [view])

        let off = DepthPreparation.prepare(dataset, provider: provider, trustMetric: false,
                                           settings: DepthPriorSettings(), seed: 3)
        let offPrior = try #require(off.dataset.views[0].depth)
        #expect(offPrior.weights == nil)

        var settings = DepthPriorSettings()
        settings.confidenceWeighting = true
        let on = DepthPreparation.prepare(dataset, provider: provider, trustMetric: false, settings: settings, seed: 3)
        let onPrior = try #require(on.dataset.views[0].depth)
        let onWeights = try #require(onPrior.weights)
        #expect(abs(onWeights[0] - 0.5) < 1e-6)
        #expect(abs(onWeights[1] - 1.5) < 1e-6)
    }

    @Test(.enabled(if: depthAnything3Package != nil, "set DA3_PACKAGE to a converted Depth Anything 3 package"))
    func smokeRunGivesConsistentMapsForEveryView() throws {
        let packageURL: URL = try #require(depthAnything3Package)
        var views: [TrainingView] = []
        for index in 0..<5 {
            views.append(gradientView(name: "view\(index)", width: 90, height: 120, shift: index))
        }
        let estimator = DepthAnything3DepthEstimator(packageURL: packageURL, views: views)
        for view in views {
            let map = try #require(try estimator.depthMap(for: view))
            #expect(map.kind == .relativeDepth)
            #expect(map.width == 378 && map.height == 504)
            let valid: Int = map.values.filter { $0.isFinite && $0 > 0 }.count
            #expect(valid > map.values.count / 2)
            let weights = try #require(map.weights)
            #expect(weights.allSatisfy { $0 >= 0 && $0.isFinite })
        }
        let outsider = gradientView(name: "outsider", width: 90, height: 120, shift: 9)
        #expect(try estimator.depthMap(for: outsider) == nil)
    }
}
