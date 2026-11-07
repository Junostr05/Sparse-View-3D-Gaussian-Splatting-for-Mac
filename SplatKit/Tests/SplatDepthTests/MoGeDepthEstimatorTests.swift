import CoreML
import CoreVideo
import Foundation
import ImageIO
import SplatCore
@testable import SplatDepth
import SplatIO
import Testing

// MARK: - Helpers

/// A depth function of the camera pixel (x, y): slanted plane between 3 and 4.9 scene units.
private func planeDepth(_ x: Int, _ y: Int) -> Float {
    3 + 0.02 * Float(x) + 0.01 * Float(y)
}

private func temporaryDirectory(_ name: String) throws -> URL {
    let url: URL = FileManager.default.temporaryDirectory
        .appendingPathComponent("MoGeDepthEstimatorTests-\(name)-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

/// Builds a 2-D multi-array with explicit strides over a Float buffer, to check that the reader honours them.
private func stridedFloatArray(shape: [Int], strides: [Int], values: [(offset: Int, value: Float)]) throws -> MLMultiArray {
    let capacity: Int = (values.map { (entry: (offset: Int, value: Float)) -> Int in entry.offset }.max() ?? 0) + 1
    let buffer: UnsafeMutableRawPointer = UnsafeMutableRawPointer.allocate(byteCount: capacity * 4, alignment: 8)
    let typed: UnsafeMutablePointer<Float> = buffer.bindMemory(to: Float.self, capacity: capacity)
    typed.initialize(repeating: -1, count: capacity)
    for entry in values {
        typed[entry.offset] = entry.value
    }
    return try MLMultiArray(dataPointer: buffer,
                            shape: shape.map { NSNumber(value: $0) },
                            dataType: .float32,
                            strides: strides.map { NSNumber(value: $0) },
                            deallocator: { (pointer: UnsafeMutableRawPointer) in pointer.deallocate() })
}

/// A small RGB image whose pixel (x, y) is (x / w, y / h, 0.25), so every position is distinguishable.
private func gradientImage(width: Int, height: Int) -> RGBImage {
    var pixels: [SIMD3<Float>] = []
    pixels.reserveCapacity(width * height)
    for row in 0..<height {
        for column in 0..<width {
            let red: Float = Float(column) / Float(width)
            let green: Float = Float(row) / Float(height)
            let pixel: SIMD3<Float> = SIMD3<Float>(red, green, 0.25)
            pixels.append(pixel)
        }
    }
    return RGBImage(width: width, height: height, pixels: pixels)
}

extension Camera {
    /// A camera with an identity pose and a centred principal point. Only the image size matters here.
    static func centred(width: Int, height: Int) -> Camera {
        Camera(width: width, height: height, fx: 60, fy: 60, cx: Float(width) / 2, cy: Float(height) / 2,
               rotation: Matrix3.identity, translation: SIMD3<Float>(0, 0, 0))
    }
}

// MARK: - Model-free tests

@Test func outputSizeKeepsTheLongSideAndTheViewAspect() {
    let landscape: (width: Int, height: Int) = MoGeDepthEstimator.outputSize(viewWidth: 4032, viewHeight: 3024, longSide: 768)
    #expect(landscape.width == 768)
    #expect(landscape.height == 576)
    let portrait: (width: Int, height: Int) = MoGeDepthEstimator.outputSize(viewWidth: 3024, viewHeight: 4032, longSide: 768)
    #expect(portrait.width == 576)
    #expect(portrait.height == 768)
    let square: (width: Int, height: Int) = MoGeDepthEstimator.outputSize(viewWidth: 500, viewHeight: 500, longSide: 768)
    #expect(square.width == 768)
    #expect(square.height == 768)
}

@Test func zChannelTakesTheThirdComponentOfEachPoint() {
    let points: [Float] = [1, 2, 3, 4, 5, 6]
    #expect(MoGeDepthConversion.zChannel(points: points, pixelCount: 2) == [3, 6])
}

@Test func depthMapIsValidOnlyWhereMaskIsConfidentAndDepthPositive() throws {
    // Mask must be above 0.5 (0.5 itself is invalid); z must be finite and positive.
    let z: [Float] = [2.0, 3.0, -1.0, Float.nan, 4.0, 5.0]
    let mask: [Float] = [0.9, 0.4, 0.9, 0.9, 0.5, 1.0]
    let map: DepthMap = try MoGeDepthConversion.depthMap(zValues: z, maskValues: mask, width: 3, height: 2)
    #expect(map.values == [2.0, 0, 0, 0, 0, 5.0])
    #expect(map.width == 3)
    #expect(map.height == 2)
    #expect(map.kind == .relativeDepth)
    #expect(map.isEstimated)
    #expect(map.confidence == nil)
}

@Test func depthMapRejectsMismatchedOutputCounts() {
    #expect(throws: MonocularDepthError.self) {
        _ = try MoGeDepthConversion.depthMap(zValues: [1], maskValues: [1, 1], width: 1, height: 1)
    }
}

@Test func multiArrayValuesFollowTheirStrides() throws {
    // Shape [2, 3] laid out column-major: element (r, c) lives at offset r + 2c.
    let array: MLMultiArray = try stridedFloatArray(shape: [2, 3], strides: [1, 2], values: [
        (offset: 0, value: 0), (offset: 1, value: 10),
        (offset: 2, value: 1), (offset: 3, value: 11),
        (offset: 4, value: 2), (offset: 5, value: 12),
    ])
    let values: [Float] = try MoGeDepthConversion.floats(from: array)
    #expect(values == [0, 1, 2, 10, 11, 12])
}

@Test func multiArrayReadsFloat16Values() throws {
    let array: MLMultiArray = try MLMultiArray(shape: [3], dataType: .float16)
    let bits: UnsafeMutablePointer<UInt16> = array.dataPointer.bindMemory(to: UInt16.self, capacity: 3)
    bits[0] = Float16(0.5).bitPattern
    bits[1] = Float16(2).bitPattern
    bits[2] = Float16(-1).bitPattern
    let values: [Float] = try MoGeDepthConversion.floats(from: array)
    #expect(values == [0.5, 2, -1])
}

@Test func inputArrayIsNCHWWithRGBInZeroToOne() throws {
    // 2x2 exact colours. Channel planes in row-major order: R = [1, 0, 0, 0.5], G = [0, 1, 0, 0.5], B = [0, 0, 1, 0.5].
    let pixels: [SIMD3<Float>] = [
        SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0),
        SIMD3<Float>(0, 0, 1), SIMD3<Float>(0.5, 0.5, 0.5),
    ]
    let image: RGBImage = RGBImage(width: 2, height: 2, pixels: pixels)
    let buffer: CVPixelBuffer = try MonocularDepthImageSupport.inputBuffer(from: image, width: 2, height: 2)
    let array: MLMultiArray = try MoGeDepthConversion.inputArray(from: buffer, width: 2, height: 2)
    #expect(array.shape.map { $0.intValue } == [1, 3, 2, 2])
    let values: [Float] = try MoGeDepthConversion.floats(from: array)
    let expected: [Float] = [1, 0, 0, 0.5, 0, 1, 0, 0.5, 0, 0, 1, 0.5]
    #expect(values.count == expected.count)
    for index in 0..<expected.count {
        #expect(abs(values[index] - expected[index]) <= 0.01)
    }
}

@Test func absentModelReturnsNilThroughTheFactory() throws {
    let root: URL = try temporaryDirectory("absent")
    defer { try? FileManager.default.removeItem(at: root) }
    let store: ModelStore = ModelStore(rootDirectory: root)
    let provider: (any DepthProvider)? = try DepthEstimatorFactory.installed(.moge2Small, in: store)
    #expect(provider == nil)
    let depthAnything: (any DepthProvider)? = try DepthEstimatorFactory.installed(.depthAnythingV2Small, in: store)
    #expect(depthAnything == nil)
}

@Test func estimatorKindsMapToTheirCatalogueEntries() {
    #expect(DepthEstimatorKind.depthAnythingV2Small.descriptor == ModelCatalog.depthAnythingV2Small)
    #expect(DepthEstimatorKind.moge2Small.descriptor == ModelCatalog.moge2SmallInt8)
}

/// A MoGe-style map at the view's 4:3 aspect (128 × 96 for a 64 × 48 view) holds an affine relative depth
/// `z = 0.4 p + 1.0`. The relative-depth fit must recover that relation from the sparse points.
@Test func moGeStyleRelativeDepthAlignsToSparsePoints() throws {
    let camera: Camera = Camera.centred(width: 64, height: 48)
    var values: [Float] = []
    values.reserveCapacity(128 * 96)
    for row in 0..<96 {
        for column in 0..<128 {
            // Map pixel (column, row) is camera pixel (column / 2, row / 2) at this resolution.
            let z: Float = planeDepth(column / 2, row / 2)
            values.append((z - 1.0) / 0.4)
        }
    }
    let map: DepthMap = DepthMap(width: 128, height: 96, kind: .relativeDepth, values: values, isEstimated: true)

    var points: [SIMD3<Float>] = []
    var colours: [SIMD3<Float>] = []
    for y in stride(from: 0, to: 48, by: 3) {
        for x in stride(from: 0, to: 64, by: 4) {
            points.append(DepthGeometry.backProject(x: x, y: y, depth: planeDepth(x, y), camera: camera))
            colours.append(SIMD3<Float>(0.5, 0.5, 0.5))
        }
    }
    let view: TrainingView = TrainingView(name: "view0", camera: camera,
                                          image: RGBImage(width: 64, height: 48,
                                                          pixels: [SIMD3<Float>](repeating: .zero, count: 64 * 48)))
    let dataset: TrainingDataset = TrainingDataset(views: [view],
                                                   points: PointCloud(positions: points, colors: colours))
    let prepared = DepthPreparation.prepare(dataset,
                                            provider: SuppliedDepthProvider(maps: ["view0": map]),
                                            trustMetric: false, settings: DepthPriorSettings(), seed: 3)
    let status: ViewDepthStatus = try #require(prepared.report.views.first)
    guard case .aligned(let alignment) = status else {
        Issue.record("expected an aligned view, got \(status)")
        return
    }
    #expect(alignment.fittedKind == .relativeDepth)
    #expect(!alignment.kindCorrected)
    #expect(abs(alignment.scale - 0.4) < 1e-3)
    #expect(abs(alignment.shift - 1.0) < 1e-3)
    let prior: DepthPrior = try #require(prepared.dataset.views[0].depth)
    #expect(prior.domain == .depth)
}

/// The model's own 512 × 768 output is not within 1 % of a 4:3 view, so `DepthPreparation` would reject it.
/// `MoGeDepthEstimator` therefore resamples to the view aspect before returning. This test pins the rejection.
@Test func rawModelSizeMapIsRejectedForAFourByThreeView() throws {
    let camera: Camera = Camera.centred(width: 64, height: 48)
    let map: DepthMap = DepthMap(width: 768, height: 512, kind: .relativeDepth,
                                 values: [Float](repeating: 2, count: 768 * 512), isEstimated: true)
    let view: TrainingView = TrainingView(name: "view0", camera: camera,
                                          image: RGBImage(width: 64, height: 48,
                                                          pixels: [SIMD3<Float>](repeating: .zero, count: 64 * 48)))
    let dataset: TrainingDataset = TrainingDataset(views: [view], points: nil)
    let prepared = DepthPreparation.prepare(dataset,
                                            provider: SuppliedDepthProvider(maps: ["view0": map]),
                                            trustMetric: false, settings: DepthPriorSettings(), seed: 1)
    let status: ViewDepthStatus = try #require(prepared.report.views.first)
    guard case .rejected = status else {
        Issue.record("expected a rejected view, got \(status)")
        return
    }
}

// MARK: - Model-dependent tests (skipped unless SPLAT_MOGE_MODEL names the .mlpackage)

/// The `.mlpackage` named by `SPLAT_MOGE_MODEL` when it exists, or nil.
private func configuredMoGeModelURL() -> URL? {
    guard let path = ProcessInfo.processInfo.environment["SPLAT_MOGE_MODEL"], !path.isEmpty,
          FileManager.default.fileExists(atPath: path) else {
        return nil
    }
    return URL(fileURLWithPath: path)
}

/// Copies the package into a fresh temporary directory, so Core ML caches its compiled model next to the copy and
/// the configured package stays untouched.
private func copyOfMoGeModel(_ source: URL) throws -> (directory: URL, package: URL) {
    let directory: URL = try temporaryDirectory("model")
    let package: URL = directory.appendingPathComponent(source.lastPathComponent, isDirectory: true)
    try FileManager.default.copyItem(at: source, to: package)
    return (directory, package)
}

@Test(.enabled(if: configuredMoGeModelURL() != nil,
               "Set SPLAT_MOGE_MODEL to moge_vits_normal_int8.mlpackage to run the MoGe model tests"))
func mogeModelRunsOnLandscapeAndPortraitViews() throws {
    let source: URL = try #require(configuredMoGeModelURL())
    let copy = try copyOfMoGeModel(source)
    defer { try? FileManager.default.removeItem(at: copy.directory) }
    let estimator: MoGeDepthEstimator = try MoGeDepthEstimator(packageURL: copy.package)
    #expect(estimator.inputWidth == 768)
    #expect(estimator.inputHeight == 512)

    // Landscape view, 4:3: the map keeps the 768 long side and the view aspect.
    let landscapeView: TrainingView = TrainingView(name: "landscape",
                                                   camera: Camera.centred(width: 800, height: 600),
                                                   image: gradientImage(width: 800, height: 600))
    let landscape: DepthMap = try #require(try estimator.depthMap(for: landscapeView))
    #expect(landscape.width == 768)
    #expect(landscape.height == 576)
    #expect(landscape.kind == .relativeDepth)
    #expect(landscape.isEstimated)
    #expect(landscape.values.count == 768 * 576)

    // Portrait view: rotated for the model and rotated back, so the map is portrait with the view's aspect.
    let portraitView: TrainingView = TrainingView(name: "portrait",
                                                  camera: Camera.centred(width: 600, height: 800),
                                                  image: gradientImage(width: 600, height: 800))
    let portrait: DepthMap = try #require(try estimator.depthMap(for: portraitView))
    #expect(portrait.width == 576)
    #expect(portrait.height == 768)
    #expect(portrait.values.count == 576 * 768)

    for map in [landscape, portrait] {
        #expect(map.values.allSatisfy { $0.isFinite && $0 >= 0 })
        let validCount: Int = (0..<map.values.count).filter { map.isValid($0) }.count
        #expect(validCount > 0)
    }
}
