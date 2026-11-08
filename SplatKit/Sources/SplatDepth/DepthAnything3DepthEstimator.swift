import CoreGraphics
import CoreML
import Foundation
import Synchronization
import SplatCore
import SplatIO

/// Multi-view depth from the Depth Anything 3 package of the pose-free pipeline (v1.5,
/// `DepthEstimatorKind.depthAnything3`).
///
/// Unlike the single-image estimators, Depth Anything 3 sees several views at once and predicts depth that
/// is consistent across them up to one shared scale, plus a per-pixel confidence. The estimator is created
/// for a fixed set of views (the training views of a run). The first `depthMap(for:)` call runs the model
/// on all of them and keeps the maps; later calls return the kept map.
///
/// - Views are run together in contiguous groups of at most 16 (`DepthAnything3Math.depthGroups`); each
///   group uses the package function with the smallest view count that fits (4, 6, 8, 12 or 16), padded by
///   repeating the last view, whose outputs are dropped. Groups share no views: every map is aligned to the
///   SfM points of its own view later (`DepthPreparation`), so the shared scale is not needed.
/// - Orientation follows the majority of each group; a view of the other orientation is stretched.
/// - Maps are `.relativeDepth` (depth, not inverse depth) at model resolution when the view has the
///   model's aspect ratio, else resampled to the view size. The confidence becomes the map's `weights`
///   (used by the loss only when `DepthPriorSettings.confidenceWeighting` is on).
/// - A view outside the set gets no map (`nil`).
public final class DepthAnything3DepthEstimator: DepthProvider, Sendable {
    public let packageURL: URL
    private let views: [TrainingView]
    private let models: DepthAnything3ModelCache
    /// Maps by view name once the batch has run, or the failure of that run.
    private let state: Mutex<Result<[String: DepthMap], MonocularDepthError>?> = Mutex(nil)

    public init(packageURL: URL, views: [TrainingView]) {
        self.packageURL = packageURL
        self.views = views
        self.models = DepthAnything3ModelCache()
    }

    /// Builds the estimator from the store when the Depth Anything 3 package is imported, else nil.
    public static func installed(in store: ModelStore, views: [TrainingView]) -> DepthAnything3DepthEstimator? {
        guard let url = store.importedPackageURL(for: ModelDescriptor.depthAnything3Base) else {
            return nil
        }
        return DepthAnything3DepthEstimator(packageURL: url, views: views)
    }

    public func depthMap(for view: TrainingView) throws -> DepthMap? {
        let outcome: Result<[String: DepthMap], MonocularDepthError> = state.withLock { stored in
            if let stored {
                return stored
            }
            let computed: Result<[String: DepthMap], MonocularDepthError>
            do throws(MonocularDepthError) {
                computed = .success(try runAll())
            } catch {
                computed = .failure(error)
            }
            stored = computed
            return computed
        }
        return try outcome.get()[view.name]
    }

    // MARK: - Inference

    private func runAll() throws(MonocularDepthError) -> [String: DepthMap] {
        guard FileManager.default.fileExists(atPath: packageURL.path) else {
            throw .modelNotFound(packageURL.path)
        }
        var maps: [String: DepthMap] = [:]
        for group in DepthAnything3Math.depthGroups(count: views.count) {
            let members: [TrainingView] = group.map { (index: Int) -> TrainingView in views[index] }
            let groupMaps: [DepthMap] = try run(members)
            for (view, map) in zip(members, groupMaps) {
                maps[view.name] = map
            }
        }
        return maps
    }

    /// One model call over `members` (1 to 16 views). Returns one map per member, in order.
    private func run(_ members: [TrainingView]) throws(MonocularDepthError) -> [DepthMap] {
        var portraitCount: Int = 0
        for view in members where view.image.height > view.image.width {
            portraitCount += 1
        }
        let portrait: Bool = portraitCount * 2 > members.count
        // A single view still needs a function for two or more; the padding repeats it.
        guard let function = DepthAnything3Math.function(forFrameCount: max(members.count, 2), portrait: portrait) else {
            throw .predictionFailed("no Depth Anything 3 function takes \(members.count) views")
        }

        var prepared: [DepthAnything3Inference.PreparedView] = []
        prepared.reserveCapacity(members.count)
        for view in members {
            let rgba: [UInt8] = try Self.resizedRGBA(view.image, width: function.width, height: function.height)
            prepared.append(DepthAnything3Inference.PreparedView(name: view.name,
                                                                 originalWidth: view.image.width,
                                                                 originalHeight: view.image.height,
                                                                 rgba: rgba))
        }

        let planeCount: Int = function.width * function.height
        let depth: [Float]
        let confidence: [Float]
        do throws(PoseEstimationFailure) {
            let model: MLModel = try models.model(packageURL: packageURL, functionName: function.name)
            let input: MLMultiArray = try DepthAnything3Inference.makeInput(views: prepared,
                                                                            viewCount: function.viewCount,
                                                                            height: function.height,
                                                                            width: function.width)
            let features: MLDictionaryFeatureProvider
            do {
                features = try MLDictionaryFeatureProvider(
                    dictionary: [DepthAnything3Inference.imageInputName: MLFeatureValue(multiArray: input)])
            } catch {
                throw PoseEstimationFailure.failed(reason: "cannot build the input: \(error.localizedDescription)")
            }
            let prediction: MLFeatureProvider
            do {
                prediction = try model.prediction(from: features)
            } catch {
                throw PoseEstimationFailure.failed(reason: "prediction failed: \(error.localizedDescription)")
            }
            depth = try DepthAnything3Inference.outputValues(prediction, name: DepthAnything3Inference.depthOutputName,
                                                             expectedCount: function.viewCount * planeCount)
            confidence = try DepthAnything3Inference.outputValues(prediction,
                                                                  name: DepthAnything3Inference.confidenceOutputName,
                                                                  expectedCount: function.viewCount * planeCount)
        } catch {
            throw .predictionFailed("Depth Anything 3: \(Self.describe(error))")
        }

        var maps: [DepthMap] = []
        maps.reserveCapacity(members.count)
        for (index, view) in members.enumerated() {
            let start: Int = index * planeCount
            let depthPlane: [Float] = Array(depth[start..<(start + planeCount)])
            let confidencePlane: [Float] = Array(confidence[start..<(start + planeCount)])
            let map: DepthMap = Self.depthMap(depth: depthPlane, confidence: confidencePlane,
                                              width: function.width, height: function.height)
            maps.append(try Self.fitted(map, toWidth: view.image.width, height: view.image.height))
        }
        return maps
    }

    // MARK: - Conversions (internal for tests)

    /// A `.relativeDepth` map from one depth and confidence plane. Depth that is not finite or ≤ 0 becomes
    /// 0 (invalid). The weights are the confidence, with non-finite or negative values set to 0.
    static func depthMap(depth: [Float], confidence: [Float], width: Int, height: Int) -> DepthMap {
        let count: Int = width * height
        precondition(depth.count == count && confidence.count == count, "planes must match the map size")
        var values: [Float] = [Float](repeating: 0, count: count)
        var weights: [Float] = [Float](repeating: 0, count: count)
        for index in 0..<count {
            let value: Float = depth[index]
            values[index] = (value.isFinite && value > 0) ? value : 0
            let weight: Float = confidence[index]
            weights[index] = (weight.isFinite && weight > 0) ? weight : 0
        }
        return DepthMap(width: width, height: height, kind: .relativeDepth, values: values,
                        confidence: nil, isEstimated: true, weights: weights)
    }

    /// The map unchanged when its aspect ratio matches the view's (alignment resamples it later), else
    /// resampled to the view size.
    static func fitted(_ map: DepthMap, toWidth width: Int, height: Int) throws(MonocularDepthError) -> DepthMap {
        guard width > 0, height > 0 else {
            throw .predictionFailed("the view has no pixels")
        }
        let mapAspect: Double = Double(map.width) / Double(map.height)
        let viewAspect: Double = Double(width) / Double(height)
        if abs(mapAspect - viewAspect) / viewAspect <= 0.01 {
            return map
        }
        do {
            return try map.resampled(width: width, height: height, allowAspectChange: true)
        } catch {
            throw .predictionFailed("cannot resample the Depth Anything 3 map to \(width)x\(height)")
        }
    }

    /// RGBA bytes (alpha ignored) of an sRGB image stretched to `width` × `height`, row 0 at the top.
    static func resizedRGBA(_ image: RGBImage, width: Int, height: Int) throws(MonocularDepthError) -> [UInt8] {
        guard image.width > 0, image.height > 0, image.pixels.count == image.width * image.height,
              width > 0, height > 0 else {
            throw .predictionFailed("the view image is empty or inconsistent")
        }
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw .predictionFailed("the sRGB colour space is unavailable")
        }
        let sourceBytesPerRow: Int = image.width * 4
        var sourceBytes: [UInt8] = [UInt8](repeating: 255, count: sourceBytesPerRow * image.height)
        for index in 0..<(image.width * image.height) {
            let pixel: SIMD3<Float> = image.pixels[index]
            sourceBytes[index * 4] = byte(pixel.x)
            sourceBytes[index * 4 + 1] = byte(pixel.y)
            sourceBytes[index * 4 + 2] = byte(pixel.z)
        }
        guard let provider = CGDataProvider(data: Data(sourceBytes) as CFData),
              let source = CGImage(width: image.width,
                                   height: image.height,
                                   bitsPerComponent: 8,
                                   bitsPerPixel: 32,
                                   bytesPerRow: sourceBytesPerRow,
                                   space: colorSpace,
                                   bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                   provider: provider,
                                   decode: nil,
                                   shouldInterpolate: false,
                                   intent: .defaultIntent) else {
            throw .predictionFailed("could not create the source image")
        }
        var output: [UInt8] = [UInt8](repeating: 0, count: width * height * 4)
        let drawn: Bool = output.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) -> Bool in
            guard let context = CGContext(data: raw.baseAddress,
                                          width: width,
                                          height: height,
                                          bitsPerComponent: 8,
                                          bytesPerRow: width * 4,
                                          space: colorSpace,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
                return false
            }
            context.interpolationQuality = .high
            context.draw(source, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else {
            throw .predictionFailed("could not resize the view image")
        }
        return output
    }

    private static func byte(_ value: Float) -> UInt8 {
        let clamped: Float = min(max(value, 0), 1)
        return UInt8((clamped * 255).rounded())
    }

    private static func describe(_ failure: PoseEstimationFailure) -> String {
        switch failure {
        case .unavailable(let reason), .failed(let reason):
            return reason
        case .tooFewFrames(let have, let need):
            return "too few views (\(have), need \(need))"
        case .registrationFailed(let registered, let total):
            return "registered \(registered) of \(total)"
        case .cancelled:
            return "cancelled"
        }
    }
}
