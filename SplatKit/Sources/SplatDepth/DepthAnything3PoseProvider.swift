import CoreGraphics
import CoreML
import Foundation
import ImageIO
import SplatCore
import SplatIO

/// Pose-free pose provider backed by Depth Anything 3 Base, converted to a multi-function Core ML package.
///
/// One forward pass over all frames gives camera poses, intrinsics and a depth map per view. The sparse
/// points are back-projected from those depth maps. More than 16 frames are sub-sampled to 16 evenly
/// spaced frames, and the others stay unregistered. Fewer frames than the chosen function's view count are
/// filled by repeating the last frame, and the repeats' outputs are dropped.
///
/// The package is not bundled. Its location is passed in, typically from `ModelStore.importedPackageURL`.
/// The Core ML run happens on a detached task, off the caller's actor.
public struct DepthAnything3PoseProvider: PoseProvider {
    public let name = "Depth Anything 3"

    /// The `.mlpackage` directory with one function per view count and orientation.
    public let packageURL: URL
    /// Upper bound on the returned points, spread evenly over the back-projected points.
    public let maximumPoints: Int

    private let models: DepthAnything3ModelCache

    public init(packageURL: URL, maximumPoints: Int = 20_000) {
        self.packageURL = packageURL
        self.maximumPoints = maximumPoints
        self.models = DepthAnything3ModelCache()
    }

    public func estimatePoses(for frames: [CapturedFrame]) async throws(PoseEstimationFailure) -> PoseEstimationResult {
        if frames.count < PoseEstimationPolicy.minimumFrames {
            throw .tooFewFrames(have: frames.count, need: PoseEstimationPolicy.minimumFrames)
        }
        guard FileManager.default.fileExists(atPath: packageURL.path) else {
            throw .unavailable(reason: "the Depth Anything 3 package is not installed at \(packageURL.path)")
        }

        let packageURL: URL = self.packageURL
        let maximumPoints: Int = self.maximumPoints
        let models: DepthAnything3ModelCache = self.models
        let work: Task<Result<PoseEstimationResult, PoseEstimationFailure>, Never> = Task.detached(priority: .userInitiated) {
            () -> Result<PoseEstimationResult, PoseEstimationFailure> in
            do throws(PoseEstimationFailure) {
                let result: PoseEstimationResult = try DepthAnything3Inference.estimate(frames: frames,
                                                                                        packageURL: packageURL,
                                                                                        maximumPoints: maximumPoints,
                                                                                        models: models)
                return .success(result)
            } catch {
                return .failure(error)
            }
        }
        let outcome: Result<PoseEstimationResult, PoseEstimationFailure> = await withTaskCancellationHandler {
            await work.value
        } onCancel: {
            work.cancel()
        }
        return try outcome.get()
    }
}

/// Loaded Core ML functions, keyed by function name. Copies of one provider share the cache.
final class DepthAnything3ModelCache: @unchecked Sendable {
    private let lock: NSLock = NSLock()
    // Guarded by `lock`.
    private var models: [String: MLModel] = [:]

    /// Returns the loaded function, loading it on first use. Loading goes through the compiled-model cache
    /// next to the package. Failures are reported as `.unavailable`.
    func model(packageURL: URL, functionName: String) throws(PoseEstimationFailure) -> MLModel {
        lock.lock()
        defer { lock.unlock() }
        if let cached: MLModel = models[functionName] {
            return cached
        }
        let configuration: MLModelConfiguration = MLModelConfiguration()
        configuration.computeUnits = .all
        configuration.functionName = functionName
        let loaded: MLModel
        do {
            loaded = try MonocularDepthEstimator.loadModel(packageURL: packageURL, configuration: configuration)
        } catch {
            let reason: String
            switch error {
            case .modelNotFound(let message), .unexpectedModelInterface(let message), .predictionFailed(let message):
                reason = message
            }
            throw .unavailable(reason: "cannot load the Depth Anything 3 function \(functionName): \(reason)")
        }
        models[functionName] = loaded
        return loaded
    }
}

/// Frame decoding, the Core ML run and the conversion of its outputs. Runs off the main actor.
enum DepthAnything3Inference {
    /// Input and output names of the converted package.
    static let imageInputName: String = "images"
    static let depthOutputName: String = "depth"
    static let confidenceOutputName: String = "confidence"
    static let poseOutputName: String = "pose"

    /// Largest number of views one run takes. More frames are sub-sampled to this many.
    static let maximumViews: Int = 16
    /// Every `pointStride`-th pixel of each depth map becomes a candidate point.
    static let pointStride: Int = 10
    /// ImageNet normalisation, per RGB channel.
    static let channelMean: [Float] = [0.485, 0.456, 0.406]
    static let channelStandardDeviation: [Float] = [0.229, 0.224, 0.225]

    /// One input view: its original upright size and its RGBA pixels resized to the model size, row 0 at the top.
    struct PreparedView {
        var name: String
        var originalWidth: Int
        var originalHeight: Int
        var rgba: [UInt8]
    }

    static func estimate(frames: [CapturedFrame],
                         packageURL: URL,
                         maximumPoints: Int,
                         models: DepthAnything3ModelCache) throws(PoseEstimationFailure) -> PoseEstimationResult {
        // Orientation is decided by the majority of the input frames, from their headers only.
        var portraitCount: Int = 0
        for frame in frames {
            if Task.isCancelled {
                throw .cancelled
            }
            let size: (width: Int, height: Int) = try uprightHeaderSize(of: frame)
            if size.height > size.width {
                portraitCount += 1
            }
        }
        let portrait: Bool = portraitCount * 2 > frames.count

        let keptIndices: [Int]
        if frames.count > maximumViews {
            keptIndices = DepthAnything3Math.evenlySpacedIndices(count: frames.count, keep: maximumViews)
        } else {
            keptIndices = Array(frames.indices)
        }
        guard let function = DepthAnything3Math.function(forFrameCount: keptIndices.count, portrait: portrait) else {
            throw .failed(reason: "no Depth Anything 3 function takes \(keptIndices.count) frames")
        }

        var views: [PreparedView] = []
        views.reserveCapacity(keptIndices.count)
        for index in keptIndices {
            if Task.isCancelled {
                throw .cancelled
            }
            views.append(try prepareView(frames[index], width: function.width, height: function.height))
        }

        let model: MLModel = try models.model(packageURL: packageURL, functionName: function.name)
        let input: MLMultiArray = try makeInput(views: views,
                                                viewCount: function.viewCount,
                                                height: function.height,
                                                width: function.width)
        if Task.isCancelled {
            throw .cancelled
        }

        let features: MLDictionaryFeatureProvider
        do {
            features = try MLDictionaryFeatureProvider(dictionary: [imageInputName: MLFeatureValue(multiArray: input)])
        } catch {
            throw .failed(reason: "cannot build the Depth Anything 3 input: \(error.localizedDescription)")
        }
        let prediction: MLFeatureProvider
        do {
            prediction = try model.prediction(from: features)
        } catch {
            throw .failed(reason: "Depth Anything 3 prediction failed: \(error.localizedDescription)")
        }

        let planeCount: Int = function.height * function.width
        let depth: [Float] = try outputValues(prediction, name: depthOutputName,
                                              expectedCount: function.viewCount * planeCount)
        let confidence: [Float] = try outputValues(prediction, name: confidenceOutputName,
                                                   expectedCount: function.viewCount * planeCount)
        let pose: [Float] = try outputValues(prediction, name: poseOutputName,
                                             expectedCount: function.viewCount * 9)
        if Task.isCancelled {
            throw .cancelled
        }

        return try buildResult(views: views,
                               height: function.height,
                               width: function.width,
                               depth: depth,
                               confidence: confidence,
                               pose: pose,
                               maximumPoints: maximumPoints)
    }

    // MARK: - Decoding

    /// Pixel size of the first image as it appears upright (EXIF orientations 5 to 8 swap the sides).
    static func uprightHeaderSize(of frame: CapturedFrame) throws(PoseEstimationFailure) -> (width: Int, height: Int) {
        guard let source = CGImageSourceCreateWithData(frame.data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
              let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
              width > 0, height > 0 else {
            throw .failed(reason: "cannot read the pixel size of \(frame.name)")
        }
        let orientation: Int = (properties[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
        if (5...8).contains(orientation) {
            return (width: height, height: width)
        }
        return (width: width, height: height)
    }

    /// Decodes a frame with its orientation applied and resizes it to `width` × `height` with high quality.
    /// The aspect ratio is not kept: a frame of the other orientation is stretched, not rotated.
    static func prepareView(_ frame: CapturedFrame, width: Int, height: Int) throws(PoseEstimationFailure) -> PreparedView {
        let header: (width: Int, height: Int) = try uprightHeaderSize(of: frame)
        guard let source = CGImageSourceCreateWithData(frame.data as CFData, nil) else {
            throw .failed(reason: "cannot decode \(frame.name)")
        }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(header.width, header.height),
        ]
        guard let oriented = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            throw .failed(reason: "cannot decode \(frame.name)")
        }

        let colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB()
        var rgba: [UInt8] = [UInt8](repeating: 0, count: width * height * 4)
        let drawn: Bool = rgba.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) -> Bool in
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
            context.draw(oriented, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else {
            throw .failed(reason: "cannot resize \(frame.name)")
        }
        return PreparedView(name: frame.name,
                            originalWidth: oriented.width,
                            originalHeight: oriented.height,
                            rgba: rgba)
    }

    // MARK: - Input and output tensors

    /// The `(1, N, 3, H, W)` float32 input. Views past `views.count` repeat the last view.
    static func makeInput(views: [PreparedView],
                          viewCount: Int,
                          height: Int,
                          width: Int) throws(PoseEstimationFailure) -> MLMultiArray {
        let shape: [NSNumber] = [1, NSNumber(value: viewCount), 3, NSNumber(value: height), NSNumber(value: width)]
        let array: MLMultiArray
        do {
            array = try MLMultiArray(shape: shape, dataType: .float32)
        } catch {
            throw .failed(reason: "cannot allocate the Depth Anything 3 input: \(error.localizedDescription)")
        }
        guard !views.isEmpty else {
            throw .failed(reason: "no frames to run Depth Anything 3 on")
        }
        array.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer, strides: [Int]) -> Void in
            for view in 0..<viewCount {
                let source: PreparedView = views[min(view, views.count - 1)]
                for row in 0..<height {
                    for column in 0..<width {
                        let pixel: Int = (row * width + column) * 4
                        for channel in 0..<3 {
                            let byte: UInt8 = source.rgba[pixel + channel]
                            let scaled: Float = Float(byte) / 255
                            let normalised: Float = (scaled - channelMean[channel]) / channelStandardDeviation[channel]
                            let element: Int = view * strides[1] + channel * strides[2] + row * strides[3] + column * strides[4]
                            raw.storeBytes(of: normalised, toByteOffset: element * 4, as: Float.self)
                        }
                    }
                }
            }
        }
        return array
    }

    /// Values of one output in row-major logical order, read through the array's strides and data type.
    static func outputValues(_ prediction: MLFeatureProvider,
                             name: String,
                             expectedCount: Int) throws(PoseEstimationFailure) -> [Float] {
        guard let array = prediction.featureValue(for: name)?.multiArrayValue else {
            throw .failed(reason: "the Depth Anything 3 model returned no \(name) output")
        }
        guard array.count == expectedCount else {
            throw .failed(reason: "the \(name) output has \(array.count) values, expected \(expectedCount)")
        }
        return try logicalFloats(array, name: name)
    }

    /// Copies a float16, float32 or float64 array into row-major `[Float]`, honouring its strides.
    static func logicalFloats(_ array: MLMultiArray, name: String) throws(PoseEstimationFailure) -> [Float] {
        let dataType: MLMultiArrayDataType = array.dataType
        guard dataType == .float16 || dataType == .float32 || dataType == .double else {
            throw .failed(reason: "the \(name) output has an unsupported data type")
        }
        let shape: [Int] = array.shape.map { (number: NSNumber) -> Int in number.intValue }
        let count: Int = shape.reduce(1, *)
        let rank: Int = shape.count
        var result: [Float] = [Float](repeating: 0, count: count)
        guard count > 0 else {
            return result
        }
        let strides: [Int] = array.strides.map { (number: NSNumber) -> Int in number.intValue }
        array.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Void in
            var index: [Int] = [Int](repeating: 0, count: rank)
            for flat in 0..<count {
                var element: Int = 0
                for dimension in 0..<rank {
                    element += index[dimension] * strides[dimension]
                }
                let value: Float
                switch dataType {
                case .float16:
                    value = Float(raw.loadUnaligned(fromByteOffset: element * 2, as: Float16.self))
                case .double:
                    value = Float(raw.loadUnaligned(fromByteOffset: element * 8, as: Double.self))
                default:
                    value = raw.loadUnaligned(fromByteOffset: element * 4, as: Float.self)
                }
                result[flat] = value
                // Advance the multi-dimensional index, last dimension fastest.
                var dimension: Int = rank - 1
                while dimension >= 0 {
                    index[dimension] += 1
                    if index[dimension] < shape[dimension] {
                        break
                    }
                    index[dimension] = 0
                    dimension -= 1
                }
            }
        }
        return result
    }

    // MARK: - Result

    /// Cameras, poses and points for the registered views (`views`, not the repeated fill views).
    static func buildResult(views: [PreparedView],
                            height: Int,
                            width: Int,
                            depth: [Float],
                            confidence: [Float],
                            pose: [Float],
                            maximumPoints: Int) throws(PoseEstimationFailure) -> PoseEstimationResult {
        let planeCount: Int = height * width
        var cameras: [EstimatedCamera] = []
        var images: [EstimatedImagePose] = []
        var points: [EstimatedPoint] = []

        for (view, prepared) in views.enumerated() {
            let poseStart: Int = view * 9
            var poseValues: [Double] = []
            poseValues.reserveCapacity(9)
            for component in 0..<9 {
                poseValues.append(Double(pose[poseStart + component]))
            }
            let converted: (rotation: [Double], translation: SIMD3<Double>, fovH: Double, fovW: Double)
                = DepthAnything3Math.worldToCamera(pose: poseValues)
            let intrinsics: (fx: Double, fy: Double, cx: Double, cy: Double)
                = DepthAnything3Math.intrinsics(fovH: converted.fovH, fovW: converted.fovW, width: width, height: height)

            // Intrinsics are at model resolution. Scale them to the frame's own pixels.
            let scaleX: Double = Double(prepared.originalWidth) / Double(width)
            let scaleY: Double = Double(prepared.originalHeight) / Double(height)
            cameras.append(EstimatedCamera(
                modelID: 1,
                width: prepared.originalWidth,
                height: prepared.originalHeight,
                params: [intrinsics.fx * scaleX, intrinsics.fy * scaleY, intrinsics.cx * scaleX, intrinsics.cy * scaleY]
            ))
            let rotation: SIMD4<Double> = DepthAnything3Math.quaternionWXYZ(fromRowMajor: converted.rotation)
            images.append(EstimatedImagePose(name: prepared.name,
                                             cameraIndex: cameras.count - 1,
                                             rotation: rotation,
                                             translation: converted.translation))

            let planeStart: Int = view * planeCount
            let depthPlane: [Float] = Array(depth[planeStart..<(planeStart + planeCount)])
            let confidencePlane: [Float] = Array(confidence[planeStart..<(planeStart + planeCount)])
            let unprojected: (points: [SIMD3<Double>], pixelIndices: [Int])
                = DepthAnything3Math.unproject(depth: depthPlane,
                                               confidence: confidencePlane,
                                               width: width,
                                               height: height,
                                               fx: intrinsics.fx,
                                               fy: intrinsics.fy,
                                               cx: intrinsics.cx,
                                               cy: intrinsics.cy,
                                               rotationW2C: converted.rotation,
                                               translationW2C: converted.translation,
                                               stride: pointStride,
                                               minConfidence: medianFinite(confidencePlane))
            for (position, pixel) in zip(unprojected.points, unprojected.pixelIndices) {
                let channel: Int = pixel * 4
                let color: SIMD3<UInt8> = SIMD3<UInt8>(prepared.rgba[channel], prepared.rgba[channel + 1], prepared.rgba[channel + 2])
                points.append(EstimatedPoint(position: position, color: color, error: 0))
            }
        }

        if points.count > maximumPoints {
            let indices: [Int] = DepthAnything3Math.evenlySpacedIndices(count: points.count, keep: max(maximumPoints, 0))
            points = indices.map { (index: Int) -> EstimatedPoint in points[index] }
        }
        images.sort { (a: EstimatedImagePose, b: EstimatedImagePose) -> Bool in a.name < b.name }
        return PoseEstimationResult(cameras: cameras, images: images, points: points)
    }

    /// Median of the finite values, or +infinity when there are none (so no point of that view is kept).
    static func medianFinite(_ values: [Float]) -> Float {
        let finite: [Float] = values.filter { (value: Float) -> Bool in value.isFinite }.sorted()
        guard !finite.isEmpty else {
            return Float.infinity
        }
        return finite[finite.count / 2]
    }
}
