import CoreML
import CoreVideo
import Foundation
import SplatCore
import SplatIO

/// Monocular depth from MoGe-2 ViT-S (Microsoft, MIT), as the unofficial Core ML conversion
/// `ebowwa/moge-2-vits-normal-coreml` (int8). The second selectable estimator next to Depth Anything V2 Small.
///
/// The model takes a fixed 512 × 768 (height × width) RGB multi-array in NCHW layout with values 0...1. The view is
/// stretched to that size with no letterboxing. Its camera-space `points` output carries an affine-invariant depth
/// in `z`, so the map is returned as `DepthKind.relativeDepth`: `DepthAlignment` fits the scale and shift to the
/// sparse points. Pixels with `mask` ≤ 0.5, or with a non-finite or non-positive z, are invalid (0).
///
/// Portrait views are rotated clockwise before inference and the depth is rotated back. Rotation cannot make a
/// portrait photo upright in the model's landscape frame; it keeps the stretch distortion small (about 1.1×
/// instead of 2× for a plain stretch). Measured behaviour on portrait photos is not yet known, see the report.
///
/// The output is resampled to the view's own aspect ratio here, not in `DepthPreparation`, which only accepts maps
/// within 1 % of the view's aspect. This is safe because the model input is a normalised stretch, so the normalised
/// image coordinates of the map still match those of the view.
///
/// `MLModel` is thread-safe for prediction, so one estimator can serve several threads.
public final class MoGeDepthEstimator: DepthProvider, @unchecked Sendable {
    private let model: MLModel
    private let inputName: String
    /// Model input size in pixels (768 × 512 for the shipped conversion, width × height).
    let inputWidth: Int
    let inputHeight: Int

    static let pointsOutputName: String = "points"
    static let maskOutputName: String = "mask"
    /// Number of channels the model takes (RGB).
    static let channelCount: Int = 3

    /// Loads the model from a `.mlpackage` directory, through the same compiled-model cache as Depth Anything.
    public init(packageURL: URL, computeUnits: MLComputeUnits = .all) throws(MonocularDepthError) {
        guard FileManager.default.fileExists(atPath: packageURL.path) else {
            throw .modelNotFound(packageURL.path)
        }
        let configuration: MLModelConfiguration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        model = try MonocularDepthEstimator.loadModel(packageURL: packageURL, configuration: configuration)

        let description: MLModelDescription = model.modelDescription
        let inputs: [String: MLFeatureDescription] = description.inputDescriptionsByName
        guard inputs.count == 1, let input = inputs.first else {
            throw .unexpectedModelInterface("expected one input, found \(inputs.count)")
        }
        guard input.value.type == .multiArray, let constraint = input.value.multiArrayConstraint else {
            throw .unexpectedModelInterface("input \(input.key) is not a multi-array")
        }
        let shape: [Int] = constraint.shape.map { (dimension: NSNumber) -> Int in dimension.intValue }
        guard shape.count == 4, shape[0] == 1, shape[1] == MoGeDepthEstimator.channelCount,
              shape[2] > 0, shape[3] > 0 else {
            throw .unexpectedModelInterface("input \(input.key) has shape \(shape), expected [1, 3, height, width]")
        }

        let outputs: [String: MLFeatureDescription] = description.outputDescriptionsByName
        for requiredName in [MoGeDepthEstimator.pointsOutputName, MoGeDepthEstimator.maskOutputName]
        where outputs[requiredName] == nil {
            throw .unexpectedModelInterface("output \(requiredName) is missing")
        }

        inputName = input.key
        inputHeight = shape[2]
        inputWidth = shape[3]
    }

    /// Builds the estimator from the store when the model is installed and verified.
    /// Returns nil when the model is missing, corrupt or not configured. Throws only when an installed model fails to load.
    public static func installed(in store: ModelStore,
                                 descriptor: ModelDescriptor) throws(MonocularDepthError) -> MoGeDepthEstimator? {
        guard case let .ready(packageURL) = store.status(for: descriptor) else {
            return nil
        }
        return try MoGeDepthEstimator(packageURL: packageURL)
    }

    /// Depth for one view, in the view's own orientation, resampled to the view's aspect ratio.
    public func depthMap(for view: TrainingView) throws -> DepthMap? {
        let image: RGBImage = view.image
        let isPortrait: Bool = image.height > image.width
        let oriented: RGBImage = isPortrait ? MonocularDepthImageSupport.rotatedClockwise(image) : image

        let input: CVPixelBuffer = try MonocularDepthImageSupport.inputBuffer(from: oriented,
                                                                              width: inputWidth,
                                                                              height: inputHeight)
        let array: MLMultiArray = try MoGeDepthConversion.inputArray(from: input,
                                                                     width: inputWidth,
                                                                     height: inputHeight)
        let features: MLDictionaryFeatureProvider
        do {
            features = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: array)])
        } catch {
            throw MonocularDepthError.predictionFailed("could not build the model input: \(error.localizedDescription)")
        }

        let result: MLFeatureProvider
        do {
            result = try model.prediction(from: features)
        } catch {
            throw MonocularDepthError.predictionFailed(error.localizedDescription)
        }
        guard let points = result.featureValue(for: MoGeDepthEstimator.pointsOutputName)?.multiArrayValue else {
            throw MonocularDepthError.predictionFailed("the model returned no \(MoGeDepthEstimator.pointsOutputName)")
        }
        guard let mask = result.featureValue(for: MoGeDepthEstimator.maskOutputName)?.multiArrayValue else {
            throw MonocularDepthError.predictionFailed("the model returned no \(MoGeDepthEstimator.maskOutputName)")
        }

        let pointValues: [Float] = try MoGeDepthConversion.floats(from: points)
        let maskValues: [Float] = try MoGeDepthConversion.floats(from: mask)
        let pixelCount: Int = inputWidth * inputHeight
        guard pointValues.count == pixelCount * MoGeDepthEstimator.channelCount, maskValues.count == pixelCount else {
            throw MonocularDepthError.unexpectedModelInterface(
                "outputs have \(pointValues.count) point values and \(maskValues.count) mask values, expected \(pixelCount * 3) and \(pixelCount)")
        }
        let depthValues: [Float] = MoGeDepthConversion.zChannel(points: pointValues, pixelCount: pixelCount)

        var map: DepthMap = try MoGeDepthConversion.depthMap(zValues: depthValues,
                                                             maskValues: maskValues,
                                                             width: inputWidth,
                                                             height: inputHeight)
        if isPortrait {
            map = map.rotatedCounterClockwise()
        }
        let target: (width: Int, height: Int) = MoGeDepthEstimator.outputSize(viewWidth: view.camera.width,
                                                                              viewHeight: view.camera.height,
                                                                              longSide: max(inputWidth, inputHeight))
        do {
            map = try map.resampled(width: target.width, height: target.height, allowAspectChange: true)
        } catch {
            throw MonocularDepthError.predictionFailed("could not resample the depth map: \(error)")
        }
        return map
    }

    /// Size of the output map: the model's long side kept, the short side set by the view's aspect ratio.
    static func outputSize(viewWidth: Int, viewHeight: Int, longSide: Int) -> (width: Int, height: Int) {
        guard viewWidth > 0, viewHeight > 0, longSide > 0 else {
            return (longSide, longSide)
        }
        if viewWidth >= viewHeight {
            let height: Double = (Double(longSide) * Double(viewHeight) / Double(viewWidth)).rounded()
            return (longSide, max(1, Int(height)))
        }
        let width: Double = (Double(longSide) * Double(viewWidth) / Double(viewHeight)).rounded()
        return (max(1, Int(width)), longSide)
    }
}

/// Pure conversions of the MoGe inputs and outputs. Internal, so the tests check them without a model.
enum MoGeDepthConversion {
    /// Reads the BGRA buffer of `MonocularDepthImageSupport.inputBuffer` into an NCHW float32 multi-array of
    /// shape [1, 3, height, width] with values in 0...1 (R, G, B order).
    static func inputArray(from buffer: CVPixelBuffer, width: Int, height: Int) throws(MonocularDepthError) -> MLMultiArray {
        guard CVPixelBufferGetWidth(buffer) == width, CVPixelBufferGetHeight(buffer) == height else {
            throw .predictionFailed("the input buffer does not have the model size")
        }
        let shape: [NSNumber] = [1, NSNumber(value: MoGeDepthEstimator.channelCount),
                                 NSNumber(value: height), NSNumber(value: width)]
        let array: MLMultiArray
        do {
            array = try MLMultiArray(shape: shape, dataType: .float32)
        } catch {
            throw .predictionFailed("could not allocate the model input: \(error.localizedDescription)")
        }
        let strides: [Int] = array.strides.map { (stride: NSNumber) -> Int in stride.intValue }
        let destination: UnsafeMutablePointer<Float> = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base: UnsafeRawPointer = CVPixelBufferGetBaseAddress(buffer).map({ UnsafeRawPointer($0) }) else {
            throw .predictionFailed("the input buffer has no memory")
        }
        let bytesPerRow: Int = CVPixelBufferGetBytesPerRow(buffer)
        for row in 0..<height {
            for column in 0..<width {
                let offset: Int = row * bytesPerRow + column * 4
                // 32BGRA: blue, green, red, unused.
                let blue: Float = Float(base.load(fromByteOffset: offset, as: UInt8.self)) / 255
                let green: Float = Float(base.load(fromByteOffset: offset + 1, as: UInt8.self)) / 255
                let red: Float = Float(base.load(fromByteOffset: offset + 2, as: UInt8.self)) / 255
                let spatialOffset: Int = row * strides[2] + column * strides[3]
                destination[spatialOffset] = red
                destination[strides[1] + spatialOffset] = green
                destination[2 * strides[1] + spatialOffset] = blue
            }
        }
        return array
    }

    /// The values of a multi-array in row-major logical order, honouring its strides. Supports float32, float16,
    /// double and int32 arrays.
    static func floats(from array: MLMultiArray) throws(MonocularDepthError) -> [Float] {
        let shape: [Int] = array.shape.map { (dimension: NSNumber) -> Int in dimension.intValue }
        let strides: [Int] = array.strides.map { (stride: NSNumber) -> Int in stride.intValue }
        guard shape.count == strides.count, !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else {
            throw .unexpectedModelInterface("output has shape \(shape) and strides \(strides)")
        }
        let count: Int = shape.reduce(1, *)
        let pointer: UnsafeMutableRawPointer = array.dataPointer
        var output: [Float] = [Float](repeating: 0, count: count)
        var index: [Int] = [Int](repeating: 0, count: shape.count)
        for linear in 0..<count {
            var offset: Int = 0
            for axis in 0..<shape.count {
                offset += index[axis] * strides[axis]
            }
            output[linear] = try element(pointer, offset: offset, type: array.dataType)
            // Advance the logical index, last axis first.
            var axis: Int = shape.count - 1
            while axis >= 0 {
                index[axis] += 1
                if index[axis] < shape[axis] {
                    break
                }
                index[axis] = 0
                axis -= 1
            }
        }
        return output
    }

    private static func element(_ pointer: UnsafeMutableRawPointer,
                                offset: Int,
                                type: MLMultiArrayDataType) throws(MonocularDepthError) -> Float {
        switch type {
        case .float32:
            return pointer.load(fromByteOffset: offset * 4, as: Float.self)
        case .float16:
            let bits: UInt16 = pointer.load(fromByteOffset: offset * 2, as: UInt16.self)
            return Float(Float16(bitPattern: bits))
        case .double:
            return Float(pointer.load(fromByteOffset: offset * 8, as: Double.self))
        case .int32:
            return Float(pointer.load(fromByteOffset: offset * 4, as: Int32.self))
        case .int8:
            return Float(pointer.load(fromByteOffset: offset, as: Int8.self))
        @unknown default:
            throw .unexpectedModelInterface("unsupported output data type \(type.rawValue)")
        }
    }

    /// The z component of row-major (x, y, z) triples: `pixelCount` values from `points`.
    static func zChannel(points: [Float], pixelCount: Int) -> [Float] {
        var z: [Float] = [Float](repeating: 0, count: pixelCount)
        for index in 0..<pixelCount {
            z[index] = points[index * MoGeDepthEstimator.channelCount + 2]
        }
        return z
    }

    /// Depth map of `width` × `height` from the z values and mask probabilities, both row-major. A pixel is valid
    /// when its mask is above 0.5 and its z is finite and positive. Invalid pixels are 0.
    static func depthMap(zValues: [Float], maskValues: [Float], width: Int, height: Int) throws(MonocularDepthError) -> DepthMap {
        let count: Int = width * height
        guard zValues.count == count, maskValues.count == count else {
            throw .unexpectedModelInterface("expected \(count) z and mask values, found \(zValues.count) and \(maskValues.count)")
        }
        var values: [Float] = [Float](repeating: 0, count: count)
        for index in 0..<count {
            let z: Float = zValues[index]
            let isValid: Bool = maskValues[index] > 0.5 && z.isFinite && z > 0
            values[index] = isValid ? z : 0
        }
        return DepthMap(width: width, height: height, kind: .relativeDepth, values: values, isEstimated: true)
    }
}
