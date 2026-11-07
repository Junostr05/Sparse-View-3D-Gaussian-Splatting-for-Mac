import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import SplatCore
import SplatIO

/// Why the monocular depth model could not be loaded or run.
public enum MonocularDepthError: Error, Equatable {
    /// The model package is absent, or Core ML could not compile or load it.
    case modelNotFound(String)
    /// The model does not have the one image input and one image output this estimator expects.
    case unexpectedModelInterface(String)
    /// A prediction or a buffer conversion failed.
    case predictionFailed(String)
}

/// Monocular depth from Apple's Core ML conversion of Depth Anything V2 Small (Phase 4 Task 20).
///
/// The output is relative inverse depth (normalised disparity, bright = near) at the model's output
/// size. Resampling to the view size happens later in `DepthPreparation`. The model is not bundled:
/// it is downloaded by `ModelStore` on first use and loaded through `installed(in:descriptor:)`.
///
/// `MLModel` is thread-safe for prediction, so one estimator can serve several threads.
public final class MonocularDepthEstimator: DepthProvider, @unchecked Sendable {
    private let model: MLModel
    private let inputName: String
    private let outputName: String
    /// Model input size in pixels (518 × 392 for the shipped conversion).
    let inputWidth: Int
    let inputHeight: Int

    /// Loads the model from a `.mlpackage` directory.
    ///
    /// Core ML compiles the package on first use. The compiled `.mlmodelc` is cached as a sibling of the
    /// package, inside its version directory, so later launches skip compilation. A cache that is older
    /// than the package or fails to load is removed and compiled again.
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
        guard input.value.type == .image, let inputConstraint = input.value.imageConstraint else {
            throw .unexpectedModelInterface("input \(input.key) is not an image")
        }
        guard inputConstraint.pixelsWide > 0, inputConstraint.pixelsHigh > 0 else {
            throw .unexpectedModelInterface("input \(input.key) has no fixed size")
        }

        let outputs: [String: MLFeatureDescription] = description.outputDescriptionsByName
        guard outputs.count == 1, let output = outputs.first else {
            throw .unexpectedModelInterface("expected one output, found \(outputs.count)")
        }
        guard output.value.type == .image else {
            throw .unexpectedModelInterface("output \(output.key) is not an image")
        }

        inputName = input.key
        outputName = output.key
        inputWidth = inputConstraint.pixelsWide
        inputHeight = inputConstraint.pixelsHigh
    }

    /// Builds the estimator from the store when the model is installed and verified.
    /// Returns nil when the model is missing, corrupt or not configured. Throws only when an installed
    /// model fails to load.
    public static func installed(in store: ModelStore,
                                 descriptor: ModelDescriptor) throws(MonocularDepthError) -> MonocularDepthEstimator? {
        guard case let .ready(packageURL) = store.status(for: descriptor) else {
            return nil
        }
        return try MonocularDepthEstimator(packageURL: packageURL)
    }

    /// Depth for one view, in the view's own orientation.
    ///
    /// Portrait views (height > width) are rotated clockwise before inference and the result is rotated
    /// back, so the model always sees a landscape image. The image is stretched to the model size with
    /// no letterboxing. Output values that are not finite or are ≤ 0 are invalid (0).
    public func depthMap(for view: TrainingView) throws -> DepthMap? {
        let image: RGBImage = view.image
        let isPortrait: Bool = image.height > image.width
        let oriented: RGBImage = isPortrait ? MonocularDepthImageSupport.rotatedClockwise(image) : image

        let input: CVPixelBuffer = try MonocularDepthImageSupport.inputBuffer(from: oriented,
                                                                              width: inputWidth,
                                                                              height: inputHeight)
        let features: MLDictionaryFeatureProvider
        do {
            features = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: input)])
        } catch {
            throw MonocularDepthError.predictionFailed("could not build the model input: \(error.localizedDescription)")
        }

        let result: MLFeatureProvider
        do {
            result = try model.prediction(from: features)
        } catch {
            throw MonocularDepthError.predictionFailed(error.localizedDescription)
        }
        guard let outputBuffer = result.featureValue(for: outputName)?.imageBufferValue else {
            throw MonocularDepthError.predictionFailed("the model returned no image for \(outputName)")
        }

        var map: DepthMap = try MonocularDepthImageSupport.depthMap(from: outputBuffer)
        if isPortrait {
            map = map.rotatedCounterClockwise()
        }
        return map
    }

    // MARK: - Loading and caching

    /// Name of the cached compiled model: `<package name without extension>.mlmodelc`, in the package's directory.
    static func compiledCacheURL(for packageURL: URL) -> URL {
        packageURL
            .deletingLastPathComponent()
            .appendingPathComponent(packageURL.deletingPathExtension().lastPathComponent + ".mlmodelc", isDirectory: true)
    }

    /// Loads a `.mlpackage` through the compiled-model cache next to it. Shared with the Depth Anything 3
    /// pose provider, which sets `configuration.functionName` for its multi-function package.
    static func loadModel(packageURL: URL,
                          configuration: MLModelConfiguration) throws(MonocularDepthError) -> MLModel {
        let fileManager: FileManager = FileManager.default
        let cacheURL: URL = compiledCacheURL(for: packageURL)

        if isCacheFresh(cacheURL, packageURL: packageURL),
           let cached = try? MLModel(contentsOf: cacheURL, configuration: configuration) {
            return cached
        }

        // Missing, stale or unloadable: compile again. Remove the old cache first so the move below cannot collide.
        try? fileManager.removeItem(at: cacheURL)
        let compiledURL: URL
        do {
            compiledURL = try MLModel.compileModel(at: packageURL)
        } catch {
            throw .modelNotFound("could not compile \(packageURL.lastPathComponent): \(error.localizedDescription)")
        }

        // Keep the compiled model next to the package. If the move fails, load from the temporary copy.
        var loadURL: URL = compiledURL
        do {
            try fileManager.moveItem(at: compiledURL, to: cacheURL)
            loadURL = cacheURL
        } catch {
            loadURL = compiledURL
        }

        do {
            return try MLModel(contentsOf: loadURL, configuration: configuration)
        } catch {
            throw .modelNotFound("could not load \(packageURL.lastPathComponent): \(error.localizedDescription)")
        }
    }

    /// The cache is fresh when it exists and was written after the package last changed.
    private static func isCacheFresh(_ cacheURL: URL, packageURL: URL) -> Bool {
        guard FileManager.default.fileExists(atPath: cacheURL.path) else {
            return false
        }
        let cacheDate: Date? = try? cacheURL.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
        let packageDate: Date? = try? packageURL.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
        guard let cacheDate, let packageDate else {
            return false
        }
        return cacheDate >= packageDate
    }
}

/// Pixel conversions for the depth model. Internal so the tests can check them without a model.
enum MonocularDepthImageSupport {
    /// Rotates an image 90° clockwise. The result is `height` wide and `width` high.
    static func rotatedClockwise(_ image: RGBImage) -> RGBImage {
        let outputWidth: Int = image.height
        let outputHeight: Int = image.width
        var pixels: [SIMD3<Float>] = [SIMD3<Float>](repeating: SIMD3<Float>(0, 0, 0), count: outputWidth * outputHeight)
        for row in 0..<image.height {
            for column in 0..<image.width {
                // Clockwise: source (x, y) lands at output (H - 1 - y, x).
                let outputColumn: Int = image.height - 1 - row
                let outputRow: Int = column
                pixels[outputRow * outputWidth + outputColumn] = image.pixels[row * image.width + column]
            }
        }
        return RGBImage(width: outputWidth, height: outputHeight, pixels: pixels)
    }

    /// Converts an sRGB image (values 0...1) to a 32BGRA pixel buffer of `width` × `height`, stretching
    /// it with Core Graphics. Both sides use the sRGB colour space, so the values are not altered.
    static func inputBuffer(from image: RGBImage, width: Int, height: Int) throws(MonocularDepthError) -> CVPixelBuffer {
        guard image.width > 0, image.height > 0, width > 0, height > 0,
              image.pixels.count == image.width * image.height else {
            throw .predictionFailed("the view image is empty or inconsistent")
        }
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw .predictionFailed("the sRGB colour space is unavailable")
        }

        // Source pixels as 8-bit RGBX, top row first.
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

        var created: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        let status: CVReturn = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                                   kCVPixelFormatType_32BGRA,
                                                   attributes as CFDictionary, &created)
        guard status == kCVReturnSuccess, let buffer = created else {
            throw .predictionFailed("could not allocate the input buffer (status \(status))")
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw .predictionFailed("the input buffer has no memory")
        }
        // 32-bit little endian with no alpha: bytes are B, G, R, X, which is 32BGRA.
        let bitmapInfo: UInt32 = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue
        guard let context = CGContext(data: base,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                      space: colorSpace,
                                      bitmapInfo: bitmapInfo) else {
            throw .predictionFailed("could not create the drawing context")
        }
        context.interpolationQuality = .high
        context.draw(source, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }

    /// Reads a one-component `Grayscale16Half` or `OneComponent32Float` output into a depth map.
    /// Values that are not finite or are ≤ 0 become 0 (invalid).
    static func depthMap(from buffer: CVPixelBuffer) throws(MonocularDepthError) -> DepthMap {
        let format: OSType = CVPixelBufferGetPixelFormatType(buffer)
        let isHalf: Bool = format == kCVPixelFormatType_OneComponent16Half
        guard isHalf || format == kCVPixelFormatType_OneComponent32Float else {
            throw .unexpectedModelInterface("unsupported output pixel format \(format)")
        }
        let width: Int = CVPixelBufferGetWidth(buffer)
        let height: Int = CVPixelBufferGetHeight(buffer)

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw .predictionFailed("the output buffer has no memory")
        }
        let bytesPerRow: Int = CVPixelBufferGetBytesPerRow(buffer)
        var values: [Float] = [Float](repeating: 0, count: width * height)
        for row in 0..<height {
            let rowPointer: UnsafeRawPointer = UnsafeRawPointer(base + row * bytesPerRow)
            for column in 0..<width {
                let value: Float
                if isHalf {
                    let bits: UInt16 = rowPointer.load(fromByteOffset: column * 2, as: UInt16.self)
                    value = Float(Float16(bitPattern: bits))
                } else {
                    value = rowPointer.load(fromByteOffset: column * 4, as: Float.self)
                }
                values[row * width + column] = (value.isFinite && value > 0) ? value : 0
            }
        }
        return DepthMap(width: width, height: height, kind: .relativeInverseDepth,
                        values: values, confidence: nil, isEstimated: true)
    }

    /// An sRGB value in 0...1 as an 8-bit byte.
    private static func byte(_ value: Float) -> UInt8 {
        let clamped: Float = min(max(value, 0), 1)
        return UInt8((clamped * 255).rounded())
    }
}
