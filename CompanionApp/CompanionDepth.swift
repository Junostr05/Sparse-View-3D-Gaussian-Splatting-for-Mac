import CoreML
import Foundation
import SplatCore

/// Depth on the companion: one Core ML image-to-depth model, compiled on first use.
nonisolated final class CompanionDepth: @unchecked Sendable {
    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let inputWidth: Int
    private let inputHeight: Int

    /// Loads the model package at `packageURL` (compiled to a cache next to the app's support files).
    init(packageURL: URL) async throws(CompanionDepthError) {
        guard FileManager.default.fileExists(atPath: packageURL.path) else {
            throw .modelNotFound(packageURL.lastPathComponent)
        }
        let compiled: URL
        do {
            compiled = try await MLModel.compileModel(at: packageURL)
        } catch {
            throw .predictionFailed("could not compile the model: \(error.localizedDescription)")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        do {
            model = try MLModel(contentsOf: compiled, configuration: configuration)
        } catch {
            throw .predictionFailed("could not load the model: \(error.localizedDescription)")
        }
        let description = model.modelDescription
        let inputs = description.inputDescriptionsByName
        guard inputs.count == 1, let input = inputs.first, input.value.type == .image,
              let constraint = input.value.imageConstraint, constraint.pixelsWide > 0, constraint.pixelsHigh > 0 else {
            throw .unexpectedModelInterface("expected one fixed-size image input")
        }
        let outputs = description.outputDescriptionsByName
        guard outputs.count == 1, let output = outputs.first, output.value.type == .image else {
            throw .unexpectedModelInterface("expected one image output")
        }
        inputName = input.key
        outputName = output.key
        inputWidth = constraint.pixelsWide
        inputHeight = constraint.pixelsHigh
    }

    /// Depth for one image (RGB8, row-major). Portrait images are rotated for the model and back again, as on the Mac.
    func depth(rgb: [UInt8], width: Int, height: Int) throws(CompanionDepthError) -> DepthMap {
        var pixels: [SIMD3<Float>] = []
        pixels.reserveCapacity(width * height)
        for index in 0..<(width * height) {
            pixels.append(SIMD3<Float>(Float(rgb[3 * index]) / 255, Float(rgb[3 * index + 1]) / 255,
                                       Float(rgb[3 * index + 2]) / 255))
        }
        let image = RGBImage(width: width, height: height, pixels: pixels)
        let isPortrait = height > width
        let oriented = isPortrait ? CompanionImageSupport.rotatedClockwise(image) : image
        let input = try CompanionImageSupport.inputBuffer(from: oriented, width: inputWidth, height: inputHeight)
        let features: MLDictionaryFeatureProvider
        do {
            features = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: input)])
        } catch {
            throw .predictionFailed("could not build the input: \(error.localizedDescription)")
        }
        let result: MLFeatureProvider
        do {
            result = try model.prediction(from: features)
        } catch {
            throw .predictionFailed(error.localizedDescription)
        }
        guard let buffer = result.featureValue(for: outputName)?.imageBufferValue else {
            throw .predictionFailed("the model returned no image")
        }
        var map = try CompanionImageSupport.depthMap(from: buffer)
        if isPortrait {
            map = map.rotatedCounterClockwise()
        }
        return map
    }
}
