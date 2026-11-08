import Foundation

/// Per-photo measurements used by photo triage. Sharpness and luminance come from a copy downscaled to at most
/// `PhotoQuality.analysisSide` pixels on the long side. The signature is a tiny grey thumbnail for the
/// redundancy test when a photo has no camera pose.
public struct PhotoMetrics: Sendable, Equatable {
    /// Variance of the Laplacian of the downscaled luma. Higher means sharper. Only comparable within one set.
    public var sharpness: Double
    /// Mean luma of the downscaled image, in `0...1`.
    public var meanLuminance: Double
    /// Fraction of pixels at or below `PhotoQuality.shadowLevel`.
    public var shadowFraction: Double
    /// Fraction of pixels at or above `PhotoQuality.highlightLevel`.
    public var highlightFraction: Double
    /// `signatureSide` x `signatureSide` grey thumbnail, row-major, values in `0...1`.
    public var signature: [Double]

    public init(sharpness: Double,
                meanLuminance: Double,
                shadowFraction: Double,
                highlightFraction: Double,
                signature: [Double]) {
        self.sharpness = sharpness
        self.meanLuminance = meanLuminance
        self.shadowFraction = shadowFraction
        self.highlightFraction = highlightFraction
        self.signature = signature
    }
}

/// Measures one decoded photo for photo triage. Pure computation on `RGBImage`, so it runs off the main actor
/// and on Linux.
public enum PhotoQuality {
    /// Long side of the copy used for sharpness and exposure.
    public static let analysisSide: Int = 256
    /// Side of the square grey signature.
    public static let signatureSide: Int = 8
    /// Luma at or below this counts as a crushed shadow.
    public static let shadowLevel: Double = 0.02
    /// Luma at or above this counts as a clipped highlight.
    public static let highlightLevel: Double = 0.98

    /// Metrics of `image`. An empty or malformed image measures as all zeros.
    public static func measure(_ image: RGBImage) -> PhotoMetrics {
        let emptySignature: [Double] = Array(repeating: 0, count: signatureSide * signatureSide)
        guard image.width > 0, image.height > 0, image.pixels.count == image.width * image.height else {
            return PhotoMetrics(sharpness: 0, meanLuminance: 0, shadowFraction: 0, highlightFraction: 0,
                                signature: emptySignature)
        }

        let longSide: Int = max(image.width, image.height)
        let scale: Double = min(1.0, Double(analysisSide) / Double(longSide))
        let analysisWidth: Int = max(1, Int((Double(image.width) * scale).rounded()))
        let analysisHeight: Int = max(1, Int((Double(image.height) * scale).rounded()))
        let analysisLuma: [Double] = lumaGrid(image, width: analysisWidth, height: analysisHeight)

        // Grey RGB image whose luma equals the grid value, so the existing Laplacian measure applies.
        var grey: [SIMD3<Float>] = []
        grey.reserveCapacity(analysisLuma.count)
        var lumaSum: Double = 0
        var shadowCount: Int = 0
        var highlightCount: Int = 0
        for value in analysisLuma {
            let level: Float = Float(value)
            grey.append(SIMD3<Float>(level, level, level))
            lumaSum += value
            if value <= shadowLevel {
                shadowCount += 1
            }
            if value >= highlightLevel {
                highlightCount += 1
            }
        }
        let count: Double = Double(analysisLuma.count)
        let greyImage: RGBImage = RGBImage(width: analysisWidth, height: analysisHeight, pixels: grey)

        return PhotoMetrics(
            sharpness: FrameSharpness.laplacianVariance(greyImage),
            meanLuminance: lumaSum / count,
            shadowFraction: Double(shadowCount) / count,
            highlightFraction: Double(highlightCount) / count,
            signature: lumaGrid(image, width: signatureSide, height: signatureSide)
        )
    }

    /// Box-averaged luma of `image` on a `width` x `height` grid, row-major, in `0...1`.
    /// Each target cell averages the source pixels it covers; a cell always covers at least one pixel.
    static func lumaGrid(_ image: RGBImage, width: Int, height: Int) -> [Double] {
        let sourceWidth: Int = image.width
        let sourceHeight: Int = image.height
        var result: [Double] = Array(repeating: 0, count: width * height)
        for targetY in 0..<height {
            let rowStart: Int = targetY * sourceHeight / height
            let rowEnd: Int = max(rowStart + 1, (targetY + 1) * sourceHeight / height)
            for targetX in 0..<width {
                let columnStart: Int = targetX * sourceWidth / width
                let columnEnd: Int = max(columnStart + 1, (targetX + 1) * sourceWidth / width)
                var total: Double = 0
                var samples: Int = 0
                for y in rowStart..<min(rowEnd, sourceHeight) {
                    for x in columnStart..<min(columnEnd, sourceWidth) {
                        let pixel: SIMD3<Float> = image.pixels[y * sourceWidth + x]
                        let luma: Double = 0.299 * Double(pixel.x) + 0.587 * Double(pixel.y) + 0.114 * Double(pixel.z)
                        total += luma
                        samples += 1
                    }
                }
                result[targetY * width + targetX] = samples > 0 ? total / Double(samples) : 0
            }
        }
        return result
    }
}
