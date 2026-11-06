import Foundation

/// Pure arithmetic for the training memory cap (recorded decision: Memory budget).
/// The macOS caller passes `MTLDevice.recommendedMaxWorkingSetSize`.
public enum TrainingMemoryBudget {
    /// Bytes per Gaussian for parameters, gradients, Adam m/v, scatter scratch and renderer
    /// scratch. With K = (d+1)² and P = 11 + 3K: 16·P + 12·K + 340.
    public static func bytesPerGaussian(shDegree: Int) -> Int {
        let k = (shDegree + 1) * (shDegree + 1)
        let p = 11 + 3 * k
        return 16 * p + 12 * k + 340
    }

    /// Bytes reserved for image-sized buffers: target images, render texture, image gradient,
    /// spare, and loss scratch. Targets are summed; the per-pixel scratch uses the largest view.
    public static func reservedImageBytes(viewSizes: [SIMD2<Int>]) -> Int {
        var targetBytes = 0
        var maxWidth = 0
        var maxHeight = 0
        for size in viewSizes {
            targetBytes += 16 * size.x * size.y
            maxWidth = max(maxWidth, size.x)
            maxHeight = max(maxHeight, size.y)
        }
        let perPixelScratch = 3 * 16 + 24 * 4
        return targetBytes + perPixelScratch * maxWidth * maxHeight
    }

    /// Bytes reserved for depth buffers (Phase 4): per training view, the target and metric
    /// depth (2 × float, 8 bytes per pixel); one float2 depth output and one float2 depth
    /// gradient sized to the largest training view (16 bytes per pixel); and one float per
    /// pixel for each pseudo-view prior.
    public static func reservedDepthBytes(viewSizes: [SIMD2<Int>], pseudoViewSizes: [SIMD2<Int>]) -> Int {
        var viewBytes: Int = 0
        var maxWidth: Int = 0
        var maxHeight: Int = 0
        for size in viewSizes {
            let pixels: Int = size.x * size.y
            viewBytes += 8 * pixels
            maxWidth = max(maxWidth, size.x)
            maxHeight = max(maxHeight, size.y)
        }
        let largestBytes: Int = 16 * maxWidth * maxHeight
        var pseudoBytes: Int = 0
        for size in pseudoViewSizes {
            let pixels: Int = size.x * size.y
            pseudoBytes += 4 * pixels
        }
        return viewBytes + largestBytes + pseudoBytes
    }

    /// Largest Gaussian count that fits in `fraction` of the working set after reserving
    /// `reservedBytes`, capped at `configuredMax`. Never negative.
    public static func maxGaussians(workingSetBytes: UInt64,
                                    shDegree: Int,
                                    reservedBytes: Int,
                                    configuredMax: Int,
                                    fraction: Double = 0.6) -> Int {
        let available = fraction * Double(workingSetBytes) - Double(reservedBytes)
        let perGaussian = Double(bytesPerGaussian(shDegree: shDegree))
        let budgetBound = Int((available / perGaussian).rounded(.down))
        return min(configuredMax, max(0, budgetBound))
    }
}
