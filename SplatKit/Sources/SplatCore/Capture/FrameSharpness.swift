import Foundation

/// Sharpness measure for candidate video frames. Motion blur removes high-frequency detail,
/// so the variance of the Laplacian is a cheap proxy for how sharp a frame is.
public enum FrameSharpness {
    /// Variance of the 3x3 Laplacian ([0 1 0; 1 -4 1; 0 1 0]) of the luma image
    /// (0.299 R + 0.587 G + 0.114 B) over the interior pixels. Images smaller than 3x3 return 0.
    /// Pixel values are 0...1 floats in `RGBImage.pixels`. Accumulation is done in Double.
    public static func laplacianVariance(_ image: RGBImage) -> Double {
        let width: Int = image.width
        let height: Int = image.height
        guard width >= 3, height >= 3, image.pixels.count == width * height else { return 0 }

        var luma: [Double] = []
        luma.reserveCapacity(width * height)
        for pixel in image.pixels {
            let r: Double = Double(pixel.x)
            let g: Double = Double(pixel.y)
            let b: Double = Double(pixel.z)
            luma.append(0.299 * r + 0.587 * g + 0.114 * b)
        }

        var responses: [Double] = []
        responses.reserveCapacity((width - 2) * (height - 2))
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let centre: Double = luma[y * width + x]
                let neighbours: Double = luma[y * width + x - 1]
                    + luma[y * width + x + 1]
                    + luma[(y - 1) * width + x]
                    + luma[(y + 1) * width + x]
                responses.append(neighbours - 4 * centre)
            }
        }

        // Two-pass variance avoids the cancellation of E[x^2] - E[x]^2.
        let n: Double = Double(responses.count)
        var mean: Double = 0
        for value in responses {
            mean += value
        }
        mean /= n

        var sumSquares: Double = 0
        for value in responses {
            let d: Double = value - mean
            sumSquares += d * d
        }
        return sumSquares / n
    }
}
