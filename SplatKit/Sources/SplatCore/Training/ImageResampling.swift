import Foundation

extension RGBImage {
    /// Bilinear sample at pixel-index coordinates (pixel `i` has its value at `i`).
    /// Coordinates outside the image are clamped to the edge.
    public func bilinearSample(x: Double, y: Double) -> SIMD3<Float> {
        precondition(width > 0 && height > 0, "cannot sample an empty image")
        let cx = min(max(x, 0), Double(width - 1))
        let cy = min(max(y, 0), Double(height - 1))
        let x0 = Int(cx.rounded(.down))
        let y0 = Int(cy.rounded(.down))
        let x1 = min(x0 + 1, width - 1)
        let y1 = min(y0 + 1, height - 1)
        let tx = Float(cx - Double(x0))
        let ty = Float(cy - Double(y0))

        let top = pixels[y0 * width + x0] * (1 - tx) + pixels[y0 * width + x1] * tx
        let bottom = pixels[y1 * width + x0] * (1 - tx) + pixels[y1 * width + x1] * tx
        return top * (1 - ty) + bottom * ty
    }

    /// Box average over `factor × factor` blocks; output size is `width / factor × height / factor`
    /// (floor). Trailing rows and columns that do not fill a block are dropped.
    public func downsampled(by factor: Int) -> RGBImage {
        precondition(factor >= 1, "downsample factor must be at least 1")
        let outWidth = width / factor
        let outHeight = height / factor
        let area = Double(factor * factor)
        var out = [SIMD3<Float>](repeating: .zero, count: outWidth * outHeight)
        for oy in 0..<outHeight {
            for ox in 0..<outWidth {
                var sum = SIMD3<Double>.zero
                for dy in 0..<factor {
                    let row = (oy * factor + dy) * width + ox * factor
                    for dx in 0..<factor {
                        let p = pixels[row + dx]
                        sum += SIMD3<Double>(Double(p.x), Double(p.y), Double(p.z))
                    }
                }
                let avg: SIMD3<Double> = sum / area
                out[oy * outWidth + ox] = SIMD3<Float>(Float(avg.x), Float(avg.y), Float(avg.z))
            }
        }
        return RGBImage(width: outWidth, height: outHeight, pixels: out)
    }
}

/// Removes lens distortion. The output is a pinhole image with the same size and the same
/// `fx, fy, cx, cy` as the input model, sampled bilinearly with clamp-to-edge.
public enum Undistortion {
    public static func undistort(_ image: RGBImage, model: CameraModel) -> RGBImage {
        guard model.hasDistortion else { return image }

        let focal = model.focal
        let center = model.principalPoint
        let w = image.width
        let h = image.height
        var out = [SIMD3<Float>](repeating: .zero, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                let nu = (Double(x) + 0.5 - center.x) / focal.x
                let nv = (Double(y) + 0.5 - center.y) / focal.y
                let d = model.distort(SIMD2(nu, nv))
                let sx = focal.x * d.x + center.x - 0.5
                let sy = focal.y * d.y + center.y - 0.5
                out[y * w + x] = image.bilinearSample(x: sx, y: sy)
            }
        }
        return RGBImage(width: w, height: h, pixels: out)
    }
}

public enum ImageResampling {
    /// Smallest integer factor that brings the long side down to at most `maxSide`; at least 1.
    public static func downsampleFactor(width: Int, height: Int, maxSide: Int) -> Int {
        precondition(maxSide >= 1, "maxSide must be at least 1")
        let longSide = max(width, height)
        return max(1, (longSide + maxSide - 1) / maxSide)
    }
}
