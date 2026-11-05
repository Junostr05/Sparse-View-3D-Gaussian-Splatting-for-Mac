import Foundation

/// Row-major RGB image.
public struct RGBImage: Sendable, Equatable {
    public var width: Int
    public var height: Int
    public var pixels: [SIMD3<Float>]

    public init(width: Int, height: Int, pixels: [SIMD3<Float>]) {
        self.width = width
        self.height = height
        self.pixels = pixels
    }
}

/// Brute-force CPU reference for the forward rasterizer. Clarity over speed: the Metal
/// kernels must match its output within 1e-4.
public enum CPURasterizer {
    public static func render(_ cloud: GaussianCloud,
                              camera: Camera,
                              settings: RasterSettings) -> RGBImage {
        // Project everything, keeping the original index for the tie-break.
        var projected: [(index: Int, gaussian: ProjectedGaussian)] = []
        projected.reserveCapacity(cloud.count)
        for i in 0..<cloud.count {
            if let g = Projection.project(cloud, index: i, camera: camera, settings: settings) {
                projected.append((i, g))
            }
        }
        // Ascending depth, ties by ascending index.
        projected.sort { lhs, rhs in
            if lhs.gaussian.depth != rhs.gaussian.depth {
                return lhs.gaussian.depth < rhs.gaussian.depth
            }
            return lhs.index < rhs.index
        }

        let width = camera.width
        let height = camera.height
        let tileSize = settings.tileSize
        var pixels = [SIMD3<Float>](repeating: .zero, count: width * height)

        for y in 0..<height {
            let ty = Int32(y / tileSize)
            for x in 0..<width {
                let tx = Int32(x / tileSize)
                let center = SIMD2<Float>(Float(x) + 0.5, Float(y) + 0.5)
                var transmittance: Float = 1
                var rgb = SIMD3<Float>.zero

                for entry in projected {
                    let g = entry.gaussian
                    guard tx >= g.tileMin.x, tx < g.tileMax.x,
                          ty >= g.tileMin.y, ty < g.tileMax.y else { continue }

                    let d = g.mean2D - center
                    let sigma = 0.5 * (g.conic.x * d.x * d.x + g.conic.z * d.y * d.y)
                        + g.conic.y * d.x * d.y
                    guard sigma >= 0 else { continue }

                    let alpha = min(0.999, g.opacity * exp(-sigma))
                    guard alpha >= 1.0 / 255.0 else { continue }

                    let nextT = transmittance * (1 - alpha)
                    if nextT <= 1e-4 { break }

                    rgb += g.color * alpha * transmittance
                    transmittance = nextT
                }

                pixels[y * width + x] = rgb + transmittance * settings.background
            }
        }

        return RGBImage(width: width, height: height, pixels: pixels)
    }
}
