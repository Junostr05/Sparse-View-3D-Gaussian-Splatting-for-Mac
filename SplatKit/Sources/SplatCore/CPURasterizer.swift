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
        composite(cloud, camera: camera, settings: settings, collectDepth: false).image
    }

    /// Same image as `render` (bit-identical), plus the rendered depth channel.
    public static func renderWithDepth(_ cloud: GaussianCloud,
                                       camera: Camera,
                                       settings: RasterSettings) -> (image: RGBImage, depth: DepthRender) {
        let result = composite(cloud, camera: camera, settings: settings, collectDepth: true)
        return (result.image, result.depth)
    }

    /// Shared pixel loop of `render` and `renderWithDepth`. The colour arithmetic does not
    /// depend on `collectDepth`; the depth accumulators are only written when it is set.
    private static func composite(_ cloud: GaussianCloud,
                                  camera: Camera,
                                  settings: RasterSettings,
                                  collectDepth: Bool) -> (image: RGBImage, depth: DepthRender) {
        let projected = projectedAndSorted(cloud, camera: camera, settings: settings)

        let width = camera.width
        let height = camera.height
        let tileSize = settings.tileSize
        let count = width * height
        var pixels = [SIMD3<Float>](repeating: .zero, count: count)
        var depthRaw = [Float](repeating: 0, count: collectDepth ? count : 0)
        var depthAlpha = [Float](repeating: 0, count: collectDepth ? count : 0)

        for y in 0..<height {
            let ty = Int32(y / tileSize)
            for x in 0..<width {
                let tx = Int32(x / tileSize)
                let center = SIMD2<Float>(Float(x) + 0.5, Float(y) + 0.5)
                var transmittance: Float = 1
                var rgb = SIMD3<Float>.zero
                var zSum: Double = 0

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
                    if collectDepth {
                        let weight: Double = Double(alpha) * Double(transmittance)
                        zSum += weight * Double(g.depth)
                    }
                    transmittance = nextT
                }

                let index = y * width + x
                pixels[index] = rgb + transmittance * settings.background
                if collectDepth {
                    depthRaw[index] = Float(zSum)
                    depthAlpha[index] = Float(1 - Double(transmittance))
                }
            }
        }

        let image = RGBImage(width: width, height: height, pixels: pixels)
        let depth = DepthRender(raw: depthRaw, alpha: depthAlpha)
        return (image, depth)
    }

    /// Projects every Gaussian, drops culled ones and sorts by ascending depth
    /// (ties by ascending original index).
    static func projectedAndSorted(_ cloud: GaussianCloud,
                                   camera: Camera,
                                   settings: RasterSettings)
        -> [(index: Int, gaussian: ProjectedGaussian)] {
        var projected: [(index: Int, gaussian: ProjectedGaussian)] = []
        projected.reserveCapacity(cloud.count)
        for i in 0..<cloud.count {
            if let g = Projection.project(cloud, index: i, camera: camera, settings: settings) {
                projected.append((i, g))
            }
        }
        projected.sort { lhs, rhs in
            if lhs.gaussian.depth != rhs.gaussian.depth {
                return lhs.gaussian.depth < rhs.gaussian.depth
            }
            return lhs.index < rhs.index
        }
        return projected
    }
}
