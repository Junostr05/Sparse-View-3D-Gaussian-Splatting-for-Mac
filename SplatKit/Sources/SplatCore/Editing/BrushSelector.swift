/// Screen-space brush selection of Gaussians.
public enum BrushSelector {
    /// Selects every Gaussian whose centre projects in front of `camera` (depth > `near`) and lies
    /// within `radiusPixels` (inclusive, Euclidean) of any stroke point.
    ///
    /// - Parameters:
    ///   - positions: xyz-interleaved centres of the display cloud (3 floats per Gaussian).
    ///   - camera: world-to-camera pinhole camera used for rendering.
    ///   - stroke: brush centres in that camera's pixel coordinates.
    ///   - radiusPixels: brush radius in pixels. Zero, negative or non-finite selects nothing.
    ///   - near: depth threshold; Gaussians with depth ≤ `near` are never selected.
    /// - Returns: mask with one bit per Gaussian. Non-finite positions are never selected.
    public static func select(positions: [Float],
                              camera: Camera,
                              stroke: [SIMD2<Float>],
                              radiusPixels: Float,
                              near: Float = 0.01) -> SelectionMask {
        precondition(positions.count % 3 == 0,
                     "BrushSelector positions must be xyz triples, got \(positions.count) floats")
        let gaussianCount: Int = positions.count / 3
        var mask = SelectionMask(count: gaussianCount)
        guard radiusPixels.isFinite, radiusPixels > 0 else { return mask }

        // Bounding box of the finite stroke points.
        var minX: Double = .infinity
        var minY: Double = .infinity
        var maxX: Double = -.infinity
        var maxY: Double = -.infinity
        for point in stroke where point.x.isFinite && point.y.isFinite {
            let x: Double = Double(point.x)
            let y: Double = Double(point.y)
            minX = Swift.min(minX, x)
            minY = Swift.min(minY, y)
            maxX = Swift.max(maxX, x)
            maxY = Swift.max(maxY, y)
        }
        guard minX <= maxX, minY <= maxY else { return mask }

        let radius: Double = Double(radiusPixels)
        let radiusSquared: Double = radius * radius
        // Candidate search uses 1% slack over the radius, so rounding in the cell arithmetic can never
        // move a point that is within the radius out of its neighbouring cells. The exact inclusive
        // test is still `distance² <= radius²`. Cells are at least `reach` wide, so every candidate
        // lies in the 3×3 block around the point's cell. The lower bound on the cell size keeps cell
        // indices bounded for very small radii over a large stroke.
        let reach: Double = radius * 1.01
        let extent: Double = Swift.max(maxX - minX, maxY - minY)
        let cellSize: Double = Swift.max(reach, extent / 4_294_967_296.0)

        // Uniform grid over the stroke points.
        var grid: [BrushCellKey: [Int]] = [:]
        for (strokeIndex, point) in stroke.enumerated() where point.x.isFinite && point.y.isFinite {
            let key: BrushCellKey = BrushCellKey(
                x: cellCoordinate(Double(point.x), origin: minX, cellSize: cellSize),
                y: cellCoordinate(Double(point.y), origin: minY, cellSize: cellSize)
            )
            grid[key, default: []].append(strokeIndex)
        }

        for gaussian in 0..<gaussianCount {
            let baseIndex: Int = 3 * gaussian
            let x: Float = positions[baseIndex]
            let y: Float = positions[baseIndex + 1]
            let z: Float = positions[baseIndex + 2]
            guard x.isFinite, y.isFinite, z.isFinite else { continue }
            let center: SIMD3<Float> = SIMD3<Float>(x, y, z)
            guard let projected = DepthGeometry.project(center, camera: camera, near: near) else {
                continue
            }
            let u: Double = Double(projected.u)
            let v: Double = Double(projected.v)
            guard u.isFinite, v.isFinite else { continue }

            // Cheap rejection: outside the stroke bounding box grown by the radius.
            if u < minX - reach || u > maxX + reach || v < minY - reach || v > maxY + reach {
                continue
            }

            let cellX: Int = cellCoordinate(u, origin: minX, cellSize: cellSize)
            let cellY: Int = cellCoordinate(v, origin: minY, cellSize: cellSize)
            var selected = false
            neighbourhood: for offsetX in -1...1 {
                for offsetY in -1...1 {
                    let key: BrushCellKey = BrushCellKey(x: cellX + offsetX, y: cellY + offsetY)
                    guard let bucket = grid[key] else { continue }
                    for strokeIndex in bucket {
                        let point: SIMD2<Float> = stroke[strokeIndex]
                        let deltaX: Double = u - Double(point.x)
                        let deltaY: Double = v - Double(point.y)
                        if deltaX * deltaX + deltaY * deltaY <= radiusSquared {
                            selected = true
                            break neighbourhood
                        }
                    }
                }
            }
            if selected {
                mask[gaussian] = true
            }
        }
        return mask
    }

    private static func cellCoordinate(_ value: Double, origin: Double, cellSize: Double) -> Int {
        let scaled: Double = ((value - origin) / cellSize).rounded(.down)
        return Int(scaled)
    }
}

/// Grid cell index used by `BrushSelector`.
private struct BrushCellKey: Hashable {
    let x: Int
    let y: Int
}
