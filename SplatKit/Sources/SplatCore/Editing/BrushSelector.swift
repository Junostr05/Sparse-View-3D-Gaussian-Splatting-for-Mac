/// Settings for front-only brush selection: only the Gaussians that form the visible surface under the
/// brush are selected, so painting a floater in front of a wall does not also select the wall.
public struct FrontOnlySettings: Sendable, Equatable {
    /// Sigmoid opacity per Gaussian, one value per Gaussian of the display cloud. Non-finite values count as 0.
    public var opacities: [Float]
    /// A candidate at least this opaque can define the surface depth of its screen cell. Less opaque
    /// candidates never do, so a semi-transparent layer does not hide the surface behind it. A NaN
    /// value means no candidate is opaque enough.
    public var minimumOpacity: Float
    /// Relative depth slack. A candidate is kept when its depth is at most `surfaceDepth · (1 + depthTolerance)`.
    /// Negative and non-finite values are treated as 0.
    public var depthTolerance: Float
    /// Side length of the square screen cells, in pixels, that group the candidates. Values below 1 pixel
    /// and non-finite values are clamped (non-finite becomes 6).
    public var cellPixels: Float

    public init(opacities: [Float],
                minimumOpacity: Float = 0.3,
                depthTolerance: Float = 0.08,
                cellPixels: Float = 6) {
        self.opacities = opacities
        self.minimumOpacity = minimumOpacity
        self.depthTolerance = depthTolerance
        self.cellPixels = cellPixels
    }
}

/// Screen-space brush selection of Gaussians.
public enum BrushSelector {
    /// Selects every Gaussian whose centre projects in front of `camera` (depth > `near`) and lies
    /// within `radiusPixels` (inclusive, Euclidean) of any stroke point.
    ///
    /// With `frontOnly` set, only the candidates that form the visible surface are kept: per screen cell
    /// the surface depth is the smallest depth among candidates that are opaque enough, and a candidate
    /// is kept when its depth is at most that surface depth times `1 + depthTolerance`.
    ///
    /// - Parameters:
    ///   - positions: xyz-interleaved centres of the display cloud (3 floats per Gaussian).
    ///   - camera: world-to-camera pinhole camera used for rendering.
    ///   - stroke: brush centres in that camera's pixel coordinates.
    ///   - radiusPixels: brush radius in pixels. Zero, negative or non-finite selects nothing.
    ///   - near: depth threshold; Gaussians with depth ≤ `near` are never selected.
    ///   - frontOnly: `nil` selects through all depths (x-ray). Otherwise `opacities` must have one entry per Gaussian.
    /// - Returns: mask with one bit per Gaussian. Non-finite positions are never selected.
    public static func select(positions: [Float],
                              camera: Camera,
                              stroke: [SIMD2<Float>],
                              radiusPixels: Float,
                              near: Float = 0.01,
                              frontOnly: FrontOnlySettings? = nil) -> SelectionMask {
        precondition(positions.count % 3 == 0,
                     "BrushSelector positions must be xyz triples, got \(positions.count) floats")
        let gaussianCount: Int = positions.count / 3
        if let frontOnly {
            precondition(frontOnly.opacities.count == gaussianCount,
                         "BrushSelector front-only needs one opacity per Gaussian, got \(frontOnly.opacities.count) for \(gaussianCount)")
        }
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

        // Screen cells for front-only selection. Cell origin and size follow the stroke bounding box, like the
        // grid above, and the size has the same bounded lower limit.
        let frontCellSize: Double = Swift.max(frontOnlyCellSize(frontOnly), extent / 4_294_967_296.0)
        var candidates: [FrontCandidate] = []

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
                if frontOnly == nil {
                    mask[gaussian] = true
                } else {
                    let cell: BrushCellKey = BrushCellKey(
                        x: cellCoordinate(u, origin: minX, cellSize: frontCellSize),
                        y: cellCoordinate(v, origin: minY, cellSize: frontCellSize)
                    )
                    candidates.append(FrontCandidate(gaussian: gaussian, depth: Double(projected.z), cell: cell))
                }
            }
        }

        if let frontOnly {
            keepVisibleSurface(candidates, settings: frontOnly, into: &mask)
        }
        return mask
    }

    /// Sets the mask bit of every candidate that lies on the visible surface of its screen cell.
    ///
    /// The surface depth of a cell is the smallest depth among its candidates with opacity at least
    /// `minimumOpacity`. A cell without such a candidate falls back to the smallest depth of all its
    /// candidates. The result does not depend on the order of `candidates`.
    private static func keepVisibleSurface(_ candidates: [FrontCandidate],
                                           settings: FrontOnlySettings,
                                           into mask: inout SelectionMask) {
        var opaqueDepth: [BrushCellKey: Double] = [:]
        var nearestDepth: [BrushCellKey: Double] = [:]
        for candidate in candidates {
            let opacity: Float = settings.opacities[candidate.gaussian]
            let isOpaque: Bool = opacity.isFinite && opacity >= settings.minimumOpacity
            nearestDepth[candidate.cell] = Swift.min(nearestDepth[candidate.cell] ?? .infinity, candidate.depth)
            if isOpaque {
                opaqueDepth[candidate.cell] = Swift.min(opaqueDepth[candidate.cell] ?? .infinity, candidate.depth)
            }
        }

        let tolerance: Double = settings.depthTolerance.isFinite
            ? Double(Swift.max(settings.depthTolerance, 0))
            : 0
        for candidate in candidates {
            let surfaceDepth: Double = opaqueDepth[candidate.cell] ?? nearestDepth[candidate.cell] ?? candidate.depth
            if candidate.depth <= surfaceDepth * (1 + tolerance) {
                mask[candidate.gaussian] = true
            }
        }
    }

    /// Screen cell size for front-only selection in pixels: at least 1 pixel, 6 when non-finite.
    private static func frontOnlyCellSize(_ settings: FrontOnlySettings?) -> Double {
        guard let settings else { return 6 }
        guard settings.cellPixels.isFinite else { return 6 }
        return Swift.max(Double(settings.cellPixels), 1)
    }

    private static func cellCoordinate(_ value: Double, origin: Double, cellSize: Double) -> Int {
        let scaled: Double = ((value - origin) / cellSize).rounded(.down)
        return Int(scaled)
    }
}

/// A Gaussian inside the brush radius, with its camera depth and screen cell, for front-only selection.
private struct FrontCandidate {
    let gaussian: Int
    let depth: Double
    let cell: BrushCellKey
}

/// Grid cell index used by `BrushSelector`.
private struct BrushCellKey: Hashable {
    let x: Int
    let y: Int
}
