import Foundation

/// Uniform hash grid over 3D points for radius counts and exact k-th neighbour distances (P10).
///
/// Cells are cubes of side `cellSize` keyed by `floor(p / cellSize)`. Each occupied cell keeps its
/// point indices in ascending order. Queries are deterministic and do not depend on insertion
/// order beyond that.
///
/// The hot paths use scalar components and a plain key struct instead of SIMD vectors: they run
/// about an order of magnitude faster in debug builds, which the test suite uses.
public struct SpatialHashGrid: Sendable {
    /// Integer cell coordinates.
    struct CellKey: Hashable, Sendable {
        var x: Int32
        var y: Int32
        var z: Int32
    }

    /// Flat point coordinates, 3 floats per point.
    private let coords: [Float]
    private let cellSize: Float
    /// Cell key of every point.
    private let pointCells: [CellKey]
    /// Occupied cells: key of each slot, and the ascending point indices stored in it.
    private let slotKeys: [CellKey]
    private let slotPoints: [[Int32]]
    /// Cell key → slot in `slotKeys` / `slotPoints`.
    private let slotLookup: [CellKey: Int]
    /// Componentwise bounds of the occupied cell keys.
    private let minKey: CellKey
    private let maxKey: CellKey

    public init(points: [SIMD3<Float>], cellSize: Float) {
        precondition(cellSize > 0 && cellSize.isFinite, "cell size must be positive and finite")
        var coords: [Float] = []
        coords.reserveCapacity(3 * points.count)
        var pointCells: [CellKey] = []
        pointCells.reserveCapacity(points.count)
        var slotKeys: [CellKey] = []
        var slotPoints: [[Int32]] = []
        var slotLookup: [CellKey: Int] = [:]
        for (index, point) in points.enumerated() {
            coords.append(point.x)
            coords.append(point.y)
            coords.append(point.z)
            let key = Self.cellKey(point, cellSize: cellSize)
            pointCells.append(key)
            if let slot = slotLookup[key] {
                slotPoints[slot].append(Int32(index))
            } else {
                slotLookup[key] = slotKeys.count
                slotKeys.append(key)
                slotPoints.append([Int32(index)])
            }
        }
        var lowKey = CellKey(x: 0, y: 0, z: 0)
        var highKey = CellKey(x: 0, y: 0, z: 0)
        if let first = slotKeys.first {
            lowKey = first
            highKey = first
            for key in slotKeys {
                lowKey = CellKey(x: min(lowKey.x, key.x), y: min(lowKey.y, key.y), z: min(lowKey.z, key.z))
                highKey = CellKey(x: max(highKey.x, key.x), y: max(highKey.y, key.y), z: max(highKey.z, key.z))
            }
        }

        self.coords = coords
        self.cellSize = cellSize
        self.pointCells = pointCells
        self.slotKeys = slotKeys
        self.slotPoints = slotPoints
        self.slotLookup = slotLookup
        self.minKey = lowKey
        self.maxKey = highKey
    }

    /// Number of points within `radius` of `points[index]`, excluding the point itself.
    /// Stops counting once `limit` is reached and returns `limit` in that case.
    /// Requires `radius ≤ cellSize`, so all candidates lie in the 3×3×3 cells around the query.
    public func countNeighbors(of index: Int, within radius: Float, limit: Int) -> Int {
        precondition(pointCells.indices.contains(index), "point index out of range")
        precondition(radius <= cellSize, "radius must not exceed the cell size")
        guard limit > 0 else { return 0 }
        let qx = coords[3 * index]
        let qy = coords[3 * index + 1]
        let qz = coords[3 * index + 2]
        let radiusSquared: Float = radius * radius
        let center = pointCells[index]
        var count = 0
        // Offsets are visited centre first, so dense queries reach `limit` after few candidates.
        var a = 0
        while a < 27 {
            let key = CellKey(x: center.x &+ Self.neighborOffset(a % 3),
                              y: center.y &+ Self.neighborOffset((a / 3) % 3),
                              z: center.z &+ Self.neighborOffset(a / 9))
            a += 1
            if let slot = slotLookup[key] {
                let bucket = slotPoints[slot]
                var b = 0
                while b < bucket.count {
                    let j = Int(bucket[b])
                    b += 1
                    if j == index { continue }
                    if squaredDistance(j, qx, qy, qz) <= radiusSquared {
                        count += 1
                        if count >= limit { return count }
                    }
                }
            }
        }
        return count
    }

    /// Exact distance from `points[index]` to its k-th nearest other point.
    ///
    /// Cubic shells of cells around the query cell are visited for r = 0, 1, 2, …. After shell `r`
    /// the search stops when at least `k` candidates were found and the k-th smallest distance is at
    /// most `r · cellSize`, because every unvisited point is at least that far away. Returns `+∞`
    /// when that does not happen within `maxRings` shells or when fewer than `k` other points exist.
    public func kthNeighborDistance(of index: Int, k: Int, maxRings: Int) -> Float {
        precondition(pointCells.indices.contains(index), "point index out of range")
        guard k > 0, maxRings >= 0, pointCells.count > 1 else { return .infinity }
        let center = pointCells[index]
        let farthestRing: Int = farthestOccupiedRing(from: center)
        var squared: [Float] = []
        var ring = 0
        while ring <= maxRings {
            collectShell(center: center, ring: ring, index: index, into: &squared)
            if squared.count >= k {
                let kth: Float = Self.kthSmallest(squared, k).squareRoot()
                if kth <= Float(ring) * cellSize {
                    return kth
                }
            }
            if ring >= farthestRing {
                // Every point has been visited, so the k-th distance is final. The stop condition
                // can only hold at a larger ring; the search would reach it at this ring index.
                guard squared.count >= k else { return .infinity }
                let kth: Float = Self.kthSmallest(squared, k).squareRoot()
                var needed: Int = max(Int((kth / cellSize).rounded(.up)), ring + 1)
                while Float(needed) * cellSize < kth {
                    needed += 1
                }
                return needed <= maxRings ? kth : .infinity
            }
            ring += 1
        }
        return .infinity
    }

    // MARK: - Shell traversal

    /// Appends the squared distances from the query point to every point in the cells at Chebyshev
    /// distance exactly `ring` from `center`, skipping `index`.
    private func collectShell(center: CellKey,
                              ring: Int,
                              index: Int,
                              into squared: inout [Float]) {
        let shellCellCount: Int = ring == 0 ? 1 : 24 * ring * ring + 2
        if shellCellCount <= slotKeys.count {
            if ring == 0 {
                appendCell(center, index: index, into: &squared)
                return
            }
            let r = Int32(ring)
            for dz in -r...r {
                let edgeZ = abs(dz) == r
                for dy in -r...r {
                    let edgeY = abs(dy) == r
                    if edgeZ || edgeY {
                        for dx in -r...r {
                            appendCell(CellKey(x: center.x &+ dx, y: center.y &+ dy, z: center.z &+ dz),
                                       index: index, into: &squared)
                        }
                    } else {
                        appendCell(CellKey(x: center.x &- r, y: center.y &+ dy, z: center.z &+ dz),
                                   index: index, into: &squared)
                        appendCell(CellKey(x: center.x &+ r, y: center.y &+ dy, z: center.z &+ dz),
                                   index: index, into: &squared)
                    }
                }
            }
        } else {
            // The shell is larger than the number of occupied cells: scan the occupied cells instead.
            let qx = coords[3 * index]
            let qy = coords[3 * index + 1]
            let qz = coords[3 * index + 2]
            for slot in slotKeys.indices where Self.chebyshevDistance(slotKeys[slot], center) == ring {
                for candidate in slotPoints[slot] {
                    let j = Int(candidate)
                    if j == index { continue }
                    squared.append(squaredDistance(j, qx, qy, qz))
                }
            }
        }
    }

    private func appendCell(_ key: CellKey, index: Int, into squared: inout [Float]) {
        guard let slot = slotLookup[key] else { return }
        let qx = coords[3 * index]
        let qy = coords[3 * index + 1]
        let qz = coords[3 * index + 2]
        for candidate in slotPoints[slot] {
            let j = Int(candidate)
            if j == index { continue }
            squared.append(squaredDistance(j, qx, qy, qz))
        }
    }

    /// Largest Chebyshev cell distance from `center` to any occupied cell.
    private func farthestOccupiedRing(from center: CellKey) -> Int {
        let dx = max(Int(center.x) - Int(minKey.x), Int(maxKey.x) - Int(center.x))
        let dy = max(Int(center.y) - Int(minKey.y), Int(maxKey.y) - Int(center.y))
        let dz = max(Int(center.z) - Int(minKey.z), Int(maxKey.z) - Int(center.z))
        return max(dx, max(dy, dz))
    }

    // MARK: - Helpers

    /// Squared distance between point `j` and the query (qx, qy, qz). Same operation order as the
    /// brute-force reference `(a − b)` components summed as `dx² + dy² + dz²`.
    private func squaredDistance(_ j: Int, _ qx: Float, _ qy: Float, _ qz: Float) -> Float {
        let dx: Float = coords[3 * j] - qx
        let dy: Float = coords[3 * j + 1] - qy
        let dz: Float = coords[3 * j + 2] - qz
        return dx * dx + dy * dy + dz * dz
    }

    private static func cellKey(_ point: SIMD3<Float>, cellSize: Float) -> CellKey {
        let scaledX: Float = (point.x / cellSize).rounded(.down)
        let scaledY: Float = (point.y / cellSize).rounded(.down)
        let scaledZ: Float = (point.z / cellSize).rounded(.down)
        return CellKey(x: clampedInt32(scaledX), y: clampedInt32(scaledY), z: clampedInt32(scaledZ))
    }

    /// Keeps keys inside a range that cannot overflow; NaN maps to cell 0.
    private static func clampedInt32(_ value: Float) -> Int32 {
        if value.isNaN { return 0 }
        let limit: Float = 1e9
        let clamped: Float = min(max(value, -limit), limit)
        return Int32(clamped)
    }

    /// Cell offset for index 0, 1, 2 along one axis: 0, −1, +1 (centre first).
    private static func neighborOffset(_ index: Int) -> Int32 {
        if index == 0 { return 0 }
        return index == 1 ? -1 : 1
    }

    private static func chebyshevDistance(_ a: CellKey, _ b: CellKey) -> Int {
        let dx = abs(Int(a.x) - Int(b.x))
        let dy = abs(Int(a.y) - Int(b.y))
        let dz = abs(Int(a.z) - Int(b.z))
        return max(dx, max(dy, dz))
    }

    /// k-th smallest value (1-based k) of an array with at least k entries.
    private static func kthSmallest(_ values: [Float], _ k: Int) -> Float {
        let sorted = values.sorted()
        return sorted[k - 1]
    }
}
