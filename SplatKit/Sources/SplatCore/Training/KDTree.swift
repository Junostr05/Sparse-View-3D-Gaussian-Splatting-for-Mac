import Foundation

/// Static k-d tree over 3D points for exact k-nearest-neighbour queries.
///
/// Built once from a point array; each node covers a contiguous range of a permutation array.
/// Queries are exact and deterministic: neighbours are ordered by ascending squared distance,
/// with ties broken by the smaller point index.
public struct KDTree: Sendable {
    private struct Node: Sendable {
        /// Range `lower..<upper` of `order` covered by this node.
        var lower: Int
        var upper: Int
        /// Split axis (0, 1, 2), or -1 for a leaf.
        var axis: Int
        /// Coordinate of the median point on `axis`.
        var split: Float
        /// Child node indices; unused for leaves.
        var left: Int
        var right: Int
    }

    private struct Candidate {
        var distance: Float
        var index: Int

        /// Strict "comes after" order: larger distance first, then larger index.
        func isWorse(than other: Candidate) -> Bool {
            distance > other.distance || (distance == other.distance && index > other.index)
        }
    }

    /// Bounded max-heap holding the k best candidates; the root is the worst of them.
    private struct BoundedHeap {
        let capacity: Int
        private(set) var entries: [Candidate] = []

        init(capacity: Int) {
            self.capacity = capacity
            entries.reserveCapacity(capacity)
        }

        var isFull: Bool { entries.count == capacity }

        /// Squared distance of the current worst kept candidate. Only meaningful when full.
        var worstDistance: Float { entries[0].distance }

        mutating func offer(_ candidate: Candidate) {
            guard capacity > 0 else { return }
            if entries.count < capacity {
                entries.append(candidate)
                siftUp(entries.count - 1)
            } else if entries[0].isWorse(than: candidate) {
                entries[0] = candidate
                siftDown(0)
            }
        }

        /// Kept indices sorted by (distance, index) ascending.
        func sortedIndices() -> [Int] {
            entries
                .sorted { lhs, rhs in
                    lhs.distance < rhs.distance || (lhs.distance == rhs.distance && lhs.index < rhs.index)
                }
                .map { $0.index }
        }

        private mutating func siftUp(_ start: Int) {
            var child = start
            while child > 0 {
                let parent = (child - 1) / 2
                guard entries[child].isWorse(than: entries[parent]) else { break }
                entries.swapAt(child, parent)
                child = parent
            }
        }

        private mutating func siftDown(_ start: Int) {
            var parent = start
            while true {
                let left = 2 * parent + 1
                let right = left + 1
                var worst = parent
                if left < entries.count, entries[left].isWorse(than: entries[worst]) {
                    worst = left
                }
                if right < entries.count, entries[right].isWorse(than: entries[worst]) {
                    worst = right
                }
                if worst == parent { break }
                entries.swapAt(parent, worst)
                parent = worst
            }
        }
    }

    /// Maximum number of points stored in a leaf.
    private static let leafSize = 8

    private let points: [SIMD3<Float>]
    /// Permutation of point indices; each node's range is contiguous in this array.
    private let order: [Int]
    /// Node 0 is the root.
    private let nodes: [Node]

    public init(points: [SIMD3<Float>]) {
        var order = Array(0..<points.count)
        var nodes: [Node] = []
        _ = Self.build(points: points, order: &order, nodes: &nodes,
                       lower: 0, upper: points.count)
        self.points = points
        self.order = order
        self.nodes = nodes
    }

    /// Indices of the k nearest points to points[index], excluding index itself; ties broken by
    /// smaller index; fewer if count − 1 < k.
    public func nearestNeighbors(of index: Int, k: Int) -> [Int] {
        precondition(points.indices.contains(index), "point index out of range")
        guard k > 0, points.count > 1 else { return [] }
        var heap = BoundedHeap(capacity: k)
        search(node: 0, query: points[index], excluding: index, heap: &heap)
        return heap.sortedIndices()
    }

    /// Indices of the k nearest points to an arbitrary location, ordered by ascending distance (ties by smaller index).
    /// Fewer than k when the tree holds fewer points.
    public func nearestNeighbors(to query: SIMD3<Float>, k: Int) -> [Int] {
        guard k > 0, !points.isEmpty else { return [] }
        var heap = BoundedHeap(capacity: k)
        search(node: 0, query: query, excluding: -1, heap: &heap)
        return heap.sortedIndices()
    }

    // MARK: - Build

    /// Recursively builds the subtree for `order[lower..<upper]` and returns its node index.
    private static func build(points: [SIMD3<Float>],
                              order: inout [Int],
                              nodes: inout [Node],
                              lower: Int,
                              upper: Int) -> Int {
        let nodeIndex = nodes.count
        if upper - lower <= leafSize {
            nodes.append(Node(lower: lower, upper: upper, axis: -1, split: 0, left: -1, right: -1))
            return nodeIndex
        }

        // Split on the axis of largest extent of the range's bounding box.
        var low = points[order[lower]]
        var high = low
        for position in lower..<upper {
            let p = points[order[position]]
            low = SIMD3<Float>(min(low.x, p.x), min(low.y, p.y), min(low.z, p.z))
            high = SIMD3<Float>(max(high.x, p.x), max(high.y, p.y), max(high.z, p.z))
        }
        let extent = high - low
        let axis: Int
        if extent.x >= extent.y && extent.x >= extent.z {
            axis = 0
        } else if extent.y >= extent.z {
            axis = 1
        } else {
            axis = 2
        }

        // Order the range by (coordinate on axis, index) and split at the median.
        order[lower..<upper].sort { a, b in
            let ca = component(points[a], axis)
            let cb = component(points[b], axis)
            return ca < cb || (ca == cb && a < b)
        }
        let mid = (lower + upper) / 2
        let split = component(points[order[mid]], axis)

        // Placeholder; children are appended after it and the entry is completed below.
        nodes.append(Node(lower: lower, upper: upper, axis: axis, split: split, left: -1, right: -1))
        let left = build(points: points, order: &order, nodes: &nodes, lower: lower, upper: mid)
        let right = build(points: points, order: &order, nodes: &nodes, lower: mid, upper: upper)
        nodes[nodeIndex] = Node(lower: lower, upper: upper, axis: axis, split: split,
                                left: left, right: right)
        return nodeIndex
    }

    // MARK: - Query

    private func search(node nodeIndex: Int,
                        query: SIMD3<Float>,
                        excluding excluded: Int,
                        heap: inout BoundedHeap) {
        let node = nodes[nodeIndex]
        if node.axis < 0 {
            for position in node.lower..<node.upper {
                let candidate = order[position]
                if candidate == excluded { continue }
                let distance = Self.squaredDistance(points[candidate], query)
                heap.offer(Candidate(distance: distance, index: candidate))
            }
            return
        }

        // Points on the near side are visited first. Every point on the far side lies at least
        // |offset| away from the query, so the far child is skipped only when that bound is
        // strictly worse than the current k-th distance (equal distances may still win on index).
        let offset = Self.component(query, node.axis) - node.split
        let nearChild = offset < 0 ? node.left : node.right
        let farChild = offset < 0 ? node.right : node.left
        search(node: nearChild, query: query, excluding: excluded, heap: &heap)
        if !heap.isFull || offset * offset <= heap.worstDistance {
            search(node: farChild, query: query, excluding: excluded, heap: &heap)
        }
    }

    // MARK: - Helpers

    /// Squared Euclidean distance. Used by both the tree and brute-force verification, so the
    /// operation order is fixed.
    private static func squaredDistance(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        let delta = a - b
        return delta.x * delta.x + delta.y * delta.y + delta.z * delta.z
    }

    private static func component(_ p: SIMD3<Float>, _ axis: Int) -> Float {
        switch axis {
        case 0: return p.x
        case 1: return p.y
        default: return p.z
        }
    }
}
