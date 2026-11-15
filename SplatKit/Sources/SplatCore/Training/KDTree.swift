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
    /// Permutation of point indices; each node's range is contiguous in this array. Walking it visits points in tree
    /// order, so queries made in that order touch the same nodes one after another.
    let order: [Int]
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

        // Place the median (by coordinate on axis, then index) at mid; everything before it is not larger,
        // everything after it is not smaller. Quickselect is linear per level, so the build is O(n log n) overall.
        let mid = (lower + upper) / 2
        selectMedian(points: points, order: &order, axis: axis, lower: lower, upper: upper, target: mid)
        let split = component(points[order[mid]], axis)

        // Placeholder; children are appended after it and the entry is completed below.
        nodes.append(Node(lower: lower, upper: upper, axis: axis, split: split, left: -1, right: -1))
        let left = build(points: points, order: &order, nodes: &nodes, lower: lower, upper: mid)
        let right = build(points: points, order: &order, nodes: &nodes, lower: mid, upper: upper)
        nodes[nodeIndex] = Node(lower: lower, upper: upper, axis: axis, split: split,
                                left: left, right: right)
        return nodeIndex
    }

    /// Hoare-partition quickselect on `order[lower..<upper]`: afterwards `order[target]` holds the element that a full
    /// sort by (coordinate on axis, index) would place there, with smaller-or-equal elements before it and
    /// larger-or-equal elements after it. The total order (index as tie-break) keeps every partition step well-defined.
    private static func selectMedian(points: [SIMD3<Float>], order: inout [Int], axis: Int,
                                     lower: Int, upper: Int, target: Int) {
        func less(_ a: Int, _ b: Int) -> Bool {
            let ca = component(points[a], axis)
            let cb = component(points[b], axis)
            return ca < cb || (ca == cb && a < b)
        }
        var lo = lower
        var hi = upper - 1
        while hi > lo {
            let pivot = order[(lo + hi) / 2]
            var i = lo
            var j = hi
            while i <= j {
                while less(order[i], pivot) { i += 1 }
                while less(pivot, order[j]) { j -= 1 }
                if i <= j {
                    order.swapAt(i, j)
                    i += 1
                    j -= 1
                }
            }
            // Now order[lo...j] is not larger than the pivot and order[i...hi] is not smaller; j < i.
            if target <= j {
                hi = j
            } else if target >= i {
                lo = i
            } else {
                // The target lies between j and i, where the elements equal the pivot: already in place.
                return
            }
        }
    }

    // MARK: - Query

    /// Squared distance from `query` to its k-th nearest point, skipping the point with index `excluded` (-1 skips
    /// nothing). With fewer than k points available, the squared distance of the farthest available point is returned
    /// instead. Nil only when no point is available. `scratch` is reused across calls, so a loop over many queries
    /// allocates nothing; its contents afterwards are unspecified.
    public func kthNearestSquaredDistance(to query: SIMD3<Float>, k: Int, excluding excluded: Int,
                                          scratch: inout [Float]) -> Float? {
        guard k > 0, !points.isEmpty else { return nil }
        scratch.removeAll(keepingCapacity: true)
        kthSearch(node: 0, query: query, excluded: excluded, k: k, heap: &scratch)
        // The heap keeps the largest kept value at its root, so the root is the k-th distance, or the farthest
        // available point when fewer than k were found.
        return scratch.first
    }

    private func kthSearch(node nodeIndex: Int, query: SIMD3<Float>, excluded: Int, k: Int,
                           heap: inout [Float]) {
        let node = nodes[nodeIndex]
        if node.axis < 0 {
            for position in node.lower..<node.upper {
                let candidate = order[position]
                if candidate == excluded { continue }
                Self.offerToMaxHeap(Self.squaredDistance(points[candidate], query), capacity: k, heap: &heap)
            }
            return
        }
        let offset = Self.component(query, node.axis) - node.split
        let nearChild = offset < 0 ? node.left : node.right
        let farChild = offset < 0 ? node.right : node.left
        kthSearch(node: nearChild, query: query, excluded: excluded, k: k, heap: &heap)
        if heap.count < k || offset * offset <= heap[0] {
            kthSearch(node: farChild, query: query, excluded: excluded, k: k, heap: &heap)
        }
    }

    /// Keeps the `capacity` smallest values in a max-heap whose root is the largest kept value.
    private static func offerToMaxHeap(_ value: Float, capacity: Int, heap: inout [Float]) {
        if heap.count < capacity {
            heap.append(value)
            var child = heap.count - 1
            while child > 0 {
                let parent = (child - 1) / 2
                if heap[parent] >= heap[child] { break }
                heap.swapAt(parent, child)
                child = parent
            }
        } else if value < heap[0] {
            heap[0] = value
            var parent = 0
            while true {
                let left = 2 * parent + 1
                let right = left + 1
                var largest = parent
                if left < heap.count && heap[left] > heap[largest] { largest = left }
                if right < heap.count && heap[right] > heap[largest] { largest = right }
                if largest == parent { break }
                heap.swapAt(parent, largest)
                parent = largest
            }
        }
    }

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
