import Metal
import SplatCore

// CPU compaction of Gaussian stores (Phase 4, Task 16). Used for the parameters and for both Adam moment
// stores with the same mask, so the three stay aligned. Isolation pruning decides the mask on the CPU.

extension GaussianBuffers {
    /// Removes the Gaussians whose `keep` flag is false. The survivors keep their relative order and move to
    /// the front of every group in place, and `count` becomes the number kept. Spare capacity beyond the new
    /// count is not cleared. `keep.count` must equal `count`, and at least one Gaussian must be kept, so the
    /// model is never emptied. GPU work using these buffers must have completed.
    public func compact(keeping keep: [Bool]) {
        let total = count
        precondition(keep.count == total, "mask has \(keep.count) entries for \(total) Gaussians")
        var kept = 0
        for flag in keep where flag {
            kept += 1
        }
        precondition(kept > 0 || total == 0, "compaction must keep at least one Gaussian")
        if kept == total {
            return
        }

        let floats = GaussianBuffers.floatsPerGaussian(shDegree: shDegree)
        let buffers = groups
        for group in 0..<buffers.count {
            let stride = floats[group] * MemoryLayout<Float>.size
            let base = buffers[group].contents()
            var write = 0
            for read in 0..<total where keep[read] {
                if write != read {
                    // Source and destination slots never overlap, because write < read.
                    (base + write * stride).copyMemory(from: base + read * stride, byteCount: stride)
                }
                write += 1
            }
        }
        count = kept
    }

    /// Positions of the first `count` Gaussians. GPU work must have completed.
    public func readPositions() -> [SIMD3<Float>] {
        let n = count
        guard n > 0 else { return [] }
        let pointer = positions.contents().bindMemory(to: Float.self, capacity: 3 * n)
        var result: [SIMD3<Float>] = []
        result.reserveCapacity(n)
        for i in 0..<n {
            result.append(SIMD3<Float>(pointer[3 * i], pointer[3 * i + 1], pointer[3 * i + 2]))
        }
        return result
    }
}
