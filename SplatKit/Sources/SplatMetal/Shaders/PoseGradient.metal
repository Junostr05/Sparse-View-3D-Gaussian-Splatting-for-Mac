// Camera pose gradient reduction (pose refinement).
//   reducePoseGradient: sums the six per-Gaussian pose contributions written by projectBackward (scratch layout
//                       6 floats per Gaussian: dL/dω, then dL/dτ) into six view sums, with a threadgroup tree
//                       reduction and one float atomic add per threadgroup and component.
// Concatenated after Backward.metal. Dispatched with exactly 256 threads per threadgroup (GPUPoseGradient).

kernel void reducePoseGradient(
    device const float* scratch [[buffer(0)]],
    device atomic_float* sums [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& base [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup float partial[6][256];
    const bool active = gid < count;
    for (uint c = 0u; c < 6u; ++c) {
        partial[c][lid] = active ? scratch[6u * gid + c] : 0.0f;
    }
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < stride) {
            for (uint c = 0u; c < 6u; ++c) {
                partial[c][lid] += partial[c][lid + stride];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        for (uint c = 0u; c < 6u; ++c) {
            atomic_fetch_add_explicit(&sums[base + c], partial[c][0], memory_order_relaxed);
        }
    }
}
