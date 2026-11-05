// Exclusive scan of uint. Block = SCAN_BLOCK elements, SCAN_THREADS threads, 4 elements per thread.

kernel void scanBlocks(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device uint* blockSums [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint numSG [[simdgroups_per_threadgroup]])
{
    threadgroup uint sgTotals[32];

    ulong base = ulong(block) * ulong(SCAN_BLOCK) + ulong(lid) * 4ul;
    uint v0 = (base + 0ul < ulong(count)) ? input[base + 0ul] : 0u;
    uint v1 = (base + 1ul < ulong(count)) ? input[base + 1ul] : 0u;
    uint v2 = (base + 2ul < ulong(count)) ? input[base + 2ul] : 0u;
    uint v3 = (base + 3ul < ulong(count)) ? input[base + 3ul] : 0u;
    uint threadTotal = v0 + v1 + v2 + v3;

    uint simdExclusive = simd_prefix_exclusive_sum(threadTotal);
    uint simdTotal = simd_sum(threadTotal);
    if (lane == 0) {
        sgTotals[sg] = simdTotal;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sgOffset = 0;
    for (uint s = 0; s < sg; ++s) {
        sgOffset += sgTotals[s];
    }
    uint prefix = sgOffset + simdExclusive;

    if (base + 0ul < ulong(count)) { output[base + 0ul] = prefix; }
    prefix += v0;
    if (base + 1ul < ulong(count)) { output[base + 1ul] = prefix; }
    prefix += v1;
    if (base + 2ul < ulong(count)) { output[base + 2ul] = prefix; }
    prefix += v2;
    if (base + 3ul < ulong(count)) { output[base + 3ul] = prefix; }

    if (lid == 0) {
        uint total = 0;
        for (uint s = 0; s < numSG; ++s) {
            total += sgTotals[s];
        }
        blockSums[block] = total;
    }
}

kernel void addBlockOffsets(
    device uint* output [[buffer(0)]],
    device const uint* scannedBlockSums [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    output[gid] += scannedBlockSums[gid / uint(SCAN_BLOCK)];
}
