// Stable LSD radix sort of ulong keys with uint values, 8-bit digits.
// Block = SORT_BLOCK elements processed by SORT_THREADS threads in SORT_CHUNKS chunks.
// Requires a simdgroup width of 32 (SORT_SIMDGROUPS = SORT_THREADS / 32).

// params: x = count, y = shift, z = numBlocks, w = unused
kernel void radixHistogram(
    device const ulong* keys [[buffer(0)]],
    device uint* hist [[buffer(1)]],
    constant uint4& params [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]])
{
    threadgroup atomic_uint sHist[256];
    atomic_store_explicit(&sHist[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong count = ulong(params.x);
    uint shift = params.y;
    ulong base = ulong(block) * ulong(SORT_BLOCK);
    for (uint i = 0; i < SORT_CHUNKS; ++i) {
        ulong idx = base + ulong(i) * ulong(SORT_THREADS) + ulong(lid);
        if (idx < count) {
            uint d = uint((keys[idx] >> shift) & 0xFFul);
            atomic_fetch_add_explicit(&sHist[d], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Digit-major layout: hist[digit * numBlocks + block].
    hist[lid * params.z + block] = atomic_load_explicit(&sHist[lid], memory_order_relaxed);
}

// offsets: exclusive scan of the digit-major histogram.
kernel void radixScatter(
    device const ulong* inKeys [[buffer(0)]],
    device const uint* inValues [[buffer(1)]],
    device ulong* outKeys [[buffer(2)]],
    device uint* outValues [[buffer(3)]],
    device const uint* offsets [[buffer(4)]],
    constant uint4& params [[buffer(5)]],
    uint lid [[thread_position_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint sCnt[SORT_SIMDGROUPS * 256];
    threadgroup uint sCounter[256];

    ulong count = ulong(params.x);
    uint shift = params.y;
    uint numBlocks = params.z;
    ulong base = ulong(block) * ulong(SORT_BLOCK);

    sCounter[lid] = offsets[lid * numBlocks + block];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong laneLess = (ulong(1) << ulong(lane)) - ulong(1);

    for (uint chunk = 0; chunk < SORT_CHUNKS; ++chunk) {
        ulong idx = base + ulong(chunk) * ulong(SORT_THREADS) + ulong(lid);
        bool valid = idx < count;
        ulong key = 0ul;
        uint val = 0u;
        if (valid) {
            key = inKeys[idx];
            val = inValues[idx];
        }
        uint d = uint((key >> shift) & 0xFFul);

        for (uint i = 0; i < SORT_SIMDGROUPS; ++i) {
            sCnt[i * 256 + lid] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Lanes of this simdgroup that are valid and share this thread's digit.
        ulong mask = static_cast<ulong>(simd_ballot(valid));
        for (uint b = 0; b < 8; ++b) {
            bool bit = ((d >> b) & 1u) != 0u;
            ulong ballot = static_cast<ulong>(simd_ballot(valid && bit));
            mask &= bit ? ballot : ~ballot;
        }
        uint rank = uint(popcount(mask & laneLess));
        if (valid && rank == 0u) {
            sCnt[sg * 256 + d] = uint(popcount(mask));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {
            uint prior = 0u;
            for (uint s = 0; s < sg; ++s) {
                prior += sCnt[s * 256 + d];
            }
            uint pos = sCounter[d] + prior + rank;
            outKeys[pos] = key;
            outValues[pos] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Thread lid owns digit lid: advance its running counter.
        uint total = 0u;
        for (uint s = 0; s < SORT_SIMDGROUPS; ++s) {
            total += sCnt[s * 256 + lid];
        }
        sCounter[lid] += total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
