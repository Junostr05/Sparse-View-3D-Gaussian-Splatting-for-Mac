// Sampling rate of the Mip-Splatting 3D filter (CPU reference: SamplingRate.compute in SplatCore).
//   samplingRateCompute: per Gaussian, the largest fx / z over the training cameras that see its centre
//                        (depth > near, centre within the image enlarged by 15 % per side); writes 0 when none.
//                        Sums and counts the visible values with float/uint atomics.
//   samplingRateFill:    replaces the zeros by the mean over the visible Gaussians (1 when nothing is visible).
// Concatenated after Regularizers.metal, so Common.metal's structs are available.

// Mirrors `SamplingCameraUniforms` in ShaderTypes.swift (80 bytes).
struct SamplingCamera {
    float4 rotRow0;      // xyz: row 0 of R, w: t.x
    float4 rotRow1;      // xyz: row 1 of R, w: t.y
    float4 rotRow2;      // xyz: row 2 of R, w: t.z
    float4 intrinsics;   // fx, fy, cx, cy
    float4 size;         // width, height, near, unused
};

// Mirrors `SamplingUniforms` in ShaderTypes.swift (16 bytes).
struct SamplingUniforms {
    uint4 counts;        // Gaussian count, camera count, unused, unused
};

kernel void samplingRateCompute(
    device const float* positions [[buffer(0)]],
    device const SamplingCamera* cameras [[buffer(1)]],
    constant SamplingUniforms& u [[buffer(2)]],
    device float* rates [[buffer(3)]],
    device atomic_float* visibleSum [[buffer(4)]],
    device atomic_uint* visibleCount [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.counts.x) return;
    float3 pw = float3(positions[3 * gid], positions[3 * gid + 1], positions[3 * gid + 2]);
    const float margin = 0.15f;

    float best = 0.0f;
    for (uint c = 0u; c < u.counts.y; ++c) {
        SamplingCamera cam = cameras[c];
        float3 pc = float3(dot(cam.rotRow0.xyz, pw) + cam.rotRow0.w,
                           dot(cam.rotRow1.xyz, pw) + cam.rotRow1.w,
                           dot(cam.rotRow2.xyz, pw) + cam.rotRow2.w);
        float z = pc.z;
        if (!(z > cam.size.z)) continue;
        float fx = cam.intrinsics.x;
        float fy = cam.intrinsics.y;
        float cx = cam.intrinsics.z;
        float cy = cam.intrinsics.w;
        float W = cam.size.x;
        float H = cam.size.y;
        float meanX = fx * pc.x / z + cx;
        float meanY = fy * pc.y / z + cy;
        if (meanX < -margin * W || meanX > (1.0f + margin) * W) continue;
        if (meanY < -margin * H || meanY > (1.0f + margin) * H) continue;
        best = max(best, fx / z);
    }

    rates[gid] = best;
    if (best > 0.0f) {
        atomic_fetch_add_explicit(visibleSum, best, memory_order_relaxed);
        atomic_fetch_add_explicit(visibleCount, 1u, memory_order_relaxed);
    }
}

kernel void samplingRateFill(
    device float* rates [[buffer(0)]],
    device atomic_float* visibleSum [[buffer(1)]],
    device atomic_uint* visibleCount [[buffer(2)]],
    constant SamplingUniforms& u [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.counts.x) return;
    if (rates[gid] > 0.0f) return;
    uint visible = atomic_load_explicit(visibleCount, memory_order_relaxed);
    float sum = atomic_load_explicit(visibleSum, memory_order_relaxed);
    rates[gid] = visible > 0u ? sum / float(visible) : 1.0f;
}
