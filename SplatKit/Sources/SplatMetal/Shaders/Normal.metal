// Normal.metal — v1.5: surface-normal consistency loss between the normals of the rendered depth and a
// camera-frame normal prior (MoGe-2). Mirrors SplatCore's `NormalLoss.evaluateDouble`; the Swift wrapper is
// GPUNormalLoss.swift. Concatenated after Depth.metal (see `MetalContext.shaderFiles`).
//
// Rendered depth D = Draw / A is back-projected per pixel centre, P = D · ((x + 0.5 − cx)/fx, (y + 0.5 − cy)/fy, 1).
// The stencil of pixel p = (x, y) uses its right (x + 1, y) and lower (x, y + 1) neighbours:
//   tx = P_r − P_p, ty = P_d − P_p, c = ty × tx, n = c / |c|   (towards the camera)
// A stencil is valid when the prior normal of p is set (|m|² > 0.25), it lies inside the image, all three pixels
// have A > alphaMask (and A > 0) and |c| > 1e-12. The loss is λ · (1/N) Σ (1 − n·m) over the N valid stencils.
//
// Passes, each its own compute encoder (serial, in order):
//   1. normalMoments    per threadgroup partial sums (N, Σ n·m)
//   2. normalFinalize   1 thread: Kahan-sum the partials, loss value, degenerate flag
//   3. normalGradient   per pixel: gathers the gradient of the (up to) three stencils the pixel belongs to and
//                       ADDS (∂L/∂Draw, ∂L/∂A) into the depth gradient written by the depth loss before it.

// Mirrors `NormalLossUniforms` in GPUNormalLoss.swift (48 bytes).
struct NormalLossUniforms {
    float4 intrinsics;  // fx, fy, cx, cy
    float4 params;      // x: λ_n, y: alphaMask, z, w: unused
    uint4 size;         // x: width, y: height, z: minimumPixels, w: unused
};

// Mirrors `NormalPartial` in GPUNormalLoss.swift (32 bytes). One entry per threadgroup.
struct NormalPartial {
    float4 sums;        // x: Σ n·m
    uint4 counts;       // x: N
};

// Mirrors `NormalStats` in GPUNormalLoss.swift (32 bytes).
struct NormalStats {
    uint4 counts;       // x: N, y: degenerate (1/0)
    float4 values;      // x: loss, y: mean cosine
};

inline float3 normalPixelRay(uint x, uint y, float4 k) {
    return float3((float(x) + 0.5f - k.z) / k.x, (float(y) + 0.5f - k.w) / k.y, 1.0f);
}

inline bool normalInMask(float2 raw, float alphaMask) {
    return raw.y > 0.0f && raw.y > alphaMask;
}

// The stencil of pixel (x, y). Returns false when it is not valid.
inline bool normalStencil(device const float2* renderedDepth, device const float4* prior, uint x, uint y,
                          constant NormalLossUniforms& u,
                          thread float3& tx, thread float3& ty, thread float3& c, thread float3& m) {
    uint w = u.size.x;
    uint h = u.size.y;
    if (x + 1u >= w || y + 1u >= h) {
        return false;
    }
    uint p = y * w + x;
    m = prior[p].xyz;
    if (!(dot(m, m) > 0.25f) || !all(isfinite(m))) {
        return false;
    }
    float2 a = renderedDepth[p];
    float2 b = renderedDepth[p + 1u];
    float2 d = renderedDepth[p + w];
    float alphaMask = u.params.y;
    if (!normalInMask(a, alphaMask) || !normalInMask(b, alphaMask) || !normalInMask(d, alphaMask)) {
        return false;
    }
    float3 pointP = (a.x / a.y) * normalPixelRay(x, y, u.intrinsics);
    float3 pointR = (b.x / b.y) * normalPixelRay(x + 1u, y, u.intrinsics);
    float3 pointD = (d.x / d.y) * normalPixelRay(x, y + 1u, u.intrinsics);
    tx = pointR - pointP;
    ty = pointD - pointP;
    c = cross(ty, tx);
    float len = length(c);
    return len > 1e-12f && isfinite(len);
}

// Stage 1: per-threadgroup N and Σ n·m. Every thread reaches the barrier.
kernel void normalMoments(device const float2* renderedDepth [[buffer(0)]],
                          device const float4* prior [[buffer(1)]],
                          device NormalPartial* partials [[buffer(2)]],
                          constant NormalLossUniforms& u [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]],
                          uint2 tgid [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint simdIndex [[simdgroup_index_in_threadgroup]],
                          uint localIndex [[thread_index_in_threadgroup]],
                          uint simdWidth [[threads_per_simdgroup]])
{
    threadgroup float groupCos[32];
    threadgroup uint groupN[32];

    float pixelCos = 0.0f;
    uint pixelN = 0u;
    if (gid.x < u.size.x && gid.y < u.size.y) {
        float3 tx, ty, c, m;
        if (normalStencil(renderedDepth, prior, gid.x, gid.y, u, tx, ty, c, m)) {
            pixelCos = dot(c / length(c), m);
            pixelN = 1u;
        }
    }
    float simdCos = simd_sum(pixelCos);
    uint simdN = simd_sum(pixelN);
    if (lane == 0) {
        groupCos[simdIndex] = simdCos;
        groupN[simdIndex] = simdN;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (localIndex == 0) {
        uint groups = 256u / simdWidth;
        float totalCos = 0.0f;
        uint totalN = 0u;
        for (uint i = 0; i < groups; ++i) {
            totalCos += groupCos[i];
            totalN += groupN[i];
        }
        uint tilesX = (u.size.x + 15u) / 16u;
        uint tgIndex = tgid.y * tilesX + tgid.x;
        partials[tgIndex].sums = float4(totalCos, 0.0f, 0.0f, 0.0f);
        partials[tgIndex].counts = uint4(totalN, 0u, 0u, 0u);
    }
}

// Stage 2: one thread. N, the loss value and the degenerate flag.
kernel void normalFinalize(device const NormalPartial* partials [[buffer(0)]],
                           device NormalStats* stats [[buffer(1)]],
                           constant NormalLossUniforms& u [[buffer(2)]])
{
    uint tilesX = (u.size.x + 15u) / 16u;
    uint tilesY = (u.size.y + 15u) / 16u;
    uint groups = tilesX * tilesY;
    float sumCos = 0.0f, compCos = 0.0f;
    uint n = 0u;
    for (uint g = 0; g < groups; ++g) {
        NormalPartial p = partials[g];
        float y = p.sums.x - compCos;
        float t = sumCos + y;
        compCos = (t - sumCos) - y;
        sumCos = t;
        n += p.counts.x;
    }
    NormalStats s;
    bool degenerate = n == 0u || n < u.size.z;
    s.counts = uint4(n, degenerate ? 1u : 0u, 0u, 0u);
    float meanCos = n > 0u ? sumCos / float(n) : 0.0f;
    float loss = degenerate ? 0.0f : u.params.x * (1.0f - meanCos);
    s.values = float4(loss, meanCos, 0.0f, 0.0f);
    stats[0] = s;
}

// dL/dtx and dL/dty of one valid stencil, for dL/dn = −λ m / N.
inline void normalStencilGradient(float3 tx, float3 ty, float3 c, float3 m, float scale,
                                  thread float3& gTx, thread float3& gTy) {
    float len = length(c);
    float3 n = c / len;
    float3 gN = -scale * m;
    float3 gC = (gN - dot(n, gN) * n) / len;
    gTy = cross(tx, gC);
    gTx = cross(gC, ty);
}

// Stage 3: per pixel q = (x, y), the sum of the contributions of the stencils at (x, y), (x − 1, y) (q is its
// right neighbour) and (x, y − 1) (q is its lower neighbour), chained to (Draw, A) and added to the gradient.
kernel void normalGradient(device const float2* renderedDepth [[buffer(0)]],
                           device const float4* prior [[buffer(1)]],
                           device const NormalStats* stats [[buffer(2)]],
                           device float2* gradient [[buffer(3)]],
                           constant NormalLossUniforms& u [[buffer(4)]],
                           uint2 gid [[thread_position_in_grid]])
{
    uint w = u.size.x;
    uint h = u.size.y;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    NormalStats s = stats[0];
    if (s.counts.y != 0u) {
        return;
    }
    float scale = u.params.x / float(s.counts.x);
    float3 rayQ = normalPixelRay(gid.x, gid.y, u.intrinsics);
    float g = 0.0f;
    bool touched = false;
    float3 tx, ty, c, m, gTx, gTy;
    if (normalStencil(renderedDepth, prior, gid.x, gid.y, u, tx, ty, c, m)) {
        normalStencilGradient(tx, ty, c, m, scale, gTx, gTy);
        g += dot(rayQ, -(gTx + gTy));
        touched = true;
    }
    if (gid.x > 0u && normalStencil(renderedDepth, prior, gid.x - 1u, gid.y, u, tx, ty, c, m)) {
        normalStencilGradient(tx, ty, c, m, scale, gTx, gTy);
        g += dot(rayQ, gTx);
        touched = true;
    }
    if (gid.y > 0u && normalStencil(renderedDepth, prior, gid.x, gid.y - 1u, u, tx, ty, c, m)) {
        normalStencilGradient(tx, ty, c, m, scale, gTx, gTy);
        g += dot(rayQ, gTy);
        touched = true;
    }
    if (touched) {
        uint idx = gid.y * w + gid.x;
        float2 raw = renderedDepth[idx];
        float d = raw.x / raw.y;
        gradient[idx] += float2(g / raw.y, -g * d / raw.y);
    }
}
