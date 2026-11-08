// Depth.metal — Phase 4: depth loss kernels (Pearson correlation and L1 against a per-view depth prior).
//
// Concatenated after Densify.metal (see `MetalContext.shaderFiles`), so everything in Common.metal is
// visible here. Mirrors SplatCore's `DepthLoss.evaluateDouble`; the Swift wrapper is GPUDepthLoss.swift.
//
// Passes, each its own compute encoder (serial, in order):
//   1. depthMomentsA    per threadgroup partial sums (n, Σx, Σy, n1, Σ|D−m|)
//   2. depthFinalizeA   1 thread: Kahan-sum the partials, means, normalised L1 value
//   3. depthMomentsB    per threadgroup centred partial sums Σ(x−μx)², Σ(y−μy)², Σ(x−μx)(y−μy)
//   4. depthFinalizeB   1 thread: σx, σy, ρ, degenerate flag, loss value into the stats slot
//   5. depthGradient    per pixel (∂L/∂Draw, ∂L/∂A), zero outside the mask
//
// Pixel layout: rendered depth is float2 (Draw, A) per pixel, prior target, metric and weight are one
// float per pixel, row-major W·H. Pixel index idx = y·W + x. All reductions use 16×16 threadgroups.
//
// Per-pixel weights w (v1.5 confidence weighting; all 1 without weights) weight every sum: the means,
// variances and covariance are divided by W = Σw over the mask instead of n, the L1 sum by W1 = Σw over
// the measured pixels. With w = 1 the sums and W = n are exact, so the result equals the unweighted loss.

// Mirrors `DepthLossUniforms` in GPUDepthLoss.swift (48 bytes).
struct DepthLossUniforms {
    float4 weights;   // x: λ_P, y: λ_L1, z: alphaMask, w: nearPlane
    float4 extra;     // x: l1Normalizer, y: domain (0 depth, 1 inverse depth), z: L1 applies (1/0), w: unused
    uint4 size;       // x: width, y: height, z: minimumPixels, w: slot
};

// Mirrors `DepthPartial` in GPUDepthLoss.swift (48 bytes). One entry per threadgroup.
struct DepthPartial {
    float4 sums;      // pass A: (Σwx, Σwy, Σw|D−m|, 0); pass B: (Σw(x−μx)², Σw(y−μy)², Σw(x−μx)(y−μy), 0)
    uint4 counts;     // pass A: (n, n1, 0, 0); pass B unused
    float4 weights;   // pass A: (W = Σw, W1 = Σw over measured pixels, 0, 0); pass B unused
};

// Mirrors `DepthStats` in GPUDepthLoss.swift (64 bytes). One entry per slot.
struct DepthStats {
    uint4 counts;     // x: n (valid pixels), y: n1 (measured pixels in the mask), z: degenerate (1/0), w: unused
    float4 means;     // x: μx, y: μy, z: normalised L1 value (0 when not applied), w: unused
    float4 spread;    // x: σx, y: σy, z: ρ (0 when degenerate), w: loss
    float4 weightSums; // x: W (Σw over the mask), y: W1 (Σw over measured pixels), z, w: unused
};

// Compensated (Kahan) float accumulation for the single-thread finalizers.
inline void depthKahanAdd(thread float& sum, thread float& compensation, float value) {
    float y = value - compensation;
    float t = sum + y;
    compensation = (t - sum) - y;
    sum = t;
}

// Mask M: opacity above the threshold (and strictly positive, so D = Draw / A is finite) and a finite,
// positive prior target. Identical in every pass.
inline bool depthMaskPixel(float2 raw, float target, float alphaMask) {
    return raw.y > 0.0f && raw.y > alphaMask && isfinite(target) && target > 0.0f;
}

// x = D (depth domain) or x = 1/max(D, near) (inverse-depth domain), and dx/dD.
inline float depthCorrelationX(float d, float nearPlane, bool inverseDomain) {
    if (!inverseDomain) {
        return d;
    }
    return 1.0f / max(d, nearPlane);
}

inline float depthCorrelationDX(float d, float nearPlane, bool inverseDomain) {
    if (!inverseDomain) {
        return 1.0f;
    }
    return d > nearPlane ? -1.0f / (d * d) : 0.0f;
}

// Stage 1: per-threadgroup partial sums over the mask. Every thread reaches both barriers.
kernel void depthMomentsA(device const float2* renderedDepth [[buffer(0)]],
                          device const float* target [[buffer(1)]],
                          device const float* metric [[buffer(2)]],
                          device DepthPartial* partials [[buffer(3)]],
                          constant DepthLossUniforms& u [[buffer(4)]],
                          device const float* pixelWeights [[buffer(5)]],
                          uint2 gid [[thread_position_in_grid]],
                          uint2 tgid [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint simdIndex [[simdgroup_index_in_threadgroup]],
                          uint localIndex [[thread_index_in_threadgroup]],
                          uint simdWidth [[threads_per_simdgroup]])
{
    threadgroup float groupX[32];
    threadgroup float groupY[32];
    threadgroup float groupAbs[32];
    threadgroup uint groupN[32];
    threadgroup uint groupN1[32];
    threadgroup float groupW[32];
    threadgroup float groupW1[32];

    uint w = u.size.x;
    uint h = u.size.y;
    bool inside = gid.x < w && gid.y < h;
    bool inverseDomain = u.extra.y > 0.5f;
    bool l1Applies = u.extra.z > 0.5f;

    float pixelX = 0.0f;
    float pixelY = 0.0f;
    float pixelAbs = 0.0f;
    uint pixelN = 0u;
    uint pixelN1 = 0u;
    float pixelW = 0.0f;
    float pixelW1 = 0.0f;
    if (inside) {
        uint idx = gid.y * w + gid.x;
        float2 raw = renderedDepth[idx];
        float t = target[idx];
        if (depthMaskPixel(raw, t, u.weights.z)) {
            float d = raw.x / raw.y;
            float weight = pixelWeights[idx];
            pixelN = 1u;
            pixelW = weight;
            pixelX = weight * depthCorrelationX(d, u.weights.w, inverseDomain);
            pixelY = weight * t;
            if (l1Applies) {
                float m = metric[idx];
                if (m > 0.0f) {
                    pixelN1 = 1u;
                    pixelW1 = weight;
                    pixelAbs = weight * fabs(d - m);
                }
            }
        }
    }

    float simdX = simd_sum(pixelX);
    float simdY = simd_sum(pixelY);
    float simdAbs = simd_sum(pixelAbs);
    uint simdN = simd_sum(pixelN);
    uint simdN1 = simd_sum(pixelN1);
    float simdW = simd_sum(pixelW);
    float simdW1 = simd_sum(pixelW1);
    if (lane == 0) {
        groupX[simdIndex] = simdX;
        groupY[simdIndex] = simdY;
        groupAbs[simdIndex] = simdAbs;
        groupN[simdIndex] = simdN;
        groupN1[simdIndex] = simdN1;
        groupW[simdIndex] = simdW;
        groupW1[simdIndex] = simdW1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (localIndex == 0) {
        uint groups = 256u / simdWidth;
        float totalX = 0.0f;
        float totalY = 0.0f;
        float totalAbs = 0.0f;
        uint totalN = 0u;
        uint totalN1 = 0u;
        float totalW = 0.0f;
        float totalW1 = 0.0f;
        for (uint i = 0; i < groups; ++i) {
            totalX += groupX[i];
            totalY += groupY[i];
            totalAbs += groupAbs[i];
            totalN += groupN[i];
            totalN1 += groupN1[i];
            totalW += groupW[i];
            totalW1 += groupW1[i];
        }
        uint tilesX = (w + 15u) / 16u;
        uint tgIndex = tgid.y * tilesX + tgid.x;
        partials[tgIndex].sums = float4(totalX, totalY, totalAbs, 0.0f);
        partials[tgIndex].counts = uint4(totalN, totalN1, 0u, 0u);
        partials[tgIndex].weights = float4(totalW, totalW1, 0.0f, 0.0f);
    }
}

// Stage 2: one thread. Sums the partials with Kahan compensation, writes the means and the normalised
// L1 value into the stats slot.
kernel void depthFinalizeA(device const DepthPartial* partials [[buffer(0)]],
                           device DepthStats* stats [[buffer(1)]],
                           constant DepthLossUniforms& u [[buffer(2)]])
{
    uint tilesX = (u.size.x + 15u) / 16u;
    uint tilesY = (u.size.y + 15u) / 16u;
    uint groups = tilesX * tilesY;

    float sumX = 0.0f, compX = 0.0f;
    float sumY = 0.0f, compY = 0.0f;
    float sumAbs = 0.0f, compAbs = 0.0f;
    float sumW = 0.0f, compW = 0.0f;
    float sumW1 = 0.0f, compW1 = 0.0f;
    uint n = 0u;
    uint n1 = 0u;
    for (uint g = 0; g < groups; ++g) {
        DepthPartial p = partials[g];
        depthKahanAdd(sumX, compX, p.sums.x);
        depthKahanAdd(sumY, compY, p.sums.y);
        depthKahanAdd(sumAbs, compAbs, p.sums.z);
        depthKahanAdd(sumW, compW, p.weights.x);
        depthKahanAdd(sumW1, compW1, p.weights.y);
        n += p.counts.x;
        n1 += p.counts.y;
    }

    DepthStats s;
    s.counts = uint4(n, n1, 0u, 0u);
    s.means = float4(0.0f, 0.0f, 0.0f, 0.0f);
    s.spread = float4(0.0f, 0.0f, 0.0f, 0.0f);
    s.weightSums = float4(sumW, sumW1, 0.0f, 0.0f);
    if (n > 0u && sumW > 0.0f) {
        s.means.x = sumX / sumW;
        s.means.y = sumY / sumW;
    }
    float l1Value = 0.0f;
    if (u.extra.z > 0.5f && n1 > 0u && sumW1 > 0.0f && u.extra.x > 0.0f) {
        l1Value = sumAbs / sumW1 / u.extra.x;
    }
    s.means.z = l1Value;
    stats[0] = s;
}

// Stage 3: per-threadgroup centred second moments over the mask, about the means of stage 2.
kernel void depthMomentsB(device const float2* renderedDepth [[buffer(0)]],
                          device const float* target [[buffer(1)]],
                          device const DepthStats* stats [[buffer(2)]],
                          device DepthPartial* partials [[buffer(3)]],
                          constant DepthLossUniforms& u [[buffer(4)]],
                          device const float* pixelWeights [[buffer(5)]],
                          uint2 gid [[thread_position_in_grid]],
                          uint2 tgid [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint simdIndex [[simdgroup_index_in_threadgroup]],
                          uint localIndex [[thread_index_in_threadgroup]],
                          uint simdWidth [[threads_per_simdgroup]])
{
    threadgroup float groupXX[32];
    threadgroup float groupYY[32];
    threadgroup float groupXY[32];

    uint w = u.size.x;
    uint h = u.size.y;
    bool inside = gid.x < w && gid.y < h;
    bool inverseDomain = u.extra.y > 0.5f;
    float meanX = stats[0].means.x;
    float meanY = stats[0].means.y;

    float pixelXX = 0.0f;
    float pixelYY = 0.0f;
    float pixelXY = 0.0f;
    if (inside) {
        uint idx = gid.y * w + gid.x;
        float2 raw = renderedDepth[idx];
        float t = target[idx];
        if (depthMaskPixel(raw, t, u.weights.z)) {
            float d = raw.x / raw.y;
            float dx = depthCorrelationX(d, u.weights.w, inverseDomain) - meanX;
            float dy = t - meanY;
            float weight = pixelWeights[idx];
            pixelXX = weight * (dx * dx);
            pixelYY = weight * (dy * dy);
            pixelXY = weight * (dx * dy);
        }
    }

    float simdXX = simd_sum(pixelXX);
    float simdYY = simd_sum(pixelYY);
    float simdXY = simd_sum(pixelXY);
    if (lane == 0) {
        groupXX[simdIndex] = simdXX;
        groupYY[simdIndex] = simdYY;
        groupXY[simdIndex] = simdXY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (localIndex == 0) {
        uint groups = 256u / simdWidth;
        float totalXX = 0.0f;
        float totalYY = 0.0f;
        float totalXY = 0.0f;
        for (uint i = 0; i < groups; ++i) {
            totalXX += groupXX[i];
            totalYY += groupYY[i];
            totalXY += groupXY[i];
        }
        uint tilesX = (w + 15u) / 16u;
        uint tgIndex = tgid.y * tilesX + tgid.x;
        partials[tgIndex].sums = float4(totalXX, totalYY, totalXY, 0.0f);
        partials[tgIndex].counts = uint4(0u, 0u, 0u, 0u);
        partials[tgIndex].weights = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

// Stage 4: one thread. σx, σy, ρ, the degenerate flag and the loss value. Degenerate (too few pixels or
// a near-constant signal) skips the Pearson term; the L1 term still applies when it is enabled, as in
// the CPU reference.
kernel void depthFinalizeB(device const DepthPartial* partials [[buffer(0)]],
                           device DepthStats* stats [[buffer(1)]],
                           constant DepthLossUniforms& u [[buffer(2)]])
{
    uint tilesX = (u.size.x + 15u) / 16u;
    uint tilesY = (u.size.y + 15u) / 16u;
    uint groups = tilesX * tilesY;

    float sumXX = 0.0f, compXX = 0.0f;
    float sumYY = 0.0f, compYY = 0.0f;
    float sumXY = 0.0f, compXY = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        DepthPartial p = partials[g];
        depthKahanAdd(sumXX, compXX, p.sums.x);
        depthKahanAdd(sumYY, compYY, p.sums.y);
        depthKahanAdd(sumXY, compXY, p.sums.z);
    }

    DepthStats s = stats[0];
    uint n = s.counts.x;
    float nd = s.weightSums.x;
    float lambdaP = u.weights.x;
    float lambdaL1 = u.weights.y;

    float sx = 0.0f;
    float sy = 0.0f;
    float rho = 0.0f;
    float pearsonTerm = 0.0f;
    bool degenerate = true;
    if (n > 0u && n >= u.size.z && nd > 0.0f) {
        float varX = sumXX / nd;
        float varY = sumYY / nd;
        float covXY = sumXY / nd;
        sx = sqrt(varX);
        sy = sqrt(varY);
        bool flatX = !(sx > 1e-6f * (fabs(s.means.x) + 1e-6f));
        bool flatY = !(sy > 1e-6f * (fabs(s.means.y) + 1e-6f));
        if (!flatX && !flatY) {
            degenerate = false;
            rho = covXY / (sx * sy);
            pearsonTerm = lambdaP * (1.0f - rho);
        }
    }

    float loss = pearsonTerm + lambdaL1 * s.means.z;
    s.counts.z = degenerate ? 1u : 0u;
    s.spread = float4(sx, sy, rho, loss);
    stats[0] = s;
}

// Stage 5: per-pixel gradient with respect to (Draw, A). Mirrors the CPU chain rule: the Pearson term
// goes through dx/dD, the L1 term acts on D directly, and D = Draw / A.
kernel void depthGradient(device const float2* renderedDepth [[buffer(0)]],
                          device const float* target [[buffer(1)]],
                          device const float* metric [[buffer(2)]],
                          device const DepthStats* stats [[buffer(3)]],
                          device float2* gradient [[buffer(4)]],
                          constant DepthLossUniforms& u [[buffer(5)]],
                          device const float* pixelWeights [[buffer(6)]],
                          uint2 gid [[thread_position_in_grid]])
{
    uint w = u.size.x;
    uint h = u.size.y;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    uint idx = gid.y * w + gid.x;
    float2 raw = renderedDepth[idx];
    float t = target[idx];

    float2 result = float2(0.0f, 0.0f);
    if (depthMaskPixel(raw, t, u.weights.z)) {
        float nearPlane = u.weights.w;
        bool inverseDomain = u.extra.y > 0.5f;
        float d = raw.x / raw.y;
        float x = depthCorrelationX(d, nearPlane, inverseDomain);
        float dxdD = depthCorrelationDX(d, nearPlane, inverseDomain);

        DepthStats s = stats[0];
        float weight = pixelWeights[idx];
        float g = 0.0f;
        if (s.counts.z == 0u) {
            float nd = s.weightSums.x;
            float sx = s.spread.x;
            float sy = s.spread.y;
            float rho = s.spread.z;
            float xhat = (x - s.means.x) / sx;
            float yhat = (t - s.means.y) / sy;
            float dTermDx = -u.weights.x * weight * (yhat - rho * xhat) / (nd * sx);
            g = dTermDx * dxdD;
        }
        if (u.extra.z > 0.5f) {
            float m = metric[idx];
            if (m > 0.0f && s.weightSums.y > 0.0f) {
                float diff = d - m;
                float sgn = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
                g += u.weights.y * weight * sgn / (s.weightSums.y * u.extra.x);
            }
        }
        result = float2(g / raw.y, -g * d / raw.y);
    }
    gradient[idx] = result;
}
