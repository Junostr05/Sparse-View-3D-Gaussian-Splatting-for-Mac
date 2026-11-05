// Loss kernels: L1 + D-SSIM forward and backward (Phase 3, Task 13).
// Concatenated after Backward (see MetalContext.shaderFiles); mirrors SplatCore's ImageLoss.
//
// Layout: planar float32 scratch, plane = W·H floats. Pixel index idx = y·W + x.
// Separable 11-tap Gaussian blur with zero padding, applied as a horizontal pass followed by a
// vertical pass. Every kernel writes whole pixels (or whole planes entries) from one thread.

// Mirrors `LossUniforms` in GPUImageLoss.swift (48 bytes).
struct LossUniforms {
    uint4 size;          // width, height, tilesX (16×16 groups), unused
    float4 scales;       // x: l1 gradient scale (1−λ)/N, y: SSIM gradient scale −λ/N, z: λ, w: unused
    float4 stability;    // x: C1, y: C2, z/w unused
};

// Vertical pass of the separable blur over one plane. `plane` points at the first element of the plane.
inline float lossBlurVertical(device const float* plane, uint w, uint h, uint x, uint y,
                              constant float* taps) {
    float acc = 0.0f;
    for (int k = 0; k < 11; ++k) {
        int sy = int(y) + k - 5;
        if (sy >= 0 && sy < int(h)) {
            acc += taps[k] * plane[uint(sy) * w + x];
        }
    }
    return acc;
}

// Stage 1: horizontal blur of {x, y, x², y², xy} for each channel.
// moments plane index: c·5 + m, with m = 0 x, 1 y, 2 x², 3 y², 4 xy. Writes 15 planes.
kernel void ssimMomentsH(texture2d<float, access::read> rendered [[texture(0)]],
                         device const float4* target [[buffer(0)]],
                         device float* moments [[buffer(1)]],
                         constant LossUniforms& u [[buffer(2)]],
                         constant float* taps [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]])
{
    uint w = u.size.x;
    uint h = u.size.y;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    uint n = w * h;
    uint row = gid.y * w;

    float3 sumX = float3(0.0f);
    float3 sumY = float3(0.0f);
    float3 sumXX = float3(0.0f);
    float3 sumYY = float3(0.0f);
    float3 sumXY = float3(0.0f);
    for (int k = 0; k < 11; ++k) {
        int sx = int(gid.x) + k - 5;
        if (sx < 0 || sx >= int(w)) {
            continue;
        }
        uint2 p = uint2(uint(sx), gid.y);
        float3 xv = rendered.read(p).xyz;
        float3 yv = target[row + p.x].xyz;
        float tap = taps[k];
        sumX += tap * xv;
        sumY += tap * yv;
        sumXX += tap * (xv * xv);
        sumYY += tap * (yv * yv);
        sumXY += tap * (xv * yv);
    }

    uint idx = row + gid.x;
    for (uint c = 0; c < 3; ++c) {
        uint base = c * 5u * n + idx;
        moments[base] = sumX[c];
        moments[base + n] = sumY[c];
        moments[base + 2u * n] = sumXX[c];
        moments[base + 3u * n] = sumYY[c];
        moments[base + 4u * n] = sumXY[c];
    }
}

// Stage 2: vertical blur of the 15 moment planes, SSIM map S, and the 9 per-pixel SSIM backward
// terms M (already multiplied by the gradient scale g = −λ/N). Plane index of M: c·3 + j.
// Also writes per-threadgroup partial sums (Σ S over channels, Σ |x−y| over channels).
kernel void ssimMapsV(texture2d<float, access::read> rendered [[texture(0)]],
                      device const float4* target [[buffer(0)]],
                      device const float* moments [[buffer(1)]],
                      device float* maps [[buffer(2)]],
                      device float2* partials [[buffer(3)]],
                      constant LossUniforms& u [[buffer(4)]],
                      constant float* taps [[buffer(5)]],
                      uint2 gid [[thread_position_in_grid]],
                      uint2 tgid [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint simdIndex [[simdgroup_index_in_threadgroup]],
                      uint localIndex [[thread_index_in_threadgroup]],
                      uint simdWidth [[threads_per_simdgroup]])
{
    threadgroup float groupS[32];
    threadgroup float groupL[32];

    uint w = u.size.x;
    uint h = u.size.y;
    uint n = w * h;
    bool inside = gid.x < w && gid.y < h;

    float pixelS = 0.0f;
    float pixelL = 0.0f;
    if (inside) {
        uint x = gid.x;
        uint y = gid.y;
        uint idx = y * w + x;
        float3 xv = rendered.read(gid).xyz;
        float3 yv = target[idx].xyz;
        float3 absDiff = abs(xv - yv);
        pixelL = absDiff.x + absDiff.y + absDiff.z;

        float c1 = u.stability.x;
        float c2 = u.stability.y;
        float ssimScale = u.scales.y;

        for (uint c = 0; c < 3; ++c) {
            device const float* base = moments + c * 5u * n;
            float mx = lossBlurVertical(base, w, h, x, y, taps);
            float my = lossBlurVertical(base + n, w, h, x, y, taps);
            float exx = lossBlurVertical(base + 2u * n, w, h, x, y, taps);
            float eyy = lossBlurVertical(base + 3u * n, w, h, x, y, taps);
            float exy = lossBlurVertical(base + 4u * n, w, h, x, y, taps);

            float mx2 = mx * mx;
            float my2 = my * my;
            float sxx = exx - mx2;
            float syy = eyy - my2;
            float sxy = exy - mx * my;

            float a1 = 2.0f * mx * my + c1;
            float a2 = 2.0f * sxy + c2;
            float b1 = mx2 + my2 + c1;
            float b2 = sxx + syy + c2;

            float numerator = a1 * a2;
            float denominator = b1 * b2;
            float s = numerator / denominator;
            pixelS += s;

            float term1 = 2.0f * my / a1;
            float term2 = 2.0f * my / a2;
            float term3 = 2.0f * mx / b1;
            float term4 = 2.0f * mx / b2;
            float muTerm = term1 - term2 - term3 + term4;

            float m0 = ssimScale * s * muTerm;
            float m1 = ssimScale * (-s / b2);
            float m2 = ssimScale * (2.0f * s / a2);

            maps[(c * 3u + 0u) * n + idx] = m0;
            maps[(c * 3u + 1u) * n + idx] = m1;
            maps[(c * 3u + 2u) * n + idx] = m2;
        }
    }

    // Threadgroup reduction. Every thread reaches both barriers (no early return above).
    float simdS = simd_sum(pixelS);
    float simdL = simd_sum(pixelL);
    if (lane == 0) {
        groupS[simdIndex] = simdS;
        groupL[simdIndex] = simdL;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (localIndex == 0) {
        uint groups = 256u / simdWidth;
        float totalS = 0.0f;
        float totalL = 0.0f;
        for (uint i = 0; i < groups; ++i) {
            totalS += groupS[i];
            totalL += groupL[i];
        }
        uint tgIndex = tgid.y * u.size.z + tgid.x;
        partials[tgIndex] = float2(totalS, totalL);
    }
}

// Stage 3: horizontal blur of the 9 SSIM backward planes. Reads `src` (maps), writes `dst`
// (the first 9 planes of the moment scratch, which are free after stage 2).
kernel void blurH9(device const float* src [[buffer(0)]],
                   device float* dst [[buffer(1)]],
                   constant LossUniforms& u [[buffer(2)]],
                   constant float* taps [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]])
{
    uint w = u.size.x;
    uint h = u.size.y;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    uint n = w * h;
    uint row = gid.y * w;
    uint idx = row + gid.x;
    for (uint p = 0; p < 9; ++p) {
        device const float* plane = src + p * n;
        float acc = 0.0f;
        for (int k = 0; k < 11; ++k) {
            int sx = int(gid.x) + k - 5;
            if (sx >= 0 && sx < int(w)) {
                acc += taps[k] * plane[row + uint(sx)];
            }
        }
        dst[p * n + idx] = acc;
    }
}

// Stage 4: vertical blur of the blurred planes, fused with
// dL/dx = (1−λ)/N·sign(x−y) + B0 + 2x·B1 + y·B2 for each channel.
// Writes one RGBA float per pixel with A = 0.
kernel void lossGradient(texture2d<float, access::read> rendered [[texture(0)]],
                         device const float4* target [[buffer(0)]],
                         device const float* blurred [[buffer(1)]],
                         device float4* gradient [[buffer(2)]],
                         constant LossUniforms& u [[buffer(3)]],
                         constant float* taps [[buffer(4)]],
                         uint2 gid [[thread_position_in_grid]])
{
    uint w = u.size.x;
    uint h = u.size.y;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    uint n = w * h;
    uint idx = gid.y * w + gid.x;
    float3 xv = rendered.read(gid).xyz;
    float3 yv = target[idx].xyz;
    float l1Scale = u.scales.x;

    float3 result = float3(0.0f);
    for (uint c = 0; c < 3; ++c) {
        float b0 = lossBlurVertical(blurred + (c * 3u + 0u) * n, w, h, gid.x, gid.y, taps);
        float b1 = lossBlurVertical(blurred + (c * 3u + 1u) * n, w, h, gid.x, gid.y, taps);
        float b2 = lossBlurVertical(blurred + (c * 3u + 2u) * n, w, h, gid.x, gid.y, taps);
        float x = xv[c];
        float y = yv[c];
        float d = x - y;
        float sgn = (d > 0.0f) ? 1.0f : ((d < 0.0f) ? -1.0f : 0.0f);
        result[c] = l1Scale * sgn + b0 + 2.0f * x * b1 + y * b2;
    }
    gradient[idx] = float4(result, 0.0f);
}
