// Backward pass of the tile rasterizer.
//   rasterizeBackward: per tile, per pixel, accumulates the screen-space gradients of every blended
//                      Gaussian into a per-Gaussian slot (12 floats, float atomics).
//   projectBackward:   per Gaussian, recomputes the projection and applies the chain rule down to the
//                      stored parameters.
// The math mirrors CPUBackward.swift (pixel stage and GaussianChain.propagate); keep them in sync.
// Concatenated after Rasterizer.metal, so Common.metal's structs and SPLAT_TILE are available.

// Accumulator layout per Gaussian (12 floats): 0..2 colour, 3 opacity, 4..6 conic (A, B, C),
// 7..8 mean2D, 9 depth (∂L/∂z_cam, Phase 4 rendered depth), 10..11 unused.
// Parameter gradients are overwritten, or added to when `ProjectUniforms.shInfo.w == 1` (accumulate).

inline void addGradient(device atomic<float>* slot, float value)
{
    atomic_fetch_add_explicit(slot, value, memory_order_relaxed);
}

kernel void rasterizeBackward(
    device const float4* meanDepth [[buffer(0)]],
    device const float4* conicOpacity [[buffer(1)]],
    device const float4* colors [[buffer(2)]],
    device const uint* values [[buffer(3)]],
    device const uint* ranges [[buffer(4)]],
    constant RasterUniforms& u [[buffer(5)]],
    device const float4* imageGradient [[buffer(6)]],
    device const float* finalT [[buffer(7)]],
    device const uint* lastContributor [[buffer(8)]],
    device atomic<float>* accum [[buffer(9)]],
    device const float2* depthGradient [[buffer(10)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float2 sMean[SPLAT_TILE * SPLAT_TILE];
    threadgroup float4 sConic[SPLAT_TILE * SPLAT_TILE];
    // xyz: colour; w: camera-space depth z (same layout as the forward kernel).
    threadgroup float4 sColor[SPLAT_TILE * SPLAT_TILE];

    const uint batch = uint(SPLAT_TILE * SPLAT_TILE);
    const uint width = u.info.x;
    const uint height = u.info.y;
    // Depth mode: RasterUniforms.background.w = 1 when the caller supplies a depth gradient.
    const bool depthMode = u.background.w > 0.5f;

    uint px = tgid.x * uint(SPLAT_TILE) + tid.x;
    uint py = tgid.y * uint(SPLAT_TILE) + tid.y;
    // Threads outside the image take part in every barrier and simd_sum but contribute zeros.
    bool inside = px < width && py < height;
    uint pixel = py * width + px;
    float2 p = float2(float(px) + 0.5f, float(py) + 0.5f);

    uint tileIndex = tgid.y * u.info.z + tgid.x;
    uint rangeStart = ranges[2u * tileIndex];
    uint rangeEnd = ranges[2u * tileIndex + 1u];

    // Per-pixel reverse state: transmittance after the last blended sample (T_final, kept constant
    // and walked back as T), the colour integral of everything behind the current sample (S), the
    // depth integral behindZ = Σ_{j>i} z_j α_j T_j (the depth counterpart of S), the pixel's image
    // gradient, its depth gradient (∂L/∂Draw, ∂L/∂A) and its cut-off.
    float T = 0.0f;
    float Tfinal = 0.0f;
    float3 behind = float3(0.0f);
    float behindZ = 0.0f;
    float3 gradPixel = float3(0.0f);
    float gD = 0.0f;
    float gA = 0.0f;
    uint lastSorted = 0u;
    if (inside) {
        T = finalT[pixel];
        Tfinal = T;
        behind = T * u.background.xyz;
        gradPixel = imageGradient[pixel].xyz;
        if (depthMode) {
            float2 gDepth = depthGradient[pixel];
            gD = gDepth.x;
            gA = gDepth.y;
        }
        lastSorted = lastContributor[pixel];
    }

    uint rangeCount = rangeEnd - rangeStart;
    uint batchCount = (rangeCount + batch - 1u) / batch;
    for (uint bi = batchCount; bi > 0u; --bi) {
        uint batchStart = rangeStart + (bi - 1u) * batch;
        uint n = min(batch, rangeEnd - batchStart);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < n) {
            uint gi = values[batchStart + lid];
            sMean[lid] = meanDepth[gi].xy;
            sConic[lid] = conicOpacity[gi];
            sColor[lid] = float4(colors[gi].xyz, meanDepth[gi].z);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Every thread of the threadgroup walks the batch in the same order, so the simdgroup
        // reduction below always combines contributions to the same Gaussian.
        for (uint jj = n; jj > 0u; --jj) {
            uint j = jj - 1u;
            uint sortedIndex = batchStart + j;
            uint gaussian = values[sortedIndex];

            float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f;   // dL/dcolor
            float r3 = 0.0f;                         // dL/dopacity
            float r4 = 0.0f, r5 = 0.0f, r6 = 0.0f;   // dL/dconic
            float r7 = 0.0f, r8 = 0.0f;              // dL/dmean2D
            float r9 = 0.0f;                         // dL/dz (camera depth), depth mode only

            // Only samples that were blended in the forward pass (sorted index below the pixel's
            // cut-off) and pass the same gating as the forward pass contribute.
            if (inside && sortedIndex < lastSorted) {
                float2 d = sMean[j] - p;
                float4 co = sConic[j];
                float sigma = 0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) + co.y * d.x * d.y;
                if (sigma >= 0.0f) {
                    float e = exp(-sigma);
                    float rawAlpha = co.w * e;
                    float alpha = min(0.999f, rawAlpha);
                    if (alpha >= 1.0f / 255.0f) {
                        float oneMinusAlpha = 1.0f - alpha;
                        float Ti = T / oneMinusAlpha;          // transmittance before this sample
                        float weight = alpha * Ti;
                        float3 color = sColor[j].xyz;
                        float z = sColor[j].w;
                        float3 mixed = color * Ti - behind / oneMinusAlpha;
                        float dAlpha = dot(mixed, gradPixel);
                        // Rendered depth: Draw = Σ αᵢ Tᵢ zᵢ, A = 1 − T_final.
                        //   ∂Draw/∂αᵢ = zᵢ Tᵢ − behindZ / (1 − αᵢ),   ∂A/∂αᵢ = T_final / (1 − αᵢ),
                        // with behindZ = Σ_{j>i} z_j α_j T_j taken before this sample is added.
                        // (The clamp at 0.999 is handled below, as for the colour term.)
                        if (depthMode) {
                            dAlpha += (z * Ti - behindZ / oneMinusAlpha) * gD
                                    + (Tfinal / oneMinusAlpha) * gA;
                            // ∂Draw/∂zᵢ = αᵢ Tᵢ. Taken before the clamp guard: z is not gated by α's clamp.
                            r9 = weight * gD;
                        }
                        float3 dColor = gradPixel * weight;
                        behind += color * weight;
                        behindZ += z * weight;
                        T = Ti;

                        r0 = dColor.x;
                        r1 = dColor.y;
                        r2 = dColor.z;
                        // Clamped alpha (0.999) has no gradient to opacity or sigma.
                        if (rawAlpha <= 0.999f) {
                            float dSigma = -co.w * e * dAlpha;
                            r3 = e * dAlpha;
                            float3 dConic = dSigma * float3(0.5f * d.x * d.x, d.x * d.y, 0.5f * d.y * d.y);
                            r4 = dConic.x;
                            r5 = dConic.y;
                            r6 = dConic.z;
                            float2 dMean = dSigma * float2(co.x * d.x + co.y * d.y, co.y * d.x + co.z * d.y);
                            r7 = dMean.x;
                            r8 = dMean.y;
                        }
                    }
                }
            }

            float s0 = simd_sum(r0);
            float s1 = simd_sum(r1);
            float s2 = simd_sum(r2);
            float s3 = simd_sum(r3);
            float s4 = simd_sum(r4);
            float s5 = simd_sum(r5);
            float s6 = simd_sum(r6);
            float s7 = simd_sum(r7);
            float s8 = simd_sum(r8);
            float s9 = simd_sum(r9);
            if (lane == 0u) {
                device atomic<float>* slot = accum + 12u * gaussian;
                addGradient(slot + 0, s0);
                addGradient(slot + 1, s1);
                addGradient(slot + 2, s2);
                addGradient(slot + 3, s3);
                addGradient(slot + 4, s4);
                addGradient(slot + 5, s5);
                addGradient(slot + 6, s6);
                addGradient(slot + 7, s7);
                addGradient(slot + 8, s8);
                if (depthMode) {
                    addGradient(slot + 9, s9);
                }
            }
        }
    }
}

// SH basis and its derivatives with respect to the (unit) direction components, degree <= 3.
// Mirrors `shBasis` in CPUBackward.swift. Entries above (degree + 1)^2 stay zero.
inline void shBasisWithGradient(uint degree, float3 d, thread float* basis, thread float3* basisGrad)
{
    const float c0 = 0.28209479177387814f;
    const float c1 = 0.4886025119029199f;
    const float c2_0 = 1.0925484305920792f;
    const float c2_1 = -1.0925484305920792f;
    const float c2_2 = 0.31539156525252005f;
    const float c2_3 = -1.0925484305920792f;
    const float c2_4 = 0.5462742152960396f;
    const float c3_0 = -0.5900435899266435f;
    const float c3_1 = 2.890611442640554f;
    const float c3_2 = -0.4570457994644658f;
    const float c3_3 = 0.3731763325901154f;
    const float c3_4 = -0.4570457994644658f;
    const float c3_5 = 1.445305721320277f;
    const float c3_6 = -0.5900435899266435f;

    for (uint k = 0u; k < 16u; ++k) {
        basis[k] = 0.0f;
        basisGrad[k] = float3(0.0f);
    }

    float x = d.x;
    float y = d.y;
    float z = d.z;
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;

    basis[0] = c0;
    if (degree >= 1u) {
        basis[1] = -c1 * y;
        basisGrad[1] = float3(0.0f, -c1, 0.0f);
        basis[2] = c1 * z;
        basisGrad[2] = float3(0.0f, 0.0f, c1);
        basis[3] = -c1 * x;
        basisGrad[3] = float3(-c1, 0.0f, 0.0f);
    }
    if (degree >= 2u) {
        basis[4] = c2_0 * x * y;
        basisGrad[4] = float3(c2_0 * y, c2_0 * x, 0.0f);
        basis[5] = c2_1 * y * z;
        basisGrad[5] = float3(0.0f, c2_1 * z, c2_1 * y);
        basis[6] = c2_2 * (2.0f * zz - xx - yy);
        basisGrad[6] = float3(c2_2 * (-2.0f * x), c2_2 * (-2.0f * y), c2_2 * 4.0f * z);
        basis[7] = c2_3 * x * z;
        basisGrad[7] = float3(c2_3 * z, 0.0f, c2_3 * x);
        basis[8] = c2_4 * (xx - yy);
        basisGrad[8] = float3(c2_4 * 2.0f * x, c2_4 * (-2.0f * y), 0.0f);
    }
    if (degree >= 3u) {
        basis[9] = c3_0 * y * (3.0f * xx - yy);
        basisGrad[9] = float3(c3_0 * 6.0f * x * y, c3_0 * (3.0f * xx - 3.0f * yy), 0.0f);
        basis[10] = c3_1 * x * y * z;
        basisGrad[10] = float3(c3_1 * y * z, c3_1 * x * z, c3_1 * x * y);
        basis[11] = c3_2 * y * (4.0f * zz - xx - yy);
        basisGrad[11] = float3(c3_2 * (-2.0f * x * y), c3_2 * (4.0f * zz - xx - 3.0f * yy), c3_2 * 8.0f * y * z);
        basis[12] = c3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
        basisGrad[12] = float3(c3_3 * (-6.0f * x * z), c3_3 * (-6.0f * y * z),
                               c3_3 * (6.0f * zz - 3.0f * xx - 3.0f * yy));
        basis[13] = c3_4 * x * (4.0f * zz - xx - yy);
        basisGrad[13] = float3(c3_4 * (4.0f * zz - 3.0f * xx - yy), c3_4 * (-2.0f * x * y), c3_4 * 8.0f * x * z);
        basis[14] = c3_5 * z * (xx - yy);
        basisGrad[14] = float3(c3_5 * 2.0f * x * z, c3_5 * (-2.0f * y * z), c3_5 * (xx - yy));
        basis[15] = c3_6 * x * (xx - 3.0f * yy);
        basisGrad[15] = float3(c3_6 * (3.0f * xx - 3.0f * yy), c3_6 * (-6.0f * x * y), 0.0f);
    }
}

// Writes `value` to a parameter gradient, or adds it when accumulating into existing gradients.
inline void storeGradient(device float* target, float value, bool accumulate)
{
    if (accumulate) {
        *target += value;
    } else {
        *target = value;
    }
}

// Writes zero gradients for one Gaussian (culled or degenerate). In accumulate mode adding zero is a
// no-op, so nothing is written.
inline void zeroParameterGradient(uint gid, uint K, ulong shBase, bool accumulate,
                                  device float* gradPositions,
                                  device float* gradLogScales,
                                  device float* gradRotations,
                                  device float* gradOpacityLogits,
                                  device float* gradSH)
{
    if (accumulate) {
        return;
    }
    gradPositions[3u * gid] = 0.0f;
    gradPositions[3u * gid + 1u] = 0.0f;
    gradPositions[3u * gid + 2u] = 0.0f;
    gradLogScales[3u * gid] = 0.0f;
    gradLogScales[3u * gid + 1u] = 0.0f;
    gradLogScales[3u * gid + 2u] = 0.0f;
    gradRotations[4u * gid] = 0.0f;
    gradRotations[4u * gid + 1u] = 0.0f;
    gradRotations[4u * gid + 2u] = 0.0f;
    gradRotations[4u * gid + 3u] = 0.0f;
    gradOpacityLogits[gid] = 0.0f;
    for (uint i = 0u; i < 3u * K; ++i) {
        gradSH[shBase + ulong(i)] = 0.0f;
    }
}

// Writes the six camera-pose contributions of one Gaussian into its scratch slots: dL/dω (0..2), then dL/dτ (3..5).
// Used only when `ProjectUniforms.limits.w` is 1 (pose refinement); see PoseGradient.metal for the reduction.
inline void storePoseContribution(device float* poseScratch, uint gid, float3 dTau, float3 dOmega)
{
    poseScratch[6u * gid] = dOmega.x;
    poseScratch[6u * gid + 1u] = dOmega.y;
    poseScratch[6u * gid + 2u] = dOmega.z;
    poseScratch[6u * gid + 3u] = dTau.x;
    poseScratch[6u * gid + 4u] = dTau.y;
    poseScratch[6u * gid + 5u] = dTau.z;
}

kernel void projectBackward(
    device const float* positions [[buffer(0)]],
    device const float* logScales [[buffer(1)]],
    device const float* rotations [[buffer(2)]],
    device const float* opacityLogits [[buffer(3)]],
    device const float* shCoeffs [[buffer(4)]],
    device const float* accum [[buffer(5)]],
    device const uint* tilesTouched [[buffer(6)]],
    device float* gradPositions [[buffer(7)]],
    device float* gradLogScales [[buffer(8)]],
    device float* gradRotations [[buffer(9)]],
    device float* gradOpacityLogits [[buffer(10)]],
    device float* gradSH [[buffer(11)]],
    constant ProjectUniforms& u [[buffer(12)]],
    constant MipFilterUniforms& mip [[buffer(13)]],
    device const float* samplingRates [[buffer(14)]],
    device float* poseScratch [[buffer(15)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.counts.x) return;
    // limits.w == 1: also write the camera-pose contribution of this Gaussian to poseScratch (6 floats).
    const bool poseOn = u.limits.w > 0.5f;

    uint K = u.shInfo.z;                       // stored SH coefficients per Gaussian
    uint evalDegree = u.shInfo.y;              // min(shDegree, shDegreeLimit)
    uint used = (evalDegree + 1u) * (evalDegree + 1u);
    ulong shBase = ulong(gid) * ulong(K) * 3ul;
    // shInfo.w == 1: add into the parameter gradients instead of overwriting them (Phase 4 accumulate).
    bool accumulate = u.shInfo.w == 1u;

    // Camera rows, and columns of the camera rotation.
    float3 R0 = u.rotRow0.xyz;
    float3 R1 = u.rotRow1.xyz;
    float3 R2 = u.rotRow2.xyz;
    float3 camT = float3(u.rotRow0.w, u.rotRow1.w, u.rotRow2.w);
    float3 cam0 = float3(R0.x, R1.x, R2.x);
    float3 cam1 = float3(R0.y, R1.y, R2.y);
    float3 cam2 = float3(R0.z, R1.z, R2.z);

    float3 pw = float3(positions[3u * gid], positions[3u * gid + 1u], positions[3u * gid + 2u]);
    float3 pc = float3(dot(R0, pw), dot(R1, pw), dot(R2, pw)) + camT;
    float x = pc.x;
    float y = pc.y;
    float z = pc.z;

    // Culled Gaussians (no tiles touched, or behind the camera) receive no gradient.
    if (tilesTouched[gid] == 0u || !(z > 0.0f)) {
        zeroParameterGradient(gid, K, shBase, accumulate, gradPositions, gradLogScales, gradRotations,
                              gradOpacityLogits, gradSH);
        if (poseOn) {
            storePoseContribution(poseScratch, gid, float3(0.0f), float3(0.0f));
        }
        return;
    }

    // Screen-space gradients accumulated by rasterizeBackward.
    device const float* slot = accum + 12u * gid;
    float3 gColor = float3(slot[0], slot[1], slot[2]);
    float gOpacity = slot[3];
    float3 gConic = float3(slot[4], slot[5], slot[6]);
    float2 gMean = float2(slot[7], slot[8]);

    // Normalized quaternion (w, x, y, z); zero quaternion maps to identity.
    float4 qr = float4(rotations[4u * gid], rotations[4u * gid + 1u],
                       rotations[4u * gid + 2u], rotations[4u * gid + 3u]);
    float qlen2 = dot(qr, qr);
    bool hasNorm = qlen2 > 0.0f;
    float qnorm = sqrt(qlen2);
    float qw = 1.0f, qx = 0.0f, qy = 0.0f, qz = 0.0f;
    if (hasNorm) {
        float inv = 1.0f / qnorm;
        qw = qr.x * inv;
        qx = qr.y * inv;
        qy = qr.z * inv;
        qz = qr.w * inv;
    }

    // Rotation matrix of the unit quaternion, column-major (columns are R's columns).
    float3x3 Rq = float3x3(
        float3(1.0f - 2.0f * (qy * qy + qz * qz), 2.0f * (qx * qy + qw * qz), 2.0f * (qx * qz - qw * qy)),
        float3(2.0f * (qx * qy - qw * qz), 1.0f - 2.0f * (qx * qx + qz * qz), 2.0f * (qy * qz + qw * qx)),
        float3(2.0f * (qx * qz + qw * qy), 2.0f * (qy * qz - qw * qx), 1.0f - 2.0f * (qx * qx + qy * qy)));

    // Mip mode (Mip-Splatting), as in the forward kernel: s' = sqrt(s² + v3/ν²) and the opacity factor κ = Π s/s'.
    // `chain` is ∂s'/∂logScale per axis (s² / s' in mip mode, s otherwise).
    const bool mipOn = mip.info.x > 0.5f;
    float3 logS = float3(logScales[3u * gid], logScales[3u * gid + 1u], logScales[3u * gid + 2u]);
    float3 rawS = exp(logS);
    float3 s = rawS;
    float opacityFactor = 1.0f;
    if (mipOn) {
        float filterVariance = mip.info.y / (samplingRates[gid] * samplingRates[gid]);
        s = sqrt(rawS * rawS + filterVariance);
        opacityFactor = (rawS.x / s.x) * (rawS.y / s.y) * (rawS.z / s.z);
    }
    float3 chain = mipOn ? (rawS * rawS / s) : s;
    float3x3 M = float3x3(Rq[0] * s.x, Rq[1] * s.y, Rq[2] * s.z);     // R_q · diag(s)
    float3x3 Sigma3 = M * transpose(M);

    // Opacity: sigmoid of the raw logit. In mip mode the used opacity is o·κ·c2.
    float opacityRaw = 1.0f / (1.0f + exp(-opacityLogits[gid]));

    // Projection with the clamped Jacobian (same limits as the forward kernel).
    float fx = u.intrinsics.x;
    float fy = u.intrinsics.y;
    float cx = u.intrinsics.z;
    float cy = u.intrinsics.w;
    float W = float(u.counts.y);
    float H = float(u.counts.z);
    float eps2D = u.limits.z;
    float tanFovX = 0.5f * W / fx;
    float tanFovY = 0.5f * H / fy;
    float limXPos = (W - cx) / fx + 0.3f * tanFovX;
    float limXNeg = cx / fx + 0.3f * tanFovX;
    float limYPos = (H - cy) / fy + 0.3f * tanFovY;
    float limYNeg = cy / fy + 0.3f * tanFovY;

    float xz = x / z;
    float yz = y / z;
    bool clampX = false;
    bool clampY = false;
    float xLimit = 0.0f;
    float yLimit = 0.0f;
    if (xz > limXPos) {
        clampX = true;
        xLimit = limXPos;
    } else if (xz < -limXNeg) {
        clampX = true;
        xLimit = -limXNeg;
    }
    if (yz > limYPos) {
        clampY = true;
        yLimit = limYPos;
    } else if (yz < -limYNeg) {
        clampY = true;
        yLimit = -limYNeg;
    }
    float tx = z * (clampX ? xLimit : xz);
    float ty = z * (clampY ? yLimit : yz);
    float3 J0 = float3(fx / z, 0.0f, -fx * tx / (z * z));
    float3 J1 = float3(0.0f, fy / z, -fy * ty / (z * z));

    // T = J · R (2×3), one row per screen axis.
    float3 T0 = float3(dot(J0, cam0), dot(J0, cam1), dot(J0, cam2));
    float3 T1 = float3(dot(J1, cam0), dot(J1, cam1), dot(J1, cam2));

    // Σ2D = T Σ3 Tᵀ + eps·I (classic), or + v2·I with the compensation c2 (mip), and its inverse (the conic).
    float3 U0 = Sigma3 * T0;
    float3 U1 = Sigma3 * T1;
    float a0 = dot(T0, U0);
    float b = dot(T0, U1);
    float c0 = dot(T1, U1);
    float a = a0 + eps2D;
    float c = c0 + eps2D;
    float det = a * c - b * b;
    if (!(det > 0.0f)) {
        zeroParameterGradient(gid, K, shBase, accumulate, gradPositions, gradLogScales, gradRotations,
                              gradOpacityLogits, gradSH);
        if (poseOn) {
            storePoseContribution(poseScratch, gid, float3(0.0f), float3(0.0f));
        }
        return;
    }
    // Mip 2D compensation c2 = sqrt(D0 / D1) and its derivatives with respect to the raw entries (a0, b, c0):
    // ∂log c2 = −½ ∂log D1 + ½ ∂log D0 (the second term only when D0 is not clamped).
    float compensation = 1.0f;
    float dC2a = 0.0f;
    float dC2b = 0.0f;
    float dC2c = 0.0f;
    if (mipOn) {
        float rawDet = a0 * c0 - b * b;
        bool clampedDet = rawDet < 1.0e-12f;
        float d0 = clampedDet ? 1.0e-12f : rawDet;
        float ratio0 = clampedDet ? 0.0f : 1.0f;
        float invD1 = 1.0f / det;
        compensation = sqrt(d0 / det);
        dC2a = compensation * (-0.5f * c * invD1 + ratio0 * 0.5f * c0 / d0);
        dC2b = compensation * (b * invD1 - ratio0 * b / d0);
        dC2c = compensation * (-0.5f * a * invD1 + ratio0 * 0.5f * a0 / d0);
    }
    // Conic matrix K = Σ2D⁻¹ = [[k00, k01], [k01, k11]].
    float k00 = c / det;
    float k01 = -b / det;
    float k11 = a / det;

    // Conic gradient → Σ2D gradient: g2 = −K · gm · K with gm = [[gA, gB/2], [gB/2, gC]].
    float m00 = gConic.x;
    float m01 = 0.5f * gConic.y;
    float m11 = gConic.z;
    float p00 = k00 * m00 + k01 * m01;
    float p01 = k00 * m01 + k01 * m11;
    float p10 = k01 * m00 + k11 * m01;
    float p11 = k01 * m01 + k11 * m11;
    float q00 = p00 * k00 + p01 * k01;
    float q01 = p00 * k01 + p01 * k11;
    float q10 = p10 * k00 + p11 * k01;
    float q11 = p10 * k01 + p11 * k11;
    float g2a = -q00;
    float g2b = -0.5f * (q01 + q10);
    float g2c = -q11;
    if (mipOn) {
        // Compensation path: ∂L/∂c2 = gOpacity·o·κ, into the Σ2D entries (off-diagonal with ½, as dL = tr(g2·dΣ)).
        float gCompensation = slot[3] * opacityRaw * opacityFactor;
        g2a += gCompensation * dC2a;
        g2b += 0.5f * gCompensation * dC2b;
        g2c += gCompensation * dC2c;
    }

    // dL/dT = 2·g2·T·Σ3, rows dT0, dT1.
    float3 dT0 = 2.0f * (g2a * U0 + g2b * U1);
    float3 dT1 = 2.0f * (g2b * U0 + g2c * U1);

    // dL/dJ = dL/dT · Rᵀ.
    float3 dJ0 = float3(dot(dT0, R0), dot(dT0, R1), dot(dT0, R2));
    float3 dJ1 = float3(dot(dT1, R0), dot(dT1, R1), dot(dT1, R2));

    // dL/dΣ3 = Tᵀ g2 T, returned as columns; dL/dM = 2·(dL/dΣ3)·M.
    float3 G3c0 = (g2a * T0.x) * T0 + (g2b * T1.x) * T0 + (g2b * T0.x) * T1 + (g2c * T1.x) * T1;
    float3 G3c1 = (g2a * T0.y) * T0 + (g2b * T1.y) * T0 + (g2b * T0.y) * T1 + (g2c * T1.y) * T1;
    float3 G3c2 = (g2a * T0.z) * T0 + (g2b * T1.z) * T0 + (g2b * T0.z) * T1 + (g2c * T1.z) * T1;
    float3x3 G3 = float3x3(G3c0, G3c1, G3c2);
    float3x3 dM = 2.0f * (G3 * M);

    // M = R_q·diag(s): scale gradient and dL/dR_q.
    float dLogS0 = chain.x * dot(Rq[0], dM[0]);
    float dLogS1 = chain.y * dot(Rq[1], dM[1]);
    float dLogS2 = chain.z * dot(Rq[2], dM[2]);
    if (mipOn) {
        // Opacity through κ: ∂o_used/∂logScaleⱼ = o_used · (1 − sⱼ²/s'ⱼ²), with o_used = o·κ·c2.
        float usedOpacity = opacityRaw * opacityFactor * compensation;
        float3 ratio2 = (rawS * rawS) / (s * s);
        dLogS0 += slot[3] * usedOpacity * (1.0f - ratio2.x);
        dLogS1 += slot[3] * usedOpacity * (1.0f - ratio2.y);
        dLogS2 += slot[3] * usedOpacity * (1.0f - ratio2.z);
    }
    float3 dRc0 = dM[0] * s.x;
    float3 dRc1 = dM[1] * s.y;
    float3 dRc2 = dM[2] * s.z;
    // Entries r_ij of dL/dR_q (row i, column j): column j is dRc_j.
    float rr00 = dRc0.x, rr01 = dRc1.x, rr02 = dRc2.x;
    float rr10 = dRc0.y, rr11 = dRc1.y, rr12 = dRc2.y;
    float rr20 = dRc0.z, rr21 = dRc1.z, rr22 = dRc2.z;

    // Rotation matrix → unit quaternion → raw quaternion.
    float dqw = 2.0f * (qz * (rr10 - rr01) + qy * (rr02 - rr20) + qx * (rr21 - rr12));
    float dqx = 2.0f * (qy * (rr10 + rr01) + qz * (rr20 + rr02) + qw * (rr21 - rr12)
                        - 2.0f * qx * (rr11 + rr22));
    float dqy = 2.0f * (qx * (rr10 + rr01) + qz * (rr21 + rr12) + qw * (rr02 - rr20)
                        - 2.0f * qy * (rr00 + rr22));
    float dqz = 2.0f * (qw * (rr10 - rr01) + qx * (rr20 + rr02) + qy * (rr21 + rr12)
                        - 2.0f * qz * (rr00 + rr11));
    float radial = qw * dqw + qx * dqx + qy * dqy + qz * dqz;

    // Camera-space mean: J terms (with the clamp rule) and the unclamped mean2D terms.
    float invZ2 = 1.0f / (z * z);
    float invZ3 = invZ2 / z;
    float dJ00 = dJ0.x;
    float dJ02 = dJ0.z;
    float dJ11 = dJ1.y;
    float dJ12 = dJ1.z;
    float dpcz = -fx * invZ2 * dJ00 - fy * invZ2 * dJ11
               + 2.0f * fx * tx * invZ3 * dJ02 + 2.0f * fy * ty * invZ3 * dJ12;
    float dtx = -fx * invZ2 * dJ02;
    float dty = -fy * invZ2 * dJ12;
    float3 dpc = float3(0.0f, 0.0f, dpcz);
    if (clampX) {
        dpc.z += dtx * xLimit;
    } else {
        dpc.x += dtx;
    }
    if (clampY) {
        dpc.z += dty * yLimit;
    } else {
        dpc.y += dty;
    }
    dpc.x += fx / z * gMean.x;
    dpc.y += fy / z * gMean.y;
    dpc.z += -fx * x * invZ2 * gMean.x - fy * y * invZ2 * gMean.y;
    // Rendered depth (slot 9 = ∂L/∂z_cam, zero when depth is off). z_cam = R2·pw + t_z, so ∂z/∂pw = R2 and
    // adding slot9 to dpc.z before the Rᵀ step gives gradPositions += slot9 · R2 (the CPU reference does the same).
    dpc.z += slot[9];

    // Camera → world (Rᵀ · dpc).
    float3 dPos = float3(dot(dpc, cam0), dot(dpc, cam1), dot(dpc, cam2));

    // Colour: SH basis, clamp at zero, and the view-direction term through normalize().
    float3 camCenter = u.camCenter.xyz;
    float3 offset = pw - camCenter;
    float len = sqrt(dot(offset, offset));
    if (!accumulate) {
        for (uint i = 0u; i < 3u * K; ++i) {
            gradSH[shBase + ulong(i)] = 0.0f;
        }
    }
    // Gradient through the view direction (w.r.t. the Gaussian position), kept apart for the pose path.
    float3 uDir = float3(0.0f);
    if (len > 0.0f) {
        float3 dir = offset / len;
        float basis[16];
        float3 basisGrad[16];
        shBasisWithGradient(evalDegree, dir, basis, basisGrad);

        float3 raw = float3(0.0f);
        for (uint k = 0u; k < used; ++k) {
            ulong o = shBase + ulong(3u * k);
            float3 coef = float3(shCoeffs[o], shCoeffs[o + 1ul], shCoeffs[o + 2ul]);
            raw += basis[k] * coef;
        }
        float3 dColor = gColor;
        if (raw.x + 0.5f <= 0.0f) dColor.x = 0.0f;
        if (raw.y + 0.5f <= 0.0f) dColor.y = 0.0f;
        if (raw.z + 0.5f <= 0.0f) dColor.z = 0.0f;

        float3 dDir = float3(0.0f);
        for (uint k = 0u; k < used; ++k) {
            ulong o = shBase + ulong(3u * k);
            float3 coef = float3(shCoeffs[o], shCoeffs[o + 1ul], shCoeffs[o + 2ul]);
            float proj = dot(coef, dColor);
            dDir += proj * basisGrad[k];
            storeGradient(&gradSH[o], basis[k] * dColor.x, accumulate);
            storeGradient(&gradSH[o + 1ul], basis[k] * dColor.y, accumulate);
            storeGradient(&gradSH[o + 2ul], basis[k] * dColor.z, accumulate);
        }
        uDir = (dDir - dir * dot(dir, dDir)) / len;
        dPos += uDir;
    }

    // Camera pose contribution at δ = 0 (the same chain as CPUBackward.swift `GaussianChain.propagate`):
    // dL/dτ = dL/dpc + W·u and dL/dω = pc × dL/dpc + 2·(M12, −M02, M01) with M = S·G − G·S, S = W·Σ3·Wᵀ, G = W·G3·Wᵀ.
    // Matrices are column-major: Mc[c][r] is row r, column c, so M12 = Mc[2][1], M02 = Mc[2][0], M01 = Mc[1][0].
    if (poseOn) {
        float3x3 Wm = float3x3(cam0, cam1, cam2);
        float3x3 Gc = Wm * G3 * transpose(Wm);
        float3x3 Sc = Wm * Sigma3 * transpose(Wm);
        float3x3 Mc = Sc * Gc - Gc * Sc;
        float3 covarianceRotation = 2.0f * float3(Mc[2][1], -Mc[2][0], Mc[1][0]);
        float3 dTau = dpc + Wm * uDir;
        float3 dOmega = cross(pc, dpc) + covarianceRotation;
        storePoseContribution(poseScratch, gid, dTau, dOmega);
    }

    // Write parameter gradients (overwrite, or add when accumulating).
    storeGradient(&gradPositions[3u * gid], dPos.x, accumulate);
    storeGradient(&gradPositions[3u * gid + 1u], dPos.y, accumulate);
    storeGradient(&gradPositions[3u * gid + 2u], dPos.z, accumulate);
    storeGradient(&gradLogScales[3u * gid], dLogS0, accumulate);
    storeGradient(&gradLogScales[3u * gid + 1u], dLogS1, accumulate);
    storeGradient(&gradLogScales[3u * gid + 2u], dLogS2, accumulate);
    if (hasNorm) {
        storeGradient(&gradRotations[4u * gid], (dqw - qw * radial) / qnorm, accumulate);
        storeGradient(&gradRotations[4u * gid + 1u], (dqx - qx * radial) / qnorm, accumulate);
        storeGradient(&gradRotations[4u * gid + 2u], (dqy - qy * radial) / qnorm, accumulate);
        storeGradient(&gradRotations[4u * gid + 3u], (dqz - qz * radial) / qnorm, accumulate);
    } else {
        storeGradient(&gradRotations[4u * gid], 0.0f, accumulate);
        storeGradient(&gradRotations[4u * gid + 1u], 0.0f, accumulate);
        storeGradient(&gradRotations[4u * gid + 2u], 0.0f, accumulate);
        storeGradient(&gradRotations[4u * gid + 3u], 0.0f, accumulate);
    }
    float opacityGradient = gOpacity * opacityRaw * (1.0f - opacityRaw);
    if (mipOn) {
        opacityGradient *= compensation * opacityFactor;
    }
    storeGradient(&gradOpacityLogits[gid], opacityGradient, accumulate);
}
