// Densification kernels (Phase 3, Task 14): gradient statistics, per-Gaussian decision, budget
// finalization and scatter of parameters and Adam moments into the compacted layout.
// Mirrors SplatCore/Training/Densification.swift line by line. Host side: GPUDensifier.swift.
// Concatenated after Backward (see MetalContext.shaderFiles); every helper is prefixed `densify`
// and every macro `DENSIFY_` because all shader files share one translation unit.
//
// Action codes equal the raw values of `DensifyAction`: 0 prune, 1 keep, 2 clone, 3 split.

#define DENSIFY_PRUNE 0u
#define DENSIFY_KEEP 1u
#define DENSIFY_CLONE 2u
#define DENSIFY_SPLIT 3u

// Mirrors `DensifyStatsUniforms` in GPUDensifier.swift (32 bytes).
struct DensifyStatsUniforms {
    float4 halfSize;   // x: width / 2, y: height / 2, zw unused
    uint4 info;        // x: Gaussian count, yzw unused
};

// Mirrors `DensifyDecideUniforms` in GPUDensifier.swift (48 bytes).
struct DensifyDecideUniforms {
    float4 params;     // x: scene radius, y: prune opacity, z: prune scale fraction, w: gradient threshold
    float4 params2;    // x: split scale fraction, yzw unused
    int4 info;         // x: Gaussian count, y: iteration, z: prune scale start iteration, w unused
};

// Mirrors `DensifyFinalizeUniforms` in GPUDensifier.swift (16 bytes).
struct DensifyFinalizeUniforms {
    uint4 info;        // x: Gaussian count, y: growths allowed, z: 1 when every Gaussian is kept, w unused
};

// Mirrors `DensifyScatterUniforms` in GPUDensifier.swift (32 bytes).
struct DensifyScatterUniforms {
    uint4 info;        // x: Gaussian count, y: floats per Gaussian in this group, z: stream seed, w unused
    float4 extra;      // x: log(split scale divisor), yzw unused
};

// Same integer hash as `SplatRandom.hash` (PCG-RXS-M-XS style). Bit-exact with the Swift version.
inline uint densifyHash(uint x) {
    uint s = x * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

// Same as `SplatRandom.uniform`, including the clamp that keeps the result strictly inside (0, 1).
inline float densifyUniform(uint seed, uint index, uint stream) {
    uint key = densifyHash(seed ^ densifyHash(index ^ densifyHash(stream)));
    float value = (float(key >> 8u) + 0.5f) / 16777216.0f;
    // 1 - 2^-24 is the largest float below 1 (Swift: 1 - Float.ulpOfOne / 2).
    return min(value, 1.0f - 5.9604644775390625e-08f);
}

// Same as `SplatRandom.normal` (Box-Muller on streams 2·stream and 2·stream + 1).
// `log` and `cos` are not bit-identical to the Swift libm calls, so normals agree only to a few ULP.
inline float densifyNormal(uint seed, uint index, uint stream) {
    float u1 = densifyUniform(seed, index, stream * 2u);
    float u2 = densifyUniform(seed, index, stream * 2u + 1u);
    // Bit pattern of Swift's Float.pi: 0x40490FDA (3.1415925, verified on the Swift 6.2 toolchain). This
    // is one ULP BELOW the float nearest to pi (0x40490FDB), so the usual decimal literal would not match
    // the CPU reference in SeededRandom.swift, which uses `2 * Float.pi * u2`. Spelling the bits keeps
    // the match exact.
    const float piF = as_type<float>(0x40490FDAu);
    return sqrt(-2.0f * log(u1)) * cos(2.0f * piF * u2);
}

// Rotation matrix of a quaternion stored as (w, x, y, z), normalised as `Matrix3.rotation(quaternion:)`.
// Columns: c0, c1, c2. A zero quaternion maps to the identity, as on the CPU.
inline float3x3 densifyRotationMatrix(float4 q) {
    float lengthSquared = dot(q, q);
    if (!(lengthSquared > 0.0f)) {
        return float3x3(float3(1.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), float3(0.0f, 0.0f, 1.0f));
    }
    float4 n = q / sqrt(lengthSquared);
    float w = n.x;
    float x = n.y;
    float y = n.z;
    float z = n.w;
    float3 c0 = float3(1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y + w * z), 2.0f * (x * z - w * y));
    float3 c1 = float3(2.0f * (x * y - w * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z + w * x));
    float3 c2 = float3(2.0f * (x * z + w * y), 2.0f * (y * z - w * x), 1.0f - 2.0f * (x * x + y * y));
    return float3x3(c0, c1, c2);
}

inline uint densifyOutputCount(uint action) {
    return action == DENSIFY_PRUNE ? 0u : (action == DENSIFY_KEEP ? 1u : 2u);
}

// Screen-space statistics for one view: Σ ‖(dL/dmean2D.x · W/2, dL/dmean2D.y · H/2)‖ and the
// visible count. Slots 7-8 of the Phase 2 accumulator hold dL/dmean2D in pixels. Culled
// Gaussians (no tiles touched) are not visible. Each thread owns its index, so there are no atomics.
kernel void accumulateDensifyStats(
    device const uint* tilesTouched [[buffer(0)]],
    device const float* accum [[buffer(1)]],
    device float* gradientNormSums [[buffer(2)]],
    device uint* visibleCounts [[buffer(3)]],
    constant DensifyStatsUniforms& u [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.info.x) return;
    if (tilesTouched[gid] == 0u) return;
    device const float* slot = accum + 12u * gid;
    float sx = slot[7] * u.halfSize.x;
    float sy = slot[8] * u.halfSize.y;
    gradientNormSums[gid] += sqrt(sx * sx + sy * sy);
    visibleCounts[gid] += 1u;
}

// Per-Gaussian decision: `Densification.action`, with the same branch order.
kernel void densifyDecide(
    device const float* logScales [[buffer(0)]],
    device const float* opacityLogits [[buffer(1)]],
    device const float* gradientNormSums [[buffer(2)]],
    device const uint* visibleCounts [[buffer(3)]],
    device uint* actions [[buffer(4)]],
    device uint* growFlags [[buffer(5)]],
    device uint* keptFlags [[buffer(6)]],
    constant DensifyDecideUniforms& u [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(u.info.x)) return;
    float3 ls = float3(logScales[3u * gid], logScales[3u * gid + 1u], logScales[3u * gid + 2u]);
    float maxLogScale = max(ls.x, max(ls.y, ls.z));
    float maxScale = exp(maxLogScale);
    float opacity = 1.0f / (1.0f + exp(-opacityLogits[gid]));
    float sceneRadius = u.params.x;

    uint action = DENSIFY_KEEP;
    if (opacity < u.params.y) {
        action = DENSIFY_PRUNE;
    } else if (u.info.y > u.info.z && maxScale > u.params.z * sceneRadius) {
        action = DENSIFY_PRUNE;
    } else {
        uint visible = visibleCounts[gid];
        float average = visible > 0u ? gradientNormSums[gid] / float(visible) : 0.0f;
        if (average > u.params.w) {
            action = (maxScale <= u.params2.x * sceneRadius) ? DENSIFY_CLONE : DENSIFY_SPLIT;
        }
    }
    actions[gid] = action;
    growFlags[gid] = (action == DENSIFY_CLONE || action == DENSIFY_SPLIT) ? 1u : 0u;
    keptFlags[gid] = (action != DENSIFY_PRUNE) ? 1u : 0u;
}

// Budget and never-empty rules, after the scans of the grow and keep flags.
// Growth candidates with rank (exclusive scan of growFlags) >= allowed become keeps, and so does
// every Gaussian when nothing survives (u.info.z). outCounts holds the outputs per input.
kernel void densifyFinalize(
    device uint* actions [[buffer(0)]],
    device const uint* growRanks [[buffer(1)]],
    device uint* outCounts [[buffer(2)]],
    constant DensifyFinalizeUniforms& u [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.info.x) return;
    uint action = actions[gid];
    if (u.info.z != 0u) {
        action = DENSIFY_KEEP;
    } else if ((action == DENSIFY_CLONE || action == DENSIFY_SPLIT) && growRanks[gid] >= u.info.y) {
        action = DENSIFY_KEEP;
    }
    actions[gid] = action;
    outCounts[gid] = densifyOutputCount(action);
}

// Scatter for groups with no per-output rule (rotations, opacity, SH): every output is a copy.
kernel void densifyScatterCopy(
    device const float* src [[buffer(0)]],
    device const uint* actions [[buffer(1)]],
    device const uint* offsets [[buffer(2)]],
    device float* dst [[buffer(3)]],
    constant DensifyScatterUniforms& u [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.info.x) return;
    uint action = actions[gid];
    uint outputs = densifyOutputCount(action);
    uint floats = u.info.y;
    for (uint k = 0u; k < outputs; ++k) {
        uint base = (offsets[gid] + k) * floats;
        for (uint f = 0u; f < floats; ++f) {
            dst[base + f] = src[gid * floats + f];
        }
    }
}

// Scatter for Adam moments: keep and the first clone child copy the moment, the second clone child
// and both split children start at zero (`Densification.apply`).
kernel void densifyScatterMoments(
    device const float* src [[buffer(0)]],
    device const uint* actions [[buffer(1)]],
    device const uint* offsets [[buffer(2)]],
    device float* dst [[buffer(3)]],
    constant DensifyScatterUniforms& u [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.info.x) return;
    uint action = actions[gid];
    uint outputs = densifyOutputCount(action);
    uint floats = u.info.y;
    for (uint k = 0u; k < outputs; ++k) {
        bool zero = (action == DENSIFY_SPLIT) || (action == DENSIFY_CLONE && k == 1u);
        uint base = (offsets[gid] + k) * floats;
        for (uint f = 0u; f < floats; ++f) {
            dst[base + f] = zero ? 0.0f : src[gid * floats + f];
        }
    }
}

// Scatter for log scales: split children use log scale − log(split scale divisor).
kernel void densifyScatterLogScales(
    device const float* src [[buffer(0)]],
    device const uint* actions [[buffer(1)]],
    device const uint* offsets [[buffer(2)]],
    device float* dst [[buffer(3)]],
    constant DensifyScatterUniforms& u [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.info.x) return;
    uint action = actions[gid];
    uint outputs = densifyOutputCount(action);
    for (uint k = 0u; k < outputs; ++k) {
        uint base = (offsets[gid] + k) * 3u;
        for (uint f = 0u; f < 3u; ++f) {
            float value = src[gid * 3u + f];
            if (action == DENSIFY_SPLIT) {
                value = value - u.extra.x;
            }
            dst[base + f] = value;
        }
    }
}

// Scatter for positions. Keep and clone copy the position. A split child j is
// p + R(q) · (exp(logScale) ⊙ ε_j), with ε_j[a] = densifyNormal(seed, gid, 3j + a).
// Reads the OLD logScales and rotations: the host scatters positions before those groups are replaced.
kernel void densifyScatterPositions(
    device const float* positions [[buffer(0)]],
    device const float* logScales [[buffer(1)]],
    device const float* rotations [[buffer(2)]],
    device const uint* actions [[buffer(3)]],
    device const uint* offsets [[buffer(4)]],
    device float* dst [[buffer(5)]],
    constant DensifyScatterUniforms& u [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.info.x) return;
    uint action = actions[gid];
    float3 p = float3(positions[3u * gid], positions[3u * gid + 1u], positions[3u * gid + 2u]);
    if (action != DENSIFY_SPLIT) {
        uint outputs = densifyOutputCount(action);
        for (uint k = 0u; k < outputs; ++k) {
            uint base = (offsets[gid] + k) * 3u;
            dst[base] = p.x;
            dst[base + 1u] = p.y;
            dst[base + 2u] = p.z;
        }
        return;
    }

    float3 parentLogScale = float3(logScales[3u * gid], logScales[3u * gid + 1u], logScales[3u * gid + 2u]);
    float3 scale = float3(exp(parentLogScale.x), exp(parentLogScale.y), exp(parentLogScale.z));
    float4 q = float4(rotations[4u * gid], rotations[4u * gid + 1u],
                      rotations[4u * gid + 2u], rotations[4u * gid + 3u]);
    float3x3 rotation = densifyRotationMatrix(q);
    uint seed = u.info.z;
    for (uint j = 0u; j < 2u; ++j) {
        uint stream = 3u * j;
        float3 eps = float3(densifyNormal(seed, gid, stream),
                            densifyNormal(seed, gid, stream + 1u),
                            densifyNormal(seed, gid, stream + 2u));
        float3 scaled = scale * eps;
        float3 offset = rotation * scaled;
        float3 child = p + offset;
        uint base = (offsets[gid] + j) * 3u;
        dst[base] = child.x;
        dst[base + 1u] = child.y;
        dst[base + 2u] = child.z;
    }
}
