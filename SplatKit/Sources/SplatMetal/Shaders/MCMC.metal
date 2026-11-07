// MCMC.metal — v1.2 MCMC densification kernels: the opacity and scale L1 regularizer gradients, and the SGLD
// position noise.
//
// Concatenated after Regularizers.metal (see `MetalContext.shaderFiles`); reuses `densifyNormal` and
// `densifyRotationMatrix` from Densify.metal. Uniform structs are mirrored in GPUMCMC.swift. Mirrors `MCMCNoise`
// and `MCMCRegularizers` in SplatCore. One thread per Gaussian; each thread writes only its own entries.

// Mirrors `MCMCRegularizerUniforms` in GPUMCMC.swift (32 bytes).
struct MCMCRegularizerUniforms {
    /// x: opacityWeight / n, y: scaleWeight / n, zw unused.
    float4 params;
    /// x: Gaussian count, yzw unused.
    uint4 info;
};

// Mirrors `MCMCNoiseUniforms` in GPUMCMC.swift (32 bytes).
struct MCMCNoiseUniforms {
    /// x: step (noise learning rate × position learning rate), y: gate steepness, z: gate opacity x0, w unused.
    float4 params;
    /// x: Gaussian count, y: seed, zw unused.
    uint4 info;
};

// Adds ∂/∂l = (w_o/n) σ(l)(1 − σ(l)) to the opacity-logit gradient and ∂/∂s_a = (w_s/n) exp(s_a) to the log-scale
// gradients. Accumulates, so it must run after the backward pass has written the gradients.
kernel void mcmcRegularizerGradient(
    device const float* opacityLogits [[buffer(0)]],
    device const float* logScales [[buffer(1)]],
    device float* gradOpacityLogits [[buffer(2)]],
    device float* gradLogScales [[buffer(3)]],
    constant MCMCRegularizerUniforms& u [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.info.x) {
        return;
    }
    float sigma = 1.0f / (1.0f + exp(-opacityLogits[gid]));
    gradOpacityLogits[gid] += u.params.x * sigma * (1.0f - sigma);
    for (uint axis = 0; axis < 3u; ++axis) {
        uint index = 3u * gid + axis;
        gradLogScales[index] += u.params.y * exp(logScales[index]);
    }
}

// SGLD noise, in place: p += step · M(o) · R S² Rᵀ ε with ε[a] = densifyNormal(seed, gid, a) and
// M(o) = 1 / (1 + exp(k (o − x0))). Reads the log-scales, rotations and opacity logits, which the same
// kernel does not write, so there is no race.
kernel void mcmcNoise(
    device float* positions [[buffer(0)]],
    device const float* logScales [[buffer(1)]],
    device const float* rotations [[buffer(2)]],
    device const float* opacityLogits [[buffer(3)]],
    constant MCMCNoiseUniforms& u [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.info.x) {
        return;
    }
    float opacity = 1.0f / (1.0f + exp(-opacityLogits[gid]));
    float gate = 1.0f / (1.0f + exp(u.params.y * (opacity - u.params.z)));
    uint seed = u.info.y;
    float3 eps = float3(densifyNormal(seed, gid, 0u), densifyNormal(seed, gid, 1u), densifyNormal(seed, gid, 2u));
    float4 q = float4(rotations[4u * gid], rotations[4u * gid + 1u],
                      rotations[4u * gid + 2u], rotations[4u * gid + 3u]);
    float3x3 rotation = densifyRotationMatrix(q);
    // Rᵀ ε: component i is the dot product of column i with ε.
    float3 projected = float3(dot(rotation[0], eps), dot(rotation[1], eps), dot(rotation[2], eps));
    float3 s = float3(exp(logScales[3u * gid]), exp(logScales[3u * gid + 1u]), exp(logScales[3u * gid + 2u]));
    float3 scaled = projected * (s * s);
    float3 noise = rotation * scaled;
    uint base = 3u * gid;
    positions[base] += u.params.x * gate * noise.x;
    positions[base + 1u] += u.params.x * gate * noise.y;
    positions[base + 2u] += u.params.x * gate * noise.z;
}
