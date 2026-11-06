// Regularizers.metal — Phase 4: artifact regularizers (anisotropy gradient, scale cap).
//
// Concatenated after Depth.metal (see `MetalContext.shaderFiles`). Uniform structs for these kernels are
// declared at the top of this file and mirrored in GPURegularizers.swift. Buffer compaction is CPU-side
// (GaussianBuffers+Compaction.swift), so it has no kernel here.
//
// Mirrors `Regularizers` in SplatCore (Phase 4 P10). Ties in the argmax and argmin of the log-scales
// resolve to the lowest axis index, as on the CPU.

struct AnisotropyUniforms {
    /// x: weight λ, y: maxRatio, z: 1 / count, w unused.
    float4 params;
    /// x: Gaussian count, yzw unused.
    uint4 info;
};

struct ScaleCapUniforms {
    /// x: fraction, y: extent radius R, z: centre x, w: centre y.
    float4 a;
    /// x: centre z, yzw unused.
    float4 b;
    /// x: Gaussian count, yzw unused.
    uint4 info;
};

// One thread per Gaussian. Each thread writes only its own three gradient entries, so no atomics are needed.
kernel void anisotropyGradient(
    device const float* logScales [[buffer(0)]],
    device float* gradLogScales [[buffer(1)]],
    constant AnisotropyUniforms& u [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.info.x) {
        return;
    }
    uint base = 3 * gid;
    uint axisMax = 0;
    uint axisMin = 0;
    float maxLog = logScales[base];
    float minLog = maxLog;
    for (uint axis = 1; axis < 3; ++axis) {
        float value = logScales[base + axis];
        if (value > maxLog) {
            maxLog = value;
            axisMax = axis;
        }
        if (value < minLog) {
            minLog = value;
            axisMin = axis;
        }
    }
    if (axisMax == axisMin) {
        return;
    }
    float ratio = exp(maxLog - minLog);
    if (!(ratio > u.params.y)) {
        return;
    }
    float slope = u.params.x * u.params.z * ratio;
    gradLogScales[base + axisMax] += slope;
    gradLogScales[base + axisMin] -= slope;
}

// Hard projection after Adam: logScale_a = min(logScale_a, log(fraction · max(R, ‖μ − c‖))).
// Reads positions and writes logScales, which are different buffers. A non-positive bound is left unchanged.
kernel void scaleCap(
    device float* logScales [[buffer(0)]],
    device const float* positions [[buffer(1)]],
    constant ScaleCapUniforms& u [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.info.x) {
        return;
    }
    uint base = 3 * gid;
    float3 centre = float3(u.a.z, u.a.w, u.b.x);
    float3 offset = float3(positions[base], positions[base + 1], positions[base + 2]) - centre;
    float distance = sqrt(dot(offset, offset));
    float bound = u.a.x * max(u.a.y, distance);
    if (!(bound > 0.0f)) {
        return;
    }
    float cap = log(bound);
    for (uint axis = 0; axis < 3; ++axis) {
        if (logScales[base + axis] > cap) {
            logScales[base + axis] = cap;
        }
    }
}
