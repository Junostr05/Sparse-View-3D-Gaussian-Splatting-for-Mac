// Adam optimizer kernel (Phase 3, Task 12). Mirrors CPUAdam.update in SplatCore: the arithmetic order
// is kept identical, so the GPU result differs only by Float rounding of the MSL compiler.
// Concatenated after Backward (see MetalContext.shaderFiles).
//
// One thread per float of a parameter group. The host dispatches the kernel once per group.
// Host-side precomputed values (Float, same as the CPU reference):
//   rates.x = headRate / bc1      (stepSize for floats with index % stride < headCount)
//   rates.y = tailRate / bc1      (stepSize for the other floats)
//   rates.z = 1 / sqrt(bc2)       (invSqrtBC2)
//   rates.w = epsilon
//   betas.x = beta1, betas.y = beta2
//   layout.x = total floats of the group (count * stride), layout.y = stride, layout.z = headCount

struct AdamUniforms {
    float4 rates;
    float4 betas;
    uint4 layout;
};

kernel void adamStep(
    device float* parameters [[buffer(0)]],
    device const float* gradients [[buffer(1)]],
    device float* firstMoment [[buffer(2)]],
    device float* secondMoment [[buffer(3)]],
    constant AdamUniforms& u [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.layout.x) {
        return;
    }
    uint stride = u.layout.y;
    uint headCount = u.layout.z;
    float stepSize = ((gid % stride) < headCount) ? u.rates.x : u.rates.y;
    float invSqrtBC2 = u.rates.z;
    float epsilon = u.rates.w;
    float beta1 = u.betas.x;
    float beta2 = u.betas.y;

    float g = gradients[gid];
    float mNew = beta1 * firstMoment[gid] + (1.0f - beta1) * g;
    float vNew = beta2 * secondMoment[gid] + (1.0f - beta2) * g * g;
    firstMoment[gid] = mNew;
    secondMoment[gid] = vNew;

    float denominator = sqrt(vNew) * invSqrtBC2 + epsilon;
    float delta = stepSize * mNew / denominator;
    parameters[gid] = parameters[gid] - delta;
}
