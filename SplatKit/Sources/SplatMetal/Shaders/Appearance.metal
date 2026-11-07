// Per-image appearance compensation (v1.2). CPU reference: SplatCore `AppearanceTransform` and `AppearanceState`.
// Concatenated after Loss (see MetalContext.shaderFiles).
//
// Parameter buffers hold 6 floats per training view: gain r, g, b, then bias r, g, b.
// Per-view Adam moments use the same 6-float layout. The host counts the Adam steps of each view.
//
// Kernels, in the order the trainer encodes them for one view:
//   appearanceForward   A = R ⊙ gain + bias into a second render texture (the image loss reads it)
//   (image loss writes G = dL/dA into the gradient buffer)
//   appearanceBackward  in place: gradient = G ⊙ gain (dL/dR for the rasterizer, alpha 0); reduces Σ G ⊙ R and Σ G
//                       into one partial per threadgroup
//   appearanceUpdate    one threadgroup of 6 threads: sums the partials, adds the identity pull and runs Adam
// Reduction: `appearanceBackward` is dispatched with `groupCount` threadgroups of 256 threads; each thread strides
// over the pixels, and `appearanceUpdate` sums the `groupCount` partials of each of the 6 values in order.

constant uint kAppearanceThreads = 256;

struct AppearancePixelUniforms {
    uint pixelCount;
    uint width;
    uint viewIndex;
    uint groupCount;
};

struct AppearanceUpdateUniforms {
    uint viewIndex;
    uint groupCount;
    // x: learningRate / bias correction 1, y: 1 / sqrt(bias correction 2), z: identity scale 2λ/3, w: epsilon.
    float4 rates;
    // x: beta1, y: beta2.
    float4 betas;
};

kernel void appearanceForward(
    texture2d<float, access::read> rendered [[texture(0)]],
    texture2d<float, access::write> compensated [[texture(1)]],
    device const float* parameters [[buffer(0)]],
    constant AppearancePixelUniforms& u [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.pixelCount) {
        return;
    }
    uint x = gid % u.width;
    uint y = gid / u.width;
    uint base = u.viewIndex * 6;
    float3 gain = float3(parameters[base], parameters[base + 1], parameters[base + 2]);
    float3 bias = float3(parameters[base + 3], parameters[base + 4], parameters[base + 5]);
    float4 r = rendered.read(uint2(x, y));
    float3 scaled = r.xyz * gain;
    float3 a = scaled + bias;
    compensated.write(float4(a, r.w), uint2(x, y));
}

kernel void appearanceBackward(
    texture2d<float, access::read> rendered [[texture(0)]],
    device float4* imageGradient [[buffer(0)]],
    device const float* parameters [[buffer(1)]],
    device float* partials [[buffer(2)]],
    constant AppearancePixelUniforms& u [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    threadgroup float sums[6 * kAppearanceThreads];

    uint base = u.viewIndex * 6;
    float3 gain = float3(parameters[base], parameters[base + 1], parameters[base + 2]);
    float3 gainAcc = float3(0.0f);
    float3 biasAcc = float3(0.0f);

    uint stride = u.groupCount * kAppearanceThreads;
    for (uint i = group * kAppearanceThreads + tid; i < u.pixelCount; i += stride) {
        uint x = i % u.width;
        uint y = i / u.width;
        float3 r = rendered.read(uint2(x, y)).xyz;
        float3 g = imageGradient[i].xyz;
        gainAcc += g * r;
        biasAcc += g;
        imageGradient[i] = float4(g * gain, 0.0f);
    }

    sums[0 * kAppearanceThreads + tid] = gainAcc.x;
    sums[1 * kAppearanceThreads + tid] = gainAcc.y;
    sums[2 * kAppearanceThreads + tid] = gainAcc.z;
    sums[3 * kAppearanceThreads + tid] = biasAcc.x;
    sums[4 * kAppearanceThreads + tid] = biasAcc.y;
    sums[5 * kAppearanceThreads + tid] = biasAcc.z;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = kAppearanceThreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (uint k = 0; k < 6; ++k) {
                sums[k * kAppearanceThreads + tid] += sums[k * kAppearanceThreads + tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < 6) {
        partials[group * 6 + tid] = sums[tid * kAppearanceThreads];
    }
}

kernel void appearanceUpdate(
    device float* parameters [[buffer(0)]],
    device float* firstMoment [[buffer(1)]],
    device float* secondMoment [[buffer(2)]],
    device const float* partials [[buffer(3)]],
    constant AppearanceUpdateUniforms& u [[buffer(4)]],
    uint k [[thread_position_in_grid]])
{
    if (k >= 6) {
        return;
    }
    uint index = u.viewIndex * 6 + k;
    float reduced = 0.0f;
    for (uint g = 0; g < u.groupCount; ++g) {
        reduced += partials[g * 6 + k];
    }
    float p = parameters[index];
    // Identity pull: gains towards 1, biases towards 0.
    float pull = (k < 3) ? (p - 1.0f) : p;
    float gradient = reduced + u.rates.z * pull;

    float beta1 = u.betas.x;
    float beta2 = u.betas.y;
    float mNew = beta1 * firstMoment[index] + (1.0f - beta1) * gradient;
    float vNew = beta2 * secondMoment[index] + (1.0f - beta2) * gradient * gradient;
    firstMoment[index] = mNew;
    secondMoment[index] = vNew;

    float denominator = sqrt(vNew) * u.rates.y + u.rates.w;
    parameters[index] = p - u.rates.x * mNew / denominator;
}
