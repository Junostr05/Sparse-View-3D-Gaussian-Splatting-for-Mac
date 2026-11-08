// Display.metal — viewer display helpers: the depth colormap of the depth view.
//
// Concatenated last (see `MetalContext.shaderFiles`), so everything in Common.metal is visible here.
// Swift side: `SplatRenderer.renderDepthColorized`; the colour stops mirror `ViewerColors.turbo` in SplatCore.

// Mirrors `DepthColormapUniforms` in ShaderTypes.swift (48 bytes).
struct DepthColormapUniforms {
    float4 range;        // x: depth at the warm end (near), y: depth at the cold end (far), z: minimum alpha of a valid pixel, w: unused
    float4 background;   // rgb: background colour for pixels without coverage, w: unused
    uint4 size;          // width, height, unused, unused
};

// Colour stops of the turbo-like ramp, equally spaced from t = 0 to t = 1 (same values as ViewerColors.colormapStops).
constant float3 kViewerColormapStops[6] = {
    float3(0.23f, 0.30f, 0.75f),
    float3(0.16f, 0.56f, 0.94f),
    float3(0.10f, 0.80f, 0.70f),
    float3(0.60f, 0.90f, 0.25f),
    float3(0.99f, 0.72f, 0.15f),
    float3(0.85f, 0.12f, 0.10f),
};

inline float3 viewerColormap(float t) {
    float x = clamp(t, 0.0f, 1.0f) * 5.0f;
    uint i = min(uint(x), 4u);
    float f = x - float(i);
    return kViewerColormapStops[i] + (kViewerColormapStops[i + 1u] - kViewerColormapStops[i]) * f;
}

// Depth view: maps the rendered depth Draw / A of each pixel to the colormap, near = warm. Pixels whose alpha is
// below the minimum show the background.
kernel void depthColormap(device const float2* renderedDepth [[buffer(0)]],
                          constant DepthColormapUniforms& u [[buffer(1)]],
                          texture2d<float, access::write> outTexture [[texture(0)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.size.x || gid.y >= u.size.y) return;
    float3 rgb = u.background.xyz;
    float2 v = renderedDepth[gid.y * u.size.x + gid.x];
    if (v.y >= u.range.z && v.y > 0.0f) {
        float depth = v.x / v.y;
        if (isfinite(depth)) {
            float span = max(u.range.y - u.range.x, 1.0e-6f);
            float t = 1.0f - clamp((depth - u.range.x) / span, 0.0f, 1.0f);
            rgb = viewerColormap(t);
        }
    }
    outTexture.write(float4(rgb, 1.0f), gid);
}
