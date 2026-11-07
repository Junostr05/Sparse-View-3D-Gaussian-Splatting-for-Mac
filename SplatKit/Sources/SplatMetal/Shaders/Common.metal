// Common.metal
// The four shader files are concatenated into one translation unit in the order
// Common, PrefixSum, RadixSort, Rasterizer. Only this file includes headers and
// defines shared structs.
#include <metal_stdlib>
using namespace metal;

#define SPLAT_TILE 16
#define SCAN_THREADS 256
#define SCAN_BLOCK 1024
#define SORT_THREADS 256
#define SORT_CHUNKS 8
#define SORT_BLOCK 2048
#define SORT_SIMDGROUPS 8

// Mirrors `ProjectUniforms` in ShaderTypes.swift (128 bytes).
struct ProjectUniforms {
    float4 rotRow0;      // xyz: row 0 of R, w: t.x
    float4 rotRow1;      // xyz: row 1 of R, w: t.y
    float4 rotRow2;      // xyz: row 2 of R, w: t.z
    float4 camCenter;    // xyz: camera center in world space
    float4 intrinsics;   // fx, fy, cx, cy
    float4 limits;       // near, far, eps2D, unused
    uint4 counts;        // gaussian count, width, height, tilesX
    uint4 shInfo;        // tilesY, evaluated SH degree, stored coefficients per gaussian K,
                         // w: 1 = accumulate parameter gradients (add) instead of overwriting them (Phase 4)
};

// Mirrors `MipFilterUniforms` in ShaderTypes.swift (16 bytes). Read by the projection kernels only in mip mode.
struct MipFilterUniforms {
    float4 info;         // x: 1 = mip filters on (Mip-Splatting 2D filter and 3D filter), 0 = classic; y: v3 (3D variance)
};

// Mirrors `DuplicateUniforms` in ShaderTypes.swift (16 bytes).
struct DuplicateUniforms {
    uint4 info;          // gaussian count, tilesX, tilesY, unused
};

// Mirrors `RasterUniforms` in ShaderTypes.swift (32 bytes).
struct RasterUniforms {
    float4 background;   // rgb; w: 1 = depth mode (Phase 4: write / read the rendered depth channel), else 0
    uint4 info;          // width, height, tilesX, tilesY
};

inline float3 loadSH(device const float* sh, ulong base, uint k) {
    ulong o = base + ulong(k) * 3ul;
    return float3(sh[o], sh[o + 1ul], sh[o + 2ul]);
}

// Raw SH sum (without +0.5), layout [gaussian][coefficient][channel].
inline float3 evaluateSH(device const float* sh, ulong base, uint degree, float3 dir) {
    float3 result = 0.28209479177387814f * loadSH(sh, base, 0);
    if (degree >= 1) {
        float dx = dir.x, dy = dir.y, dz = dir.z;
        result += 0.4886025119029199f * (-dy * loadSH(sh, base, 1)
                                         + dz * loadSH(sh, base, 2)
                                         - dx * loadSH(sh, base, 3));
        if (degree >= 2) {
            float xx = dx * dx, yy = dy * dy, zz = dz * dz;
            float xy = dx * dy, yz = dy * dz, xz = dx * dz;
            result += 1.0925484305920792f * xy * loadSH(sh, base, 4)
                    + (-1.0925484305920792f) * yz * loadSH(sh, base, 5)
                    + 0.31539156525252005f * (2.0f * zz - xx - yy) * loadSH(sh, base, 6)
                    + (-1.0925484305920792f) * xz * loadSH(sh, base, 7)
                    + 0.5462742152960396f * (xx - yy) * loadSH(sh, base, 8);
            if (degree >= 3) {
                result += (-0.5900435899266435f) * dy * (3.0f * xx - yy) * loadSH(sh, base, 9)
                        + 2.890611442640554f * xy * dz * loadSH(sh, base, 10)
                        + (-0.4570457994644658f) * dy * (4.0f * zz - xx - yy) * loadSH(sh, base, 11)
                        + 0.3731763325901154f * dz * (2.0f * zz - 3.0f * xx - 3.0f * yy) * loadSH(sh, base, 12)
                        + (-0.4570457994644658f) * dx * (4.0f * zz - xx - yy) * loadSH(sh, base, 13)
                        + 1.445305721320277f * dz * (xx - yy) * loadSH(sh, base, 14)
                        + (-0.5900435899266435f) * dx * (xx - 3.0f * yy) * loadSH(sh, base, 15);
            }
        }
    }
    return result;
}
