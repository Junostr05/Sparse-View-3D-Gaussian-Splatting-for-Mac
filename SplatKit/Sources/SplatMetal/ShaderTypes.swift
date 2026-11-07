/// Swift mirrors of the uniform structs in `Shaders/Common.metal`.
/// Only `SIMD4<Float>` / `SIMD4<UInt32>` fields are used so the layouts match MSL exactly.

/// MSL `ProjectUniforms`, 128 bytes.
struct ProjectUniforms {
    /// xyz: row 0 of the camera rotation, w: translation x.
    var rotRow0: SIMD4<Float>
    var rotRow1: SIMD4<Float>
    var rotRow2: SIMD4<Float>
    /// xyz: camera center in world space.
    var camCenter: SIMD4<Float>
    /// fx, fy, cx, cy.
    var intrinsics: SIMD4<Float>
    /// near, far, eps2D, unused.
    var limits: SIMD4<Float>
    /// gaussian count, width, height, tilesX.
    var counts: SIMD4<UInt32>
    /// tilesY, evaluated SH degree, stored SH coefficients per gaussian, w: 1 = accumulate parameter
    /// gradients instead of overwriting them (Phase 4). Layout unchanged.
    var shInfo: SIMD4<UInt32>
}

/// MSL `MipFilterUniforms`, 16 bytes. `info.x`: 1 = mip filters on, `info.y`: v3 (3D filter variance).
struct MipFilterUniforms {
    var info: SIMD4<Float>
}

/// MSL `SamplingCamera`, 80 bytes: one training camera for the sampling-rate kernel.
struct SamplingCameraUniforms {
    /// xyz: row 0 of the camera rotation, w: t.x.
    var rotRow0: SIMD4<Float>
    /// xyz: row 1 of the camera rotation, w: t.y.
    var rotRow1: SIMD4<Float>
    /// xyz: row 2 of the camera rotation, w: t.z.
    var rotRow2: SIMD4<Float>
    /// fx, fy, cx, cy.
    var intrinsics: SIMD4<Float>
    /// width, height, near, unused.
    var size: SIMD4<Float>
}

/// MSL `SamplingUniforms`, 16 bytes.
struct SamplingUniforms {
    /// Gaussian count, camera count, unused, unused.
    var counts: SIMD4<UInt32>
}

/// MSL `DuplicateUniforms`, 16 bytes.
struct DuplicateUniforms {
    /// gaussian count, tilesX, tilesY, unused.
    var info: SIMD4<UInt32>
}

/// MSL `RasterUniforms`, 32 bytes.
struct RasterUniforms {
    /// rgb background; w: 1 = depth mode (Phase 4 rendered depth), 0 otherwise. Layout unchanged.
    var background: SIMD4<Float>
    /// width, height, tilesX, tilesY.
    var info: SIMD4<UInt32>
}
