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
    /// tilesY, evaluated SH degree, stored SH coefficients per gaussian, unused.
    var shInfo: SIMD4<UInt32>
}

/// MSL `DuplicateUniforms`, 16 bytes.
struct DuplicateUniforms {
    /// gaussian count, tilesX, tilesY, unused.
    var info: SIMD4<UInt32>
}

/// MSL `RasterUniforms`, 32 bytes.
struct RasterUniforms {
    /// rgb background, w unused.
    var background: SIMD4<Float>
    /// width, height, tilesX, tilesY.
    var info: SIMD4<UInt32>
}
