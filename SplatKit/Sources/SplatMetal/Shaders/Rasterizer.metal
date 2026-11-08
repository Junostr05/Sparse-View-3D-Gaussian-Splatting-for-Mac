// Forward splatting pipeline: projection, key duplication, tile ranges, tile rasterization.

kernel void projectGaussians(
    device const float* positions [[buffer(0)]],
    device const float* logScales [[buffer(1)]],
    device const float* rotations [[buffer(2)]],
    device const float* opacityLogits [[buffer(3)]],
    device const float* shCoeffs [[buffer(4)]],
    device float4* outMeanDepth [[buffer(5)]],
    device float4* outConicOpacity [[buffer(6)]],
    device float4* outColor [[buffer(7)]],
    device int4* outTileRect [[buffer(8)]],
    device uint* outTilesTouched [[buffer(9)]],
    constant ProjectUniforms& u [[buffer(10)]],
    constant MipFilterUniforms& mip [[buffer(11)]],
    device const float* samplingRates [[buffer(12)]],
    constant ViewModeUniforms& view [[buffer(13)]],
    device const float4* colorOverride [[buffer(14)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.counts.x) return;
    outTilesTouched[gid] = 0u;
    // Mip mode (Mip-Splatting): 3D filter on the scales, 2D filter with compensation. The mip 2D variance v2 is
    // passed in limits.z, in place of eps2D.
    const bool mipOn = mip.info.x > 0.5f;
    // Viewer display mode (VIEW_MODE_SPLAT leaves every step below unchanged).
    const uint viewMode = view.modeInfo.x;
    const bool useColorOverride = view.modeInfo.y != 0u;

    float3 pw = float3(positions[3 * gid], positions[3 * gid + 1], positions[3 * gid + 2]);
    // Viewer crop (off for training and every export path that does not set it): Gaussians whose centre lies
    // outside the box are culled, the same as the splats behind the near plane.
    if (view.cropCentre.w > 0.5f) {
        float3 offset = pw - view.cropCentre.xyz;
        float3 local = float3(dot(view.cropRow0.xyz, offset), dot(view.cropRow1.xyz, offset), dot(view.cropRow2.xyz, offset));
        if (any(fabs(local) > view.cropHalfExtents.xyz)) return;
    }
    float3 pc = float3(dot(u.rotRow0.xyz, pw) + u.rotRow0.w,
                       dot(u.rotRow1.xyz, pw) + u.rotRow1.w,
                       dot(u.rotRow2.xyz, pw) + u.rotRow2.w);
    float z = pc.z;
    if (z < u.limits.x || z > u.limits.y) return;

    float fx = u.intrinsics.x, fy = u.intrinsics.y, cx = u.intrinsics.z, cy = u.intrinsics.w;
    float W = float(u.counts.y), H = float(u.counts.z);
    float2 mean = float2(fx * pc.x / z + cx, fy * pc.y / z + cy);

    // Covariance.
    float4 qr = float4(rotations[4 * gid], rotations[4 * gid + 1], rotations[4 * gid + 2], rotations[4 * gid + 3]);
    float qlen2 = dot(qr, qr);
    float qw = 1.0f, qx = 0.0f, qy = 0.0f, qz = 0.0f;
    if (qlen2 > 0.0f) {
        float inv = 1.0f / sqrt(qlen2);
        qw = qr.x * inv; qx = qr.y * inv; qy = qr.z * inv; qz = qr.w * inv;
    }
    float3x3 Rq = float3x3(
        float3(1.0f - 2.0f * (qy * qy + qz * qz), 2.0f * (qx * qy + qw * qz), 2.0f * (qx * qz - qw * qy)),
        float3(2.0f * (qx * qy - qw * qz), 1.0f - 2.0f * (qx * qx + qz * qz), 2.0f * (qy * qz + qw * qx)),
        float3(2.0f * (qx * qz + qw * qy), 2.0f * (qy * qz - qw * qx), 1.0f - 2.0f * (qx * qx + qy * qy)));
    float3 s = float3(exp(logScales[3 * gid]), exp(logScales[3 * gid + 1]), exp(logScales[3 * gid + 2]));
    if (viewMode != VIEW_MODE_SPLAT) {
        s *= view.styleInfo.z;
    }
    float opacityFactor = 1.0f;
    if (mipOn) {
        // 3D filter: s' = sqrt(s² + v3/ν²), opacity × Π s/s'.
        float filterVariance = mip.info.y / (samplingRates[gid] * samplingRates[gid]);
        float3 sRaw = s;
        s = sqrt(sRaw * sRaw + filterVariance);
        opacityFactor = (sRaw.x / s.x) * (sRaw.y / s.y) * (sRaw.z / s.z);
    }
    float3x3 M = float3x3(Rq[0] * s.x, Rq[1] * s.y, Rq[2] * s.z);
    float3x3 sigma3 = M * transpose(M);

    // Clamped Jacobian.
    float tanFovX = 0.5f * W / fx;
    float tanFovY = 0.5f * H / fy;
    float limXPos = (W - cx) / fx + 0.3f * tanFovX;
    float limXNeg = cx / fx + 0.3f * tanFovX;
    float limYPos = (H - cy) / fy + 0.3f * tanFovY;
    float limYNeg = cy / fy + 0.3f * tanFovY;
    float tx = z * clamp(pc.x / z, -limXNeg, limXPos);
    float ty = z * clamp(pc.y / z, -limYNeg, limYPos);
    float3 j0 = float3(fx / z, 0.0f, -fx * tx / (z * z));
    float3 j1 = float3(0.0f, fy / z, -fy * ty / (z * z));

    // Camera rotation as a column-major matrix (columns of R).
    float3x3 Rc = float3x3(float3(u.rotRow0.x, u.rotRow1.x, u.rotRow2.x),
                           float3(u.rotRow0.y, u.rotRow1.y, u.rotRow2.y),
                           float3(u.rotRow0.z, u.rotRow1.z, u.rotRow2.z));
    float3 r0 = float3(dot(j0, Rc[0]), dot(j0, Rc[1]), dot(j0, Rc[2]));
    float3 r1 = float3(dot(j1, Rc[0]), dot(j1, Rc[1]), dot(j1, Rc[2]));
    float a = dot(r0, sigma3 * r0);
    float b = dot(r0, sigma3 * r1);
    float c = dot(r1, sigma3 * r1);
    float compensation = 1.0f;
    if (mipOn) {
        // 2D filter: Σ' = Σ + v2·I, opacity × sqrt(det Σ / det Σ') with det Σ clamped below.
        float rawDet = max(a * c - b * b, 1.0e-12f);
        a += u.limits.z;
        c += u.limits.z;
        float filteredDet = a * c - b * b;
        if (!(filteredDet > 0.0f)) return;
        compensation = sqrt(rawDet / filteredDet);
    } else {
        a += u.limits.z;
        c += u.limits.z;
    }

    // Points mode: a round disc of pointRadiusPixels. The covariance is s·I with s = r² / (2 ln 255), so a fully
    // opaque point falls to 1/255 alpha at its rim, and the tile rectangle uses the disc radius.
    float radiusOverride = 0.0f;
    if (viewMode == VIEW_MODE_POINTS) {
        float discRadius = view.styleInfo.x;
        float discVariance = discRadius * discRadius / (2.0f * VIEW_LOG_255);
        a = discVariance;
        b = 0.0f;
        c = discVariance;
        radiusOverride = ceil(discRadius);
    }

    float det = a * c - b * b;
    if (det <= 0.0f) return;
    float3 conic = float3(c / det, -b / det, a / det);

    float mid = 0.5f * (a + c);
    float lambda = mid + sqrt(max(0.1f, mid * mid - det));
    float radius = radiusOverride > 0.0f ? radiusOverride : ceil(3.0f * sqrt(lambda));
    if (radius <= 0.0f) return;

    int tilesX = int(u.counts.w);
    int tilesY = int(u.shInfo.x);
    float tile = float(SPLAT_TILE);
    float lo = 1.0e8f;
    int minX = clamp(int(clamp((mean.x - radius) / tile, -lo, lo)), 0, tilesX);
    int maxX = clamp(int(clamp((mean.x + radius + 15.0f) / tile, -lo, lo)), 0, tilesX);
    int minY = clamp(int(clamp((mean.y - radius) / tile, -lo, lo)), 0, tilesY);
    int maxY = clamp(int(clamp((mean.y + radius + 15.0f) / tile, -lo, lo)), 0, tilesY);
    if (maxX <= minX || maxY <= minY) return;

    // Color: the SH colour, or the per-Gaussian override of the points, shapes and heatmap modes.
    float3 color;
    if (useColorOverride) {
        color = colorOverride[gid].xyz;
    } else {
        float3 dir = normalize(pw - u.camCenter.xyz);
        ulong shBase = ulong(gid) * ulong(u.shInfo.z) * 3ul;
        float3 raw = evaluateSH(shCoeffs, shBase, u.shInfo.y, dir);
        color = max(raw + 0.5f, float3(0.0f));
    }

    float opacity = 1.0f / (1.0f + exp(-opacityLogits[gid]));
    if (mipOn) {
        opacity *= opacityFactor * compensation;
    }
    if (viewMode == VIEW_MODE_POINTS || viewMode == VIEW_MODE_HEATMAP) {
        opacity = 1.0f;
    } else if (viewMode == VIEW_MODE_SHAPES) {
        opacity *= view.styleInfo.y;
    }

    outMeanDepth[gid] = float4(mean.x, mean.y, z, radius);
    outConicOpacity[gid] = float4(conic.x, conic.y, conic.z, opacity);
    outColor[gid] = float4(color, 1.0f);
    outTileRect[gid] = int4(minX, minY, maxX, maxY);
    outTilesTouched[gid] = uint((maxX - minX) * (maxY - minY));
}

kernel void duplicateKeys(
    device const uint* tilesTouched [[buffer(0)]],
    device const uint* offsets [[buffer(1)]],
    device const int4* tileRect [[buffer(2)]],
    device const float4* meanDepth [[buffer(3)]],
    device ulong* keys [[buffer(4)]],
    device uint* values [[buffer(5)]],
    constant DuplicateUniforms& u [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.info.x) return;
    if (tilesTouched[gid] == 0u) return;
    uint pos = offsets[gid];
    int4 rect = tileRect[gid];
    uint depthBits = as_type<uint>(meanDepth[gid].z);
    uint tilesX = u.info.y;
    for (int ty = rect.y; ty < rect.w; ++ty) {
        for (int tx = rect.x; tx < rect.z; ++tx) {
            uint tileID = uint(ty) * tilesX + uint(tx);
            keys[pos] = (ulong(tileID) << 32) | ulong(depthBits);
            values[pos] = gid;
            pos += 1u;
        }
    }
}

// ranges[2*tile] = start, ranges[2*tile+1] = end into the sorted arrays; untouched tiles keep (0, 0).
// Start and end are written as separate uints: they are usually written by different threads, and a
// component store into a shared uint2 is not guaranteed to be free of read-modify-write races.
kernel void tileRanges(
    device const ulong* keys [[buffer(0)]],
    device uint* ranges [[buffer(1)]],
    constant uint& total [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total) return;
    uint tile = uint(keys[gid] >> 32);
    if (gid == 0u || uint(keys[gid - 1u] >> 32) != tile) {
        ranges[2u * tile] = gid;
    }
    if (gid == total - 1u || uint(keys[gid + 1u] >> 32) != tile) {
        ranges[2u * tile + 1u] = gid + 1u;
    }
}

kernel void rasterizeTiles(
    device const float4* meanDepth [[buffer(0)]],
    device const float4* conicOpacity [[buffer(1)]],
    device const float4* colors [[buffer(2)]],
    device const uint* values [[buffer(3)]],
    device const uint* ranges [[buffer(4)]],
    constant RasterUniforms& u [[buffer(5)]],
    texture2d<float, access::write> outTexture [[texture(0)]],
    device float* outFinalT [[buffer(6)]],
    device uint* outLastContributor [[buffer(7)]],
    device float2* outDepth [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup float2 sMean[SPLAT_TILE * SPLAT_TILE];
    threadgroup float4 sConic[SPLAT_TILE * SPLAT_TILE];
    // xyz: colour; w: camera-space depth z of the Gaussian (meanDepth.z). Only xyz feeds the colour
    // integral; the w channel is read by the depth accumulation alone.
    threadgroup float4 sColor[SPLAT_TILE * SPLAT_TILE];
    threadgroup atomic_uint doneCount;

    // Depth mode (Phase 4): RasterUniforms.background.w = 1 writes outDepth. Uniform for the whole grid.
    const bool depthMode = u.background.w > 0.5f;

    uint px = tgid.x * uint(SPLAT_TILE) + tid.x;
    uint py = tgid.y * uint(SPLAT_TILE) + tid.y;
    bool inside = px < u.info.x && py < u.info.y;
    float2 p = float2(float(px) + 0.5f, float(py) + 0.5f);

    uint tileIndex = tgid.y * u.info.z + tgid.x;
    uint2 range = uint2(ranges[2u * tileIndex], ranges[2u * tileIndex + 1u]);
    const uint batch = uint(SPLAT_TILE * SPLAT_TILE);

    if (lid == 0u) {
        atomic_store_explicit(&doneCount, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float3 rgb = float3(0.0f);
    // Σ αᵢ Tᵢ zᵢ over the blended samples (rendered depth, depth mode only).
    float depthAcc = 0.0f;
    float T = 1.0f;
    uint lastContributor = 0u;
    bool done = !inside;
    bool counted = false;

    for (uint base = range.x; base < range.y; base += batch) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint idx = base + lid;
        if (idx < range.y) {
            uint g = values[idx];
            sMean[lid] = meanDepth[g].xy;
            sConic[lid] = conicOpacity[g];
            sColor[lid] = float4(colors[g].xyz, meanDepth[g].z);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint n = min(batch, range.y - base);
        if (!done) {
            for (uint j = 0; j < n; ++j) {
                float2 d = sMean[j] - p;
                float4 co = sConic[j];
                float sigma = 0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) + co.y * d.x * d.y;
                if (sigma < 0.0f) continue;
                float alpha = min(0.999f, co.w * exp(-sigma));
                if (alpha < (1.0f / 255.0f)) continue;
                float nextT = T * (1.0f - alpha);
                if (nextT <= 1.0e-4f) {
                    done = true;
                    break;
                }
                rgb += sColor[j].xyz * (alpha * T);
                if (depthMode) {
                    depthAcc += sColor[j].w * (alpha * T);
                }
                T = nextT;
                lastContributor = base + j + 1u;
            }
        }
        if (done && !counted) {
            counted = true;
            atomic_fetch_add_explicit(&doneCount, 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (atomic_load_explicit(&doneCount, memory_order_relaxed) >= batch) {
            break;
        }
    }

    if (inside) {
        float3 outRGB = rgb + T * u.background.xyz;
        outTexture.write(float4(outRGB, 1.0f), uint2(px, py));
        // Training state for the backward pass: transmittance after the last blended sample and
        // the exclusive sorted index one past it (0 when nothing was blended).
        uint pixel = py * u.info.x + px;
        outFinalT[pixel] = T;
        outLastContributor[pixel] = lastContributor;
        // Rendered depth (gsplat RGB+ED convention): Draw = Σ αᵢ Tᵢ zᵢ and A = 1 − T_final.
        if (depthMode) {
            outDepth[pixel] = float2(depthAcc, 1.0f - T);
        }
    }
}
