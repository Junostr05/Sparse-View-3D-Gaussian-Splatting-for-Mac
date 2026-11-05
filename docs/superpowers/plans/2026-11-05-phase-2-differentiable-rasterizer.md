# Phase 2 — Differentiable Rasterizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Per `CLAUDE.md`: workers never commit — they return their diff; the orchestrator reviews and commits.

**Goal:** Gradients of the rendered image with respect to every Gaussian parameter (positions, log-scales, raw quaternions, opacity logits, SH coefficients), computed by a Swift CPU reference and by hand-written Metal kernels, both verified against PyTorch-autograd goldens and finite differences.

**Architecture:** A Python tool (`Tools/golden/`) re-implements the Phase 1 forward math in PyTorch, cross-checks its projection and SH against gsplat's own PyTorch reference, and writes JSON golden cases (inputs, image, upstream image gradient, parameter gradients via autograd). `SplatCore` gains `CPURasterizer.backward` (pure Swift, Linux-testable). `SplatMetal` gains training-mode forward outputs (per-pixel final transmittance and last contributor) and backward kernels (rasterize backward with simdgroup-reduced float atomics, then projection/SH backward per Gaussian).

**Tech Stack:** Swift 6.2, Swift Testing, Metal (`atomic<float>`), Python 3 + PyTorch (CPU) + gsplat 1.5.3 (reference only, not shipped).

**Spec:** `docs/ARCHITECTURE.md` (decisions 6, 7; Phase 2) and the *Shared math* section of `docs/superpowers/plans/2026-11-05-phase-1-viewer.md`, which defines the forward pass being differentiated.

## Global Constraints

- Everything in Phase 1's Global Constraints still applies (macOS 26, Swift 6.2, library modules nonisolated, `SplatCore` stdlib + Foundation only, no third-party Swift dependencies, shared Swift/MSL structs without `SIMD3`, shaders in `mathMode = .safe`).
- Gradient conventions (identical in CPU, Metal and goldens):
  - `α = min(0.999, o·exp(−σ))`: when the clamp is active (`o·exp(−σ) > 0.999`), the gradients to `o` and `σ` are 0.
  - `color = max(SH + 0.5, 0)`: per channel, gradient 0 where the clamp is active.
  - Jacobian clamping of `x/z`, `y/z`: gradient of the clamped coordinate is 0 when clamped (the mean2D gradient still flows through `mean2D = f·x/z + c`, which is unclamped).
  - Gating decisions (culling, tile rect, `σ < 0`, `α < 1/255`, early termination) are treated as constants.
  - Gradients flow through the view direction of SH into the positions.
  - SH coefficients of bands above `min(shDegree, shDegreeLimit)` get gradient 0.
  - The background colour gets no gradient.
- Python tooling is never shipped and never imported by Swift. gsplat code is a reference only — do not copy it into the repository. Reference sources (Apache-2.0) for workers: `/tmp/claude-0/-home-user-Sparse-View-3D-Gaussian-Splatting-for-Mac/5719d6a1-3352-58bd-abda-ce56777b2fb7/scratchpad/gsplat-ref/` (`csrc/RasterizeToPixels3DGSBwd.cu`, `csrc/ProjectionEWA3DGSFused.cu`, `include/Utils.cuh`, `_torch_impl.py`). gsplat 1.5.3 uses a per-axis 3.33σ radius; **our** radius/tile rules from Phase 1 stay authoritative.
- Never port code from `graphdeco-inria/gaussian-splatting`.

## Review Focus

1. **Early termination pixels** (T drops below 1e-4) — backward must start from the last *blended* Gaussian, not the one that triggered termination. Golden case `occluded` exercises it; CPU/Metal tests must assert it.
2. **Clamped alpha (0.999)** — opacity and σ gradients must be exactly 0 for that sample. Golden case `occluded` (opacity logit ≥ 8) exercises it.
3. **Non-identity camera rotation and translation** — gradient transport through `Rᵀ` must be right. Golden case `rotated` uses a rotated, translated camera.
4. **Gaussians projected near the image border / clamped Jacobian** — golden case `rotated` includes Gaussians beyond 1.3× FOV so the clamp is active.
5. **Many Gaussians hitting the same pixel on the GPU** (atomic contention, >256 per tile so multiple batches) — Metal test `backwardMatchesCPUWithMultipleBatches` with ≥ 600 Gaussians in one tile.

---

### Task 1: Golden generator and fixtures

**Files:**
- Create: `Tools/golden/requirements.txt` (`torch` CPU, `gsplat==1.5.3`, `numpy`), `Tools/golden/README.md` (setup + regenerate commands), `Tools/golden/reference.py` (PyTorch forward), `Tools/golden/generate.py` (cross-check + writes fixtures)
- Create: `SplatKit/Tests/SplatCoreTests/Fixtures/golden_single.json`, `golden_multi.json`, `golden_rotated.json`, `golden_occluded.json`, and identical copies under `SplatKit/Tests/SplatMetalTests/Fixtures/`
- Create: `SplatKit/Tests/SplatCoreTests/GoldenCase.swift` (Codable loader) + `GoldenLoaderTests.swift`; copy `GoldenCase.swift` into `SplatKit/Tests/SplatMetalTests/`
- Modify: `SplatKit/Package.swift` (test targets get `resources: [.copy("Fixtures")]`), root `.gitignore` (`!**/Tests/**/Fixtures/**` instead of `!Tests/**/Fixtures/**`)

**Interfaces:**
- Produces: JSON schema (all arrays flat, Float32 values):
  `name, width, height, fx, fy, cx, cy, rotation (9, row-major world→camera), translation (3), background (3), shDegree, shDegreeLimit, positions, logScales, rotations, opacityLogits, sh, image (W·H·3, row-major), imageGradient (W·H·3), gradients { positions, logScales, rotations, opacityLogits, sh }`.
  Swift: `struct GoldenCase: Decodable { …same fields…; static func load(_ name: String) throws -> GoldenCase; func cloud() throws -> GaussianCloud; func camera() -> Camera; func settings() -> RasterSettings; var imageGradientPixels: [SIMD3<Float>] }`.
- Cases (fixed seeds): `single` (1 Gaussian, degree 0, 16×16, identity camera); `multi` (30 Gaussians, degree 3, 40×28, identity camera); `rotated` (40 Gaussians, degree 2, 32×24, camera rotated ~20° about two axes and translated, ≥ 3 Gaussians outside 1.3× FOV); `occluded` (60 Gaussians, degree 1, 32×32, several with opacity logit ≥ 8 stacked on the axis so early termination and the 0.999 clamp both occur). `imageGradient` uniform random in [−1, 1].

- [ ] **Step 1:** Create a venv (`python3 -m venv .venv-golden`, ignored via `.venv` rule — add `.venv-golden/` to `.gitignore`), install `torch` from `https://download.pytorch.org/whl/cpu`, `gsplat==1.5.3 --no-deps`, `numpy jaxtyping typing_extensions rich`.
- [ ] **Step 2:** `reference.py`: `forward(params, camera, settings) -> image` in float32 PyTorch implementing the Phase 1 *Shared math* literally, vectorized over Gaussians and pixels (alpha matrix `[pixels, gaussians]`, gating masks computed under `torch.no_grad()`, per-pixel depth order, transmittance via cumulative product, termination mask "first sample whose nextT ≤ 1e-4 and everything after it is excluded"). Clamps via `torch.clamp` so autograd yields the conventions above.
- [ ] **Step 3:** `generate.py --out <dir>...`: for each case build params, run `reference.forward`, assert against gsplat's `_quat_scale_to_covar_preci`, `_world_to_cam`, `_persp_proj` (means2d, cov2d + eps2d) and `_eval_sh_bases_fast` (colors) with max abs diff ≤ 1e-5, then `image.backward(imageGradient)` and write JSON. Print a summary (max |grad| per parameter group, number of clamped-alpha and terminated pixels per case; abort if `occluded` has zero of either, or `rotated` has no clamped-Jacobian Gaussian).
- [ ] **Step 4:** Run it for both fixture directories; files must be < 400 KB each.
- [ ] **Step 5:** Swift: `GoldenLoaderTests.loadsAllCases` (all four decode; array lengths consistent with `GaussianCloud` init) and `forwardMatchesGolden` (`CPURasterizer.render` vs `image`, max abs diff ≤ 1e-4 for every case). Run `cd SplatKit && swift test`.
- [ ] **Step 6:** Return diff (commit: `Add PyTorch golden generator and gradient fixtures`).

### Task 2: CPU reference backward

**Files:**
- Create: `SplatKit/Sources/SplatCore/GaussianGradients.swift`, `SplatKit/Sources/SplatCore/CPUBackward.swift`
- Create: `SplatKit/Tests/SplatCoreTests/CPUBackwardTests.swift`

**Interfaces:**
- Consumes: `GaussianCloud`, `Camera`, `RasterSettings`, `Projection`, `SphericalHarmonics`, `GoldenCase`.
- Produces:
  - `public struct GaussianGradients: Sendable, Equatable { var positions, logScales, rotations, opacityLogits, sh: [Float]; init(zeroLike cloud: GaussianCloud) }` (same layouts as `GaussianCloud`).
  - `extension CPURasterizer { static func backward(_ cloud: GaussianCloud, camera: Camera, settings: RasterSettings, imageGradient: [SIMD3<Float>]) -> GaussianGradients }`.

Approach: re-run the forward per pixel recording the blended samples (index, α, T, unclamped flags); traverse them back to front accumulating `dL/dcolor_i`, `dL/dα_i`, then `dL/dopacity`, `dL/dσ` → `dL/dconic`, `dL/dmean2D`; per Gaussian chain into `Σ2D` (via `dΣ2D = −C·dC·C` for conic `C`, symmetrised), `T = J·R`, `Σ3D`, `M = R_q·S`, scales (`·scale` for log), normalized and raw quaternion, camera-space mean (incl. the `J` terms and the unclamped `mean2D` terms), world position (`Rᵀ`), and SH (basis × colour gradient, plus direction gradient through `normalize`). Use `Double` accumulators internally.

- [ ] **Step 1: Write the failing tests:**

```swift
@Test(arguments: ["single", "multi", "rotated", "occluded"])
func backwardMatchesGolden(name: String) throws
    // every gradient entry: |cpu − golden| ≤ 1e-4 + 1e-3·|golden|
@Test func finiteDifferencesMatch() throws
    // "multi" case; loss = Σ imageGradient·image; for 40 randomly chosen parameters (seed 7) with
    // |analytic| > 1e-3: central difference with h = 1e-3 in Float; relative error ≤ 5e-2
@Test func clampedSamplesHaveZeroOpacityGradient() throws
    // single Gaussian, logit 10, pixel at its mean only: opacity gradient == 0
@Test func higherSHBandsGetZeroGradientWhenLimited() throws
    // degree-3 cloud, shDegreeLimit 1: sh gradients of coefficients 4…15 are exactly 0
```

- [ ] **Step 2–4:** run (fail) → implement → `swift test --filter CPUBackwardTests` (pass) and full `swift test`.
- [ ] **Step 5: Return diff** (commit: `Add CPU reference backward pass`).

### Task 3: Metal backward

**Files:**
- Modify: `SplatKit/Sources/SplatMetal/Shaders/Rasterizer.metal` (forward writes per-pixel `finalT` and `lastContributor`), `SplatKit/Sources/SplatMetal/SplatRenderer.swift`
- Create: `SplatKit/Sources/SplatMetal/Shaders/Backward.metal` (append to the concatenation order after `Rasterizer`), `SplatKit/Sources/SplatMetal/GradientBuffers.swift`
- Create: `SplatKit/Tests/SplatMetalTests/BackwardTests.swift`
- Modify: `SplatKit/Sources/SplatMetal/MetalContext.swift` (`shaderFiles` += `Backward`)

**Interfaces:**
- Consumes: Phase 1 renderer, `GaussianGradients`, `CPURasterizer.backward`, `GoldenCase`.
- Produces:
  - `public final class GradientBuffers { positions, logScales, rotations, opacityLogits, sh: MTLBuffer; func toGaussianGradients(count: Int, shDegree: Int) -> GaussianGradients }` (`.storageModeShared`, Float).
  - `SplatRenderer.encodeBackward(imageGradient: MTLBuffer, into gradients: GradientBuffers, commandBuffer: MTLCommandBuffer) throws(MetalError)` — uses the state of the most recent `render` call (same camera/settings); `imageGradient` is `W·H·4` floats (RGBA layout, A ignored).
  - Convenience `SplatRenderer.backward(imageGradient: [SIMD3<Float>]) throws(MetalError) -> GaussianGradients` (allocates, encodes, waits).

Kernels: (1) `rasterizeBackward`: same tile/threadgroup layout as forward; each pixel starts at its `lastContributor` and walks the tile range backwards in batches (staged in threadgroup memory, last batch first); per Gaussian sample it computes `dmean2D (2)`, `dconic (3)`, `dopacity (1)`, `dcolor (3)`, reduces them across the simdgroup with `simd_sum` and lets lane 0 `atomic_fetch_add_explicit` into per-Gaussian 2D-gradient buffers (`device atomic<float>*`). (2) `projectBackward`: one thread per Gaussian, recomputes the projection and applies the same chain rule as the CPU reference to write parameter gradients (no atomics needed). Gradient buffers are zeroed with a blit `fill` before encoding.

- [ ] **Step 1: Write the failing tests** (suite gated on Metal like Phase 1):

```swift
@Test(arguments: ["single", "multi", "rotated", "occluded"])
func backwardMatchesGolden(name: String) throws   // ≤ 1e-4 + 1e-3·|golden|
@Test func backwardMatchesCPU() throws             // random 200-Gaussian scene (Phase 1 helper), 64×64
@Test func backwardMatchesCPUWithMultipleBatches() throws // 600 Gaussians inside one 16×16 tile, 16×16 image
@Test func finiteDifferencesOnGPU() throws         // "multi": 20 parameters, GPU forward via renderImage on a
                                                   // rebuilt renderer per perturbation, h = 1e-3, rel ≤ 5e-2
```

- [ ] **Step 2: Run on CI (macOS)** → compile failure.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: CI green** (Linux + macOS jobs).
- [ ] **Step 5: Return diff** (commit: `Add Metal backward pass for the tile rasterizer`).
