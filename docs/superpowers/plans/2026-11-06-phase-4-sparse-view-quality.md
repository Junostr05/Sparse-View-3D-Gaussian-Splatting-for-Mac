# Phase 4 — Sparse-View Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Per `CLAUDE.md`: workers never commit or push. They return their diff, and the orchestrator (Sonnet) reviews it, runs the tests and commits. Tasks marked `Escalate-to: opus-review` get an additional Opus review before commit.

**Goal:** Turn the Phase 3 trainer into a sparse-view (3–6 images) reconstructor that is free of needles and floaters and gives geometrically consistent novel views. It adds a depth prior (Pearson, plus L1 for LiDAR), dense depth initialization, depth-supervised pseudo-views, artifact regularizers (anisotropy, scale cap, isolation pruning) and leave-one-out early stopping followed by a full retrain.

**Architecture:** Everything numeric first exists as a pure-Swift CPU reference in `SplatCore` (Linux-tested, with finite differences and PyTorch goldens): depth types, alignment, rendered depth forward/backward, the depth loss, the regularizers, dense init, pseudo-views, the leave-one-out scheduler and a CPU end-to-end run on a synthetic ground-truth scene. Depth enters through a `DepthProvider` seam, so nothing downstream needs a neural network. Supplied depth files (`.sdpt`), LiDAR from ARKit captures and a synthetic provider all go through it. The Core ML model (Depth Anything V2 Small) is the last, separable provider, in the macOS-only `SplatDepth` module. On the GPU, the existing rasterizer kernels gain a depth mode: expected depth becomes an extra composited channel, which reuses the same tile traversal and the free accumulator slot 9. The depth loss and the regularizers are small new kernel files. `GPUTrainer` gets staged hooks, and `TrainingSession` runs the two-phase leave-one-out schedule.

**Tech Stack:** Swift 6.2 (strict concurrency), Swift Testing, Metal (runtime-compiled MSL 3.0, `mathMode = .safe`), Core ML (only `SplatDepth`), SwiftUI (Liquid Glass), XcodeGen, GitHub Actions (`macos-26`, paravirtual GPU), Python 3 + PyTorch CPU (goldens only).

**Spec:** `docs/ARCHITECTURE.md` (decisions 3, 4, 5, 13, 14, 15; Phase 4; *Training defaults*; *Licensing rules*), and the Phase 2/3 plans for gradient conventions (`docs/superpowers/plans/2026-11-05-phase-2-differentiable-rasterizer.md`) and the trainer structure (`docs/superpowers/plans/2026-11-05-phase-3-training.md`, D1–D13).

## Architectural decisions (summary, with justification)

| # | Topic | Decision | Why |
|---|---|---|---|
| P1 | Depth source seam | `protocol DepthProvider: Sendable { func depthMap(for view: TrainingView) throws -> DepthMap? }` in `SplatCore`. Implementations: `SuppliedDepthProvider` (files from the project: LiDAR or precomputed), `SyntheticScene` ground truth (tests and the benchmark), `MonocularDepthEstimator` (Core ML, `SplatDepth`, macOS) and `FirstAvailableDepthProvider` (chain: supplied first, then model). `DepthPreparation` turns maps into per-view `DepthPrior`s at view resolution. Synchronous API: Core ML prediction is synchronous, and callers already run in `@concurrent` code. | Everything downstream (alignment, loss, init, pseudo-views) is testable on Linux without a network or a model. |
| P2 | Depth file format | **`.sdpt`**, little-endian: 24-byte header `"SDPT"`, `UInt16 version = 1`, `UInt8 kind` (0 metric, 1 relative depth, 2 relative inverse depth), `UInt8 flags` (bit 0: confidence plane present, bit 1: estimated by a model), `UInt32 width`, `UInt32 height`, `8` reserved zero bytes. Then `width·height` `Float32` row-major (non-finite or ≤ 0 = invalid), then optionally `width·height` `UInt8` confidence (ARKit 0/1/2). Stored in the project as `depth/<image name>.sdpt`. | Trivial to decode with Foundation on Linux and identical on macOS (no PNG/zlib); carries the kind, which a 16-bit PNG or PFM does not; Python writes it with `struct`. |
| P3 | Resolution | Depth is resampled on the CPU to the **final view size** at preparation time. The aspect ratio must match within 1 %, otherwise the view's depth is rejected with a reported reason (not an error). Bilinear, but only when all 4 taps are valid and their max/min ratio ≤ 1.1; otherwise nearest valid tap. Confidence uses nearest. LiDAR pixels with confidence 0 are invalid. | Prevents flying pixels at depth edges; the GPU only ever sees priors aligned with its render grid. |
| P4 | Alignment domain | Relative inverse depth (Depth Anything's output) is fitted **in inverse-depth space**: `s·p + t ≈ 1/z_SfM`. Relative depth is fitted in depth space. Metric depth is trusted as-is only for ARKit captures (metric poses). Metric depth on a COLMAP project is fitted scale-only (`t = 0`, depth space), because COLMAP's scale is arbitrary. | The model's affine invariance holds in disparity. Fitting in depth space would use the wrong noise model and bias near/far. |
| P5 | Alignment solver | Deterministic LMedS over point pairs (all pairs for n ≤ 64, else 1 024 seeded pairs), then MAD inlier gating (2.5·1.4826·MAD) and two least-squares refits on the inliers. Occlusion pre-filter: per 8×8-pixel cell keep SfM points with `z ≤ 1.15·min z`. It needs ≥ 8 points, inliers ≥ max(8, 0.3 n) and `s > 0`. With a wrong-sign or failed fit, the other relative kind is tried once (`kindCorrected`). With < 8 points, the view is `unaligned`: Pearson still works (it is affine-invariant), but no dense init, L1 or pseudo-view prior comes from that view. | Robust with tens of points and heavy occlusion outliers; no randomness except a seeded generator. |
| P6 | Rendered depth | Expected depth **as an extra composited channel of the existing kernels** in a depth mode: `Draw = Σ αᵢTᵢzᵢ`, `A = Σ αᵢTᵢ = 1 − T_final`, written as `float2` per pixel. The loss uses `D = Draw / A`. The backward input is a per-pixel `float2 (∂L/∂Draw, ∂L/∂A)`. `∂L/∂zᵢ` goes into accumulator **slot 9** (unused today), and `projectBackward` adds `slot9 · W[2,:]` to the position gradient. Depth mode is a uniform flag (`RasterUniforms.background.w = 1`). `zᵢ` travels in `sColor.w` of the existing threadgroup arrays (the forward writes `colors.w = 1`, which no kernel reads). | A second kernel pass would repeat the full tile traversal (≈ 2× backward cost). As a channel it costs ~3 % forward and ~10–15 % backward (one more float atomic per sample, 8 B written + 8 B read per pixel). The viewer path is unchanged (flag off). |
| P7 | Depth loss | Per view, over the mask `A > 0.3 ∧ prior valid`: `L = λ_P·(1 − ρ(x, y)) + λ_L1·mean|D − m| / median(m)`. Here `x = D` (depth-domain prior) or `x = 1/max(D, near)` (inverse-domain prior), `y` is the prior in its own domain, and ρ is the Pearson coefficient (population moments). L1 applies **only to measured metric (LiDAR) depth**. Degenerate cases (n < 64, σ ≈ 0) give loss 0 and gradient 0 and are flagged. Weights *(initial)*: λ_P = 0.05 from iteration 0, λ_L1 = 0.1. | Decision 3 verbatim; no raw L1 against relative monocular depth; the NaN guard is explicit. |
| P8 | Dense init | Back-project each **aligned/metric** view on a stride grid (targeting `maxPoints / views` per view), drop depth-edge pixels (3×3 max/min > 1.05) and points that violate free space in other views (point more than 5 % in front of another view's surface, with violations ≥ supports), dedupe with a voxel hash (voxel = median pixel footprint, first point wins, SfM points inserted first), cap with a seeded subset at `min(150 000, budget/2)`, then `GaussianInitializer.fromPoints` (k-d-tree scales, opacity 0.1). With no aligned view it falls back to the Phase 3 initializer. | Linux-testable CPU code; deterministic; leaves room for densification under the budget. |
| P9 | Pseudo-views | A fixed pool built once at trainer init: for each metric view, its nearest metric view by camera centre (unique pairs); skip a pair whose optical axes differ by > 60°. For t = k/4 (k = 1…3): centre lerp, camera-to-world quaternion slerp, intrinsics lerp, half resolution. Prior = the parents' metric depth **forward-warped** with a z-buffer (Pearson in depth domain, no RGB loss). Rejected if coverage < 30 % or if the 5th-percentile depth is < 0.2× the parents' median (camera inside or too close to geometry). Used every 2nd iteration from iteration 2 000, λ = 0.05. The gradient **accumulates** into the training view's gradients before the single Adam step. It does **not** feed densification statistics. | FSGS-style regularization without running a network during training (FSGS runs monocular depth on renders; our warp needs only the aligned priors). One Adam step per iteration keeps D5 semantics. |
| P10 | Regularizers | **Anisotropy**: `λ_a/n · Σ max(0, s_max/s_min − 10)`, λ_a = 0.1, analytic gradient added to `∂L/∂logScale` by a small kernel after `projectBackward`. **Scale cap**: a hard projection after Adam, `logScale ≤ log(0.05·max(R, ‖μ − c‖))` (cap grows with distance from the extent centre, so far background is not shredded); no gradient. **Isolation pruning**: every 500 iterations from 1 000, prune Gaussians whose 4th-NN distance > 8 × median 4th-NN distance (median from 2 048 seeded samples, exact via a uniform hash grid; test via a radius count in a grid with cell = threshold). At most 5 % per pass, and never empties the model. It runs on the CPU over the shared-memory positions while the GPU is idle, then compacts params/m/v in place. | The scale cap as a projection needs no gradient verification. Isolation runs ≤ 20 times per training; an O(n) CPU hash grid takes ~0.1–0.3 s at 300 k on an M1 and reuses Phase 3's compaction rules. The GPU grid hash is deferred unless profiling shows > 2 % of wall time (see Open risks). |
| P11 | Leave-one-out | Metric: **SSIM** (same window as the loss) on the held-out view, plus PSNR for display. **LPIPS deferred** (pretrained VGG/AlexNet weights, a Core ML conversion, 10–60 MB, a separate licence review). Held-out view: PCA of camera centres, exclude both extremes along the principal axis, pick the camera nearest the median projection; `heldOutViews[0]` overrides. Disabled with < 4 views (3 views would leave two extreme cameras). Run 1 excludes the view (also from dense init and pseudo-views), evaluates every 250 iterations, uses an EMA (0.5) of SSIM with min-delta 0.002 and patience 4, and allows no stop before 1 500. Run 2 starts from a fresh initial cloud on all views with the **same nominal schedule** (LR decay over `iterations`) and stops at `round(best·N/(N−1))`. Cost ≈ 1.8–2.2× a single run. | Decision 5 with a defensible metric; the N/(N−1) factor keeps the per-view visit count of the chosen iteration. |
| P12 | Trainer hooks | `AuxiliaryLoss` becomes staged: `.pixelGradient` (before backward: depth loss), `.parameterGradient` (after backward, before Adam: anisotropy), `.postOptimizer` (after Adam: scale cap). Each gets an `AuxiliaryContext` with buffers. Pseudo-views and isolation pruning live in `GPUTrainer` itself (they need extra forward passes or a CPU wait). | The Phase 3 hook had no access to any buffer and could not express the stages Phase 4 needs. |
| P13 | Module placement | Alignment, preparation, dense init, pseudo-views, regularizers and the scheduler live in `SplatCore/Depth/` and `SplatCore/Training/` (Linux). `SplatDepth` holds **only** Core ML inference. | Deviates from the ARCHITECTURE module sketch ("SplatDepth: … scale/shift alignment"): alignment must be Linux-testable. |
| P14 | Configuration migration | `TrainingConfiguration` gains `configurationRevision` (current 2). A manifest without it (Phase 3) decodes `regularizers` as the **new default set** (all features on), because Phase 3 always stored `[]` for the unused extension point. | Decision 14 (on by default) must apply to existing projects too. |
| P15 | Model | Depth Anything V2 **Small** only (Apache-2.0). Obtained by `Tools/models/fetch_depth_model.sh` from Apple's Core ML conversion on Hugging Face (`apple/coreml-depth-anything-v2-small`, `DepthAnythingV2SmallF16.mlpackage`, pinned revision + SHA-256, licence checked by the person running it). The fallback is `Tools/models/convert_depth_anything.py` (coremltools, from `depth-anything/Depth-Anything-V2-Small-hf`, ImageNet normalization baked into the model). It is never committed (`App/Models/` is gitignored); it is bundled when present (`optional: true` in `project.yml`). Input 518×392 RGB, landscape (portrait rotated 90° before and after), stretched; output = relative inverse depth, resampled to view size. **Model absent → the provider is not created and depth comes from supplied files only; with none, depth features stay off.** | Decision 13 licensing; the cloud container cannot fetch or convert (no Mac, likely no model-host access). |

## Global Constraints

- Everything in the Phase 1–3 Global Constraints still applies: macOS 26, `swift-tools-version: 6.2`, Swift 6 strict concurrency. Library modules (`SplatCore`, `SplatIO`, `SplatMetal`, `SplatTraining`, **`SplatDepth`**) use default nonisolated isolation; never set `defaultIsolation(MainActor.self)` on a library target. Only the app is `@MainActor` by default, with `NonisolatedNonsendingByDefault` on, so CPU-heavy app work must be `@concurrent`.
- `SplatCore` and `SplatIO` import only the Swift stdlib + `Foundation` and build and test on Linux (`cd SplatKit && swift test`). No `simd`, `ImageIO`, `CoreGraphics`, `Metal`, `CoreML` there, not even behind `#if`. Core ML only in `SplatDepth` and the app.
- No third-party dependencies. No `DispatchQueue` in new code.
- **Xcode type-checker rule:** Xcode rejects expressions the Linux toolchain accepts. Write numeric code (alignment, Pearson, pose interpolation, back-projection, test fixtures) as explicitly typed `let` steps and loops. No long mixed SIMD/Float expressions and no `map { SIMD3(Float(...) / 255, ...) }` closures. A green Linux `swift test` does not prove the macOS build compiles.
- **Gradient verification (non-negotiable):** every new backward path (rendered-depth backward CPU+Metal, Pearson/L1 gradient CPU+Metal, anisotropy gradient CPU+Metal) is checked three ways: against the Swift CPU reference, against PyTorch goldens (`Tools/golden/`, Task 3) and with central finite differences in Double. No backward kernel is merged without these tests. The scale cap is a projection (no gradient), and isolation pruning is a discrete decision (tested against a brute-force/k-d-tree oracle).
- **`ShaderTypes.swift` / `Common.metal` layouts are not changed.** Phase 4 uses two fields that are unused today: `RasterUniforms.background.w` (1 = depth mode) and `ProjectUniforms.shInfo.w` (1 = accumulate parameter gradients instead of overwriting). Only Task 14 edits their doc comments. New kernel families declare their uniform structs at the top of their own `.metal` file and mirror them in their own Swift file (Phase 3 rule). Shared structs contain only `Float`, `UInt32`, `Int32`, `SIMD4<Float>`, `SIMD4<UInt32>`.
- New shader files `Depth.metal` and `Regularizers.metal` are appended after `Densify` in `MetalContext.shaderFiles` **once, in Task 13**. No other task edits `MetalContext.swift`.
- Accumulator layout per Gaussian (12 floats) after Phase 4: 0..2 colour, 3 opacity, 4..6 conic, 7..8 mean2D, **9 depth (∂L/∂z_cam)**, 10..11 unused.
- All buffers `.storageModeShared` via `MetalContext.makeBuffer`. CPU reads or writes of GPU buffers happen only after the last command buffer touching them has completed (isolation pruning and compaction run only at a wait point).
- Determinism: randomness only via `SeededGenerator` / `SplatRandom`; CPU references accumulate in `Double`; ties are always broken by lowest index.
- Typed throws where a module uses them. New error enums: `DepthMapError` (SplatIO), `DepthResampleError` (SplatCore). **Do not add cases to `SplatIOError` or `TrainingError` beyond those listed in a task's Interfaces.** Depth problems are **reported, never thrown**, once a dataset loads: missing, unalignable or mismatched depth turns that feature off for that view.
- CI budget (paravirtual GPU): each new macOS unit test ≤ 10 s, images ≤ 64×64; the three quality tests in Task 21 ≤ 60 s each. Metal suites are gated on Metal availability (`MetalTestSupport.swift`). Linux CPU end-to-end tests ≤ 30 s in debug.
- Licensing: gsplat (Apache-2.0) is the reference for the rendered depth convention (camera z, expected depth = accumulated depth / alpha, `RGB+ED`). FSGS / SparseGS / DNGaussian are implemented **from the papers**; **never port code from `graphdeco-inria/gaussian-splatting`** or from repositories derived from it (FSGS and SparseGS code is derived from it, so read only the papers). Only Depth Anything V2 **Small** may ship. No model files, depth datasets, `*.ply`, `*.splat` or `.mlpackage` in git.
- Never claim a macOS/Metal test passed unless CI ran it. Workers on Linux return macOS-only diffs unverified and say so.

## Review Focus

1. **No depth available at all** (COLMAP project, no supplied files, model not installed). Training must behave exactly like Phase 3: depth/pseudo/dense-init silently off, regularizers still on, the inspector says why. Tests: Task 5 `prepareWithoutProviderLeavesViewsUntouched`; Task 11 `cpuTrainerWithoutDepthMatchesPhase3Order`; Task 17 `trainsWithoutAnyDepthPrior`.
2. **Depth maps whose resolution differs from the images** (LiDAR 256×192 vs 1920×1440, maxImageSide downsampling, a portrait/landscape mismatch). Resample when the aspect matches within 1 %, otherwise reject that view's depth with a reason. Tests: Task 2 `resamplesLiDARToImageSize`, `rejectsAspectMismatch`, `edgeAwareBilinearKeepsDiscontinuity`; Task 5 `mismatchedDepthIsReportedNotThrown`.
3. **Relative depth with the wrong sign or inversion** (a file declared inverse that holds depth, or vice versa). The fit detects `s ≤ 0` or a poor fit and tries the other kind, and the Pearson domain follows the corrected kind. Tests: Task 5 `detectsDepthDeclaredAsInverse`, `rejectsNegativeScale`; Task 6 `pearsonDomainFollowsPrior`.
4. **Too few SfM points for alignment** (0, 1–7, or mostly occluded). The view is `unaligned`, Pearson still applies, and dense init and pseudo-views skip it. Tests: Task 5 `tooFewPointsLeavesViewUnaligned`, `occludedPointsAreRejectedAsOutliers`; Task 8 `fallsBackWithoutAlignedViews`.
5. **Pearson with constant or empty depth** (a fresh model with A < 0.3 everywhere, a flat prior, a single valid pixel). Loss 0, gradient 0, `degenerate = true`, never NaN. Tests: Task 6 `constantRenderedDepthIsDegenerateNotNaN`, `emptyMaskIsDegenerate`; Task 15 `gpuDegenerateCasesAreZero`.
6. **Held-out view selection with 3 views / a straight line of cameras / identical centres.** Early stopping is disabled below 4 views (a single run with the configured iterations), an extreme camera is never held out, and a degenerate configuration returns nil. Tests: Task 10 `disabledBelowFourViews`, `neverHoldsOutAnExtremeCamera`, `coincidentCentresReturnNil`.
7. **Early stopping on a noisy metric.** A single lucky eval must not end run 1, and no stop happens before `minimumIterations`. Tests: Task 10 `singleSpikeDoesNotStop`, `noStopBeforeMinimumIterations`, `stopsAfterPatienceWithoutImprovement`.
8. **Pseudo-view cameras behind or inside objects** (interpolating across a wide baseline). Rejected by the angle, coverage and near-depth checks. Tests: Task 9 `rejectsWideAnglePairs`, `rejectsCameraInsideGeometry`, `coverageBelowThresholdIsRejected`.

## Task overview and dependencies

| Task | Title | Platform | Depends on | Shared files touched |
|---|---|---|---|---|
| 1 | Phase 4 configuration and depth types | Linux-testable | — | `TrainingConfiguration.swift`, `TrainingDataset.swift`, `TrainingMemoryBudget.swift` |
| 2 | `.sdpt` codec, depth resampling, project depth files | Linux-testable | 1 | `ProjectBundle.swift` |
| 3 | PyTorch goldens: rendered depth, Pearson, anisotropy | Linux (Python + torch) | — | `Tools/golden/*`, both `GoldenCase.swift`, both `Fixtures/` |
| 4 | CPU rendered depth forward/backward | Linux-testable | 3 | `CPURasterizer.swift`, `CPUBackward.swift` |
| 5 | Depth alignment, providers, preparation | Linux-testable | 1, 2 | — |
| 6 | CPU depth loss (Pearson + L1) | Linux-testable | 1, 3 | — |
| 7 | CPU regularizers and isolation pruning | Linux-testable | 1, 3 | — |
| 8 | Dense depth initialization | Linux-testable | 1 | — |
| 9 | Pseudo-view generation | Linux-testable | 1, 8 | — |
| 10 | Held-out selection, scheduler, SSIM metric | Linux-testable | 1 | `ImageLoss.swift` |
| 11 | Synthetic scene + CPU trainer Phase 4 integration | Linux-testable | 4, 6, 7, 8, 9 | `CPUTrainer.swift` |
| 12 | Synthetic dataset tool: depth + novel views | Linux (Python) | 2 | `Tools/synthetic/*` |
| 13 | Metal setup: register shader files | macOS-CI-only | — | `MetalContext.swift` |
| 14 | Metal rendered depth fwd/bwd | macOS-CI-only, **Escalate-to: opus-review** | 3, 4 | `Rasterizer.metal`, `Backward.metal`, `SplatRenderer.swift`, `ShaderTypes.swift` (comments), `Common.metal` (comments) |
| 15 | Metal depth loss | macOS-CI-only | 6, 13 | `Depth.metal` |
| 16 | Metal regularizers + buffer compaction | macOS-CI-only | 7, 13 | `Regularizers.metal` |
| 17 | `GPUTrainer` A: staged hooks, depth, regularizers, isolation | macOS-CI-only, **Escalate-to: opus-review** | 11, 14, 15, 16 | `GPUTrainer.swift` |
| 18 | `GPUTrainer` B: pseudo-views, held-out evaluation | macOS-CI-only | 9, 10, 17 | `GPUTrainer.swift` |
| 19 | `TrainingSession` leave-one-out | macOS-CI-only | 10, 18 | `TrainingSession.swift`, `TrainingProgress.swift` |
| 20 | `SplatDepth`: Core ML Depth Anything V2 Small | macOS-CI-only (+ manual on a Mac) | 1, 5 | `Package.swift`, `.gitignore` |
| 21 | Quality tests and benchmark | macOS-CI-only | 11, 19 | `.github/workflows/benchmark.yml` (new) |
| 22 | App: inspector, depth import/estimation, metrics | macOS-CI-only (build) + manual | 2, 5, 8, 19, 20 | `project.yml`, `App/Project/*` |

Parallel waves (no two tasks in a wave touch the same file): **A** = 1, 3, 13 · **B** = 2, 4, 6, 7, 8, 10 · **C** = 5, 9, 12, 14, 15, 16 · **D** = 11, 20 · **E** = 17 · **F** = 18 · **G** = 19 · **H** = 21, 22. Serialized hot files: `GPUTrainer.swift` (17 → 18), `TrainingConfiguration.swift` (only 1), `MetalContext.swift` (only 13), `Package.swift` (only 20), `SplatRenderer.swift`/`Rasterizer.metal`/`Backward.metal` (only 14), `CPUTrainer.swift` (only 11), `ProjectBundle.swift` (only 2), goldens (only 3). Tasks 4, 6 and 7 read fixtures from Task 3 but do not edit the generator.

New SplatCore code goes into `SplatKit/Sources/SplatCore/Depth/` (depth-related) or `Training/` (regularizers, scheduler). One new test file per task in `SplatKit/Tests/SplatCoreTests/`, `SplatIOTests/`, `SplatMetalTests/` or `SplatTrainingTests/`.

Verification commands:
- Linux: the `export PATH=…` printed by `Tools/install-swift-linux.sh`, then `cd SplatKit && swift test --filter <Suite>` and the full `swift test`.
- macOS: the orchestrator pushes; CI job `macos` (`swift test` in `SplatKit`, then the app build) must be green. Batch macOS tasks into as few pushes as the dependencies allow.

---

### Task 1: Phase 4 configuration and depth types

**Platform:** Linux-testable · **Depends on:** —

**Files:**
- Modify: `SplatKit/Sources/SplatCore/Training/TrainingConfiguration.swift`, `TrainingDataset.swift`, `TrainingMemoryBudget.swift`
- Create: `SplatKit/Sources/SplatCore/Depth/DepthTypes.swift`
- Test: `SplatKit/Tests/SplatCoreTests/Phase4ConfigurationTests.swift`

**Interfaces (Produces):**
```swift
// DepthTypes.swift
public enum DepthKind: UInt8, Codable, Sendable { case metric = 0, relativeDepth = 1, relativeInverseDepth = 2 }
public struct DepthMap: Sendable, Equatable {
    public var width: Int; public var height: Int; public var kind: DepthKind
    public var values: [Float]            // row-major; non-finite or ≤ 0 = invalid
    public var confidence: [UInt8]?       // ARKit 0/1/2, same size
    public var isEstimated: Bool          // produced by a model (flags bit 1)
    public init(width: Int, height: Int, kind: DepthKind, values: [Float], confidence: [UInt8]? = nil, isEstimated: Bool = false)
    public func isValid(_ index: Int, minimumConfidence: UInt8 = 1) -> Bool   // finite, > 0, confidence ≥ min when present
}
public enum DepthDomain: Sendable, Equatable { case depth, inverseDepth }
/// Per-view prior at view resolution, ready for the loss.
public struct DepthPrior: Sendable, Equatable {
    public var width: Int; public var height: Int
    public var correlationTarget: [Float]   // in `domain`; ≤ 0 or non-finite = invalid
    public var domain: DepthDomain
    public var metricDepth: [Float]?        // camera-z depth in scene units (aligned or LiDAR); 0 = invalid
    public var isMeasuredMetric: Bool       // LiDAR only: enables the L1 term
    public var l1Normalizer: Float          // median of valid metricDepth when isMeasuredMetric, else 0
    public init(width: Int, height: Int, correlationTarget: [Float], domain: DepthDomain,
                metricDepth: [Float]? = nil, isMeasuredMetric: Bool = false)       // computes l1Normalizer
}

// TrainingDataset.swift — type change of the unused Phase 3 extension point:
public var depth: DepthPrior?      // was [Float]?; init parameter `depth: DepthPrior? = nil`

// TrainingConfiguration.swift
public enum RegularizerKind: String, Codable, Sendable, CaseIterable { case depthPearson, depthL1, anisotropy, scaleCap, isolationPrune, pseudoViews }
public struct DepthPriorSettings: Codable, Sendable, Equatable {
    public var pearsonWeight: Float = 0.05; public var l1Weight: Float = 0.1
    public var alphaMask: Float = 0.3; public var minimumPixels = 64; public var startIteration = 0
    public var minimumAlignmentPoints = 8; public var minimumInlierFraction: Float = 0.3
    public init()
}
public struct PseudoViewSettings: Codable, Sendable, Equatable {
    public var startIteration = 2_000; public var interval = 2; public var weight: Float = 0.05
    public var perPair = 3; public var resolutionDivisor = 2; public var maxAngleDegrees: Float = 60
    public var minimumCoverage: Float = 0.3; public var minimumDepthFraction: Float = 0.2
    public init()
}
public struct ArtifactRegularizerSettings: Codable, Sendable, Equatable {
    public var anisotropyWeight: Float = 0.1; public var maxAxisRatio: Float = 10
    public var scaleCapFraction: Float = 0.05
    public var isolationNeighbors = 4; public var isolationFactor: Float = 8
    public var isolationStartIteration = 1_000; public var isolationInterval = 500
    public var isolationMaxFraction: Float = 0.05; public var isolationSampleCount = 2_048
    public init()
}
public struct DenseInitSettings: Codable, Sendable, Equatable {
    public var isEnabled = true; public var maxPoints = 150_000; public var edgeRatio: Float = 1.05
    public var consistencyTolerance: Float = 0.05; public var includeSfMPoints = true
    public init()
}
public struct EarlyStoppingPolicy: Codable, Sendable, Equatable {
    public var isEnabled = true; public var minimumViews = 4; public var evalInterval = 250
    public var patience = 4; public var minDelta: Double = 0.002; public var minimumIterations = 1_500
    public var emaWeight: Double = 0.5
    public init()
}
// New TrainingConfiguration properties (all decodeIfPresent with defaults):
public static let currentRevision = 2
public var configurationRevision = TrainingConfiguration.currentRevision
public var regularizers: Set<RegularizerKind> = Set(RegularizerKind.allCases)   // default changes from []
public var depthPrior = DepthPriorSettings()
public var pseudoViews = PseudoViewSettings()
public var artifacts = ArtifactRegularizerSettings()
public var denseInit = DenseInitSettings()
public var earlyStopping = EarlyStoppingPolicy()
public func isEnabled(_ kind: RegularizerKind) -> Bool

// TrainingMemoryBudget.swift
public static func reservedDepthBytes(viewSizes: [SIMD2<Int>], pseudoViewSizes: [SIMD2<Int>]) -> Int
// = Σ_views 8·w·h (target + metric) + 16·wmax·hmax (float2 depth out + float2 depth grad) + Σ_pseudo 4·w·h
```
Migration rule (P14): when `configurationRevision` is absent from the JSON, `regularizers` = `Set(RegularizerKind.allCases)` regardless of the stored value, and the decoded revision is set to `currentRevision`. When present, the stored set is used.

- [ ] **Step 1: Write the failing tests** (`@Suite struct Phase4ConfigurationTests`)
```swift
@Test func defaultsMatchArchitecture()            // all six regularizers on, maxAxisRatio 10, earlyStopping.minimumViews 4, pearsonWeight 0.05, shDegree 1, opacityResetInterval 0
@Test func phase3ManifestGetsNewDefaults()        // decode {"regularizers": [], "iterations": 50} → regularizers == all cases, iterations 50, revision 2
@Test func revision2KeepsStoredRegularizers()     // {"configurationRevision": 2, "regularizers": ["anisotropy"]} → [.anisotropy]
@Test func phase4ConfigurationRoundTrips()
@Test func depthMapValidity()                     // 0, −1, NaN, +inf invalid; confidence 0 invalid with min 1; 2.5 valid
@Test func priorComputesL1Normalizer()            // metric [1, 2, 3, 0] measured → 2; not measured → 0
@Test func reservedDepthBytesFormula()            // views 4×(10×10), pseudo 2×(5×5): 4·800 + 16·100 + 2·100 = 5000
```
- [ ] **Step 2: Run** `swift test --filter Phase4ConfigurationTests` → fails to compile.
- [ ] **Step 3: Implement.** Grep for `depth:` call sites of `TrainingView.init` (none pass it today) and keep the Phase 3 tests green (`defaultsMatchArchitecture` in `TrainingFoundationTests` asserts the Phase 3 defaults; update its `regularizers` expectation to the new default set and say so in the diff).
- [ ] **Step 4: Run** the suite and the full `swift test` → pass.
- [ ] **Step 5: Return diff** (commit: `Add Phase 4 configuration, depth types and depth memory budget`).

### Task 2: `.sdpt` codec, depth resampling, project depth files

**Platform:** Linux-testable · **Depends on:** 1

**Files:**
- Create: `SplatKit/Sources/SplatIO/DepthMapCodec.swift`, `SplatKit/Sources/SplatCore/Depth/DepthResampling.swift`
- Modify: `SplatKit/Sources/SplatIO/ProjectBundle.swift`
- Test: `SplatKit/Tests/SplatIOTests/DepthMapCodecTests.swift`, `SplatKit/Tests/SplatCoreTests/DepthResamplingTests.swift`

**Interfaces (Produces):**
```swift
public enum DepthMapError: Error, Equatable { case badMagic, unsupportedVersion(Int), unknownKind(UInt8), invalidSize(width: Int, height: Int), truncated(expected: Int, actual: Int) }
public enum DepthMapCodec {
    public static let fileExtension = "sdpt"
    public static func encode(_ map: DepthMap) -> Data
    public static func decode(_ data: Data) throws(DepthMapError) -> DepthMap   // width, height in 1...16384; exact length check
}
public enum DepthResampleError: Error, Equatable { case aspectMismatch(source: SIMD2<Int>, target: SIMD2<Int>) }
extension DepthMap {
    /// P3 rules. `allowAspectChange` (model output only) skips the 1 % check.
    public func resampled(width: Int, height: Int, allowAspectChange: Bool = false) throws(DepthResampleError) -> DepthMap
    public func rotatedClockwise() -> DepthMap            // for portrait model inference (Task 20)
    public func rotatedCounterClockwise() -> DepthMap
}
// ProjectBundle.swift
extension ProjectLayout { public static let depthDirectory = "depth"; public static func depthPath(forImage name: String) -> String }   // "depth/<name>.sdpt"
public struct CheckpointRecord { …; public var heldOutView: String?; public var heldOutPSNR: Double?; public var heldOutSSIM: Double?; public var bestIteration: Int? }  // optional, synthesized Codable stays backward-compatible
extension ProjectBundle {
    public static func importARKit(capture: Data, imageFiles: [String: Data], depthFiles: [String: Data], into contents: ProjectContents) -> ProjectContents
    // depthFiles keyed by image name; the existing 3-argument overload forwards with [:]. Clearing for import also removes depth/.
    public static func withDepthMaps(_ contents: ProjectContents, _ maps: [String: DepthMap]) -> ProjectContents   // writes depth/<name>.sdpt
    public static func depthMaps(_ contents: ProjectContents) -> (maps: [String: DepthMap], failures: [String: DepthMapError])
}
```
Resampling algorithm (target pixel centre → source coordinate `u = (x + 0.5)·sw/tw − 0.5`, same for v): take the 4 integer neighbours (clamped). If all 4 are valid and `max/min ≤ 1.1`, use the bilinear value. Otherwise use the nearest valid tap by the rounded position, falling back to the nearest valid of the 4 in raster order, else 0. Confidence: nearest. Aspect check: `|sw/sh − tw/th| / (tw/th) ≤ 0.01`.

- [ ] **Step 1: Write the failing tests**
```swift
// DepthMapCodecTests
@Test func roundTripsAllKindsWithAndWithoutConfidence()
@Test func headerIsTwentyFourBytesLittleEndian()     // encode 2×1 metric [1, 2]: bytes 0..3 "SDPT", 4..5 = 01 00, 6 = 0, 12..15 = width 2 LE, total 24 + 8
@Test func rejectsBadMagicVersionKindAndTruncation()
@Test func estimatedFlagRoundTrips()
// DepthResamplingTests
@Test func identityResampleIsExact()
@Test func resamplesLiDARToImageSize()               // 4×3 plane z = 2 → 16×12, all 2 (1e-6)
@Test func rejectsAspectMismatch()                   // 4×3 → 16×9 throws aspectMismatch
@Test func edgeAwareBilinearKeepsDiscontinuity()     // left half 1, right half 10 → no value in (1.2, 9) in the output
@Test func invalidTapsAreNotInterpolated()           // a 0 tap is never averaged; isolated invalid → nearest valid
@Test func rotationsAreInverse()
// ProjectBundleTests additions (in DepthMapCodecTests to avoid editing the existing file)
@Test func arkitImportStoresDepthFiles()
@Test func depthMapsReportsCorruptFilesByName()
@Test func checkpointRecordDecodesWithoutHeldOutFields()
```
- [ ] **Step 2–4:** run (fail) → implement → `swift test --filter "DepthMapCodecTests|DepthResamplingTests"` and full `swift test` pass.
- [ ] **Step 5: Return diff** (commit: `Add sdpt depth map format, edge-aware resampling and project depth files`). The format table (P2) goes into the doc comment of `DepthMapCodec`; no separate doc file.

### Task 3: PyTorch goldens — rendered depth, Pearson, anisotropy

**Platform:** Linux (Python 3 + PyTorch CPU; set up per `Tools/golden/README.md`) · **Depends on:** —

**Files:**
- Modify: `Tools/golden/reference.py`, `Tools/golden/generate.py`, `Tools/golden/README.md`, `SplatKit/Tests/SplatCoreTests/GoldenCase.swift`, `SplatKit/Tests/SplatMetalTests/GoldenCase.swift`
- Create (generated): `golden_depth_single.json`, `golden_depth_occluded.json`, `golden_pearson.json`, `golden_anisotropy.json` in both `Fixtures/` directories

**Interfaces (Produces):**
- `reference.forward(..., with_depth=True)` additionally returns `depth_raw (H, W) = Σ αᵢTᵢzᵢ` and `alpha (H, W) = 1 − T_final`, with the existing gating conventions (early termination excludes the terminating sample, constant gating, alpha clamp). `zᵢ` = camera-space z of the mean.
- Depth cases: the Phase 2 `single` and `occluded` scenes plus seeded upstream gradients `imageGradient (H·W·3)`, `depthGradient (H·W·2) = (∂L/∂Draw, ∂L/∂A)`. Autograd of `Σ g·image + Σ gD·Draw + Σ gA·A`. JSON adds `renderedDepth`, `renderedAlpha`, `depthGradient` next to the existing fields. Cross-check against gsplat 1.5.3 `rasterization(render_mode="RGB+D")` (accumulated depth) and its alphas, within 1e-5. Exit non-zero otherwise.
- `golden_pearson.json`: two sub-cases (`depth` and `inverseDepth` domain), 12×9 pixels, seeded `Draw`, `A` (some below 0.3), prior with some invalid pixels, a metric prior with normalizer for the L1 term, weights λ_P = 0.05, λ_L1 = 0.1, alphaMask 0.3, near 0.01. Outputs `loss`, `pearson`, `l1`, `gradDraw`, `gradA` (formulas of Task 6, written directly in PyTorch, autograd for the gradient).
- `golden_anisotropy.json`: 16 Gaussians with seeded logScales (half above ratio 10), λ 0.1, max ratio 10. Outputs `loss`, `gradLogScales` (autograd of `λ/n Σ relu(max/min − 10)`).
- Swift `GoldenCase` gains optional `renderedDepth: [Float]?`, `renderedAlpha: [Float]?`, `depthGradient: [Float]?`; new `PearsonGolden` and `AnisotropyGolden` Decodable structs in both `GoldenCase.swift` files with a `static func load(_ name: String) throws -> Self` matching the existing loader.

- [ ] **Step 1:** Extend `reference.py` (no change to existing outputs; the Phase 2 fixtures must regenerate byte-identically: verify with `git diff --stat` on the old fixtures = empty).
- [ ] **Step 2:** Add the four cases to `generate.py`; keep each file < 400 KB.
- [ ] **Step 3:** Run `python Tools/golden/generate.py --out SplatKit/Tests/SplatCoreTests/Fixtures SplatKit/Tests/SplatMetalTests/Fixtures` → exits 0, prints the gsplat depth cross-check differences (≤ 1e-5).
- [ ] **Step 4:** Add the Swift decoding structs plus one test `GoldenLoaderTests.phase4GoldensDecode` (in the existing `GoldenLoaderTests.swift`: allowed, as only this task edits it) asserting the array lengths. `swift test --filter GoldenLoaderTests` passes on Linux.
- [ ] **Step 5: Return diff** (commit: `Add Phase 4 goldens for rendered depth, Pearson loss and anisotropy`).

### Task 4: CPU rendered depth forward/backward

**Platform:** Linux-testable · **Depends on:** 3

**Files:**
- Modify: `SplatKit/Sources/SplatCore/CPURasterizer.swift`, `SplatKit/Sources/SplatCore/CPUBackward.swift`
- Test: `SplatKit/Tests/SplatCoreTests/DepthRasterizerTests.swift`

**Interfaces (Produces):**
```swift
public struct DepthRender: Sendable, Equatable {
    public var raw: [Float]     // Σ αᵢTᵢzᵢ, row-major W·H
    public var alpha: [Float]   // 1 − T_final
    public func normalized(minimumAlpha: Float) -> [Float]   // raw/alpha where alpha > minimumAlpha, else 0
}
extension CPURasterizer {
    public static func renderWithDepth(_ cloud: GaussianCloud, camera: Camera, settings: RasterSettings) -> (image: RGBImage, depth: DepthRender)
    /// `depthGradient`: per pixel (∂L/∂Draw, ∂L/∂A); nil = Phase 3 behaviour, bit-identical.
    public static func backwardWithScreenSpace(_ cloud: GaussianCloud, camera: Camera, settings: RasterSettings,
                                               imageGradient: [SIMD3<Float>], depthGradient: [SIMD2<Float>]?) -> ScreenSpaceBackward
}
```
The existing `backwardWithScreenSpace(_:camera:settings:imageGradient:)` forwards with `nil`. `renderWithDepth.image` must equal `render` bit-for-bit.

Backward algorithm (pixel stage, back to front, Double). It extends the existing loop; `T_f` = final transmittance, `behindZ` starts at 0:
```text
for sample i (reversed):
    screen[i].depth += α_i · T_i · gD                      // ∂L/∂z_i  (z_i = ProjectedGaussian.depth)
    dAlpha += (z_i·T_i − behindZ/(1 − α_i)) · gD           // added to the existing colour dAlpha
            + (T_f / (1 − α_i)) · gA                       // ∂A/∂α_i, closed form
    behindZ += z_i · α_i · T_i
    (then the existing clamp guard and the σ/opacity/conic/mean2D chain on the combined dAlpha)
GaussianChain.propagate: position gradient += screen.depth · (W[2][0], W[2][1], W[2][2])   // ∂z_cam/∂p_world
```
Note: the `screen.depth` accumulation happens **before** the `clamped` guard (z is not gated by the alpha clamp).

- [ ] **Step 1: Write the failing tests**
```swift
@Test func renderWithDepthImageMatchesRender()                 // bit-identical on the Phase 2 golden scenes
@Test func singleOpaqueGaussianDepthIsItsZ()                    // one large Gaussian, opacity logit 8, z = 3: normalized depth at centre ≈ 3 (1e-4)
@Test func depthMatchesGoldens()                                // golden_depth_single/occluded: raw, alpha (1e-5)
@Test func depthBackwardMatchesGoldens()                        // all parameter groups (rel 1e-4, abs 1e-6)
@Test func depthBackwardMatchesFiniteDifferences()              // occluded scene, image gradient 0, random depthGradient; central FD in Double, h 1e-3 on positions/logScales/rotations/opacityLogits; rel 2e-3
@Test func combinedGradientIsSumOfParts()                       // (img, depth) == (img, nil) + (0, depth) within 1e-6
@Test func nilDepthGradientIsBitIdenticalToPhase3()
```
- [ ] **Step 2–4:** run (fail) → implement → `swift test --filter "DepthRasterizerTests|CPUBackwardTests"` and full `swift test` pass.
- [ ] **Step 5: Return diff** (commit: `Add rendered depth to the CPU reference rasterizer and backward`).

### Task 5: Depth alignment, providers, preparation

**Platform:** Linux-testable · **Depends on:** 1, 2

**Files:**
- Create: `SplatKit/Sources/SplatCore/Depth/DepthAlignment.swift`, `DepthProvider.swift`, `DepthPreparation.swift`
- Test: `SplatKit/Tests/SplatCoreTests/DepthAlignmentTests.swift`

**Interfaces (Produces):**
```swift
public protocol DepthProvider: Sendable { func depthMap(for view: TrainingView) throws -> DepthMap? }
public struct SuppliedDepthProvider: DepthProvider { public init(maps: [String: DepthMap]) }          // keyed by view name
public struct FirstAvailableDepthProvider: DepthProvider { public init(_ providers: [any DepthProvider]) } // first non-nil; a throwing provider is skipped
public enum AlignmentFailure: Sendable, Equatable { case tooFewPoints(Int), noPositiveFit, tooFewInliers(inliers: Int, of: Int) }
public struct AlignmentResult: Sendable, Equatable {
    public var scale: Double; public var shift: Double; public var fittedKind: DepthKind
    public var inliers: Int; public var samples: Int; public var medianRelativeError: Double
    public var kindCorrected: Bool
}
public enum DepthAlignment {
    /// Samples: prior value at the projected pixel (nearest) of each visible SfM point and that point's camera z.
    public static func samples(points: [SIMD3<Float>], camera: Camera, map: DepthMap) -> [(prior: Double, z: Double)]  // occlusion pre-filter per 8×8 cell
    public static func fit(_ samples: [(prior: Double, z: Double)], kind: DepthKind, scaleOnly: Bool,
                           minimumPoints: Int, minimumInlierFraction: Double, seed: UInt64) -> Result<AlignmentResult, AlignmentFailure>
    /// Metric camera-z depth from an aligned relative map; invalid → 0. Inverse: z = 1/(s·p + t) if s·p + t > 1/zMax.
    public static func metricDepth(_ map: DepthMap, alignment: AlignmentResult, zMax: Double) -> [Float]
}
public enum ViewDepthStatus: Sendable, Equatable { case none, measuredMetric, aligned(AlignmentResult), unaligned(AlignmentFailure), rejected(String) }
public struct DepthPreparationReport: Sendable, Equatable {
    public var views: [ViewDepthStatus]
    public var alignedOrMetricCount: Int { get }
    public var hasAnyPrior: Bool { get }
}
public enum DepthPreparation {
    /// Resamples each provided map to its view, aligns, and returns the dataset with `TrainingView.depth` set.
    /// `trustMetric`: true for ARKit sources (P4). Provider errors and resample errors become `.rejected(reason)`.
    public static func prepare(_ dataset: TrainingDataset, provider: (any DepthProvider)?, trustMetric: Bool,
                               settings: DepthPriorSettings, seed: UInt64) -> (dataset: TrainingDataset, report: DepthPreparationReport)
}
```
Fit algorithm (P5), in Double. Target `q = 1/z` for `.relativeInverseDepth`, else `q = z`:
```text
n < minimumPoints → .tooFewPoints(n)
candidates: scaleOnly → s_k = q_k/p_k for all k (t = 0); else pairs (all i<j if n ≤ 64, else 1024 pairs from SeededGenerator(seed)),
            skip |p_i − p_j| < 1e-9·max(|p_i|, |p_j|, 1e-12); s = (q_i − q_j)/(p_i − p_j), t = q_i − s·p_i; keep only s > 0
none → .noPositiveFit
best = argmin over candidates of median_k |s·p_k + t − q_k|   (first wins on ties)
repeat 2×: σ = 1.4826·median|r| + 1e-12; inliers = |r| ≤ 2.5σ; least squares on inliers (scaleOnly: s = Σpq/Σpp);
           if the 2×2 system is singular keep the previous (s, t); if s ≤ 0 → .noPositiveFit
inliers < max(minimumPoints, ceil(minimumInlierFraction·n)) → .tooFewInliers
medianRelativeError = median over inliers of |r|/|q|
```
Kind correction in `prepare`: for a relative map, if the declared kind fails or has `inliers/n < 0.5`, fit the other relative kind (reinterpret the same values). Accept it if it succeeds with `inliers/n ≥ 0.5` and `inliers > 1.5 ×` the declared kind's inliers (0 for a failed fit); then `kindCorrected = true` and `fittedKind` = the other kind. `zMax = 100 × median SfM z of the samples`. Prior construction: measured metric → `correlationTarget = metric`, domain `.depth`, `isMeasuredMetric = true`; aligned → `correlationTarget` = the map values in the fitted kind's domain (`.inverseDepth` for inverse), `metricDepth` = aligned; unaligned relative → `correlationTarget` = values in the declared domain (`.depth` for relativeDepth or metric, `.inverseDepth` for inverse), `metricDepth = nil`.

- [ ] **Step 1: Write the failing tests** (synthetic plane/box geometry built in the test file; points from a known scene)
```swift
@Test func recoversAffineInverseDepth()             // prior = 0.7/z + 0.2 at 200 points → scale 0.7, shift 0.2 (rel 1e-6), inliers 200
@Test func recoversWithThirtyPercentOutliers()      // 30 % of samples replaced by random q → scale within 1 %
@Test func occludedPointsAreRejectedAsOutliers()    // a back wall's points behind a front box are removed by the cell pre-filter or gating; scale within 2 %
@Test func tooFewPointsLeavesViewUnaligned()        // 0 and 7 points → .tooFewPoints; prepare → .unaligned, prior still set (Pearson), metricDepth nil
@Test func rejectsNegativeScale()                   // fit(samples with prior = −0.5/z + 3, kind .relativeInverseDepth) → .noPositiveFit
@Test func detectsDepthDeclaredAsInverse()          // values = 2z + 1 declared .relativeInverseDepth → kindCorrected, fittedKind .relativeDepth, domain .depth
@Test func scaleOnlyForMetricOnCOLMAP()             // metric map p = 3·z, trustMetric false → scale-only fit q = z ≈ s·p: s = 1/3 (rel 1e-6), shift 0
@Test func trustedMetricIsMeasured()                // trustMetric true → .measuredMetric, isMeasuredMetric, no fit
@Test func prepareWithoutProviderLeavesViewsUntouched()
@Test func mismatchedDepthIsReportedNotThrown()     // 16:9 map for a 4:3 view → .rejected, view.depth nil
@Test func fitIsDeterministic()                     // same seed → identical result; n > 64 path
```
- [ ] **Step 2–4:** run (fail) → implement → `swift test --filter DepthAlignmentTests` and full `swift test` pass.
- [ ] **Step 5: Return diff** (commit: `Add depth providers, robust scale/shift alignment and depth preparation`).

### Task 6: CPU depth loss (Pearson + L1)

**Platform:** Linux-testable · **Depends on:** 1, 3

**Files:**
- Create: `SplatKit/Sources/SplatCore/Depth/DepthLoss.swift`
- Test: `SplatKit/Tests/SplatCoreTests/DepthLossTests.swift`

**Interfaces (Produces):**
```swift
public struct DepthLossWeights: Sendable, Equatable {
    public var pearson: Float; public var l1: Float; public var alphaMask: Float; public var minimumPixels: Int; public var nearPlane: Float
    public init(pearson: Float, l1: Float, alphaMask: Float, minimumPixels: Int, nearPlane: Float)
    public init(settings: DepthPriorSettings, nearPlane: Float)
}
public struct DepthLossResult: Sendable {
    public var loss: Double; public var pearson: Double /* ρ, 0 when degenerate */; public var l1: Double
    public var validPixels: Int; public var degenerate: Bool
    public var gradient: [SIMD2<Float>]   // per pixel (∂L/∂Draw, ∂L/∂A), zero outside the mask
}
public enum DepthLoss {
    public static func evaluate(depth: DepthRender, prior: DepthPrior, weights: DepthLossWeights) -> DepthLossResult
}
```
Algorithm (Double; the mask is a constant gate):
```text
M = { p : A_p > alphaMask ∧ target_p finite ∧ target_p > 0 };  n = |M|;  D_p = Draw_p / A_p
x_p = D_p (domain .depth) | 1/max(D_p, near) (domain .inverseDepth);  dx/dD = 1 | (D_p > near ? −1/D_p² : 0)
Pearson: μx, μy, σx² = mean((x−μx)²), σy² likewise, cov = mean((x−μx)(y−μy))
         degenerate if n < minimumPixels or σx ≤ 1e-6·(|μx| + 1e-6) or σy ≤ 1e-6·(|μy| + 1e-6) → term 0, gradient 0
         ρ = cov/(σx·σy);  term = λ_P·(1 − ρ)
         ∂term/∂x_p = −λ_P·(ŷ_p − ρ·x̂_p)/(n·σx),  x̂ = (x−μx)/σx, ŷ = (y−μy)/σy
L1 (only if prior.isMeasuredMetric ∧ λ_L1 > 0 ∧ l1Normalizer > 0): M1 = M ∩ {metric_p > 0}, n1 = |M1|
         l1 = mean_{M1} |D_p − m_p| / l1Normalizer;  ∂/∂D_p = λ_L1·sign(D_p − m_p)/(n1·l1Normalizer), sign(0) = 0
gD_p = ∂term/∂x_p · dx/dD + ∂L1/∂D_p;   gradient_p = (gD_p / A_p, −gD_p · D_p / A_p)
loss = term + λ_L1·l1;   `degenerate` refers to the Pearson term only
```

- [ ] **Step 1: Write the failing tests**
```swift
@Test func perfectCorrelationHasZeroLoss()              // target = 3·D + 1 (depth domain) → ρ = 1 (1e-9), loss 0, gradient ≈ 0
@Test func anticorrelationHasLossTwiceWeight()          // target = −D + c → loss ≈ 2·λ_P
@Test func pearsonDomainFollowsPrior()                  // inverse prior = 1/D: ρ = 1 in .inverseDepth, < 1 in .depth for a non-linear D range
@Test func matchesPearsonGolden()                       // both sub-cases: loss, pearson, l1 (1e-6), gradDraw/gradA (rel 1e-5)
@Test func gradientMatchesFiniteDifferences()           // perturb Draw_p and A_p, central FD h 1e-4 in Double, rel 1e-4, mask fixed
@Test func constantRenderedDepthIsDegenerateNotNaN()    // all D equal → degenerate, loss finite (0 without L1), gradient zero
@Test func emptyMaskIsDegenerate()                      // all A < 0.3 → n 0, degenerate, gradient all zero
@Test func l1OnlyForMeasuredMetric()                    // same metric array, isMeasuredMetric false → l1 0
```
- [ ] **Step 2–4:** run (fail) → implement → `swift test --filter DepthLossTests` and full `swift test` pass.
- [ ] **Step 5: Return diff** (commit: `Add CPU reference Pearson and L1 depth loss`).

### Task 7: CPU regularizers and isolation pruning

**Platform:** Linux-testable · **Depends on:** 1, 3

**Files:**
- Create: `SplatKit/Sources/SplatCore/Training/Regularizers.swift`, `SplatKit/Sources/SplatCore/Training/SpatialHashGrid.swift`, `SplatKit/Sources/SplatCore/Training/IsolationPruning.swift`
- Test: `SplatKit/Tests/SplatCoreTests/RegularizerTests.swift`

**Interfaces (Produces):**
```swift
public enum Regularizers {
    /// Adds λ/n·∂(Σ hinge)/∂logScale into `gradients.logScales`; returns the loss value.
    @discardableResult
    public static func addAnisotropyGradient(_ cloud: GaussianCloud, weight: Float, maxRatio: Float, into gradients: inout GaussianGradients) -> Double
    /// logScale_a = min(logScale_a, log(fraction · max(R, ‖μ − c‖))) for every axis.
    public static func applyScaleCap(_ cloud: inout GaussianCloud, fraction: Float, extent: SceneExtent)
}
public struct SpatialHashGrid: Sendable {
    public init(points: [SIMD3<Float>], cellSize: Float)       // Dictionary<SIMD3<Int32>, [Int32]>, ascending indices per cell
    public func countNeighbors(of index: Int, within radius: Float, limit: Int) -> Int   // excludes self, early exit at limit; requires radius ≤ cellSize
    public func kthNeighborDistance(of index: Int, k: Int, maxRings: Int) -> Float      // exact; +inf if not found within maxRings
}
public enum IsolationPruning {
    /// true = prune. Never marks all; at most floor(maxFraction·n).
    public static func pruneMask(positions: [SIMD3<Float>], neighbors k: Int, factor: Float, maxFraction: Float,
                                 sampleCount: Int, seed: UInt64) -> [Bool]
}
```
Anisotropy: `s = exp(logScale)`, `a_max = argmax`, `a_min = argmin` (lowest axis index on ties), `r = s_max/s_min`. If `r > maxRatio`: `loss += λ/n·(r − maxRatio)`, `g[a_max] += λ·r/n`, `g[a_min] −= λ·r/n`. If `a_max == a_min` there is no term.

Isolation algorithm:
```text
n ≤ k → all false
c0 = 2·cbrt(V/n), V = product over axes of max(P99 − P1, 1e-6·max extent) of the positions; grid0 = SpatialHashGrid(cellSize: c0)
S = GaussianInitializer.subsetIndices(total: n, maxCount: sampleCount, seed: seed)   (internal helper, same module)
d_s = grid0.kthNeighborDistance(of: s, k: k, maxRings: 64) for s ∈ S;  m = median of finite d_s; m ≤ 0 or none finite → all false
T = factor·m;  gridT = SpatialHashGrid(cellSize: T);  isolated_i = gridT.countNeighbors(of: i, within: T, limit: k) < k
cap = floor(maxFraction·n); if count(isolated) > cap: rank the isolated by gridT.kthNeighborDistance(k, maxRings: 64) descending (index ascending on ties), keep the first cap
if every point would be pruned → all false
```
Ring expansion for `kthNeighborDistance`: visit cubic shells r = 0, 1, 2, …; after shell r, stop when ≥ k candidates are found and the k-th smallest distance ≤ r·cellSize.

- [ ] **Step 1: Write the failing tests**
```swift
@Test func anisotropyZeroBelowRatio()                     // ratio 9.9 → loss 0, gradient 0
@Test func anisotropyMatchesGolden()                       // golden_anisotropy: loss 1e-6, gradients rel 1e-5
@Test func anisotropyMatchesFiniteDifferences()            // away from the kink and argmax ties, h 1e-4, rel 1e-4
@Test func scaleCapClampsOnlyLargeScales()                 // R 2, centre 0, μ at 0: cap 0.1 → logScale log(0.5) → log(0.1); log(0.05) unchanged
@Test func scaleCapGrowsWithDistance()                     // μ at distance 10 from centre → cap 0.5
@Test func kthNeighborMatchesKDTree()                      // 500 random points: kthNeighborDistance == KDTree distance of the k-th neighbour (1e-6)
@Test func countNeighborsMatchesBruteForce()
@Test func prunesFarOutliers()                             // 2 000 points on a unit sphere + 10 points at distance 20 → exactly the 10 marked
@Test func densePointsAreNeverPruned()                     // uniform grid → none
@Test func pruningRespectsMaxFraction()                    // 100 outliers among 1 000, maxFraction 0.05 → 50 marked, the farthest
@Test func pruneMaskIsDeterministic()
```
- [ ] **Step 2–4:** run (fail) → implement → `swift test --filter RegularizerTests` and full `swift test` pass. `prunesFarOutliers` on 200 k points must take < 2 s in debug (measure with `ContinuousClock` in a local run, not asserted).
- [ ] **Step 5: Return diff** (commit: `Add anisotropy loss, scale cap and isolation pruning references`).

### Task 8: Dense depth initialization

**Platform:** Linux-testable · **Depends on:** 1

**Files:**
- Create: `SplatKit/Sources/SplatCore/Depth/DepthGeometry.swift`, `SplatKit/Sources/SplatCore/Depth/DenseInitializer.swift`
- Test: `SplatKit/Tests/SplatCoreTests/DenseInitializerTests.swift`

**Interfaces (Produces):**
```swift
public enum DepthGeometry {
    /// World point of pixel centre (x + 0.5, y + 0.5) at camera depth z: R^T·(z·K⁻¹·[u, v, 1] − t).
    public static func backProject(x: Int, y: Int, depth z: Float, camera: Camera) -> SIMD3<Float>
    /// Pixel coordinates and camera z of a world point; nil if z ≤ near.
    public static func project(_ p: SIMD3<Float>, camera: Camera, near: Float) -> (u: Float, v: Float, z: Float)?
    /// True if the 3×3 neighbourhood of valid depths has max/min > ratio (or fewer than 2 valid neighbours).
    public static func isDepthEdge(_ depth: [Float], width: Int, height: Int, x: Int, y: Int, ratio: Float) -> Bool
}
public struct DenseInitReport: Sendable, Equatable { public var backProjected: Int; public var rejectedEdges: Int; public var rejectedInconsistent: Int; public var afterDedup: Int; public var final: Int }
public enum DenseInitializer {
    /// Returns nil when no view has `depth?.metricDepth`.
    public static func pointCloud(_ dataset: TrainingDataset, settings: DenseInitSettings, maxCount: Int, seed: UInt64) -> (PointCloud, DenseInitReport)?
    /// Dense cloud when possible, else `GaussianInitializer.initialize` (Phase 3). maxCount = budget cap.
    public static func initialize(_ dataset: TrainingDataset, configuration: TrainingConfiguration, maxCount: Int) -> GaussianCloud
}
```
Algorithm (P8). Views with metric depth: `V`, |V| = m. Cap `C = min(settings.maxPoints, maxCount / 2)`.
```text
per view v ∈ V: valid_v = #valid metric pixels; stride g_v = max(1, ceil(sqrt(valid_v · m / C)))
  for y, x on the g_v grid (row-major), metric z > 0:
     skip if isDepthEdge(…, ratio: edgeRatio)                  → rejectedEdges
     X = backProject; colour = image pixel (x, y)
     for w ∈ V, w ≠ v: proj = project(X, w); inside and metric_w(px) > 0 (px = floor(u), floor(v)):
         if proj.z < metric_w·(1 − tol) → violation; else if |proj.z − metric_w| ≤ tol·metric_w → support
     drop if violations ≥ 1 ∧ violations ≥ supports                → rejectedInconsistent
voxel h = median over kept points of z·g_v / fx_v; key = floor(X / h) as SIMD3<Int32>
insertion order: SfM points first (if includeSfMPoints and present), then views in index order, pixels row-major; first point per voxel wins
if count > C: GaussianInitializer.subsetIndices(total:, maxCount: C, seed: seed)
```
`initialize` calls `GaussianInitializer.fromPoints(cloud, shDegree:, initialOpacity:, maxCount:, seed:)` when `settings.isEnabled` and `pointCloud` is non-nil.

- [ ] **Step 1: Write the failing tests** (two or three 32×24 views of an analytic plane + box, metric depth computed analytically in the test)
```swift
@Test func backProjectInvertsProject()                 // random pixels/depths round-trip within 1e-4 px
@Test func pointsLieOnTheSurface()                     // plane z = 0: every dense point |z| < 1e-3
@Test func overlappingViewsAreDeduplicated()           // two identical cameras → afterDedup ≈ single-view count (±5 %)
@Test func freeSpaceViolationsAreRemoved()             // inject a floater depth patch in view 0 in front of view 1's surface → rejectedInconsistent > 0, no final point in that region
@Test func depthEdgesAreSkipped()                      // box silhouette pixels are not back-projected
@Test func capIsRespected()                            // maxPoints 500 → final ≤ 500; maxCount 600 → ≤ 300
@Test func fallsBackWithoutAlignedViews()              // no metricDepth → initialize equals GaussianInitializer.initialize
@Test func denseInitIsDeterministic()
```
- [ ] **Step 2–4:** run (fail) → implement → `swift test --filter DenseInitializerTests` and full `swift test` pass.
- [ ] **Step 5: Return diff** (commit: `Add dense depth initialization with free-space filtering and voxel deduplication`).

### Task 9: Pseudo-view generation

**Platform:** Linux-testable · **Depends on:** 1, 8

**Files:**
- Create: `SplatKit/Sources/SplatCore/Depth/PseudoViews.swift`
- Test: `SplatKit/Tests/SplatCoreTests/PseudoViewTests.swift`

**Interfaces (Produces):**
```swift
public struct PseudoView: Sendable, Equatable { public var camera: Camera; public var prior: DepthPrior; public var parents: SIMD2<Int>; public var t: Float }
public enum PseudoViewRejection: Sendable, Equatable { case wideAngle, lowCoverage(Float), tooClose }
public enum PseudoViews {
    public static func pairs(_ views: [TrainingView]) -> [SIMD2<Int>]          // metric views only, nearest neighbour by centre, unique, sorted
    public static func interpolate(_ a: Camera, _ b: Camera, t: Float, resolutionDivisor: Int) -> Camera
    public static func warpedDepth(into camera: Camera, from views: [TrainingView], near: Float) -> [Float]   // z-buffer of all metric pixels of the given views
    public static func make(_ views: [TrainingView], settings: PseudoViewSettings, near: Float) -> (views: [PseudoView], rejected: [PseudoViewRejection])
    /// Deterministic pick for an iteration (SplatRandom stream 7).
    public static func index(iteration: Int, count: Int, seed: UInt64) -> Int
}
```
Rules (P9): interpolation via camera-to-world rotation `Rᵀ` → quaternion → shortest-path slerp (negate one quaternion if dot < 0; lerp+normalize when dot > 0.9995) → back to world→camera `R`. Centre lerp, `t = −R·C`. fx, fy, cx, cy lerped then divided by the divisor; size = view a's size / divisor (rounded down, ≥ 1). Angle test: optical axes (third row of R) with `acos(dot) > maxAngle` → `.wideAngle`. The prior from `warpedDepth(into:, from: [parents])` has domain `.depth`, `correlationTarget = metricDepth = warped`, `isMeasuredMetric = false`. Coverage = valid / pixels < minimumCoverage → `.lowCoverage`. `P5(warped valid) < minimumDepthFraction · min(median metric of parents)` → `.tooClose`.

- [ ] **Step 1: Write the failing tests**
```swift
@Test func pairsAreNearestNeighboursAndUnique()       // 4 cameras on an arc → [(0,1), (1,2), (2,3)]
@Test func interpolationEndpointsMatchParents()       // t 0 / 1 → parent rotation and centre (1e-5), divisor 1
@Test func interpolatedCentreAndOrientationAreMidway() // two cameras looking at the origin: t 0.5 looks at the origin (angle < 0.5°)
@Test func warpedDepthReproducesPlane()               // warp a plane's metric depth into a midway camera → valid pixels match the analytic depth (rel 1e-3)
@Test func rejectsWideAnglePairs()                     // 90° apart → .wideAngle
@Test func rejectsCameraInsideGeometry()               // parents facing a box, pseudo camera inside it → .tooClose
@Test func coverageBelowThresholdIsRejected()
@Test func viewsWithoutMetricDepthProduceNoPseudoViews()
@Test func indexIsDeterministicAndInRange()
```
- [ ] **Step 2–4:** run (fail) → implement → `swift test --filter PseudoViewTests` and full `swift test` pass.
- [ ] **Step 5: Return diff** (commit: `Add pseudo-view generation with warped depth priors`).

### Task 10: Held-out selection, scheduler, SSIM metric

**Platform:** Linux-testable · **Depends on:** 1

**Files:**
- Create: `SplatKit/Sources/SplatCore/Training/LeaveOneOut.swift`
- Modify: `SplatKit/Sources/SplatCore/Training/ImageLoss.swift` (add `ImageMetrics.ssim`)
- Test: `SplatKit/Tests/SplatCoreTests/LeaveOneOutTests.swift`

**Interfaces (Produces):**
```swift
extension ImageMetrics { public static func ssim(_ a: RGBImage, _ b: RGBImage) -> Double }   // == ImageLoss.evaluate(...).ssim, without the gradient
public struct HeldOutMetrics: Sendable, Equatable { public var psnr: Double; public var ssim: Double; public init(psnr: Double, ssim: Double) }
public enum HeldOutSelection {
    /// nil when disabled: fewer than policy.minimumViews views, or degenerate camera layout. An override name wins if present.
    public static func select(names: [String], cameraCenters: [SIMD3<Float>], override: String?, policy: EarlyStoppingPolicy) -> Int?
}
public struct LeaveOneOutScheduler: Sendable {
    public enum Decision: Sendable, Equatable { case `continue`, stop }
    public init(policy: EarlyStoppingPolicy, maxIterations: Int)
    public private(set) var bestIteration: Int?           // argmax of the EMA
    public private(set) var bestMetrics: HeldOutMetrics?
    public private(set) var history: [(iteration: Int, metrics: HeldOutMetrics, ema: Double)]
    public mutating func record(iteration: Int, metrics: HeldOutMetrics) -> Decision
    /// round(best · N/(N−1)) clamped to [evalInterval, maxIterations]; maxIterations when no eval was recorded.
    public func finalIterations(viewCount: Int) -> Int
}
```
Selection: when an override is present, return the index of `override` (overrides apply even below minimumViews if N ≥ 2). Otherwise, N < minimumViews → nil. Centroid; covariance of centres; principal axis by 50 power iterations from (1, 1, 1)/√3. If the largest eigenvalue is < 1e-12, return nil. Projections u_i; remove argmin and argmax (lowest index on ties); return the remaining index minimizing |u_i − median(u of all)|, lowest index on ties.

Scheduler: `ema_0 = ssim_0`, `ema_k = w·ssim_k + (1−w)·ema_{k−1}`. A new best requires `ema_k > bestEMA + minDelta` (the first record is always best). Return `.stop` iff `iteration ≥ maxIterations` or (`iteration ≥ minimumIterations` and the number of records since the best ≥ patience).

- [ ] **Step 1: Write the failing tests**
```swift
@Test func ssimMatchesLossSSIM()                       // random 16×12 images: equals ImageLoss.evaluate(...).ssim (1e-12)
@Test func ssimOfIdenticalImagesIsOne()
@Test func selectsInteriorCameraOnArc()                // 5 cameras on an arc → index 2; 4 → 1 (closest to median, lowest index)
@Test func neverHoldsOutAnExtremeCamera()              // shuffled order of 6 arc cameras: result is never the first/last along the arc
@Test func disabledBelowFourViews()                    // 3 cameras → nil
@Test func coincidentCentresReturnNil()
@Test func overrideWins()                              // override "view_03" → its index
@Test func stopsAfterPatienceWithoutImprovement()      // rising then flat SSIM, patience 4 → stop at the 4th flat record after minimumIterations
@Test func singleSpikeDoesNotStop()                    // one high outlier then steady improvement → bestIteration moves past the spike
@Test func noStopBeforeMinimumIterations()
@Test func finalIterationsScalesByViewRatio()          // best 3000, N 4 → 4000; clamped to maxIterations 3500 → 3500
```
- [ ] **Step 2–4:** run (fail) → implement → `swift test --filter LeaveOneOutTests` and full `swift test` pass.
- [ ] **Step 5: Return diff** (commit: `Add held-out view selection, leave-one-out scheduler and SSIM metric`).

### Task 11: Synthetic scene and CPU trainer Phase 4 integration

**Platform:** Linux-testable · **Depends on:** 4, 6, 7, 8, 9

**Files:**
- Create: `SplatKit/Sources/SplatCore/Synthetic/SyntheticScene.swift`
- Modify: `SplatKit/Sources/SplatCore/Training/CPUTrainer.swift`
- Test: `SplatKit/Tests/SplatCoreTests/CPUTrainerPhase4Tests.swift`

**Interfaces (Produces):**
```swift
public struct SyntheticScene: Sendable {
    public var groundTruth: GaussianCloud              // textured sphere (radius 1) + ground plane, SH degree 0
    public var train: [TrainingView]                   // images rendered by CPURasterizer, depth nil
    public var novel: [TrainingView]                   // interpolated cameras between the training ones (held out for evaluation)
    public var trainDepth: [DepthMap]                  // metric = normalized GT depth (alpha > 0.5), kind .metric
    public var points: PointCloud                      // `sparsePoints` seeded-random GT Gaussian means + N(0, 0.01²) noise, GT colours (SfM stand-in)
    /// Cameras on a 110° elevated arc at radius 3.5 looking at the origin (as Tools/synthetic), fx = fy = 1.3·width.
    public static func make(viewCount: Int, novelCount: Int, width: Int, height: Int, gaussians: Int, sparsePoints: Int, seed: UInt64) -> SyntheticScene
    /// Affine relative inverse depth a/z + b of a training view (invalid where GT is invalid).
    public func relativeInverseDepth(view: Int, a: Float, b: Float) -> DepthMap
    public var dataset: TrainingDataset { get }        // train + points
}
public struct DepthStepInfo: Sendable { public var training: DepthLossResult?; public var pseudo: DepthLossResult?; public var isolationPruned: Int }
// CPUStepResult gains: public var depth: DepthStepInfo
// CPUTrainer gains: public let pseudoViews: [PseudoView]   (built in init when .pseudoViews is enabled)
```
Normative step order (Task 17 mirrors it exactly):
```text
1 sample view; render (with depth if view.depth != nil ∧ .depthPearson|.depthL1 enabled ∧ iteration ≥ depthPrior.startIteration)
2 image loss; depth loss (weights: pearson if .depthPearson else 0, l1 if .depthL1 else 0) → depthGradient
3 backward(image, depth) + statistics (Phase 3 rule)
4 pseudo-view (if .pseudoViews, pool non-empty, iteration ≥ start, (iteration − start) % interval == 0):
    pick PseudoViews.index; render depth; DepthLoss with weights (pearson: pseudoViews.weight, l1: 0); backward with zero image
    gradient; ADD its gradients; no statistics
5 if .anisotropy: Regularizers.addAnisotropyGradient
6 Adam; 7 if .scaleCap: applyScaleCap; 8 iteration += 1; 9 densify at boundaries (Phase 3)
10 if .isolationPrune ∧ iteration ≥ isolationStart ∧ (iteration − isolationStart) % isolationInterval == 0 ∧ iteration < iterations:
    mask = IsolationPruning.pruneMask(…, seed: seed ^ UInt64(iteration)); apply as DensifyPlan(keep/prune) via Densification.apply; reset statistics
```

- [ ] **Step 1: Write the failing tests**
```swift
@Test func syntheticSceneIsDeterministicAndConsistent() // same seed → equal; trainDepth at the sphere centre pixel ≈ 2.5 (camera distance 3.5 − radius 1, rel 2 %)
@Test func cpuTrainerWithoutDepthMatchesPhase3Order()   // regularizers [] → identical cloud to Phase 3 CPUTrainer after 20 steps (same seed)
@Test func depthPriorImprovesNovelDepthCorrelation()    // 4 views 32×24, 120 iterations, random init 400, metric priors: Pearson(novel normalized depth, GT) with prior > without
@Test func anisotropyBoundsAxisRatio()                  // 120 iterations with .anisotropy, .scaleCap → 95th percentile ratio ≤ 12
@Test func pseudoViewsContributeGradient()              // after start, a pseudo-view step reports pseudo != nil and finite loss
@Test func isolationPruningRemovesInjectedFloaters()    // 20 Gaussians injected at distance 10 → gone after the first isolation boundary
@Test func modelStaysFinite()                           // all features on, 150 iterations: every parameter finite
```
- [ ] **Step 2–4:** run (fail) → implement → `swift test --filter CPUTrainerPhase4Tests` (< 30 s in debug) and full `swift test` pass. If `depthPriorImprovesNovelDepthCorrelation` is marginal, raise the iterations up to 200 before touching the comparison.
- [ ] **Step 5: Return diff** (commit: `Add synthetic ground-truth scene and Phase 4 terms to the CPU trainer`).

### Task 12: Synthetic dataset tool — depth and novel views

**Platform:** Linux (Python stdlib only) · **Depends on:** 2 (format)

**Files:**
- Modify: `Tools/synthetic/make_colmap_scene.py`, `Tools/synthetic/README.md`

**Produces:** new flags.
- `--depth`: writes `depth/<image name>.sdpt` with the metric z-buffer depth of the front-most disc of coverage ≥ 0.5 (kind 0, no confidence).
- `--relative-depth`: writes kind 2 instead, `a/z + b` with seeded `a ∈ [0.5, 2]`, `b ∈ [0, 0.5]`; the values are printed so alignment can be checked by hand.
- `--novel-views N`: writes `novel/images/*.png` + `novel/sparse/0/{cameras,images}.bin` and an empty valid `points3D.bin`, at azimuths midway between consecutive training cameras (the first N midpoints, then repeating at quarter points). It is loadable by the app's COLMAP loader.
- `.sdpt` writing via `struct.pack('<4sHBBII8x', b'SDPT', 1, kind, flags, w, h)` + `array('f')`.

- [ ] **Step 1:** Implement the flags; `--verify` additionally checks that each depth file decodes, has the image's size, and has the sphere-centre pixel at depth 2.5 ± 0.1 (metric case), and that the novel dataset re-reads.
- [ ] **Step 2: Run** `python3 Tools/synthetic/make_colmap_scene.py $TMP/scene --views 4 --depth --novel-views 2 --ppm` and `--relative-depth` → exit 0; `--verify` → exit 0.
- [ ] **Step 3:** README: document the flags and the benchmark use (Task 21).
- [ ] **Step 4: Return diff** (commit: `Add ground-truth depth and novel views to the synthetic scene generator`).

### Task 13: Metal setup — register shader files

**Platform:** macOS-CI-only · **Depends on:** —

**Files:**
- Modify: `SplatKit/Sources/SplatMetal/MetalContext.swift` (`shaderFiles` gets `"Depth", "Regularizers"` after `"Densify"`)
- Create: `SplatKit/Sources/SplatMetal/Shaders/Depth.metal`, `SplatKit/Sources/SplatMetal/Shaders/Regularizers.metal` (header comment only: purpose, concatenation order, "uniform structs declared here, mirrored in GPUDepthLoss.swift / GPURegularizers.swift")

- [ ] **Step 1:** Implement. **Step 2:** CI green (existing suites; library compiles with the empty files).
- [ ] **Step 3: Return diff** (commit: `Register Phase 4 shader files`).

### Task 14: Metal rendered depth forward/backward

**Platform:** macOS-CI-only · **Escalate-to: opus-review** · **Depends on:** 3, 4

**Files:**
- Modify: `SplatKit/Sources/SplatMetal/Shaders/Rasterizer.metal` (`rasterizeTiles`), `Shaders/Backward.metal` (`rasterizeBackward`, `projectBackward`), `SplatKit/Sources/SplatMetal/SplatRenderer.swift`, `ShaderTypes.swift` and `Common.metal` (doc comments only: `background.w`, `shInfo.w`, accumulator slot 9)
- Test: `SplatKit/Tests/SplatMetalTests/DepthRenderTests.swift`

**Interfaces (Produces):**
```swift
extension SplatRenderer {
    /// `depth`: float2 per pixel (Draw, A), ≥ width·height·8 bytes; nil = Phase 3 behaviour.
    public func encodeForward(camera: Camera, settings: RasterSettings, to texture: MTLTexture, depth: MTLBuffer?) throws(MetalError) -> ForwardPass
    /// `depthGradient`: float2 per pixel (∂L/∂Draw, ∂L/∂A) or nil. `accumulate`: add into `gradients` instead of overwriting
    /// (no zero-fill of the gradient buffers, projectBackward adds; culled Gaussians untouched).
    public func encodeBackward(imageGradient: MTLBuffer, depthGradient: MTLBuffer?, into gradients: GradientBuffers,
                               statistics: DensificationStatisticsBuffers?, accumulate: Bool, commandBuffer: MTLCommandBuffer) throws(MetalError)
    public func renderDepth(camera: Camera, settings: RasterSettings) throws(MetalError) -> DepthRender   // synchronous, tests/eval
    public func backward(imageGradient: [SIMD3<Float>], depthGradient: [SIMD2<Float>]) throws(MetalError) -> GaussianGradients // synchronous, tests
}
```
The existing overloads forward with `depth: nil`, `depthGradient: nil`, `accumulate: false`. Kernel changes (mirror Task 4 exactly):
- `rasterizeTiles`: on load `sColor[lid] = float4(colors[g].xyz, meanDepth[g].z)`; new `device float2* outDepth [[buffer(8)]]`. When `u.background.w > 0.5`: `depthAcc += sColor[j].w * (alpha * T)` and write `outDepth[pixel] = float2(depthAcc, 1 − T)`. Otherwise do not write. **First verify that no kernel reads `colors.w` / `sColor.w`** and record it in the diff.
- `rasterizeBackward`: new `device const float2* depthGradient [[buffer(10)]]`. Per-pixel state `gD, gA` (zero when the flag is off or outside), `behindZ`. Per sample, in the order of Task 4: slot 9 `+= α·T·gD` (before the clamp guard), `dAlpha += (z·T − behindZ/(1−α))·gD + (T_final/(1−α))·gA`, `behindZ += z·α·T`. Use the same simd/atomic pattern as the colour slots. `T_final` = the pixel's `finalT`.
- `projectBackward`: `gradPositions += slot[9] * R2` (the world→camera row 2). With `u.shInfo.w == 1`, all writes become `+=` and culled Gaussians return without writing.
- The renderer always binds `outDepth`/`depthGradient` (a 16-byte dummy buffer when nil) so API validation is satisfied.

- [ ] **Step 1: Write the failing tests** (Metal-gated; reuse `MetalTestSupport`)
```swift
@Test func depthOffIsBitIdenticalToPhase3()           // golden scenes: image and gradients equal to the existing path
@Test func depthMatchesCPUReference()                  // occluded scene: raw/alpha vs CPURasterizer.renderWithDepth (1e-4)
@Test func depthMatchesGoldens()                       // golden_depth_single/occluded (1e-4)
@Test func depthBackwardMatchesCPUReference()          // random depthGradient: all groups rel 1e-3 / abs 1e-5 (float atomics)
@Test func depthBackwardMatchesGoldens()
@Test func depthBackwardMatchesFiniteDifferences()     // GPU loss Σ gD·Draw + gA·A, FD on 4 Gaussians' positions/opacity (h 1e-3, rel 5e-3)
@Test func accumulateAddsToExistingGradients()         // backward A then B with accumulate == CPU(A) + CPU(B) (rel 1e-3)
@Test func accumulateLeavesCulledGaussiansUntouched()
```
- [ ] **Step 2–4:** CI fail → implement → CI green; record the forward/backward GPU time on/off from a 64×64 test log line (`print`) in the diff description.
- [ ] **Step 5: Return diff** (commit: `Add rendered depth channel to the Metal rasterizer and backward`). Opus reviews the reverse-pass algebra against Task 4, the dummy bindings, the slot-9 chain rule and the accumulate semantics before commit.

### Task 15: Metal depth loss

**Platform:** macOS-CI-only · **Depends on:** 6, 13

**Files:**
- Modify: `SplatKit/Sources/SplatMetal/Shaders/Depth.metal`
- Create: `SplatKit/Sources/SplatMetal/GPUDepthLoss.swift`
- Test: `SplatKit/Tests/SplatMetalTests/GPUDepthLossTests.swift`

**Interfaces (Produces):**
```swift
public struct DepthLossValue: Sendable, Equatable { public var loss: Double; public var pearson: Double; public var l1: Double; public var validPixels: Int; public var degenerate: Bool }
public final class GPUDepthPrior { public let width: Int; public let height: Int; public let domain: DepthDomain; public let isMeasuredMetric: Bool; public let l1Normalizer: Float }  // owns target + metric buffers
public final class GPUDepthLoss {
    public static let slotCount = 2                       // 0 = training view, 1 = pseudo-view
    public init(context: MetalContext) throws(MetalError)
    public func makePrior(_ prior: DepthPrior) throws(MetalError) -> GPUDepthPrior
    /// Writes depthGradient (float2 per pixel, overwritten everywhere) and the value into `slot`.
    public func encode(renderedDepth: MTLBuffer, prior: GPUDepthPrior, weights: DepthLossWeights, slot: Int,
                       depthGradient: MTLBuffer, commandBuffer: MTLCommandBuffer) throws(MetalError)
    public func readValue(slot: Int) -> DepthLossValue   // only after the command buffer completed
}
```
Kernels in `Depth.metal`, with the `DepthLossUniforms` struct at the top (`SIMD4<Float>` weights: pearson, l1, alphaMask, near; `SIMD4<Float>` extra: l1Normalizer, domain (0 depth / 1 inverse), measured (0/1), unused; `SIMD4<UInt32>` width, height, minimumPixels, slot):
1. `depthMomentsA`: per threadgroup partial sums of (n, Σx, Σy, n1, Σ|D−m|). 2. `depthFinalizeA` (1 thread): means. 3. `depthMomentsB`: centred partials Σ(x−μx)², Σ(y−μy)², Σ(x−μx)(y−μy). 4. `depthFinalizeB` (1 thread): σ, ρ, the degenerate flag, the loss value into the slot. 5. `depthGradientKernel`: per pixel, the gradient of Task 6. The finalizers sum partials in float with Kahan compensation.

- [ ] **Step 1: Write the failing tests**
```swift
@Test func matchesCPUOnRandomInputs()            // 37×23 (non-multiple of threadgroup) both domains, with L1: value rel 1e-4, gradient rel 1e-3/abs 1e-7 vs DepthLoss.evaluate
@Test func matchesPearsonGolden()
@Test func gpuDegenerateCasesAreZero()           // constant D, empty mask, single pixel: degenerate, gradient all 0, loss finite
@Test func slotsAreIndependent()                 // two encodes into slots 0 and 1 in one command buffer read back separately
@Test func largeImageMomentsAreAccurate()        // 64×64 with depth offset 1000 (cancellation case): ρ within 1e-4 of CPU
```
- [ ] **Step 2–4:** CI fail → implement → CI green.
- [ ] **Step 5: Return diff** (commit: `Add Metal Pearson and L1 depth loss`).

### Task 16: Metal regularizers and buffer compaction

**Platform:** macOS-CI-only · **Depends on:** 7, 13

**Files:**
- Modify: `SplatKit/Sources/SplatMetal/Shaders/Regularizers.metal`
- Create: `SplatKit/Sources/SplatMetal/GPURegularizers.swift`, `SplatKit/Sources/SplatMetal/GaussianBuffers+Compaction.swift`
- Test: `SplatKit/Tests/SplatMetalTests/GPURegularizerTests.swift`

**Interfaces (Produces):**
```swift
public final class GPURegularizers {
    public init(context: MetalContext) throws(MetalError)
    /// gradLogScales[i] += anisotropy gradient for i < count (λ/count as in Regularizers.addAnisotropyGradient).
    public func encodeAnisotropy(parameters: GaussianBuffers, gradients: GradientBuffers, weight: Float, maxRatio: Float, commandBuffer: MTLCommandBuffer) throws(MetalError)
    public func encodeScaleCap(parameters: GaussianBuffers, fraction: Float, extent: SceneExtent, commandBuffer: MTLCommandBuffer) throws(MetalError)
}
extension GaussianBuffers {
    /// CPU, in place, order-preserving; keep.count == count; sets count = kept. Call only while the GPU is idle.
    public func compact(keeping keep: [Bool])
    /// First `count` positions read from shared memory (GPU idle).
    public func readPositions() -> [SIMD3<Float>]
}
```
Kernels: `anisotropyGradient` (one thread per Gaussian, `RegularizerUniforms {SIMD4<Float> (weight, maxRatio, 1/count, 0); SIMD4<UInt32> (count, 0, 0, 0)}`) and `scaleCap` (`SIMD4<Float> (fraction, R, cx, cy)`, `SIMD4<Float> (cz, 0, 0, 0)`, count). Tie rule for argmax/argmin as on the CPU.

- [ ] **Step 1: Write the failing tests**
```swift
@Test func anisotropyMatchesCPU()               // 1 000 random Gaussians: rel 1e-5
@Test func anisotropyMatchesGolden()
@Test func anisotropyMatchesFiniteDifferences() // GPU gradient vs Double FD of the CPU loss on 8 Gaussians (rel 1e-4)
@Test func scaleCapMatchesCPU()
@Test func compactPreservesOrderAndCount()      // keep every 3rd → snapshot equals CPU-filtered cloud; m/v compacted the same way
@Test func compactNeverReadsBeyondCount()
```
- [ ] **Step 2–4:** CI fail → implement → CI green.
- [ ] **Step 5: Return diff** (commit: `Add Metal anisotropy gradient, scale cap and buffer compaction`).

### Task 17: `GPUTrainer` A — staged hooks, depth, regularizers, isolation

**Platform:** macOS-CI-only · **Escalate-to: opus-review** · **Depends on:** 11, 14, 15, 16

**Files:**
- Modify: `SplatKit/Sources/SplatTraining/GPUTrainer.swift`
- Create: `SplatKit/Sources/SplatTraining/Phase4Terms.swift` (the three `AuxiliaryLoss` implementations)
- Test: `SplatKit/Tests/SplatTrainingTests/GPUTrainerPhase4Tests.swift`

**Interfaces (Produces):**
```swift
public enum AuxiliaryStage: Sendable { case pixelGradient, parameterGradient, postOptimizer }
public struct AuxiliaryContext {
    public let viewIndex: Int; public let iteration: Int; public let isPseudoView: Bool; public let camera: Camera
    public let renderedDepth: MTLBuffer?; public let depthGradient: MTLBuffer?
    public let parameters: GaussianBuffers; public let gradients: GradientBuffers; public let sceneExtent: SceneExtent
    public let commandBuffer: MTLCommandBuffer
}
public protocol AuxiliaryLoss: AnyObject {           // replaces the Phase 3 protocol (no external conformers exist)
    var stage: AuxiliaryStage { get }
    func encode(_ context: AuxiliaryContext) throws(TrainingError)
}
final class DepthLossTerm: AuxiliaryLoss     // .pixelGradient; owns GPUDepthLoss + one GPUDepthPrior per view (nil for views without depth)
final class AnisotropyTerm: AuxiliaryLoss    // .parameterGradient
final class ScaleCapTerm: AuxiliaryLoss      // .postOptimizer
// StepReport gains:
public var depthLoss: DepthLossValue?        // read at the same wait points as the image loss
public var isolationPruned: Int?
```
Behaviour. `GPUTrainer.init` builds the Phase 4 terms from `configuration.isEnabled(_)` and the dataset (a depth term only if any view has a prior). It adds `TrainingMemoryBudget.reservedDepthBytes` to the reserved bytes and allocates the `float2` depth buffers (grow-only, like `imageGradient`). The step follows Task 11's order, with steps 1–3 and 5–10 here (pseudo-views in Task 18). The depth-mode forward is used only for views with a prior, and `depthGradient` is passed only then. Isolation pruning runs after the densify wait at its boundaries: `synchronize`, `readPositions`, `IsolationPruning.pruneMask`, `compact` store/m/v, `statistics.encodeReset` (own command buffer, waited). A non-finite depth loss throws `.nonFiniteLoss(iteration:)` like the image loss.

- [ ] **Step 1: Write the failing tests**
```swift
@Test func phase3ConfigurationIsUnchanged()           // regularizers [] → same loss sequence as Phase 3 GPUTrainer over 30 steps (1e-6)
@Test func trainsWithoutAnyDepthPrior()               // all features on, no priors: runs, no depth term, no NaN
@Test func depthLossMatchesCPUTrainerFirstSteps()     // SyntheticScene 32×24 metric priors, densify off: losses of steps 1–5 vs CPUTrainer (rel 1e-3)
@Test func anisotropyAndScaleCapApplied()             // 200 steps: no Gaussian above the cap; 95th percentile ratio ≤ 12
@Test func isolationPruningRemovesInjectedFloaters()  // as Task 11, GPU
@Test func nonFiniteDepthPriorIsHarmless()            // NaN prior pixels are invalid → loss finite
```
- [ ] **Step 2–4:** CI fail → implement → CI green.
- [ ] **Step 5: Return diff** (commit: `Wire depth loss and artifact regularizers into GPUTrainer`). Opus reviews the stage order versus `CPUTrainer`, the wait points around compaction, gradient-buffer replacement after growth/compaction, and budget accounting.

### Task 18: `GPUTrainer` B — pseudo-views, held-out evaluation

**Platform:** macOS-CI-only · **Depends on:** 9, 10, 17

**Files:**
- Modify: `SplatKit/Sources/SplatTraining/GPUTrainer.swift`
- Test: `SplatKit/Tests/SplatTrainingTests/GPUTrainerPseudoViewTests.swift`

**Interfaces (Produces):**
```swift
extension GPUTrainer {
    public var pseudoViewCount: Int { get }
    /// Renders `view` with the full SH degree and returns PSNR + SSIM (CPU metrics). Waits for the GPU.
    public func evaluate(view: TrainingView) throws(TrainingError) -> HeldOutMetrics
}
```
Pseudo-view step (Task 11 step 4, P9). The pool is built in `init` via `PseudoViews.make(views, …)` and its priors are uploaded with `GPUDepthLoss.makePrior`. On a pseudo iteration: encode the training part up to and including its backward (no Adam) into B and commit B. Then `encodeForward(pseudo camera, depth:)` (its projection waits behind B). Then the pseudo command buffer: depth loss (slot 1) → `encodeBackward(zero imageGradient, depthGradient, statistics: nil, accumulate: true)` → `.parameterGradient` terms → Adam → `.postOptimizer` terms. The zero image gradient is a dedicated grow-only zero-filled buffer. Non-pseudo iterations keep the single command buffer.

- [ ] **Step 1: Write the failing tests**
```swift
@Test func pseudoPoolBuiltFromMetricViews()          // SyntheticScene 4 views → pseudoViewCount == 9 minus rejections (assert 3…9)
@Test func pseudoIterationAccumulatesGradient()      // single step at start iteration: gradient == training-only + pseudo-only (via CPU references, rel 1e-3)
@Test func pseudoViewsDoNotTouchStatistics()
@Test func evaluateMatchesCPUMetrics()               // evaluate(view) == ImageMetrics on renderImage (1e-9)
```
- [ ] **Step 2–4:** CI fail → implement → CI green.
- [ ] **Step 5: Return diff** (commit: `Add depth-supervised pseudo-views and held-out evaluation to GPUTrainer`).

### Task 19: `TrainingSession` leave-one-out

**Platform:** macOS-CI-only · **Depends on:** 10, 18

**Files:**
- Modify: `SplatKit/Sources/SplatTraining/TrainingSession.swift`, `SplatKit/Sources/SplatTraining/TrainingProgress.swift`
- Test: `SplatKit/Tests/SplatTrainingTests/LeaveOneOutSessionTests.swift`

**Interfaces (Produces):**
```swift
public enum TrainingPhase: Sendable, Equatable { case single, selection(heldOutView: String), final(stopIteration: Int) }
// TrainingProgress gains (defaulted init parameters):
public var phase: TrainingPhase = .single
public var heldOut: HeldOutMetrics?         // latest held-out evaluation in .selection
public var bestIteration: Int?
public var depthLoss: Double?
// TrainingSession
public init(dataset: TrainingDataset, configuration: TrainingConfiguration,
            initializer: @escaping @Sendable (TrainingDataset) -> GaussianCloud)      // new; the old init wraps a constant cloud
public struct TrainingOutcome: Sendable { public var cloud: GaussianCloud; public var heldOutView: String?; public var bestIteration: Int?; public var heldOut: HeldOutMetrics?; public var iterations: Int }
public func runWithOutcome() async throws(TrainingError) -> TrainingOutcome
// run() returns runWithOutcome().cloud
```
Behaviour: `HeldOutSelection.select(…, override: configuration.heldOutViews.first, policy:)` when `earlyStopping.isEnabled`, else nil.
- nil → `.single`: the Phase 3 loop with `configuration.iterations`.
- Otherwise run 1 (`.selection`): dataset without the held-out view, cloud `initializer(reduced dataset)`. At every `evalInterval` it calls `trainer.evaluate(heldOut)`, then `scheduler.record`, and emits progress. On `.stop` the trainer is released. Then run 2 (`.final(stopIteration:)`): `initializer(full dataset)`, a fresh trainer, and the loop limit is `scheduler.finalIterations(viewCount: N)`. The trainer's own configuration is unchanged, so the nominal schedule is kept.
- `totalIterations` is `configuration.iterations` in run 1 and the stop iteration in run 2. Pause/cancel work in both phases. Cancel in run 1 returns run 1's cloud with `.cancelled` and `phase = .selection`. The terminal `.finished` event carries `bestIteration` and `heldOut` (best).

- [ ] **Step 1: Write the failing tests**
```swift
@Test func threeViewsRunSingle()                      // phases == [.single] only
@Test func selectionThenFinal()                       // 4 views 32×24, iterations 600, evalInterval 100, patience 2, minimumIterations 200: phases in order, final stop ≤ 600, cloud finite
@Test func heldOutViewExcludedFromInitializer()       // the initializer closure sees 3 views in run 1 and 4 in run 2
@Test func cancellationInSelectionReturnsModel()
@Test func overrideNameIsHeldOut()
```
- [ ] **Step 2–4:** CI fail → implement → CI green (each test ≤ 10 s; reduce to 400 iterations if needed).
- [ ] **Step 5: Return diff** (commit: `Add leave-one-out early stopping with full retrain to TrainingSession`).

### Task 20: `SplatDepth` — Core ML Depth Anything V2 Small

**Platform:** macOS-CI-only (build + tests skip without the model) + manual model check on a Mac · **Depends on:** 1, 5

**Files:**
- Modify: `SplatKit/Package.swift` (macOS block: `.target(name: "SplatDepth", dependencies: ["SplatCore"])`, product, `.testTarget(name: "SplatDepthTests", dependencies: ["SplatDepth", "SplatCore"])`), `.gitignore` (`App/Models/`, `*.mlpackage`, `*.mlmodelc`)
- Create: `SplatKit/Sources/SplatDepth/MonocularDepthEstimator.swift`, `SplatKit/Tests/SplatDepthTests/MonocularDepthEstimatorTests.swift`, `Tools/models/fetch_depth_model.sh`, `Tools/models/convert_depth_anything.py`, `Tools/models/README.md`

**Interfaces (Produces):**
```swift
public enum MonocularDepthError: Error, Equatable { case modelNotFound(String), unexpectedModelInterface(String), predictionFailed(String) }
public final class MonocularDepthEstimator: DepthProvider, @unchecked Sendable {   // MLModel is thread-safe for prediction; document it
    public static let bundledModelName = "DepthAnythingV2SmallF16"
    public init(compiledModelURL: URL, computeUnits: MLComputeUnits = .all) throws(MonocularDepthError)  // validates 1 image input, 1 output
    public static func bundled(in bundle: Bundle) -> MonocularDepthEstimator?    // nil when absent, never throws
    public func depthMap(for view: TrainingView) throws -> DepthMap?             // kind .relativeInverseDepth, isEstimated, at view size
}
```
Inference: read the input size from the model description (expect 518×392). Portrait views (h > w) are rotated clockwise first and the output counter-clockwise afterwards. The RGB image is converted to a `CVPixelBuffer` (32BGRA) scaled to the input size (stretch, vImage or Core Graphics). The output is a `MLMultiArray` or a `CVPixelBuffer` (OneComponent16Half/32Float), handled by inspecting the output feature type. It becomes a `DepthMap` at model resolution, then `resampled(width:height:allowAspectChange: true)`. No normalization (P4: affine-invariant). `fetch_depth_model.sh`: pinned HF revision, downloads `DepthAnythingV2SmallF16.mlpackage` into `App/Models/`, verifies the SHA-256 recorded in the script (`--record-hash` on first use prints it for the orchestrator to commit), prints the licence line of the model card and refuses unless it says Apache-2.0. `convert_depth_anything.py`: coremltools conversion of `depth-anything/Depth-Anything-V2-Small-hf` at 518×392 with ImageNet mean/std as `ImageType` scale/bias, FP16.

- [ ] **Step 1: Write the tests** (they skip with a message when `SPLAT_DEPTH_MODEL` is not set to a compiled model path)
```swift
@Test func absentModelReturnsNil()                 // bundled(in: .module) == nil (always runs)
@Test func predictsInverseDepthOrdering()          // SyntheticScene 160×120: mean output on the sphere > on the far ground (near = larger)
@Test func outputHasViewSizeForPortraitAndLandscape()
@Test func alignsToSyntheticGroundTruth()          // DepthPreparation with points → .aligned, medianRelativeError < 0.15
```
- [ ] **Step 2:** Implement; CI green (model-dependent tests skipped).
- [ ] **Step 3: Manual on a Mac (orchestrator or user):** run `Tools/models/fetch_depth_model.sh` (or the conversion), compile with `xcrun coremlcompiler compile`, run `SPLAT_DEPTH_MODEL=… swift test --filter MonocularDepthEstimatorTests`. Report the result; do not claim it otherwise.
- [ ] **Step 4: Return diff** (commit: `Add SplatDepth module with Core ML Depth Anything V2 Small provider`).

### Task 21: Quality tests and benchmark

**Platform:** macOS-CI-only · **Depends on:** 11, 19

**Files:**
- Create: `SplatKit/Tests/SplatTrainingTests/SparseViewQualityTests.swift`, `.github/workflows/benchmark.yml` (`workflow_dispatch` only, `macos-26`, sets `SPLAT_BENCHMARK=1`, runs `swift test -c release --filter SparseViewBenchmark`, uploads the printed table as an artifact)

**Produces:**
- CI quality tests (64×48, ≤ 60 s each), built on `SyntheticScene` with metric priors through `DepthPreparation`:
```swift
@Test func depthPriorImprovesNovelDepthCorrelation()   // 4 views, 400 it., random init: Pearson(novel depth, GT) with prior > without, and with ≥ 0.5
@Test func regularizersBoundNeedles()                   // 400 it., all features: share of Gaussians with axis ratio > 15 ≤ 2 %; none above the scale cap
@Test func fullPipelineIsNotWorseOnNovelViews()         // dense init + all terms vs Phase 3 config: novel PSNR ≥ baseline − 0.5 dB
```
- Benchmark suite `SparseViewBenchmark` (skipped unless `SPLAT_BENCHMARK=1`): 160×120, 3/4/6 views, 3 000 iterations, novel views 4. It prints a Markdown table of novel PSNR / SSIM / depth Pearson, the share of Gaussians with ratio > 10, and the share farther than 2·R from the centre, for the rows: Phase 3 baseline, + dense init, + depth prior, + regularizers, + pseudo-views, all, all + leave-one-out (reported iterations). The targets (reported, not asserted): all ≥ baseline + 1 dB novel PSNR at 3 and 4 views; needles and far floaters ≤ 1 %.

- [ ] **Step 1:** Implement; CI green; record the three tests' durations from the log.
- [ ] **Step 2:** Orchestrator triggers the benchmark workflow once and pastes the table into the commit message body or the PR.
- [ ] **Step 3: Return diff** (commit: `Add sparse-view quality tests and benchmark workflow`).

### Task 22: App — inspector, depth import/estimation, metrics

**Platform:** macOS-CI-only (app build) + manual check · **Depends on:** 2, 5, 8, 19, 20

Use the skills `swiftui-macos-skill:swiftui-macos` and `liquid-glass:liquid-glass`; run `scripts/check_glass.py --target 26.0` on every changed SwiftUI file.

**Files:**
- Modify: `project.yml` (app depends on `SplatDepth`; resource `App/Models/DepthAnythingV2SmallF16.mlpackage` with `optional: true`), `App/Project/ProjectModel.swift`, `App/Project/ProjectInspector.swift`
- Create: `App/Project/SparseViewSettingsSection.swift`, `App/Project/DepthPipeline.swift`

**Behaviour:**
- `DepthPipeline` (`nonisolated enum`, `@concurrent static func prepare(contents:configuration:) async -> (dataset, report, newMaps: [String: DepthMap])`). It loads the dataset, reads `ProjectBundle.depthMaps`, builds `FirstAvailableDepthProvider([supplied, MonocularDepthEstimator.bundled(in: .main)].compactMap)` (the model only when `.depthPearson` is enabled), and caches newly estimated maps (`newMaps`, written via `ProjectBundle.withDepthMaps`, so the document becomes dirty). It runs `DepthPreparation.prepare(trustMetric: source == .arkit)`. The initializer closure is `DenseInitializer.initialize`.
- ARKit import also copies each frame's `depth` file into `depthFiles` (missing depth files are listed in a non-fatal warning). New button "Import Depth Maps…" for COLMAP projects (folder of `<image name>.sdpt`; unmatched files are listed).
- Inspector section "Sparse-View Quality" (plain `Form` controls, no custom glass): Toggle "Depth prior" (disabled with a footnote "No depth: import depth maps or install the depth model" when neither exists), Slider "Depth weight" 0…0.2, Toggle "Artifact regularizers" (anisotropy + scale cap + isolation together), Toggle "Pseudo-views", Toggle "Early stopping (leave-one-out)" (disabled with "Needs at least 4 images" below 4), a `DisclosureGroup` "Depth status" listing `ViewDepthStatus` per image in plain language.
- The training readout adds the phase label ("Choosing iteration (holding out view_02)" / "Final training to 4 000"), held-out PSNR (2 decimals, dB) and SSIM (3 decimals), best iteration, and depth loss (4 decimals). The checkpoint record stores `heldOutView`, `heldOutPSNR`, `heldOutSSIM`, `bestIteration`.
- Train stays the single `.glassProminent` action; the viewport stays the content layer; icon-only controls have accessibility labels.

- [ ] **Step 1:** Implement; `xcodegen generate` + `xcodebuild … build` on CI succeed with and without `App/Models/`; glass linter clean.
- [ ] **Step 2: Manual check on a Mac (not claimed by CI):** synthetic scene from Task 12 with `--depth` and with `--relative-depth` → import, import depth maps, Train → the depth status shows aligned views, both phases run, the held-out metrics appear, and the result looks free of needles in the viewport. Then once with the model installed and no depth files.
- [ ] **Step 3: Return diff** (commit: `Add sparse-view quality controls, depth import and held-out metrics to the app`).

---

## `docs/ARCHITECTURE.md` updates (orchestrator applies after Phase 4 lands; this plan does not edit the file)

Add decision-log rows:
- 26 **Depth source & format** — `DepthProvider` seam; `.sdpt` (24-byte header, Float32, optional UInt8 confidence) in `depth/<image>.sdpt`; edge-aware resampling to view size, aspect within 1 % (P1–P3).
- 27 **Depth alignment** — inverse-depth domain for relative inverse priors, LMedS + MAD + LS, ≥ 8 points / 30 % inliers, kind correction, metric trusted only for ARKit; alignment lives in SplatCore (P4, P5, P13).
- 28 **Rendered depth** — expected depth as a composited channel (`Draw`, `A`), loss on `Draw/A`, accumulator slot 9, uniform flags `background.w` / `shInfo.w` (P6).
- 29 **Depth loss details** — Pearson in the prior's domain, mask A > 0.3, degenerate → 0, L1 only for LiDAR normalized by the median, λ 0.05 / 0.1 (P7).
- 30 **Dense init details** — P8 values (edge 1.05, free-space 5 %, voxel = pixel footprint, cap min(150 k, budget/2)).
- 31 **Pseudo-views** — nearest-neighbour pairs, slerp, t ∈ {¼, ½, ¾}, half resolution, warped metric depth, every 2nd iteration from 2 000, λ 0.05, gradient accumulation, no densify stats (P9).
- 32 **Regularizer values** — anisotropy hinge λ 0.1 at ratio 10; scale cap 0.05·max(R, distance) as a projection; isolation 4-NN > 8× median, every 500 from 1 000, ≤ 5 % per pass, CPU hash grid (P10).
- 33 **Early stopping** — SSIM (LPIPS deferred), PCA interior selection, ≥ 4 views, EMA/patience/min-delta, run 2 to best·N/(N−1) on the nominal schedule (P11). Amend row 5 ("LPIPS/SSIM" → "SSIM; LPIPS optional later").
- 34 **Model acquisition** — P15.

Also: *Module layout*: SplatDepth = Core ML inference only; *Project bundle*: `depth/` added, `CheckpointRecord` held-out fields; *Training defaults*: the Phase 4 values above (marked *(initial)*); status line: Phase 4 done.

## Self-review notes

- Spec coverage: depth prior Pearson + L1 (1, 4, 6, 14, 15, 17), dense depth init (5, 8, 22), pseudo-views (9, 11, 18), anisotropy / scale cap / isolation (7, 11, 16, 17), leave-one-out + full retrain (10, 19), evaluation harness (11, 12, 21), UI (22), model (20), licensing (Global Constraints, 20).
- Type consistency checked across tasks: `DepthPrior`, `DepthRender`, `DepthLossWeights`/`DepthLossResult`/`DepthLossValue`, `HeldOutMetrics`, `PseudoView`, `AuxiliaryContext`, `EarlyStoppingPolicy`, `TrainingPhase`.
- Open risks: (a) **model acquisition/conversion**: Apple's Core ML package name, I/O types and licence must be confirmed on a Mac; the cloud container cannot fetch or convert it, so Task 20's model tests stay skipped on CI and the feature is verified manually. (b) **LPIPS** is deferred; SSIM may pick slightly early iterations on texture-poor scenes, so the benchmark compares against fixed-iteration runs. (c) **GPU cost of rendered depth**: estimated +3 % forward / +10–15 % backward, plus ~20 % from pseudo-views (a second forward/backward at quarter pixel count and one extra CPU sync every 2nd iteration); Task 14 logs measured times. If the backward atomics dominate, a later optimization is per-tile reduction before the atomic add. (d) CPU isolation pruning stalls the GPU ~0.1–0.3 s per pass at 300 k Gaussians; move to a GPU grid hash only if the benchmark shows > 2 % wall time. (e) Weights and thresholds are *(initial)* and are tuned with the Task 21 benchmark, not on CI. (f) Pseudo-view warped priors inherit alignment errors; with only unaligned views they are off and Pearson on the training views is the only geometric prior.
