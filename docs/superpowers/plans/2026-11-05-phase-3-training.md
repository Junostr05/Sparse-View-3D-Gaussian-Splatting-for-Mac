# Phase 3 — Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Per `CLAUDE.md`: workers never commit or push — they return their diff; the orchestrator (Sonnet) reviews, runs tests and commits. Tasks marked `Escalate-to: opus-review` get an additional Opus review before commit.

**Goal:** Train a 3D Gaussian Splatting scene from COLMAP or ARKit input (3–6 views) entirely with hand-written Metal kernels — L1 + D-SSIM loss, Adam, densification/pruning — headless via an `AsyncStream`, plus a minimal `.gsproj` document with a Train button and a live preview.

**Architecture:** Every training algorithm first exists as a deterministic, slow, pure-Swift CPU reference in `SplatCore` (Linux-tested, incl. finite differences and a tiny end-to-end CPU training run). `SplatIO` gains COLMAP/ARKit import, a PLY writer and the `.gsproj` contents model, all operating on `Data` so they are Linux-testable. `SplatMetal` gains a capacity-based SoA `GaussianBuffers` store that the existing `SplatRenderer` reads, plus three small kernel families (Adam, loss, densify), each tested against its CPU reference on tiny inputs. A new macOS-only module `SplatTraining` wires them into `GPUTrainer` (synchronous, one step = one command buffer) and the `TrainingSession` actor (cancellation, pause, `AsyncStream<TrainingProgress>`). The app adds a `FileDocument` project type and a small inspector.

**Tech Stack:** Swift 6.2 (strict concurrency), Swift Testing, Metal (runtime-compiled MSL 3.0, `mathMode = .safe`), SwiftUI (`DocumentGroup`, `.inspector`, Liquid Glass), XcodeGen, GitHub Actions (`macos-26`, paravirtual GPU).

**Spec:** `docs/ARCHITECTURE.md` (decisions 2, 6, 7, 9, 12, 15, 16, 17; Phase 3; *Training defaults*; *Memory budget*; *Project bundle*), the *Shared math* section of `docs/superpowers/plans/2026-11-05-phase-1-viewer.md` and the gradient conventions of `docs/superpowers/plans/2026-11-05-phase-2-differentiable-rasterizer.md`.

## Architectural decisions (summary, with justification)

| # | Topic | Decision | Why |
|---|---|---|---|
| D1 | GPU parameter storage | **SoA**, one `MTLBuffer` per parameter group (positions 3, logScales 3, rotations 4, opacityLogits 1, sh 3K floats per Gaussian), identical to `GaussianCloud` and to what the Phase 2 kernels already read. A `GaussianBuffers` object has a **capacity** and a mutable **count**; Adam `m`/`v` are two more `GaussianBuffers` with the same layout; gradients stay `GradientBuffers` (allocated at capacity). | No kernel rewrite; one generic elementwise Adam kernel per group; densification is a per-group gather. AoS would force rewriting forward/backward. |
| D2 | Growth / compaction | Initial capacity `min(cap, max(2·n₀, 65 536))`. Densification runs only at densify boundaries (every 100 it.) after a GPU wait: decide kernel → `GPUPrefixSum` scans → finalize kernel → `outCount` scan gives output offsets → per-group **scatter into one scratch buffer, blit back**. If the output exceeds capacity, all `GaussianBuffers` grow ×1.5 (CPU `memcpy`, shared storage, GPU idle) before the scatter. No per-iteration reallocation. | Out-of-place per group needs only one scratch buffer of the largest group size, not a second parameter set. |
| D3 | Renderer generalization | `SplatRenderer.init(context:gaussians:)` reads a shared `GaussianBuffers` at render time (`count` and buffer references re-read every frame). The Phase 1 `init(context:cloud:)` becomes a convenience that uploads into a fresh store. New `encodeForward` returns the uncommitted rasterize command buffer so loss/backward/Adam join it. `encodeBackward` accepts gradient buffers with capacity ≥ count. | Viewer and Phase 1/2 tests keep their API; training gets one command buffer per step. |
| D4 | Live preview | The trainer never shares GPU buffers across isolation domains. It emits a `GaussianCloud` snapshot (Sendable) every `previewInterval` iterations; the viewer calls `SplatRenderer.setCloud(_:)`. | No data races, no count changes under a running viewer frame; snapshot cost (~25 MB per 300 k Gaussians) is fine at ≤ 2 Hz. |
| D5 | Parameter groups & optimizer | Adam (β₁ 0.9, β₂ 0.999, ε 1e-15), PyTorch formulation with bias correction, one global step counter. LRs (gsplat defaults): position 1.6e-4 → 1.6e-6 exponential decay over all iterations, **× scene radius**; logScale 5e-3; rotation 1e-3; opacity 5e-2; SH DC 2.5e-3; SH rest 1.25e-4 (DC/20). SH DC vs rest handled by `headCount` (3 floats of 3K per Gaussian). New Gaussians start with m = v = 0. | Matches the reference implementation the goldens come from; one kernel serves all groups. |
| D6 | Loss | `L = (1−λ)·mean|x−y| + λ·(1 − mean SSIM)`, λ = 0.2, means over W·H·3. SSIM: 11×11 Gaussian window σ = 1.5 applied separably, **zero padding, same size**, C₁ = 0.01², C₂ = 0.03², per channel. Backward in closed form (see Task 3). GPU: planar float32 scratch, fused passes, per-threadgroup partial sums; image gradient written as RGBA float `W·H·4` = the buffer `encodeBackward` already takes. | Linux-testable CPU reference with finite differences; the GPU kernel set is 4 small kernels. |
| D7 | Images | sRGB-encoded values /255, **no linearization**, EXIF orientation ignored (raw pixel grid, as COLMAP). Integer box downsampling so the long side ≤ `maxImageSide` (1600); intrinsics divided by the same factor. Decoding via ImageIO lives in the **app**; `SplatIO` takes an injected `(Data) throws -> RGBImage` decoder; tests use a tiny PPM codec. | Keeps `SplatIO` stdlib + Foundation only. |
| D8 | Densification | gsplat default strategy semantics, opacity reset off. Stats: per Gaussian `Σ ‖(∂L/∂mean2D.x·W/2, ∂L/∂mean2D.y·H/2)‖` and visible-count, accumulated from the Phase 2 accumulator slots 7–8 by a new kernel encoded inside `encodeBackward` (no change to `rasterizeBackward`). Rules and budget cap: see Task 4. One-pass decision (prune checked on the parent first). | Deterministic decision function on arrays (CPU) mirrored 1:1 in MSL. |
| D9 | Initialization | From COLMAP `points3D` (random cube fallback), isotropic scale = √(mean squared distance to 3 nearest neighbours) via a **k-d tree**, opacity 0.1, SH DC from point colour, identity rotation. No world re-normalization: a `SceneExtent` (camera-centroid + 1.1 × max distance) scales LRs and thresholds. | Exports stay in COLMAP coordinates; k-d tree is exact and simple to verify against brute force. |
| D10 | Import | COLMAP binary (`cameras.bin`, `images.bin`, `points3D.bin`). Supported: SIMPLE_PINHOLE, PINHOLE (as is), SIMPLE_RADIAL, RADIAL, OPENCV (**CPU undistortion** to a pinhole with the same fx, fy, cx, cy and size, bilinear, clamp-to-edge). Everything else → `DatasetError.unsupportedCameraModel`. COLMAP pose is already world→camera in OpenCV convention. ARKit: own JSON schema (Task 8), camera-to-world converted with `diag(1, −1, −1)`. Intrinsics rescaled when the image file size differs from the camera size (same aspect within 1 %), else error. Missing images → error listing **all** missing names. | With 3–6 views silently dropping a view is worse than failing. |
| D11 | `.gsproj` | `FileDocument` (value type, `nonisolated`, Sendable contents) with `ProjectContents { manifest, files: [relativePath: Data] }`. Layout: `manifest.json`, `images/…`, `poses/colmap/{cameras,images,points3D}.bin` or `poses/arkit.json`, `checkpoints/latest.ply`. Checkpoint = standard 3DGS PLY (same as export), **no optimizer state** (resume restarts Adam). `formatVersion = 1`; newer versions are rejected. | Simplest model that works with strict concurrency; checkpoints are only written at finish/stop, so whole-package writes are acceptable. |
| D12 | Training loop | `GPUTrainer` (non-Sendable class, synchronous) owned by `actor TrainingSession`. One iteration = projection command buffer (the renderer's existing CPU sync for the intersection count) + one command buffer with rasterize → loss → backward (+stats) → Adam, committed without waiting; waits only at log/eval/densify/preview boundaries. Camera order: seeded per-epoch shuffle. Cancellation checked between iterations (densification is atomic w.r.t. cancellation). Pause = cancellable 50 ms sleep loop. | Fits Swift 6 isolation without sharing Metal objects; one blocked cooperative thread is acceptable for Phase 3. |
| D13 | Memory budget | `bytesPerGaussian = 16·P + 4·3K + 340` with `P = 11 + 3K` (params, grads, m, v; scatter scratch; 340 B renderer/stat/densify scratch + ~8 tile intersections). Cap = `min(config.maxGaussians (1.5 M), ⌊(0.6·recommendedMaxWorkingSetSize − reservedImageBytes) / bytesPerGaussian⌋)`. SH 1 → 756 B. | Extends the recorded 368 B/Gaussian (params+grads+m+v) with the real auxiliary buffers. |

## Global Constraints

- Everything in the Phase 1 and Phase 2 Global Constraints still applies: macOS 26 deployment target, `swift-tools-version: 6.2`, Swift 6 strict concurrency; library modules (`SplatCore`, `SplatIO`, `SplatMetal`, **`SplatTraining`**) use default nonisolated isolation — never `defaultIsolation(MainActor.self)` on a library target; only the app is `@MainActor` by default (and has `SWIFT_APPROACHABLE_CONCURRENCY = YES`, i.e. `NonisolatedNonsendingByDefault` is on: CPU-heavy app work must be `@concurrent`).
- `SplatCore` and `SplatIO` import only the Swift stdlib + `Foundation` and must build and test on Linux (`cd SplatKit && swift test`). No `simd`, `ImageIO`, `CoreGraphics`, `Metal` there, not even behind `#if`.
- No third-party dependencies. No `DispatchQueue` in new code.
- Swift/MSL shared structs contain only `Float`, `UInt32`, `Int32`, `SIMD4<Float>`, `SIMD4<UInt32>` fields. **Phase 3 does not edit `ShaderTypes.swift` or `Common.metal`**: each new kernel family declares its uniform struct at the top of its own `.metal` file and mirrors it in its own Swift file.
- Shaders: runtime-compiled, `mathMode = .safe`, MSL 3.0. New files `Adam.metal`, `Loss.metal`, `Densify.metal` are appended after `Backward` in `MetalContext.shaderFiles` (done once, in Task 11).
- All buffers `.storageModeShared` via `MetalContext.makeBuffer` (never zero length). CPU reads/writes of GPU buffers only after the last command buffer touching them completed.
- Determinism: library code never uses `SystemRandomNumberGenerator`; all randomness goes through `SeededGenerator` (CPU-only sequences) or `SplatRandom` (index-addressable hash, identical in Swift and MSL). CPU references accumulate in `Double`.
- Gradient conventions of Phase 2 are unchanged. L1 subgradient: `sign(0) = 0`.
- Typed throws where the module already uses them (`throws(MetalError)`, `throws(SplatIOError)`); new errors: `DatasetError`, `ProjectError` (SplatIO), `TrainingError` (SplatTraining). **Do not add cases to `SplatIOError`** (the app switches over it exhaustively).
- CI budget (paravirtual GPU): each new macOS test ≤ 10 s, the two overfit tests ≤ 60 s each, images ≤ 64×64 in GPU tests. Metal test suites are gated on Metal availability like the existing ones (`MetalTestSupport.swift`).
- Licensing: gsplat (Apache-2.0) is the reference for the densification strategy and default hyperparameters; **never port code from `graphdeco-inria/gaussian-splatting`** (including its `loss_utils` SSIM and densification code). SSIM is implemented from Wang et al. 2004 and the formulas below.
- Phase 4 extension points must exist but stay unused: `TrainingConfiguration.regularizers` (empty set), `TrainingView.depth` (`nil`), `GPUTrainer`'s `AuxiliaryLoss` hook list (empty), `TrainingConfiguration.heldOutViews` (empty). No depth, pseudo-view or anisotropy code in Phase 3.
- Never claim a macOS/Metal test passed unless CI ran it. Workers on Linux return the diff for macOS-only tasks unverified and say so.

## Review Focus

1. **COLMAP images whose file resolution differs from `cameras.bin`, and views with different resolutions** — intrinsics must be rescaled (same aspect) or the import rejected with a named error; training must handle per-view sizes (grow-only buffers). Tests: Task 8 `scalesIntrinsicsWhenImageFileIsSmaller`, `rejectsAspectMismatch`; Task 15 `trainsViewsWithDifferentResolutions`.
2. **Images listed in `images.bin` but missing from the folder** — import fails with `DatasetError.missingImages` naming every missing file. Test: Task 8 `missingImagesAreReportedByName`.
3. **NaN/Inf loss** (e.g. a corrupt target pixel, exploding LR) — training stops with `TrainingError.nonFiniteLoss(iteration:)`, the stream reports `.failed`, the last good snapshot is still obtainable. Tests: Task 10 `nonFiniteTargetIsReported`, Task 15 `nonFiniteLossStopsWithError`.
4. **Gaussian count reaching the budget** — growth beyond the cap is demoted to keep, deterministically; count never exceeds `maxGaussians`; pruning never empties the model. Tests: Task 4 `planDemotesGrowthBeyondBudget`, `planNeverPrunesEverything`; Task 14 `densifyRespectsCap`; Task 15 `overfitWithDensificationStaysWithinCap`.
5. **A view in which no Gaussian is visible** — step succeeds, gradients and stats stay zero, loss finite. Tests: Task 4 `statisticsIgnoreInvisibleGaussians`; Task 15 `viewWithNoVisibleGaussiansIsHarmless`.
6. **Cancellation during a run that densifies** — `run()` returns a consistent cloud (all array lengths agree, count ≤ cap), the stream ends with `.cancelled`. Test: Task 15 `cancellationLeavesConsistentModel`.

## Task overview and dependencies

| Task | Title | Platform | Depends on | Shared files touched |
|---|---|---|---|---|
| 1 | Training foundations (config, RNG, extent, budget, dataset types) | Linux-testable | — | — |
| 2 | CPU Adam | Linux-testable | 1 | — |
| 3 | CPU loss (L1 + D-SSIM) and PSNR | Linux-testable | — | — |
| 4 | CPU densification (stats, plan, apply) | Linux-testable | 1 | `CPUBackward.swift` |
| 5 | Initialization (k-d tree, init from points / random) | Linux-testable | 1 | — |
| 6 | Camera models, undistortion, resampling | Linux-testable | — | — |
| 7 | COLMAP binary reader | Linux-testable | 6 | — |
| 8 | Dataset loading (COLMAP + ARKit + PPM) | Linux-testable | 1, 6, 7 | — |
| 9 | PLY writer and `.gsproj` contents | Linux-testable | 1, 8 | — |
| 10 | CPU training loop + smoke test | Linux-testable | 1, 2, 3, 4, 5 | — |
| 11 | `GaussianBuffers`, renderer generalization, shader registration | macOS-CI-only | — | `SplatRenderer.swift`, `GradientBuffers.swift`, `MetalContext.swift` |
| 12 | Metal Adam | macOS-CI-only | 2, 11 | — |
| 13 | Metal loss | macOS-CI-only | 3, 11 | — |
| 14 | Metal densification + stats hook | macOS-CI-only, **Escalate-to: opus-review** | 4, 11 | `SplatRenderer.swift` |
| 15 | `SplatTraining`: `GPUTrainer`, `TrainingSession` | macOS-CI-only, **Escalate-to: opus-review** | 1, 5, 12, 13, 14 | `Package.swift` |
| 16 | App: project document, import, Train, live preview | macOS-CI-only (build) | 5, 8, 9, 15 | `project.yml`, `App/Info.plist`, `App/SparseSplatApp.swift`, `App/ViewerModel.swift`, `App/Viewport/*` |

Parallel waves (no two tasks in a wave touch the same file): **A** = 1, 3, 6, 11 · **B** = 2, 4, 5, 7, 13 · **C** = 8, 10, 12, 14 · **D** = 9, 15 · **E** = 16. Linux tasks are verified locally; macOS tasks are batched onto CI pushes by the orchestrator. Only Task 11 and Task 14 edit `SplatRenderer.swift` (serialized by the dependency); only Task 11 edits `MetalContext.swift`; only Task 15 edits `Package.swift`.

New `SplatCore` training code lives in `SplatKit/Sources/SplatCore/Training/` (SwiftPM picks up subfolders; no manifest change). New Linux tests live in `SplatKit/Tests/SplatCoreTests/` and `SplatKit/Tests/SplatIOTests/`, one new file per task.

Verification commands used below:
- Linux: `export PATH=…` as printed by `Tools/install-swift-linux.sh`, then `cd SplatKit && swift test --filter <Suite>` and the full `swift test`.
- macOS: the orchestrator pushes the branch; CI job `macos` (`swift test` in `SplatKit`, then the app build) must be green.

---

### Task 1: Training foundations

**Platform:** Linux-testable · **Depends on:** —

**Files:**
- Create: `SplatKit/Sources/SplatCore/Training/TrainingConfiguration.swift`, `SeededRandom.swift`, `SceneExtent.swift`, `TrainingMemoryBudget.swift`, `TrainingDataset.swift`
- Test: `SplatKit/Tests/SplatCoreTests/TrainingFoundationTests.swift`

**Interfaces (Produces):**
```swift
public struct LearningRates: Codable, Sendable, Equatable {
    public var positionInitial: Float = 1.6e-4   // × scene radius
    public var positionFinal: Float = 1.6e-6     // × scene radius
    public var logScale: Float = 5e-3
    public var rotation: Float = 1e-3
    public var opacity: Float = 5e-2
    public var shDC: Float = 2.5e-3
    public var shRest: Float = 1.25e-4
}
public struct ResolvedLearningRates: Sendable, Equatable { public var position, logScale, rotation, opacity, shDC, shRest: Float }
public struct AdamHyperparameters: Codable, Sendable, Equatable { public var beta1: Float = 0.9; public var beta2: Float = 0.999; public var epsilon: Float = 1e-15 }
public struct DensificationPolicy: Codable, Sendable, Equatable {
    public var startIteration = 500, stopIteration = 6_000, interval = 100
    public var gradientThreshold: Float = 0.0002
    public var splitScaleFraction: Float = 0.01      // × scene radius: clone if max scale ≤ this, else split
    public var pruneOpacity: Float = 0.005
    public var pruneScaleFraction: Float = 0.1       // × scene radius
    public var pruneScaleStartIteration = 3_000
    public var splitScaleDivisor: Float = 1.6
    public var opacityResetInterval = 0              // recorded decision 15: off; must stay 0 in Phase 3
    public var isEnabled: Bool { startIteration < stopIteration && interval > 0 }
}
public enum RegularizerKind: String, Codable, Sendable { case depthPearson, depthL1, anisotropy, scaleCap, isolationPrune }  // Phase 4; unused
public struct TrainingConfiguration: Codable, Sendable, Equatable {
    public var iterations = 10_000
    public var shDegree = 1
    public var shDegreeInterval = 1_000
    public var lambdaDSSIM: Float = 0.2
    public var background: SIMD3<Float> = .zero
    public var learningRates = LearningRates()
    public var adam = AdamHyperparameters()
    public var densification = DensificationPolicy()
    public var maxGaussians = 1_500_000
    public var maxImageSide = 1_600
    public var initialOpacity: Float = 0.1
    public var randomInitCount = 10_000
    public var seed: UInt64 = 0
    public var logInterval = 10, evalInterval = 500, previewInterval = 200
    public var regularizers: Set<RegularizerKind> = []   // Phase 4 extension point
    public var heldOutViews: [String] = []               // Phase 4 extension point (leave-one-out)
    public init()
    public init(from decoder: Decoder) throws   // every key optional, missing → default
    public func activeSHDegree(iteration: Int) -> Int           // min(shDegree, iteration / shDegreeInterval)
    public func learningRates(iteration: Int, sceneRadius: Float) -> ResolvedLearningRates
}
public struct SeededGenerator: RandomNumberGenerator, Sendable { public init(seed: UInt64); public mutating func next() -> UInt64 }  // SplitMix64
public enum SplatRandom {
    public static func hash(_ x: UInt32) -> UInt32
    public static func uniform(seed: UInt32, index: UInt32, stream: UInt32) -> Float   // in (0, 1)
    public static func normal(seed: UInt32, index: UInt32, stream: UInt32) -> Float
}
public struct SceneExtent: Sendable, Equatable {
    public var center: SIMD3<Float>; public var radius: Float
    public static func estimate(cameraCenters: [SIMD3<Float>], points: [SIMD3<Float>]) -> SceneExtent
}
public enum TrainingMemoryBudget {
    public static func bytesPerGaussian(shDegree: Int) -> Int
    public static func reservedImageBytes(viewSizes: [SIMD2<Int>]) -> Int
    public static func maxGaussians(workingSetBytes: UInt64, shDegree: Int, reservedBytes: Int, configuredMax: Int, fraction: Double = 0.6) -> Int
}
public struct PointCloud: Sendable, Equatable { public var positions: [SIMD3<Float>]; public var colors: [SIMD3<Float>] /* 0…1 */; public init(positions:colors:) }
public struct TrainingView: Sendable {
    public var name: String; public var camera: Camera; public var image: RGBImage
    public var depth: [Float]?   // Phase 4 extension point, always nil in Phase 3
}
public struct TrainingDataset: Sendable { public var views: [TrainingView]; public var points: PointCloud?; public var cameraCenters: [SIMD3<Float>] { get } }
```

Exact algorithms:
```text
SplitMix64: state += 0x9E3779B97F4A7C15; z = state; z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9;
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB; return z ^ (z >> 31)
SplatRandom.hash (PCG-RXS-M-XS): s = x &* 747796405 &+ 2891336453;
            w = ((s >> ((s >> 28) &+ 4)) ^ s) &* 277803737; return (w >> 22) ^ w
key       = hash(seed ^ hash(index ^ hash(stream)))
uniform   = (Float(key >> 8) + 0.5) / 16777216
normal    = sqrt(-2·log(uniform(seed, index, 2·stream))) · cos(2π · uniform(seed, index, 2·stream + 1))
position lr(t) = radius · exp((1−r)·ln(positionInitial) + r·ln(positionFinal)), r = clamp(t / iterations, 0, 1)
SceneExtent: centroid of camera centers; radius = 1.1 · max distance to centroid.
            If radius < 1e-6 (one camera / coincident): center = mean of points, radius = half the diagonal of the points'
            5–95 % per-axis percentile box; if that is also < 1e-6 or there are no points: radius = 1.
bytesPerGaussian(d): K = (d+1)², P = 11 + 3K → 16·P + 12·K + 340      (d=0: 576, d=1: 756, d=3: 1476)
reservedImageBytes: Σ_views 16·w·h (targets) + 3·16·wmax·hmax (render texture, image gradient, spare) + 24·4·wmax·hmax (loss scratch)
maxGaussians = min(configuredMax, max(0, Int((fraction·workingSet − reserved) / bytesPerGaussian)))
```
`TrainingMemoryBudget` is pure arithmetic so it is Linux-testable; the macOS caller (Task 15) passes `device.recommendedMaxWorkingSetSize`.

- [ ] **Step 1: Write the failing tests** (`@Suite struct TrainingFoundationTests`)
```swift
@Test func defaultsMatchArchitecture()        // shDegree 1, opacityResetInterval 0, lambda 0.2, iterations 10_000
@Test func decodesWithMissingKeysUsingDefaults() // decode "{}" and {"iterations": 50} → defaults except iterations
@Test func configurationRoundTripsThroughJSON()
@Test func activeSHDegreeSteps()              // shDegree 3, interval 1000: it 0→0, 999→0, 1000→1, 5000→3
@Test func positionLearningRateDecaysExponentially() // radius 2: it 0 → 3.2e-4, it = iterations → 3.2e-6, midpoint → 3.2e-5 (rel 1e-4)
@Test func seededGeneratorIsDeterministic()   // seed 0: first output == 0xE220A8397B1DCDAF (SplitMix64 reference); same seed → equal sequences, different seeds differ
@Test func splatRandomNormalHasUnitMoments()  // 100_000 samples (index 0..<1e5): |mean| < 0.02, |var − 1| < 0.03; uniform strictly in (0,1)
@Test func extentFromCameraCenters()          // centers (±1,0,0): center 0, radius 1.1
@Test func extentFallsBackToPointsForSingleCamera()
@Test func extentFallsBackToOneWithoutData()
@Test func bytesPerGaussianValues()           // 576, 756, 1476 for degrees 0, 1, 3
@Test func maxGaussiansRespectsConfiguredCapAndBudget() // tiny working set → budget-bound; huge → configuredMax; reserved > budget → 0
```
- [ ] **Step 2: Run** `swift test --filter TrainingFoundationTests` → fails to compile.
- [ ] **Step 3: Implement** the files above (custom `init(from:)` uses `decodeIfPresent` for every key).
- [ ] **Step 4: Run** the suite and the full `swift test` → pass.
- [ ] **Step 5: Return diff** (commit: `Add training configuration, seeded RNG, scene extent and memory budget`).

### Task 2: CPU Adam

**Platform:** Linux-testable · **Depends on:** 1

**Files:**
- Create: `SplatKit/Sources/SplatCore/Training/CPUAdam.swift`
- Test: `SplatKit/Tests/SplatCoreTests/CPUAdamTests.swift`

**Interfaces:**
- Consumes: `GaussianCloud`, `GaussianGradients`, `ResolvedLearningRates`, `AdamHyperparameters`.
- Produces:
```swift
public typealias GaussianMoments = GaussianGradients   // same SoA layout, used for Adam m and v
public struct AdamState: Sendable, Equatable { public var m: GaussianMoments; public var v: GaussianMoments; public var step: Int; public init(zeroLike: GaussianCloud) }
public enum CPUAdam {
    /// Updates `parameters` in place. Float `i` belongs to Gaussian `i / stride`; floats with `i % stride < headCount` use `headRate`, the others `tailRate`.
    public static func update(_ parameters: inout [Float], gradients: [Float], m: inout [Float], v: inout [Float],
                              stride: Int, headCount: Int, headRate: Float, tailRate: Float,
                              step: Int, hyper: AdamHyperparameters)
    /// Increments `state.step`, then updates all five groups (sh: stride 3K, headCount 3 → shDC/shRest).
    public static func step(_ cloud: GaussianCloud, gradients: GaussianGradients, state: inout AdamState,
                            rates: ResolvedLearningRates, hyper: AdamHyperparameters) -> GaussianCloud
}
```
Exact update (Float arithmetic in this order — the Metal kernel copies it, Task 12):
```text
bc1 = 1 − β1^t ; bc2 = 1 − β2^t          (t = step ≥ 1, computed in Double, passed as Float)
stepSize = rate / bc1 ; invSqrtBC2 = 1 / sqrt(bc2)
m = β1·m + (1−β1)·g
v = β2·v + (1−β2)·g·g
p = p − stepSize · m / (sqrt(v)·invSqrtBC2 + ε)
```
Quaternions are not renormalized (the forward normalizes).

- [ ] **Step 1: Write the failing tests**
```swift
@Test func firstStepMovesBySignTimesRate()     // t=1, g = [0.5, −2, 1e-3]: Δp ≈ −rate·sign(g) (rel 1e-5)
@Test func matchesHandComputedSecondStep()     // two steps with g1, g2: compare to a Double re-derivation (rel 1e-6)
@Test func headAndTailRatesApplyPerStride()    // stride 6, headCount 3: floats 0..2 use headRate, 3..5 tailRate
@Test func zeroGradientWithZeroMomentsIsNoOp()
@Test func stepUpdatesEveryGroupWithItsRate()  // 2-Gaussian degree-1 cloud, unit gradients: each group moves by its own rate
```
- [ ] **Step 2–4:** run (fail) → implement → `swift test --filter CPUAdamTests` and full `swift test` pass.
- [ ] **Step 5: Return diff** (commit: `Add CPU reference Adam optimizer`).

### Task 3: CPU loss (L1 + D-SSIM) and PSNR

**Platform:** Linux-testable · **Depends on:** —

**Files:**
- Create: `SplatKit/Sources/SplatCore/Training/ImageLoss.swift`, `SplatKit/Sources/SplatCore/Training/ImageMetrics.swift`
- Test: `SplatKit/Tests/SplatCoreTests/ImageLossTests.swift`

**Interfaces (Produces):**
```swift
public struct ImageLossResult: Sendable { public var loss: Double; public var l1: Double; public var ssim: Double; public var gradient: [SIMD3<Float>] /* dL/drendered, row-major */ }
public enum ImageLoss {
    public static let windowRadius = 5
    public static let window: [Float]            // 11 taps, exp(−(i−5)²/(2·1.5²)) normalized to sum 1
    public static let c1: Float = 0.0001, c2: Float = 0.0009
    public static func evaluate(rendered: RGBImage, target: RGBImage, lambda: Float) -> ImageLossResult
}
public enum ImageMetrics {
    public static func mse(_ a: RGBImage, _ b: RGBImage) -> Double
    public static func psnr(_ a: RGBImage, _ b: RGBImage) -> Double     // 10·log10(1/mse); mse == 0 → +infinity
}
```
Math (per channel independently; `G⊛` = separable 11-tap blur, horizontal then vertical, zero padding outside the image; N = W·H·3):
```text
μx = G⊛x, μy = G⊛y, exx = G⊛x², eyy = G⊛y², exy = G⊛(x·y)
σxx = exx − μx², σyy = eyy − μy², σxy = exy − μx·μy
A1 = 2μxμy + C1, A2 = 2σxy + C2, B1 = μx² + μy² + C1, B2 = σxx + σyy + C2,  S = A1·A2 / (B1·B2)
loss = (1−λ)·Σ|x−y|/N + λ·(1 − ΣS/N)
Backward (g = −λ/N per pixel, y is constant):
  M0 = g·S·(2μy/A1 − 2μy/A2 − 2μx/B1 + 2μx/B2)        (∂S/∂μx, including μx inside σxx, σxy)
  M1 = g·(−S/B2)                                       (∂S/∂exx)
  M2 = g·(2S/A2)                                       (∂S/∂exy)
  dL/dx = (1−λ)/N·sign(x−y) + (G⊛M0) + 2x·(G⊛M1) + y·(G⊛M2)
```
(G is symmetric, so the adjoint of zero-padded `G⊛` is the same zero-padded `G⊛`.) Compute in `Double`, return the gradient as `Float`.

- [ ] **Step 1: Write the failing tests**
```swift
@Test func identicalImagesHaveZeroLossAndUnitSSIM()   // random 13×9 image vs itself: loss < 1e-7, ssim > 1 − 1e-6
@Test func windowIsNormalizedAndSymmetric()
@Test func l1OnlyGradientIsSign()                      // lambda 0: gradient == sign(x−y)/N exactly
@Test func finiteDifferencesMatch()                    // 12×10 random rendered/target (SeededGenerator seed 3), λ = 0.2,
                                                       // 40 random entries, central difference h = 1e-3 on the Float input,
                                                       // |fd − analytic| ≤ 2e-2·|analytic| + 1e-6
@Test func ssimIsTranslationSensitive()               // shifted image → ssim < 0.9
@Test func psnrOfKnownMSE()                            // constant offset 0.1 → mse 0.01 → psnr 20 dB; identical → .infinity
```
- [ ] **Step 2–4:** run (fail) → implement → `swift test --filter ImageLossTests` and full `swift test` pass.
- [ ] **Step 5: Return diff** (commit: `Add CPU reference L1 + D-SSIM loss and PSNR`).

### Task 4: CPU densification

**Platform:** Linux-testable · **Depends on:** 1

**Files:**
- Modify: `SplatKit/Sources/SplatCore/CPUBackward.swift` (expose screen-space gradients; `backward` delegates and stays bit-identical)
- Create: `SplatKit/Sources/SplatCore/Training/Densification.swift`
- Test: `SplatKit/Tests/SplatCoreTests/DensificationTests.swift`

**Interfaces:**
- Consumes: `CPURasterizer.backward`, `GaussianMoments`, `DensificationPolicy`, `SplatRandom`, `Matrix3.rotation(quaternion:)`.
- Produces:
```swift
public struct ScreenSpaceBackward: Sendable { public var gradients: GaussianGradients; public var mean2D: [SIMD2<Float>] /* dL/dmean2D in pixels, per Gaussian */; public var visible: [Bool] /* projected and tile rect non-empty */ }
extension CPURasterizer { public static func backwardWithScreenSpace(_ cloud: GaussianCloud, camera: Camera, settings: RasterSettings, imageGradient: [SIMD3<Float>]) -> ScreenSpaceBackward }
public struct DensificationStatistics: Sendable, Equatable {
    public var gradientNormSums: [Float]; public var visibleCounts: [UInt32]
    public init(count: Int)
    public mutating func accumulate(mean2D: [SIMD2<Float>], visible: [Bool], width: Int, height: Int)
    public mutating func reset()
}
public enum DensifyAction: UInt32, Sendable { case prune = 0, keep = 1, clone = 2, split = 3
    public var outputCount: Int { get } }   // 0, 1, 2, 2
public struct DensifyPlan: Sendable, Equatable { public var actions: [DensifyAction]; public var outputCount: Int; public var pruned, cloned, split, demoted: Int; public var keptAll: Bool }
public enum Densification {
    public static func action(logScale: SIMD3<Float>, opacityLogit: Float, gradientNormSum: Float, visibleCount: UInt32,
                              policy: DensificationPolicy, sceneRadius: Float, iteration: Int) -> DensifyAction
    public static func plan(_ cloud: GaussianCloud, statistics: DensificationStatistics, policy: DensificationPolicy,
                            sceneRadius: Float, iteration: Int, maxGaussians: Int) -> DensifyPlan
    public static func apply(_ plan: DensifyPlan, to cloud: GaussianCloud, state: inout AdamState,
                             policy: DensificationPolicy, seed: UInt32, iteration: Int) -> GaussianCloud
}
```
Exact rules (the Metal kernels in Task 14 mirror these line by line):
```text
accumulate: for visible i: gradientNormSums[i] += ‖(mean2D[i].x·W/2, mean2D[i].y·H/2)‖; visibleCounts[i] += 1
action:
  maxScale = exp(max(logScale)); opacity = 1/(1+exp(−logit))
  if opacity < pruneOpacity                                                  → prune
  if iteration > pruneScaleStartIteration && maxScale > pruneScaleFraction·R  → prune
  avg = visibleCount > 0 ? gradientNormSum / Float(visibleCount) : 0
  if avg > gradientThreshold: maxScale ≤ splitScaleFraction·R ? clone : split
  else keep
plan:
  1. actions = action(i) for all i
  2. if no action is non-prune and count > 0: every prune → keep, keptAll = true  (never empty the model)
  3. base = #keep + #clone + #split; allowed = max(0, maxGaussians − base)
     growth candidates (clone/split) in ascending index order; candidate rank ≥ allowed → keep, demoted += 1
  4. outputCount = Σ action.outputCount
apply (outputs of input i occupy offsets[i] ..< offsets[i] + outputCount, offsets = exclusive scan, input order kept):
  keep : copy params, copy m, v
  clone: child 0 = copy with its m, v; child 1 = copy with m = v = 0
  split: for j in 0..<2: logScale' = logScale − log(splitScaleDivisor);
         pos' = pos + R(q̂)·(exp(logScale) ⊙ ε_j), ε_j[a] = SplatRandom.normal(seed: seed &+ UInt32(iteration), index: UInt32(i), stream: UInt32(3·j + a));
         rotation, opacity, sh copied; m = v = 0; state.step unchanged
  After apply the caller resets statistics to the new count.
```
This differs from gsplat in one documented point: gsplat grows first and prunes the grown set; we decide prune on the parent, so children of a prunable parent are never created.

- [ ] **Step 1: Write the failing tests**
```swift
@Test func screenSpaceBackwardKeepsGradientsIdentical()   // all golden cases: .gradients == backward(...) exactly
@Test func mean2DGradientMatchesPositionGradientOnAxis()  // one isotropic degree-0 Gaussian on the optical axis (x = y = 0, identity camera):
                                                          // there dΣ2D/dx = 0, so mean2D.x gradient == positions gradient x · z / fx (rel ≤ 1e-4), same for y
@Test func statisticsIgnoreInvisibleGaussians()           // camera looking away: all visible == false, sums and counts stay 0
@Test func actionTable()                                  // one case per branch, incl. boundary avg == threshold → keep, scale-prune only after start iteration
@Test func planDemotesGrowthBeyondBudget()                // 10 Gaussians, 6 growth candidates, maxGaussians 13 → allowed 3, lowest 3 indices grow, demoted 3, outputCount 13
@Test func planNeverPrunesEverything()                    // all opacity 1e-4 → keptAll, outputCount == count
@Test func applyLayoutAndMoments()                        // [keep, clone, prune, split]: 5 outputs in order; clone child 1 and both split children have m = v = 0
@Test func splitChildrenAreDeterministicAndShrunk()       // same seed → identical; logScale − log 1.6; positions differ from parent by R·(s⊙ε)
@Test func applyPreservesSHAndRotationLayouts()           // degree-2 cloud: GaussianCloud init succeeds, sh of children equals parent
```
- [ ] **Step 2–4:** run (fail) → implement → `swift test --filter "DensificationTests|CPUBackwardTests"` and full `swift test` pass.
- [ ] **Step 5: Return diff** (commit: `Add CPU reference densification and screen-space gradients`).

### Task 5: Initialization

**Platform:** Linux-testable · **Depends on:** 1

**Files:**
- Create: `SplatKit/Sources/SplatCore/Training/KDTree.swift`, `SplatKit/Sources/SplatCore/Training/GaussianInitializer.swift`
- Test: `SplatKit/Tests/SplatCoreTests/InitializationTests.swift`

**Interfaces (Produces):**
```swift
public struct KDTree: Sendable {
    public init(points: [SIMD3<Float>])
    /// Indices of the k nearest points to points[index], excluding index itself; ties broken by smaller index; fewer if count − 1 < k.
    public func nearestNeighbors(of index: Int, k: Int) -> [Int]
}
public enum GaussianInitializer {
    public static func nearestNeighborScales(_ positions: [SIMD3<Float>], k: Int = 3) -> [Float]   // linear scale per point
    public static func fromPoints(_ points: PointCloud, shDegree: Int, initialOpacity: Float, maxCount: Int, seed: UInt64) -> GaussianCloud
    public static func random(count: Int, extent: SceneExtent, shDegree: Int, initialOpacity: Float, seed: UInt64) -> GaussianCloud
    public static func initialize(_ dataset: TrainingDataset, configuration: TrainingConfiguration, maxCount: Int) -> GaussianCloud
}
```
Algorithms:
```text
KDTree build: index array; recursive on ranges; split axis = largest extent of the range's bounding box;
              order the range by (coordinate on axis, index) and split at the median; leaves ≤ 8 points.
query: depth-first, near child first; bounded max-heap of size k keyed by (squared distance, index);
       prune a far child when (split plane distance)² > current k-th distance and the heap is full.
scale_i = sqrt(max(mean of squared distances to its available neighbours (≤ k), 1e-7)); a single point (no neighbours) → scale 0.01.
fromPoints: if count > maxCount, keep a uniformly random subset of maxCount (SeededGenerator(seed), ascending index order kept).
            logScale = log(scale) on all 3 axes; rotation (1,0,0,0); opacityLogit = log(o/(1−o));
            sh DC = (color − 0.5) / 0.28209479177387814, other coefficients 0.
random:     positions uniform in the cube center ± radius (SeededGenerator), colors uniform in [0,1], then as fromPoints.
initialize: points non-empty → fromPoints, else random(count: min(randomInitCount, maxCount), extent: estimate(cameraCenters, points: [])).
```

- [ ] **Step 1: Write the failing tests**
```swift
@Test func kdTreeMatchesBruteForce()          // 2_000 random points + 50 exact duplicates, k = 3: identical index sets for every point
@Test func kdTreeHandlesTinyInputs()          // 1 point → [], 2 points → [other], k larger than count − 1
@Test func scalesOnRegularGrid()              // unit-spaced 5×5×5 grid: interior point scale == 1 (rel 1e-6)
@Test func duplicatePointsGetMinimumScale()   // scale == sqrt(1e-7)
@Test func fromPointsSetsActivationsAndColor()// opacity sigmoid(logit) == 0.1, DC reproduces color via SH eval (degree 0) within 1e-6
@Test func fromPointsSubsamplesDeterministically() // 1000 points, maxCount 100: count 100, same seed → same cloud
@Test func randomFallbackStaysInCube()
```
- [ ] **Step 2–4:** run (fail) → implement → `swift test --filter InitializationTests` and full `swift test` pass. Also check build time of `kdTreeMatchesBruteForce` stays < 2 s in debug.
- [ ] **Step 5: Return diff** (commit: `Add k-d tree and Gaussian initialization from points`).

### Task 6: Camera models, undistortion, resampling

**Platform:** Linux-testable · **Depends on:** —

**Files:**
- Create: `SplatKit/Sources/SplatCore/CameraModel.swift`, `SplatKit/Sources/SplatCore/Training/ImageResampling.swift`
- Test: `SplatKit/Tests/SplatCoreTests/CameraModelTests.swift`

**Interfaces (Produces):**
```swift
public enum CameraModel: Sendable, Equatable {
    case simplePinhole(f: Double, cx: Double, cy: Double)
    case pinhole(fx: Double, fy: Double, cx: Double, cy: Double)
    case simpleRadial(f: Double, cx: Double, cy: Double, k: Double)
    case radial(f: Double, cx: Double, cy: Double, k1: Double, k2: Double)
    case opencv(fx: Double, fy: Double, cx: Double, cy: Double, k1: Double, k2: Double, p1: Double, p2: Double)
    public var focal: SIMD2<Double> { get }; public var principalPoint: SIMD2<Double> { get }
    public var hasDistortion: Bool { get }        // false for pinhole kinds and when all coefficients are 0
    public func distort(_ normalized: SIMD2<Double>) -> SIMD2<Double>
}
extension RGBImage {
    public func bilinearSample(x: Double, y: Double) -> SIMD3<Float>   // pixel-index coordinates, clamp to edge
    public func downsampled(by factor: Int) -> RGBImage                // box average of factor×factor blocks, size w/f × h/f (floor)
}
public enum Undistortion { public static func undistort(_ image: RGBImage, model: CameraModel) -> RGBImage }
extension Camera {
    public init(model: CameraModel, width: Int, height: Int, rotation: Matrix3, translation: SIMD3<Float>)   // pinhole part only
    public func downscaled(by factor: Int) -> Camera   // width/f, height/f (floor), fx, fy, cx, cy divided by f
    public func rescaled(toWidth: Int, height: Int) -> Camera   // fx·w'/w, cx·w'/w, fy·h'/h, cy·h'/h
}
public enum ImageResampling { public static func downsampleFactor(width: Int, height: Int, maxSide: Int) -> Int }  // ceil(max(w,h)/maxSide), ≥ 1
```
Distortion (COLMAP formulas, `(u, v)` normalized, `r² = u² + v²`): SIMPLE_RADIAL `(u, v)·(1 + k·r²)`; RADIAL `(u, v)·(1 + k1·r² + k2·r⁴)`; OPENCV `radial = k1·r² + k2·r⁴`, `du = u·radial + 2p1·u·v + p2·(r² + 2u²)`, `dv = v·radial + 2p2·u·v + p1·(r² + 2v²)`, result `(u + du, v + dv)`. Undistort: output pixel `(x, y)` center `(x+0.5, y+0.5)` → `n = ((x+0.5−cx)/fx, (y+0.5−cy)/fy)` → `d = distort(n)` → sample source at `(fx·d.x + cx − 0.5, fy·d.y + cy − 0.5)`.

- [ ] **Step 1: Write the failing tests**
```swift
@Test func zeroDistortionIsIdentity()          // opencv with all coefficients 0: undistort returns the same pixels (≤ 1e-6)
@Test func simpleRadialMovesPointsOutward()    // k = 0.1: distort((0.5, 0)) == (0.5125, 0)
@Test func opencvTangentialTerms()             // hand-computed value for (0.3, −0.2) with p1, p2 only
@Test func undistortRecoversSyntheticCheckerboardCorner() // render a corner at a distorted location, undistort, corner lands at the pinhole location ± 0.5 px
@Test func downsampleAveragesBlocks()          // 4×2 → factor 2 → 2×1, exact averages
@Test func downscaledCameraProjectsConsistently() // a world point at continuous pixel coordinate p in the full camera lands at p / f in the downscaled one (≤ 1e-4)
@Test func downsampleFactorForMaxSide()        // 4000×3000, 1600 → 3; 1600×900 → 1
```
- [ ] **Step 2–4:** run (fail) → implement → `swift test --filter CameraModelTests` and full `swift test` pass.
- [ ] **Step 5: Return diff** (commit: `Add camera models, undistortion and image downsampling`).

### Task 7: COLMAP binary reader

**Platform:** Linux-testable · **Depends on:** 6

**Files:**
- Create: `SplatKit/Sources/SplatIO/COLMAPReader.swift`, `SplatKit/Sources/SplatIO/DatasetError.swift`
- Create: `SplatKit/Tests/SplatIOTests/COLMAPFixture.swift` (test-only writer for synthetic binaries), `SplatKit/Tests/SplatIOTests/COLMAPReaderTests.swift`

**Interfaces (Produces):**
```swift
public enum DatasetError: Error, Equatable {
    case truncated(file: String)
    case unknownCameraModel(Int32)
    case unsupportedCameraModel(String)
    case missingCamera(image: String, cameraID: UInt32)
    case missingImages([String])
    case imageDecodingFailed(name: String, reason: String)
    case imageSizeMismatch(name: String, expected: SIMD2<Int>, actual: SIMD2<Int>)
    case invalidPose(name: String)
    case invalidARKitCapture(String)
    case noViews
}
public struct COLMAPCamera: Sendable, Equatable { public var id: UInt32; public var modelID: Int32; public var width: Int; public var height: Int; public var params: [Double] }
public struct COLMAPImage: Sendable, Equatable { public var id: UInt32; public var rotation: SIMD4<Double> /* qw qx qy qz */; public var translation: SIMD3<Double>; public var cameraID: UInt32; public var name: String }
public struct COLMAPPoint: Sendable, Equatable { public var position: SIMD3<Double>; public var color: SIMD3<UInt8> }
public enum COLMAPReader {
    public static func parameterCount(modelID: Int32) -> Int?
    public static func readCameras(_ data: Data) throws(DatasetError) -> [COLMAPCamera]
    public static func readImages(_ data: Data) throws(DatasetError) -> [COLMAPImage]   // sorted by name
    public static func readPoints3D(_ data: Data) throws(DatasetError) -> [COLMAPPoint]
}
extension CameraModel { public init(colmap camera: COLMAPCamera) throws(DatasetError) }  // ids 0–4 supported; known others → .unsupportedCameraModel(name)
```
Binary layout (little-endian): `cameras.bin`: `u64 n`; per camera `i32 id, i32 model, u64 width, u64 height, f64 params[count(model)]`. `images.bin`: `u64 n`; per image `u32 id, f64 qw qx qy qz, f64 tx ty tz, u32 camera_id, name (bytes until 0x00), u64 n2D, n2D × (f64 x, f64 y, i64 point3D_id)` (skip 2D points). `points3D.bin`: `u64 n`; per point `u64 id, f64 x y z, u8 r g b, f64 error, u64 track_len, track_len × (i32 image_id, i32 point2D_idx)`. Parameter counts by model id: 0→3, 1→4, 2→4, 3→5, 4→8, 5→8, 6→12, 7→5, 8→4, 9→5, 10→12, 11→16; names `SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, OPENCV_FISHEYE, FULL_OPENCV, FOV, SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE, THIN_PRISM_FISHEYE, RAD_TAN_THIN_PRISM_FISHEYE`. Every read is bounds-checked (`.truncated(file:)`); counts larger than `data.count / minimumRecordSize` are rejected as truncated before allocating.

- [ ] **Step 1: Write the failing tests** (fixtures written with `COLMAPFixture`)
```swift
@Test func readsCamerasOfEverySupportedModel()
@Test func readsImagesWithPoints2DAndUTF8Names()   // 2 images, one with 3 2D points, name "cam0/ä.jpg"; result sorted by name
@Test func readsPoints3DWithTracks()
@Test func truncatedFilesThrow()                    // each file cut at several offsets → .truncated, never a crash
@Test func hugeCountIsRejectedWithoutAllocation()   // n = UInt64.max → .truncated
@Test func unknownModelThrows()                     // id 99 → .unknownCameraModel(99)
@Test func fisheyeModelIsUnsupported()              // id 5 → CameraModel(colmap:) throws .unsupportedCameraModel("OPENCV_FISHEYE")
```
- [ ] **Step 2–4:** run (fail) → implement → `swift test --filter COLMAPReaderTests` and full `swift test` pass.
- [ ] **Step 5: Return diff** (commit: `Add COLMAP binary reader`).

### Task 8: Dataset loading (COLMAP, ARKit, PPM)

**Platform:** Linux-testable · **Depends on:** 1, 6, 7

**Files:**
- Create: `SplatKit/Sources/SplatIO/TrainingDatasetLoader.swift`, `SplatKit/Sources/SplatIO/ARKitCapture.swift`, `SplatKit/Sources/SplatIO/PPMCodec.swift`
- Test: `SplatKit/Tests/SplatIOTests/DatasetLoaderTests.swift`

**Interfaces:**
- Consumes: Task 7 reader and `CameraModel(colmap:)`, Task 6 undistortion/downsampling/`Camera(model:…)`, Task 1 `TrainingDataset`, `PointCloud`.
- Produces:
```swift
public enum PPMCodec { public static func decode(_ data: Data) throws -> RGBImage; public static func encode(_ image: RGBImage) -> Data }  // binary P6, maxval 255
public struct ARKitCapture: Codable, Sendable, Equatable {
    public struct Frame: Codable, Sendable, Equatable {
        public var image: String; public var width: Int; public var height: Int
        public var intrinsics: [Double]      // fx, fy, cx, cy in pixels of width×height
        public var cameraToWorld: [Double]   // 16 values, column-major (simd_float4x4 columns), ARKit camera convention
        public var depth: String?            // Phase 4, ignored
    }
    public var version: Int; public var frames: [Frame]
    public static func camera(for frame: Frame) throws(DatasetError) -> Camera
}
public enum TrainingDatasetLoader {
    public typealias ImageDecoder = (Data) throws -> RGBImage
    public static func loadCOLMAP(cameras: Data, images: Data, points3D: Data?, imageData: (String) -> Data?,
                                  maxImageSide: Int, decode: ImageDecoder) throws(DatasetError) -> TrainingDataset
    public static func loadARKit(capture: Data, imageData: (String) -> Data?, maxImageSide: Int,
                                 decode: ImageDecoder) throws(DatasetError) -> TrainingDataset
}
```
Rules:
```text
COLMAP camera: R = Matrix3.rotation(quaternion: q normalized), t = tvec (world→camera, OpenCV) — no conversion.
ARKit camera: columns c0 (right), c1 (up), c2 (backward), c3 (position) of cameraToWorld.
              OpenCV axes x = c0, y = −c1, z = −c2 → R = Matrix3(rows: [x, y, z]), t = −R·position.
              Reject (.invalidPose) if |cᵢ·cⱼ − δᵢⱼ| > 1e-3 or the last row ≠ (0,0,0,1) or values are non-finite.
Per view: decode → actual (w, h). If (w, h) ≠ camera (W, H): accept iff |w/h − W/H| ≤ 0.01·W/H, then rescale intrinsics to (w, h);
          else .imageSizeMismatch. Undistort if model.hasDistortion. Downsample by downsampleFactor(w, h, maxImageSide), camera.downscaled(by:).
All image names are checked before decoding anything; any missing → .missingImages(all missing names, sorted).
Decoder errors → .imageDecodingFailed(name, String(describing: error)). Zero views → .noViews.
points3D: positions as Float, colors /255. Views sorted by name.
```

- [ ] **Step 1: Write the failing tests** (synthetic COLMAP binaries via `COLMAPFixture`, images as PPM)
```swift
@Test func colmapPoseProjectsKnownPoint()       // camera rotated 90° about y and translated; a world point lands at the expected pixel (≤ 1e-3)
@Test func arkitIdentityLooksDownNegativeZ()    // cameraToWorld = identity: point (0,0,−2) projects to (cx, cy); point (0,1,−2) has smaller v (up is −y)
@Test func arkitRejectsNonOrthonormalPose()
@Test func scalesIntrinsicsWhenImageFileIsSmaller() // camera 64×48, image 32×24 → fx, cx halved
@Test func rejectsAspectMismatch()              // camera 64×48, image 48×48 → .imageSizeMismatch
@Test func missingImagesAreReportedByName()     // two of three missing → .missingImages(["b.ppm", "c.ppm"])
@Test func distortedCameraIsUndistorted()       // SIMPLE_RADIAL k ≠ 0 → output camera is pinhole with same fx/cx
@Test func largeImagesAreDownsampled()          // 64×32 with maxImageSide 16 → 16×8, fx / 4
@Test func ppmRoundTrip()
```
- [ ] **Step 2–4:** run (fail) → implement → `swift test --filter DatasetLoaderTests` and full `swift test` pass.
- [ ] **Step 5: Return diff** (commit: `Add COLMAP and ARKit dataset loading`).

### Task 9: PLY writer and `.gsproj` contents

**Platform:** Linux-testable · **Depends on:** 1, 8

**Files:**
- Create: `SplatKit/Sources/SplatIO/PLYWriter.swift`, `SplatKit/Sources/SplatIO/ProjectBundle.swift`
- Test: `SplatKit/Tests/SplatIOTests/ProjectBundleTests.swift`

**Interfaces (Produces):**
```swift
public enum PLYWriter { public static func write(_ cloud: GaussianCloud) -> Data }
// binary_little_endian 1.0; float properties in order: x y z nx ny nz f_dc_0..2 f_rest_0..(3(K−1)−1) opacity scale_0..2 rot_0..3
// f_rest channel-major: f_rest[c·(K−1) + (k−1)] (inverse of PLYReader); normals written as 0.
public enum PoseSource: String, Codable, Sendable { case colmap, arkit }
public struct CheckpointRecord: Codable, Sendable, Equatable { public var iteration: Int; public var gaussianCount: Int; public var psnr: Double?; public var date: Date }
public struct ProjectManifest: Codable, Sendable, Equatable {
    public static let currentFormatVersion = 1
    public var formatVersion: Int; public var source: PoseSource?; public var imageFiles: [String]
    public var training: TrainingConfiguration; public var checkpoints: [CheckpointRecord]
    public init()   // empty project
}
public enum ProjectError: Error, Equatable { case missingManifest, unsupportedFormatVersion(Int), corruptManifest(String), corruptCheckpoint(String) }
public enum ProjectLayout {
    public static let manifest = "manifest.json", imagesDirectory = "images", colmapDirectory = "poses/colmap"
    public static let arkitCapture = "poses/arkit.json", latestCheckpoint = "checkpoints/latest.ply"
}
public struct ProjectContents: Sendable, Equatable { public var manifest: ProjectManifest; public var files: [String: Data] /* relative paths, never manifest.json */ }
public enum ProjectBundle {
    public static func encodeManifest(_ manifest: ProjectManifest) throws -> Data   // pretty, sorted keys, ISO-8601 dates
    public static func decode(files: [String: Data]) throws(ProjectError) -> ProjectContents   // reads manifest.json from `files`
    public static func encode(_ contents: ProjectContents) throws -> [String: Data]             // adds manifest.json
    public static func importCOLMAP(cameras: Data, images: Data, points3D: Data?, imageFiles: [String: Data], into contents: ProjectContents) -> ProjectContents
    public static func importARKit(capture: Data, imageFiles: [String: Data], into contents: ProjectContents) -> ProjectContents
    public static func loadDataset(_ contents: ProjectContents, decode: TrainingDatasetLoader.ImageDecoder) throws(DatasetError) -> TrainingDataset
    public static func latestCheckpoint(_ contents: ProjectContents) throws(ProjectError) -> GaussianCloud?
    public static func withCheckpoint(_ contents: ProjectContents, cloud: GaussianCloud, record: CheckpointRecord) -> ProjectContents
}
```
Import replaces previous images/poses/checkpoints, sets `source`, `imageFiles` (sorted), stores images under `images/<name>` (nested names kept) and pose files under the layout paths. `loadDataset` uses `contents.manifest.training.maxImageSide`.

- [ ] **Step 1: Write the failing tests**
```swift
@Test func plyRoundTripsThroughReader()        // random degree 0, 1, 3 clouds: PLYReader.read(PLYWriter.write(c)) == c exactly
@Test func manifestRoundTrip()
@Test func newerFormatVersionIsRejected()      // formatVersion 2 → .unsupportedFormatVersion(2)
@Test func missingManifestIsRejected()
@Test func importCOLMAPThenLoadDataset()       // synthetic 2-view COLMAP + PPM images → 2 views, points present
@Test func checkpointRoundTrip()               // withCheckpoint then latestCheckpoint returns the same cloud; record appended
@Test func reimportClearsOldCheckpoint()
```
- [ ] **Step 2–4:** run (fail) → implement → `swift test --filter ProjectBundleTests` and full `swift test` pass.
- [ ] **Step 5: Return diff** (commit: `Add PLY writer and project bundle contents`).

### Task 10: CPU training loop and smoke test

**Platform:** Linux-testable · **Depends on:** 1, 2, 3, 4, 5

**Files:**
- Create: `SplatKit/Sources/SplatCore/Training/CPUTrainer.swift`
- Test: `SplatKit/Tests/SplatCoreTests/CPUTrainerTests.swift`

**Interfaces (Produces):**
```swift
public enum CPUTrainerError: Error, Equatable { case nonFiniteLoss(iteration: Int), noViews }
public struct CPUStepResult: Sendable { public var iteration: Int; public var loss: Double; public var densified: DensifyPlan? }
public struct CPUTrainer: Sendable {
    public private(set) var cloud: GaussianCloud
    public private(set) var iteration: Int
    public let sceneExtent: SceneExtent
    public init(dataset: TrainingDataset, initialCloud: GaussianCloud, configuration: TrainingConfiguration) throws(CPUTrainerError)
    public mutating func step() throws(CPUTrainerError) -> CPUStepResult
}
```
One step (this is the normative order the GPU trainer follows):
```text
view = sampler.next()     // per-epoch Fisher–Yates shuffle with SeededGenerator(configuration.seed), regenerated each epoch
settings = RasterSettings(background: configuration.background, shDegreeLimit: activeSHDegree(iteration))
rendered = CPURasterizer.render; loss = ImageLoss.evaluate(rendered, view.image, λ); non-finite → throw .nonFiniteLoss(iteration)
sb = CPURasterizer.backwardWithScreenSpace(…, imageGradient: loss.gradient)
if iteration < densification.stopIteration: statistics.accumulate(sb.mean2D, sb.visible, W, H)
cloud = CPUAdam.step(cloud, sb.gradients, &adam, learningRates(iteration, radius), hyper)
iteration += 1
if densification.isEnabled && iteration ≥ start && iteration < stop && iteration % interval == 0:
    plan → apply (seed: UInt32(truncatingIfNeeded: configuration.seed)) → statistics = DensificationStatistics(count: new count)
```

- [ ] **Step 1: Write the failing tests**
```swift
@Test func lossDecreasesOnToyScene()       // GT: 3 Gaussians (degree 0) in front of a 24×24 camera; init = GT with positions
                                           // shifted by (0.05, −0.05, 0) and colours set to 0.5; densification disabled;
                                           // 150 iterations; mean of last 10 losses ≤ 0.6 × first loss
@Test func trainingIsDeterministic()       // two trainers, same seed → identical clouds after 20 iterations
@Test func densificationRunsAtBoundaries() // start 5, stop 30, interval 5, threshold 0 → count grows at iterations 5, 10, …; ≤ maxGaussians 12
@Test func nonFiniteTargetIsReported()     // one NaN pixel in the target → throws .nonFiniteLoss(iteration: 0)
```
- [ ] **Step 2–4:** run (fail) → implement → `swift test --filter CPUTrainerTests` (< 30 s in debug) and full `swift test` pass.
- [ ] **Step 5: Return diff** (commit: `Add CPU reference training loop`).

### Task 11: `GaussianBuffers`, renderer generalization, shader registration

**Platform:** macOS-CI-only · **Depends on:** —

**Files:**
- Create: `SplatKit/Sources/SplatMetal/GaussianBuffers.swift`
- Create: `SplatKit/Sources/SplatMetal/Shaders/Adam.metal`, `Loss.metal`, `Densify.metal` (each only a header comment naming its owning task)
- Modify: `SplatKit/Sources/SplatMetal/SplatRenderer.swift`, `GradientBuffers.swift`, `MetalContext.swift` (`shaderFiles` += `"Adam", "Loss", "Densify"`)
- Test: `SplatKit/Tests/SplatMetalTests/GaussianBuffersTests.swift`

**Interfaces (Produces):**
```swift
public final class GaussianBuffers {
    public let shDegree: Int
    public private(set) var capacity: Int
    public var count: Int                         // precondition 0 ≤ count ≤ capacity
    public private(set) var positions, logScales, rotations, opacityLogits, sh: MTLBuffer
    public init(context: MetalContext, capacity: Int, shDegree: Int) throws(MetalError)   // zero-filled
    public convenience init(context: MetalContext, cloud: GaussianCloud, capacity: Int? = nil) throws(MetalError)
    public static func floatsPerGaussian(shDegree: Int) -> [Int]   // [3, 3, 4, 1, 3K] in group order
    public var groups: [MTLBuffer] { get }                         // same order
    public func upload(_ cloud: GaussianCloud)                     // same degree, count ≤ capacity; sets count
    public func upload(arrays: GaussianMoments)                    // for m/v; lengths define count
    public func snapshot() -> GaussianCloud                        // first `count` Gaussians
    public func snapshotArrays() -> GaussianMoments
    public func grow(toCapacity: Int) throws(MetalError)           // CPU copy of the first `count`, rest zero; GPU must be idle
}
public struct ForwardPass { public let commandBuffer: MTLCommandBuffer; public let visibleGaussians: Int; public let tileIntersections: Int }
extension SplatRenderer {
    public convenience init(context: MetalContext, cloud: GaussianCloud) throws(MetalError)   // unchanged call sites
    public init(context: MetalContext, gaussians: GaussianBuffers) throws(MetalError)        // designated
    public var gaussians: GaussianBuffers { get }
    public func setCloud(_ cloud: GaussianCloud) throws(MetalError)    // upload, or replace the store if degree/capacity do not fit
    /// Runs projection + scan (waits, as before), encodes duplicate/sort/ranges/rasterize into a NEW command buffer and returns it
    /// uncommitted. Sets the state `encodeBackward` uses. `render` = encodeForward + commit + wait + FrameStats.
    public func encodeForward(camera: Camera, settings: RasterSettings, to texture: MTLTexture) throws(MetalError) -> ForwardPass
}
```
`GradientBuffers` keeps its API; its stored `count` now means capacity, and `encodeBackward` requires `gradients.count >= gaussians.count && gradients.shDegree == shDegree`. `LastFrame` records the Gaussian count used; `encodeBackward` throws `.commandFailed("gaussian count changed since render")` on mismatch. All per-frame uniforms and dispatches use `gaussians.count` read at encode time.

- [ ] **Step 1: Write the failing tests** (`GaussianBuffersTests`, Metal-gated)
```swift
@Test func uploadSnapshotRoundTrip()             // degree 1, 37 Gaussians, capacity 100 → snapshot == cloud
@Test func growPreservesContents()               // grow 100 → 250: snapshot unchanged, capacity 250
@Test func rendererOverLargerCapacityMatchesCloudRenderer() // capacity 3·n: renderImage identical (max |Δ| == 0) to SplatRenderer(cloud:)
@Test func countChangeIsVisibleToRenderer()      // set count = n/2 → equals rendering the first n/2 Gaussians
@Test func encodeForwardThenBackwardMatchesBackward() // encodeForward + encodeBackward in the returned buffer == backward(imageGradient:) (≤ 1e-6)
@Test func backwardRejectsCountChange()
@Test func setCloudReplacesScene()
```
All existing `SplatMetalTests` must stay green unchanged.
- [ ] **Step 2: CI** → compile failure. **Step 3: Implement.** **Step 4: CI green** (Linux + macOS jobs).
- [ ] **Step 5: Return diff** (commit: `Generalize SplatRenderer to capacity-based Gaussian buffers`).

### Task 12: Metal Adam

**Platform:** macOS-CI-only · **Depends on:** 2, 11

**Files:**
- Modify: `SplatKit/Sources/SplatMetal/Shaders/Adam.metal`
- Create: `SplatKit/Sources/SplatMetal/GPUAdam.swift`
- Test: `SplatKit/Tests/SplatMetalTests/GPUAdamTests.swift`

**Interfaces:**
- Consumes: `GaussianBuffers`, `GradientBuffers`, `ResolvedLearningRates`, `AdamHyperparameters`, `CPUAdam`.
- Produces:
```swift
public final class AdamBuffers { public let m: GaussianBuffers; public let v: GaussianBuffers
    public init(context: MetalContext, capacity: Int, shDegree: Int) throws(MetalError); public func grow(toCapacity: Int) throws(MetalError) }
public final class GPUAdam {
    public init(context: MetalContext) throws(MetalError)
    /// One step for all five groups over `parameters.count` Gaussians; `step` ≥ 1 is the already-incremented counter.
    public func encode(parameters: GaussianBuffers, gradients: GradientBuffers, state: AdamBuffers,
                       rates: ResolvedLearningRates, step: Int, hyper: AdamHyperparameters,
                       commandBuffer: MTLCommandBuffer) throws(MetalError)
}
```
Kernel `adamStep` (one thread per float; uniform struct `AdamUniforms { SIMD4<Float> rates /* headStep, tailStep, invSqrtBC2, eps */; SIMD4<Float> betas /* β1, β2 */; SIMD4<UInt32> layout /* total floats, stride, headCount */ }`), arithmetic exactly as Task 2. Dispatched five times (one per group).

- [ ] **Step 1: Write the failing tests**
```swift
@Test func matchesCPUAdamOverThreeSteps()   // 1_000 Gaussians degree 1, random gradients (seed 5) per step; params, m, v ≤ 1e-6 + 1e-5·|cpu|
@Test func onlyFirstCountGaussiansChange()  // capacity 2·count: floats beyond count stay bit-identical
@Test func shHeadAndTailRates()             // sh DC moves with shDC, rest with shRest
```
- [ ] **Step 2–4:** CI fail → implement → CI green.
- [ ] **Step 5: Return diff** (commit: `Add Metal Adam optimizer`).

### Task 13: Metal loss

**Platform:** macOS-CI-only · **Depends on:** 3, 11

**Files:**
- Modify: `SplatKit/Sources/SplatMetal/Shaders/Loss.metal`
- Create: `SplatKit/Sources/SplatMetal/GPUImageLoss.swift`
- Test: `SplatKit/Tests/SplatMetalTests/GPUImageLossTests.swift`

**Interfaces:**
- Consumes: `ImageLoss` (window, constants, formulas), rendered `MTLTexture` (`rgba32Float`, as produced by `encodeForward`).
- Produces:
```swift
public struct ImageLossValue: Sendable, Equatable { public var loss: Double; public var l1: Double; public var ssim: Double }
public final class GPUImageLoss {
    public init(context: MetalContext) throws(MetalError)
    public func makeTarget(_ image: RGBImage) throws(MetalError) -> MTLBuffer      // float4 per pixel, A = 0
    /// Writes dL/drendered into `imageGradient` (W·H·4 floats, A = 0) and per-threadgroup partial sums.
    public func encode(rendered: MTLTexture, target: MTLBuffer, width: Int, height: Int, lambda: Float,
                       imageGradient: MTLBuffer, commandBuffer: MTLCommandBuffer) throws(MetalError)
    /// Sums the partials of the most recent completed `encode` on the CPU in Double.
    public func readValue() -> ImageLossValue
}
```
Kernels (scratch buffers grow-only; planar float32, plane = W·H floats): (1) `ssimMomentsH`: per pixel, 15 planes = horizontal blur of {x, y, x², y², xy} × 3 channels; (2) `ssimMapsV`: vertical blur of the 15 planes → μ/e terms → S, writes 9 planes {M0, M1, M2} × 3 (already × g = −λ/N) and, per 16×16 threadgroup, `partials[tg] = (ΣS, Σ|x−y|)` via `simd_sum` + threadgroup reduction (no atomics); (3) `blurH` over the 9 planes (reusing the first 9 planes of the H buffer); (4) `lossGradient`: vertical blur fused with `dL/dx = (1−λ)/N·sign(x−y) + B0 + 2x·B1 + y·B2`. `readValue`: `l1 = ΣΣ|x−y| / N`, `ssim = ΣΣS / N`, `loss = (1−λ)·l1 + λ·(1−ssim)`.

- [ ] **Step 1: Write the failing tests**
```swift
@Test func matchesCPULossOnOddSize()        // 37×29 random images (seed 9): loss, l1, ssim rel ≤ 1e-4; gradient |Δ| ≤ 1e-6 + 1e-3·|cpu|
@Test func identicalImagesGiveZeroLoss()
@Test func gradientFeedsEncodeBackward()    // render a 32×32 scene with encodeForward, loss + encodeBackward in the same command buffer:
                                            // parameter gradients == CPURasterizer.backward with ImageLoss.evaluate's gradient (Phase 2 tolerance)
```
- [ ] **Step 2–4:** CI fail → implement → CI green.
- [ ] **Step 5: Return diff** (commit: `Add Metal L1 + D-SSIM loss`).

### Task 14: Metal densification and statistics hook

**Platform:** macOS-CI-only · **Escalate-to: opus-review** · **Depends on:** 4, 11

**Files:**
- Modify: `SplatKit/Sources/SplatMetal/Shaders/Densify.metal`, `SplatKit/Sources/SplatMetal/SplatRenderer.swift` (only `encodeBackward`)
- Create: `SplatKit/Sources/SplatMetal/GPUDensifier.swift`
- Test: `SplatKit/Tests/SplatMetalTests/GPUDensifierTests.swift`

**Interfaces:**
- Consumes: `Densification` rules (Task 4), `GPUPrefixSum`, `GaussianBuffers`, `AdamBuffers` (Task 12 is NOT a dependency: this task defines the moment stores as two `GaussianBuffers` parameters).
- Produces:
```swift
public final class DensificationStatisticsBuffers {
    public let gradientNormSums: MTLBuffer   // Float per Gaussian
    public let visibleCounts: MTLBuffer      // UInt32 per Gaussian
    public private(set) var capacity: Int
    public init(context: MetalContext, capacity: Int) throws(MetalError)
    public func encodeReset(commandBuffer: MTLCommandBuffer) throws(MetalError)
    public func grow(toCapacity: Int) throws(MetalError)            // contents discarded (always reset after densify)
    public func read(count: Int) -> DensificationStatistics
}
extension SplatRenderer {
    // Existing signature gains a defaulted parameter; Phase 2 call sites unchanged.
    public func encodeBackward(imageGradient: MTLBuffer, into gradients: GradientBuffers,
                               statistics: DensificationStatisticsBuffers? = nil, commandBuffer: MTLCommandBuffer) throws(MetalError)
}
public struct DensifySummary: Sendable, Equatable { public var before: Int; public var after: Int; public var pruned, cloned, split, demoted: Int; public var keptAll: Bool; public var grewCapacity: Bool }
public final class GPUDensifier {
    public init(context: MetalContext) throws(MetalError)
    /// Synchronous (waits on its own command buffers). Precondition: the GPU is idle w.r.t. all passed buffers.
    /// Grows parameters, m, v and statistics when needed; the caller re-creates GradientBuffers if `grewCapacity`.
    public func densify(parameters: GaussianBuffers, m: GaussianBuffers, v: GaussianBuffers,
                        statistics: DensificationStatisticsBuffers, policy: DensificationPolicy,
                        sceneRadius: Float, iteration: Int, seed: UInt32, maxGaussians: Int) throws(MetalError) -> DensifySummary
}
```
Statistics hook: when `statistics != nil` and the frame has intersections, `encodeBackward` dispatches `accumulateDensifyStats` after `rasterizeBackward` (reads accumulator slots 7–8 and `tilesTouched`; `if tilesTouched[i] > 0 { sums[i] += length(float2(a7·W/2, a8·H/2)); counts[i] += 1 }`). `rasterizeBackward` and `projectBackward` are unchanged.

Densify pipeline (buffers: `actions`, `growFlag`, `growRank`, `keptFlag`, `keptScan`, `outCount`, `offsets` — UInt32 × capacity, grow-only; one float scratch of `max group floats × new capacity`):
```text
CB1: densifyDecide (Task 4 `action`, writes actions, growFlag = clone|split, keptFlag = action ≠ prune)
     scan growFlag → growRank; scan keptFlag → keptScan; commit + wait
CPU: G = total grows, K = total kept; keptAll = (K == 0 && count > 0); allowed = max(0, maxGaussians − K) (keptAll: K = count, G = 0)
     after = K + min(G, allowed); if after > capacity: grow parameters/m/v/statistics to min(maxGaussians, max(after, ⌈1.5·capacity⌉))
CB2: densifyFinalize (keptAll → prune becomes keep; grow with growRank ≥ allowed → keep; outCount)
     scan outCount → offsets
     for group in [positions, rotations, opacityLogits, sh, logScales] (this order: positions read the old logScales/rotations):
         densifyScatter(group kind) into scratch; blit scratch → group buffer
     for each of the 10 moment groups (m, v): densifyScatterMoments (keep: copy; clone child 0: copy, child 1: 0; split: 0); blit back
     statistics reset; commit + wait; set count = after on parameters, m, v
```
- [ ] **Step 1: Write the failing tests**
```swift
@Test func statisticsMatchCPU()            // 64×48 scene, 3 backward passes from different views: sums ≤ 1e-5 + 1e-4·|cpu|, counts exact
@Test func backwardWithoutStatisticsIsUnchanged() // Phase 2 BackwardTests unchanged + nil statistics path identical
@Test func decisionsAndLayoutMatchCPU()    // 300 Gaussians, crafted stats covering all four actions: params, m, v vs Densification.apply ≤ 1e-5 + 1e-5·|cpu|
@Test func densifyRespectsCap()            // maxGaussians = count + 5 with 40 candidates: after == count − pruned + 5, demoted == 35
@Test func densifyGrowsCapacity()          // capacity == count, 50 clones → grewCapacity, contents correct
@Test func neverPrunesEverything()
```
- [ ] **Step 2–4:** CI fail → implement → CI green.
- [ ] **Step 5: Return diff** (commit: `Add Metal densification and gradient statistics`). Opus reviews the compaction ordering, the cap logic and the CPU/GPU decision parity before commit.

### Task 15: `SplatTraining` — `GPUTrainer` and `TrainingSession`

**Platform:** macOS-CI-only · **Escalate-to: opus-review** · **Depends on:** 1, 5, 12, 13, 14

**Files:**
- Modify: `SplatKit/Package.swift` (inside `#if os(macOS)`: product + target `SplatTraining` depending on `SplatCore`, `SplatMetal`; test target `SplatTrainingTests` depending on `SplatTraining`, `SplatCore`, `SplatMetal`)
- Create: `SplatKit/Sources/SplatTraining/TrainingError.swift`, `TrainingProgress.swift`, `GPUTrainer.swift`, `TrainingSession.swift`
- Test: `SplatKit/Tests/SplatTrainingTests/GPUTrainerTests.swift`, `SplatKit/Tests/SplatTrainingTests/TrainingSessionTests.swift` (Metal-gated; copy the gating helper from `MetalTestSupport.swift`)

**Interfaces (Produces):**
```swift
public enum TrainingError: Error, Equatable { case metal(String), noViews, nonFiniteLoss(iteration: Int), initialCloudExceedsBudget(count: Int, cap: Int), alreadyRunning }
public struct TrainingProgress: Sendable {
    public enum State: Sendable, Equatable { case running, paused, finished, cancelled, failed(String) }
    public var state: State; public var iteration: Int; public var totalIterations: Int
    public var loss: Double?; public var psnr: Double?; public var gaussianCount: Int; public var iterationsPerSecond: Double
    public var preview: GaussianCloud?       // set every previewInterval and on finished/cancelled
}
public struct StepReport: Sendable { public var iteration: Int; public var loss: ImageLossValue?; public var visibleGaussians: Int; public var densify: DensifySummary? }
/// Phase 4 extension point: extra loss terms encoded after the photometric loss into the same image gradient / parameter gradients.
public protocol AuxiliaryLoss: AnyObject { func encode(view: Int, iteration: Int, commandBuffer: MTLCommandBuffer) throws(TrainingError) }
public final class GPUTrainer {
    public init(context: MetalContext, dataset: TrainingDataset, initialCloud: GaussianCloud, configuration: TrainingConfiguration) throws(TrainingError)
    public let maxGaussians: Int
    public let sceneExtent: SceneExtent
    public private(set) var iteration: Int
    public var gaussianCount: Int { get }
    public var auxiliaryLosses: [any AuxiliaryLoss]   // empty in Phase 3
    @discardableResult public func step() throws(TrainingError) -> StepReport
    public func evaluatePSNR() throws(TrainingError) -> Double      // mean over training views (renderImage + ImageMetrics.psnr)
    public func snapshot() throws(TrainingError) -> GaussianCloud   // waits for the last command buffer first
}
public actor TrainingSession {
    public nonisolated let progress: AsyncStream<TrainingProgress>   // .bufferingNewest(32); finished when run() returns
    public init(dataset: TrainingDataset, initialCloud: GaussianCloud, configuration: TrainingConfiguration)
    public func run() async throws(TrainingError) -> GaussianCloud   // returns final (or, on cancellation, current) cloud
    public func pause()
    public func resume()
}
```
`GPUTrainer.step()` follows Task 10's order exactly: sampler (same Fisher–Yates + `SeededGenerator`); `encodeForward` into the view's (grow-only) `rgba32Float` texture; `GPUImageLoss.encode`; `auxiliaryLosses`; `encodeBackward(statistics:)` (statistics only while `iteration < stopIteration`); `GPUAdam.encode` with `step = iteration + 1`; commit. It waits for completion only when `(iteration+1) % logInterval == 0`, at the last iteration, before densification (`GPUDensifier.densify`, then re-create `GradientBuffers` if `grewCapacity`), and in `snapshot`/`evaluatePSNR`. A non-finite loss read → `.nonFiniteLoss(iteration:)`. Cap = `TrainingMemoryBudget.maxGaussians(workingSetBytes: device.recommendedMaxWorkingSetSize, …, reservedBytes: reservedImageBytes(viewSizes))`; initial count > cap → `.initialCloudExceedsBudget`.

`TrainingSession.run()`: creates `MetalContext.makeDefault()` and `GPUTrainer` inside the actor; loop: `if Task.isCancelled` → snapshot, yield `.cancelled` (with preview), finish stream, return; while paused → yield `.paused` once, `try? await Task.sleep(for: .milliseconds(50))`, re-check cancellation; `step()`; yield `.running` progress every `logInterval` (with loss) and with preview every `previewInterval`, PSNR every `evalInterval`; `await Task.yield()` after each step. Errors → yield `.failed(message)`, finish, rethrow. End → final PSNR, yield `.finished` with preview, finish, return cloud. A second concurrent `run()` throws `.alreadyRunning`.

- [ ] **Step 1: Write the failing tests**
```swift
// GPUTrainerTests
@Test func overfitsSmallScene()                    // GT: 500 random Gaussians (MetalTestSupport random-scene helper), 2 views 64×64;
                                                   // init: different seed, 500 Gaussians; densification disabled; 300 iterations;
                                                   // CPU ImageLoss L1 over both views at end ≤ 0.5 × at start; wall time < 60 s
@Test func overfitWithDensificationStaysWithinCap() // same scene, start 50, stop 250, interval 50, maxGaussians 800, 300 it.:
                                                   // count changed, count ≤ 800 throughout, snapshot finite
@Test func matchesCPUTrainerForFirstSteps()        // toy scene of Task 10, 5 steps, densify off: params ≤ 1e-4 + 1e-3·|cpu|
@Test func viewWithNoVisibleGaussiansIsHarmless()  // one view looks away: step succeeds, loss finite, parameters change only through Adam momentum
@Test func trainsViewsWithDifferentResolutions()   // views 64×48 and 32×32: 20 steps without error, both views rendered
@Test func nonFiniteLossStopsWithError()           // NaN pixel in a target → .nonFiniteLoss at the first logged iteration
// TrainingSessionTests
@Test func streamReportsProgressAndFinishes()      // 40 iterations, logInterval 10: ≥ 4 .running, last state .finished with preview
@Test func cancellationLeavesConsistentModel()     // densify every 10 it.; cancel the run task after the 3rd progress event:
                                                   // returned cloud valid (GaussianCloud lengths), ≤ cap, last state .cancelled
@Test func pauseStopsIterating()                   // pause → iteration constant over 200 ms → resume → finishes
```
- [ ] **Step 2–4:** CI fail → implement → CI green; check the CI log for the two overfit tests' durations (< 60 s each).
- [ ] **Step 5: Return diff** (commit: `Add SplatTraining module with GPU trainer and training session`). Opus reviews isolation (no Metal object leaves the actor), the wait points and cancellation.

### Task 16: App integration

**Platform:** macOS-CI-only (app build; manual check on a Mac) · **Depends on:** 5, 8, 9, 15

Use the skills `swiftui-macos-skill:swiftui-macos` and `liquid-glass:liquid-glass`; run the glass linter (`scripts/check_glass.py --target 26.0`) on every changed SwiftUI file.

**Files:**
- Modify: `project.yml` (app depends on product `SplatTraining`), `App/Info.plist` (exported UTType `io.github.junostr05.SparseSplat.gsproj`, conforms to `com.apple.package`, extension `gsproj`; `CFBundleDocumentTypes` entry, role Editor, `LSTypeIsPackage` true), `App/SparseSplatApp.swift` (add `DocumentGroup(newDocument: ProjectDocument())` before the existing viewer group), `App/ViewerModel.swift` (`func replaceCloud(_:)`, observed `cloudRevision: Int`), `App/Viewport/ViewportView.swift` (read `cloudRevision` in `updateNSView`)
- Create: `App/Project/ProjectDocument.swift`, `App/Project/ProjectModel.swift`, `App/Project/ProjectWindow.swift`, `App/Project/ProjectInspector.swift`, `App/Project/ImageIODecoder.swift`

**Interfaces:**
- Consumes: `ProjectBundle`, `ProjectContents`, `TrainingDatasetLoader`, `GaussianInitializer.initialize`, `TrainingSession`, `TrainingProgress`, `SplatRenderer.setCloud`.
- Produces:
```swift
nonisolated struct ProjectDocument: FileDocument {      // readable/writable: [.gaussianProject]
    var contents: ProjectContents
    init()                                              // empty manifest
    init(configuration: ReadConfiguration) throws       // flatten FileWrapper tree → [path: Data] → ProjectBundle.decode
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper   // ProjectBundle.encode → nested directory wrappers
}
nonisolated enum ImageIODecoder { static func decode(_ data: Data) throws -> RGBImage }   // CGImageSource → RGBA8 sRGB CGContext → /255
@Observable final class ProjectModel {
    var progress: TrainingProgress?; var isTraining: Bool; var errorMessage: String?
    let viewer: ViewerModel?                            // created from the latest checkpoint or the first preview
    func importCOLMAP(folder: URL, into document: Binding<ProjectDocument>)    // expects sparse/0/*.bin and images/
    func importARKit(folder: URL, into document: Binding<ProjectDocument>)     // expects capture.json and the images it names
    func startTraining(document: Binding<ProjectDocument>)
    func togglePause(); func stop()
}
```
Behavior: dataset loading and initialization run in a `@concurrent` function (not on the main actor); the `run()` call runs in a `Task` stored on the model (`@ObservationIgnored`); the model iterates `session.progress` on the main actor, updates `progress`, forwards `preview` clouds to `viewer.replaceCloud`, and on `.finished`/`.cancelled` writes `ProjectBundle.withCheckpoint` into the document binding (marks it dirty). Inspector: when no images — two buttons "Import COLMAP…" / "Import ARKit Capture…" (`.fileImporter(allowedContentTypes: [.folder])`, security-scoped access); otherwise image count, source, a **Train** button (the single `.buttonStyle(.glassProminent)` action) that turns into Pause/Resume + Stop while training, `ProgressView(value:iteration, total:)`, loss (4 decimals), PSNR (2 decimals, dB), Gaussian count, iterations/s; errors in a plain `Text` with `.foregroundStyle(.red)` (not glass). Sidebar: list of image names. Detail: the existing `ViewportView` (content layer, no glass) or a `ContentUnavailableView` before any model exists. Every icon-only control has an accessibility label.

- [ ] **Step 1:** Implement; `xcodegen generate` and `xcodebuild … build` on CI must succeed; glass linter clean.
- [ ] **Step 2: Manual check on a Mac (orchestrator or user, not claimed by CI):** new project → import a COLMAP scene of 3–6 images → Train → loss falls, preview updates, Stop writes `checkpoints/latest.ply`; reopen restores the checkpoint in the viewport.
- [ ] **Step 3: Return diff** (commit: `Add gsproj project document with training inspector and live preview`).

---

## `docs/ARCHITECTURE.md` updates (orchestrator applies after Phase 3 lands; this plan does not edit the file)

Add decision-log rows:
- 18 **GPU parameter storage** — SoA `GaussianBuffers` with capacity/count; Adam m/v same layout; densify via scan + per-group scatter/blit; growth ×1.5 only at densify boundaries (D1, D2).
- 19 **Optimizer** — Adam (PyTorch form, ε 1e-15), gsplat default LRs, position LR × scene radius with exponential decay; no optimizer state in checkpoints (D5, D11).
- 20 **Photometric loss** — L1 + D-SSIM (λ 0.2, 11×11 σ 1.5 separable, zero padding), sRGB values, no linearization (D6, D7).
- 21 **Densification** — gsplat default strategy semantics (grad 2e-4 in NDC-scaled pixels, clone/split at 0.01·R, prune opacity 0.005, scale 0.1·R after it. 3000), prune-before-grow, deterministic cap demotion, never empty (D8).
- 22 **Initialization** — SfM points with 3-NN k-d tree scales, opacity 0.1; random cube fallback; no world normalization, `SceneExtent` instead (D9).
- 23 **Import** — supported COLMAP models and CPU undistortion; ARKit JSON schema v1; intrinsics rescale rule; fail on missing images (D10).
- 24 **Training concurrency & preview** — `TrainingSession` actor + synchronous `GPUTrainer`, `AsyncStream<TrainingProgress>`, previews as `GaussianCloud` snapshots (D4, D12).
- 25 **Memory budget formula** — D13 (756 B/Gaussian at SH 1, 0.6 × `recommendedMaxWorkingSetSize`, configured max 1.5 M).

Also: *Module layout* — CPU training references live in `SplatCore/Training/`, `SplatTraining` is macOS-only, ImageIO decoding lives in the app. *Project bundle* — replace with the D11 layout (`poses/colmap/` holds the `sparse/0` files, `checkpoints/latest.ply`, manifest fields). *Training defaults* — add iterations 10 000, SH degree interval 1000, densify 500–6000 every 100, max image side 1600, background black, max Gaussians 1.5 M. Status line: Phase 3 done.

## Self-review notes

- Spec coverage: COLMAP/ARKit import (7, 8), Adam (2, 12), densification/pruning (4, 14), training from SfM points (5, 10, 15), L1 + D-SSIM (3, 13), headless `AsyncStream` loop (15), `.gsproj` + Train + live viewer (9, 16), Phase 4 extension points (Global Constraints; types in 1, 15).
- Open risks: (a) one blocked cooperative thread per running session and two CPU syncs per iteration — acceptable now, revisit with completion-handler continuations if iterations/s is poor; (b) the paravirtual CI GPU may make the overfit tests slow — reduce to 200 iterations / 48×48 before relaxing the 0.5 factor; (c) clamp-to-edge undistortion leaves smeared borders for strong distortion; (d) whole-package `FileDocument` writes copy all images on each save — acceptable for 3–6 images, revisit if projects grow; (e) float atomics make GPU training non-bitwise-deterministic, so GPU tests use tolerances and trends only.
