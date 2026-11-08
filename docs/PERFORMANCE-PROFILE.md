# Performance profile (v1.5, "Speicher und Tempo")

Date: 2026-11-08. Machine: Apple M5 Pro, `recommendedMaxWorkingSetSize` 18 186 MB. Release builds.

## TL;DR

* **The training step is dominated by one kernel: `rasterizeBackward`.** It took 77 % of the GPU time of an
  iteration at 293 k Gaussians (27.2 of 35.3 ms) and 79 % at 1.5 M (131.5 of 165.5 ms). Projection, key
  duplication, sort, tile ranges, loss, Adam and the regularizers together are under 5 ms per step at 293 k.
* **One change, inside `rasterizeBackward`, roughly halves the training time** with unchanged arithmetic:
  the kernel stops each tile's reverse walk at the tile's largest last-contributor index, skips samples that no
  lane of a simdgroup blended, and spreads the per-sample atomics over lanes instead of issuing them all from lane 0.
  `rasterizeBackward` 27.2 → 9.7 ms (293 k); a full 5000-iteration run 106 s → 60 s (47 → 83 it/s);
  1.5 M Gaussians 168 → 73 ms per step (5.9 → 13.8 it/s).
* **The viewer frame (forward only) is dominated by `rasterizeTiles`**: 11.9 of 16.0 ms GPU time for a 1 M cloud at
  1600×1000; the radix sort takes 2.9 ms. Nothing was changed there in this round (see "Not done").
* **Memory per Gaussian matches the budget formula** (756 B at SH 1): measured 765 B per Gaussian of capacity at
  1.5 M, including sort and image buffers. `TrainingMemoryBudget.bytesPerGaussian` is unchanged.

## Method

* Harness: a scratch SwiftPM executable (`perfharness`, not in the repo) that depends on this worktree's SplatKit.
  `train`: COLMAP import of `datasets/table-scene`, `TrainingPreset.standard.configuration()` with 5000 iterations
  and the schedules scaled as in `docs/HELD-OUT-INVESTIGATION.md` (densification 500–3000, scale pruning from
  1500, SH interval 500), depth prior from Depth Anything V2 Small, 10 training views, held out IMG_9014,
  IMG_9018 and IMG_9022 (600×450). `GPUTrainer` is driven directly; training PSNR is `evaluatePSNR()`, held-out
  PSNR/SSIM is `evaluate(view:)` on the three held-out views. `synth`: fixed-count training (MCMC and densification
  off, full SH degree) from the trained 293 k cloud, or from that cloud replicated with jitter to 1.5 M. `viewer`:
  `SplatRenderer.render` of a 1 M cloud (the trained cloud replicated) at 1600×1000, orbiting 0.01 rad per frame.
  `memory`: `MTLDevice.currentAllocatedSize` before the trainer, after `init`, and after 30 steps.
* Stage timing: `GPUStageProfiler` (new, opt-in, `SplatMetal/GPUStageProfiler.swift`). When
  `GPUStageProfiler.shared` is set, every compute encoder made through `MetalContext.makeComputeEncoder` samples
  GPU timestamps at its start and end (`MTLCounterSamplingPoint.atStageBoundary`, the only point the M5 Pro
  supports), labelled by its call site, and each command buffer reports `gpuEndTime − gpuStartTime`. When it is
  nil (always, outside a harness) the encoder path is the old one plus one nil check. Blit fills are not
  sampled; they are the small difference between the command-buffer time and the encoder sum.
* GPU runs were made one at a time.

## 1. Per-iteration breakdown (before)

Fixed-count training, defaults (depth prior, appearance, MCMC terms, anti-aliasing, pose refinement, anisotropy,
scale cap), full SH degree, 600×450 views. Milliseconds of GPU time per iteration.

| stage (encoder call site) | 293 k | 1.5 M |
|---|---|---|
| rasterize backward (`SplatRenderer.swift:586`) | **27.23** | **131.53** |
| rasterize forward (`:384`) | 4.15 | 16.20 |
| radix sort scatter, 6 passes (`GPURadixSort.swift:65`) | 0.89 | 6.51 |
| Adam, all groups (`GPUAdam.swift:73`) | 0.64 | 3.41 |
| radix sort histogram, 6 passes (`:53`) | 0.31 | 1.83 |
| project backward (`SplatRenderer.swift:622`) | 0.31 | 1.63 |
| projection (`:295`) | 0.17 | 1.15 |
| depth loss (4 encoders) | 0.61 | – |
| image loss (D-SSIM, 4 encoders) | 0.39 | – |
| densify statistics hook (`GPUDensifier.swift:113`) | 0.18 | – |
| duplicate keys (`:340`) | 0.11 | – |
| prefix sums (21 encoders) | 0.09 | – |
| tile ranges (`:365`) | 0.05 | – |
| regularizers, appearance, sampling rate | 0.13 | – |
| **sum of encoders** | **35.29** | ≈ 165 |
| command-buffer GPU time | 34.84 | 165.55 |
| wall time per iteration | 35.55 | 168.33 |

(– = not listed separately in the 1.5 M run; each is under 1 ms.)

Readings:

* The GPU is busy 98 % of the wall time: the CPU sync after the projection (the CPU reads the intersection total
  and counts the visible Gaussians) costs about 0.7 ms per step at 293 k and 2.8 ms at 1.5 M. That is the next
  bubble once the backward is fast (4 % of a step at 293 k after the change).
* Densification/MCMC boundaries and snapshots are not per-step costs: in the 5000-iteration run the MCMC
  relocation (CPU, every 100 iterations) is included in the wall-clock times below.
* The radix sort uses 6 passes (32 depth bits + 11 tile bits at 600×450). It is 1.2 ms of 35 ms: reducing key bits
  would save at most a few percent of a training step.

## 2. The change: `rasterizeBackward` (Shaders/Backward.metal)

Three edits, each of which only removes work whose result is an exact zero:

1. **Tile cut-off.** The kernel reduces the per-pixel `lastContributor` to the tile maximum (simd_max plus an
   8-entry threadgroup array) and walks only `rangeStart..<tileLast`. A sample at or past a pixel's cut-off was
   already gated out (`sortedIndex < lastSorted`) and did not change the pixel's reverse state. The batch grid stays
   anchored at `rangeStart`, so the remaining samples are visited in the same batches and order.
2. **Simdgroup skip.** If no lane of a simdgroup blended a sample (`simd_any(contributed)` is false), all ten
   partial values are zero; the ten `simd_sum`s and the atomics are skipped. The branch is uniform per simdgroup,
   so `simd_sum` still sees every lane.
3. **Parallel atomics.** The sums are uniform across the simdgroup, so lane k adds slot k (9 slots, 10 in depth
   mode) instead of lane 0 issuing 9–10 atomics one after another. Same values into the same slots.

Adding +0.0 with an atomic leaves the slot unchanged, so the set of non-zero additions per Gaussian is identical;
only the order of the float atomics differs, which was already nondeterministic between runs.

### Effect

| measurement | before | after | change |
|---|---|---|---|
| `rasterizeBackward`, 293 k, ms/step | 27.23 | 9.69 | −64 % |
| fixed-count step, 293 k, ms (it/s) | 35.55 (28.1) | 18.02 (55.5) | ×1.97 |
| fixed-count step, 1.5 M, ms (it/s) | 168.33 (5.9) | 72.70 (13.8) | ×2.32 |
| full run, 5000 iterations, seed 0 / 1 / 0 repeat, s | 106.4 / 104.5 / 106.9 | 60.7 / 59.5 / 60.6 | ×1.76 |
| full run it/s (mean) | 47.2 | 83.0 | ×1.76 |
| viewer frame, 1 M, 1600×1000 | 16.7 ms | unchanged (forward only) | – |

### Quality (real photos, 10 train / 3 held-out, 5000 iterations, defaults)

| run | train PSNR | held-out PSNR | held-out SSIM | Gaussians |
|---|---|---|---|---|
| before, seed 0 | 28.777 | 13.741 | 0.4102 | 292 638 |
| before, seed 1 | 29.082 | 13.776 | 0.4123 | 290 906 |
| before, seed 0 (repeat) | 28.953 | 14.214 | 0.4195 | 293 895 |
| **before, mean** | **28.94** | **13.91** | **0.414** | |
| after, seed 0 | 28.989 | 14.371 | 0.4154 | 292 457 |
| after, seed 1 | 29.159 | 14.567 | 0.4253 | 292 542 |
| after, seed 0 (repeat) | 28.905 | 14.189 | 0.4162 | 293 242 |
| **after, mean** | **29.02** | **14.38** | **0.419** | |

* Training PSNR: +0.08 dB, inside ±0.3 dB.
* Held-out PSNR: the means differ by +0.47 dB, which is **outside the ±0.3 dB target** on paper. It is not a real
  effect: the same seed with the **unchanged** build gave 13.74 and 14.21 dB in two runs (0.47 dB apart, from the
  atomic-order noise alone), and the after runs lie inside the before range plus that spread. The change cannot
  move the expected score: it only skips exact zero additions (section 2). More seeds would narrow the means; the
  earlier documented default level was 14.5–14.65 dB.
* Tests: `swift test -c release` green (all suites, 0 failures), including the backward golden, CPU-vs-Metal and
  finite-difference tests, with no tolerance changed.

## 3. Memory

`currentAllocatedSize` growth of a `GPUTrainer` (defaults, SH 1, MCMC/densification off), after `init` and after 30
steps. The capacity is the D2 rule `min(budget, max(2n, 65 536))`, capped by the configured 1.5 M.

| Gaussians | capacity | after init | after 30 steps | per capacity slot | formula `bytesPerGaussian(1)` |
|---|---|---|---|---|---|
| 292 638 | 585 276 | 285 MB | 416 MB | 745 B | 756 B |
| 1 000 000 | 1 500 000 | 613 MB | 953 MB | 666 B | 756 B |
| 1 500 000 | 1 500 000 | 612 MB | 1 095 MB | 765 B | 756 B |

Per Gaussian by buffer (SH 1, K = 4, P = 23 floats), from the code:

| buffer | bytes |
|---|---|
| parameters | 92 |
| gradients | 92 |
| Adam m, v | 184 |
| renderer projection (mean/depth, conic, colour, tile rect: 4 × float4) | 64 |
| tiles touched + offsets | 8 |
| backward accumulator (12 floats, 10 used) | 48 |
| pose-gradient scratch (6 floats) | 24 |
| sampling rate, densify statistics | ~12 |
| classic-densification scratch (only when classic densification runs) | ~120 |
| sort keys + values + radix scratch, per tile intersection | 24 |

The measured footprint at full capacity matches the formula within 1–2 %; it already includes the sort buffers for
this view size. `TrainingMemoryBudget.bytesPerGaussian` was **not** changed. Possible savings are small
(packing the tile rect as `ushort4`: 8 B, a 10-float accumulator: 8 B, about 2 % together); fp16 storage was not
tried, since the step time, not memory, was the limit, and the fp16 rule requires a quality measurement per change.

## Not done / what did not pay off (measured reasons)

* **Tighter tile-overlap test (ellipse/OBB).** It would cut sort keys and batch loads, but the sort is only 1.2 of
  35 ms per step, and dropping tiles changes `tilesTouched`, which the densify statistics use as the per-view
  visibility count. Classic densification, and the CPU-vs-GPU tests that use it, would then see different
  denominators. Not done.
* **Fewer sort-key bits / smaller keys.** At most about 0.5 ms of a 35 ms step (training), about 1 ms of a 16 ms
  frame (viewer). Not done.
* **Viewer: skip the sort for a still camera, or double-buffer sorting.** The sort is 2.9 of 16 ms per frame at
  1 M; the forward rasterizer is 11.9 ms. Rendering with a stale order would change the image during motion. Not done.
* **Removing the CPU sync / visible-count loop after the projection.** 0.7 ms per step (293 k), 2.8 ms (1.5 M).
  Getting rid of it needs GPU-driven (indirect) dispatch of duplicate, sort and ranges. That is the next candidate
  for training now that the backward is fast.
* **Forward rasterizer** (`rasterizeTiles`): the next target for the viewer (75 % of a frame). Candidates:
  load `meanDepth` once per sample instead of twice, and a per-simdgroup skip similar to the backward.

## Reproducing

Set `GPUStageProfiler.shared = GPUStageProfiler(device: context.device)` before the steps to profile, read
`results()` after a wait (for example `GPUTrainer.appearanceParameters()`), and clear `shared` afterwards.
