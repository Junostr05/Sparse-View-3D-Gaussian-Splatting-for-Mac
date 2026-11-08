# Depth prior experiments (v1.5)

Three upgrades of the depth prior, all behind default-off switches:

1. **Depth Anything 3 as the depth estimator** (`DepthPriorSettings.estimator = .depthAnything3`).
2. **Confidence weighting** of the depth loss (`DepthPriorSettings.confidenceWeighting`).
3. **Surface-normal prior** from MoGe-2 (`TrainingConfiguration.normalPrior.isEnabled`, weight 0.05).

Nothing below changes a default. Decision 51 (default estimator Depth Anything V2 Small) still stands until the
user decides otherwise.

## Setup

- Real photos: the 13 iPhone photos (COLMAP poses, 450 × 600 after loading), 10 training views and 3 held-out
  views (IMG_9014, IMG_9018, IMG_9022), 5000 iterations, `--features app` (every default of decision 56),
  harness `harness4` built against this worktree. The metric is the mean PSNR / SSIM on the 3 held-out views.
- Depth Anything 3: `DepthAnything3BaseQ8.mlpackage` (the pose-free package), function `v12_504x378`: the 10
  training views in one call, padded to 12 by repeating the last view. **Only the training views go into the
  Depth Anything 3 pass**, so no held-out image influences the depth of a training view.
- MoGe-2: `moge_vits_normal_int8.mlpackage` (the same package as the decision 51 measurements).
- Seeds 0, 1, 2 per variant. At most two runs at a time on the shared GPU.

## Results on the real photos (held-out views)

| Variant | Seed 0 | Seed 1 | Seed 2 | Mean PSNR (dB) | Mean SSIM | Δ PSNR vs V2 | Gaussians | Run time* |
|---|---|---|---|---|---|---|---|---|
| Depth Anything V2 Small (default) | 14.52 / 0.414 | 14.10 / 0.405 | 14.40 / 0.408 | **14.34** | 0.409 | – | ~269 k | ~6.1 min |
| Depth Anything 3 | 16.14 / 0.497 | 16.15 / 0.496 | 16.12 / 0.496 | **16.14** | 0.496 | **+1.80** | ~450 k | ~7.0 min |
| Depth Anything 3 + confidence weighting | 16.21 / 0.498 | 16.18 / 0.498 | 16.17 / 0.497 | **16.19** | 0.498 | +1.85 | ~451 k | ~7.0 min |
| Depth Anything 3 + normal prior (MoGe-2, λ = 0.05) | 16.13 / 0.505 | 16.21 / 0.504 | 16.23 / 0.508 | **16.19** | 0.506 | +1.85 | ~489 k | ~7.7 min |

Each cell is held-out PSNR (dB) / SSIM. \*Wall time of one run including depth inference, with two runs sharing the
GPU, so it shows relative cost only. Earlier MoGe-2-as-depth result (decision 51 setup, other build): 13.45 dB.

Run-to-run spread at a fixed variant is 0.1–0.4 dB (here: Depth Anything V2 Small 14.10–14.52 dB across seeds;
the Depth Anything V2 numbers of this build match the earlier baselines 14.65 / 14.45 dB within that spread).

### Why Depth Anything 3 helps

The per-view scale/shift fit to the SfM points (`DepthAlignment`) is far better with Depth Anything 3 than with
Depth Anything V2 Small on these photos (seed 0, same SfM points, same RANSAC settings):

| Estimator | Inliers (mean over 10 views) | Mean median relative error | Kind corrections |
|---|---|---|---|
| Depth Anything V2 Small (relative inverse depth) | 55 % | 0.0118 | 1 of 10 |
| Depth Anything 3 Base (depth) | 77 % | 0.0038 | 0 of 10 |

So the aligned metric depth that drives dense initialisation, the L1-free Pearson term and the pseudo-views is
about three times more accurate, and consistent across views.

### Confidence weighting

Depth Anything 3's confidence on these photos ranges 1 … 12.5 (median 6.5, 10th percentile 1.7): low on sky,
reflections and depth edges. Used as normalised per-pixel weights of the Pearson term it changes the held-out PSNR by
+0.05 dB (16.19 vs 16.14 dB mean; per seed +0.07, +0.03, +0.05) and SSIM by +0.002, which is inside the run-to-run noise. It costs nothing measurable, but there is no evidence it helps.

### Normal prior

The MoGe-2 normals agree with the normals of the Depth Anything 3 depth after SfM alignment: mean cosine 0.78–0.91
per view (0.38–0.50 with the other portrait rotation, which confirmed the rotation convention). On top of Depth
Anything 3 the normal term leaves the held-out PSNR unchanged (+0.05 dB, within noise; per seed −0.01, +0.06, +0.11)
but raises the held-out SSIM by +0.010 on every seed (0.504–0.508 against 0.496–0.497, so the ranges do not overlap).
It costs about 10 % more training time and ends with about 9 % more Gaussians. Not measured: the normal prior on top
of Depth Anything V2, other weights than 0.05, and the combination with confidence weighting.

## Synthetic benchmark

`SPLAT_BENCHMARK=1 SPLAT_BENCHMARK_VIEWS=4 SPLAT_BENCHMARK_SEED=<5,6,7> SPLAT_BENCHMARK_ONLY="defaults ("
swift test -c release --filter SparseViewBenchmark` (new row "defaults (+ normal prior from the depth prior)"):

| Seed | defaults: novel PSNR / SSIM / depth Pearson | + normal prior: novel PSNR / SSIM / depth Pearson | Δ PSNR |
|---|---|---|---|
| 5 | 27.47 / 0.950 / 0.966 | 27.33 / 0.946 / 0.967 | −0.14 |
| 6 | 27.42 / 0.952 / 0.956 | 25.52 / 0.930 / 0.957 | −1.90 |
| 7 | 27.90 / 0.951 / 0.952 | 26.40 / 0.928 / 0.963 | −1.50 |
| mean | 27.60 / 0.951 / 0.958 | 26.42 / 0.935 / 0.962 | **−1.18** |

4 training views, 3000 iterations, 160 × 120. The normal prior slightly raises the novel-view depth correlation but
costs 0.1–1.9 dB novel PSNR. A likely cause: in this scene of small sphere Gaussians the normals of the aligned
depth prior are noisy across every depth edge (forward differences over a discontinuity), and λ = 0.05 pulls the
geometry towards them. MoGe-2 normals on the real photos are smooth, which may be why the real scene does not lose.
This was not investigated further (no edge masking, no weight sweep).

- Depth Anything 3 and confidence weighting cannot be measured there: the benchmark's depth is synthetic (derived
  from the ground truth) and has no confidence, so both are not applicable. Without weights the weighted loss is
  the unweighted loss exactly (all weights 1; checked by tests on CPU and GPU), so the benchmark is unchanged by
  construction.
- The normal-prior row uses the normals of each view's aligned depth prior (no MoGe), so it only checks that the
  term is harmless when its normals agree with the depth prior.

## Recommendation

1. **Depth Anything 3 as the depth estimator: clear gain.** +1.80 dB held-out PSNR and +0.087 SSIM over Depth
   Anything V2 Small on the real photos, with every seed of Depth Anything 3 (16.12–16.15 dB) far above every seed
   of V2 (14.10–14.52 dB). This is about 5× the run-to-run spread. It is the candidate for the new default
   estimator, **but it cannot simply become the default today**: the package is not downloadable yet (it has to be
   converted and imported by hand, `ModelDescriptor.depthAnything3Base.baseURL == nil`), it is about 600 MB and
   ~1.7× more Gaussians (≈ 15 % slower training) result. Suggested next step for the user: decide whether to publish
   the package (decision on hosting and the default), and confirm the gain on a second scene. Until then it stays
   selectable in the Depth Model picker, marked "pose-free package".
2. **Confidence weighting: keep off.** +0.05 dB is inside the noise. The code is correct and tested and costs
   nothing, so it can stay as an option, but there is no evidence for turning it on.
3. **Normal prior: keep off.** On the real photos it is PSNR-neutral with a small but consistent SSIM gain (+0.010),
   on the synthetic benchmark it loses 1.2 dB on average. Not good enough for a default; it would need an edge-aware
   mask and a weight sweep first.

## What was not verified

- Only one real scene (13 photos, one held-out split). The Depth Anything 3 gain is large and consistent across
  seeds, but it is one dataset.
- The app path (Depth Model picker → Depth Anything 3 → Train) was built (`xcodebuild` BUILD SUCCEEDED) but not
  clicked through in the app. The harness uses the same `DepthAnything3DepthEstimator` and `DepthPreparation`.
- Groups of more than 16 views (`DepthAnything3Math.depthGroups`) are unit-tested but were not run on a real scene.
- Leave-one-out in the app: Depth Anything 3 runs once on all project views, so the held-out view of a
  leave-one-out run is part of the multi-view input of the other views' depth (its pixels never enter the loss).
  This slightly flatters the leave-one-out score with Depth Anything 3; it does not affect normal training.
- The normal prior has no cache (MoGe runs on every training start) and is only implemented in the GPU trainer
  (the CPU reference trainer has none, like the other v1.2+ features).
- New configuration keys change `OptimizerState.configurationHash`, so an autosave written by an older build is
  not resumed by this build (it falls back to a fresh optimizer state, as for any configuration change).
