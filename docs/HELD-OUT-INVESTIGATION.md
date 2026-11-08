# Held-out gap investigation (v1.5)

Question: on the 13 iPhone photos of the table scene (COLMAP poses, 10 training views, held out IMG_9014,
IMG_9018 and IMG_9022, 450×600 / 600×450, 5000 iterations) the model reaches 24–31 dB on the training views
but only about 14.5 dB PSNR / 0.41 SSIM on the held-out views. Why, and what raises the held-out score?

Date: 2026-11-08. Machine: Apple M5 Pro (GPU shared with other work, at most two runs at a time).

## TL;DR

* **The gap is genuine wide-baseline generalisation, not an evaluation artefact.** Each held-out camera sits in
  the middle of a 50–60° gap of a 360° orbit, 28–29° in viewing direction and 0.42–0.59 × the scene distance
  away from the nearest training camera. A flat image in the photo's own mean colour already scores
  12.4 dB / SSIM 0.355, so the model is only about 2.2 dB better than "no model".
* Evaluation and photometric effects together explain **less than about 1.3 dB** of the gap:
  per-channel colour fit +0.8 dB, pose search +0.1–0.3 dB, and an upper bound of about +0.8 dB if the
  under-covered pixels were perfect. About 12 dB of the 24→14.5 dB difference is real.
* The held-out score is reached after **about 300–500 iterations** and then stays flat or drops slightly.
  Training PSNR keeps rising, from 20 to 29–31 dB. What the later iterations learn does not transfer.
* The main measured lever is **training at half resolution: +0.84 dB PSNR, +0.016 SSIM** (2 seeds, every
  held-out view improves). DropGaussian (p = 0.3) and early stopping give SSIM gains of about +0.02, but
  their PSNR changes are inside the noise.

## Datasets and harness

* **Dataset A**: the PNG export used for the earlier measurements (portrait 450×600, COLMAP model with
  1418 points). The machine rebooted during the investigation, which wiped `/private/tmp`, including
  dataset A and the raw logs of the first runs. The numbers marked *A* below were copied from those logs
  before the wipe and cannot be re-checked.
* **Dataset B**: `datasets/table-scene` (the same 13 photos as landscape JPGs, 600×450, a separate COLMAP
  model with 1526 points). All later runs used B. The defaults give the same result on both datasets
  (A: 14.42 / 14.47, B: 14.66 / 14.46, earlier known value 14.65 / 14.45), so A and B are comparable within
  the noise.
* Harness: a scratch SwiftPM executable (`h5`, not in the repo). It uses the same configuration rules as
  `harness4` and calls `GPUTrainer` directly so it can evaluate during training. "Defaults" means the app
  defaults (every regularizer, appearance, MCMC, anti-aliasing and pose refinement on). Unless noted:
  5000 iterations; held-out PSNR is measured from a classic render of the baked snapshot (it matches the
  trainer's own mip evaluation within 0.03 dB).
* Noise: the run-to-run spread is 0.1–0.7 dB. Differences under 0.5 dB with 2 seeds are not conclusive.

## 1. How hard are the held-out views? (no model)

| view | nearest training camera | view angle | baseline / scene distance | nearest photo PSNR | nearest photo + colour fit | flat mean image PSNR / SSIM |
|---|---|---|---|---|---|---|
| IMG_9014 | IMG_9013 | 27.8° | 0.45 | 10.90 | 13.71 | 13.60 / 0.367 |
| IMG_9018 | IMG_9019 | 27.6° | 0.42 | 10.12 | 12.10 | 11.62 / 0.380 |
| IMG_9022 | IMG_9016 | 29.0° | 0.59 | 9.42 | 11.86 | 11.85 / 0.318 |
| **mean** | | | | **10.15** | **12.56** | **12.36 / 0.355** |

Camera layout (azimuth around the centroid of the camera centres, dataset A): the 13 photos cover 360°.
IMG_9014 (−97°) lies between 9013 (−48°) and 9024/9015 (−124°/−159°). IMG_9018 (61°) lies between 9019 (25°)
and 9017 (100°). IMG_9022 (131°) lies between 9017 (100°) and 9016 (152°), at almost twice the radius of
its neighbours. The model's 14.5 dB / 0.41 is therefore only +2.2 dB / +0.06 SSIM above a flat image.

Calibration of the image texture (dataset B): a training photo compared with itself shifted by 1, 3, 5 and
10 px gives 23.7, 18.7, 17.1 and 15.3 dB. A 1–3 px misregistration alone costs 6–10 dB on this scene.

## 2. Evaluation artefacts (defaults, `--diag`)

| | A s0 | A s1 | B s0 | B s1 | mean |
|---|---|---|---|---|---|
| held-out PSNR (as reported) | 14.42 | 14.47 | 14.66 | 14.46 | **14.50** |
| + per-view, per-channel affine colour fit | 15.20 | 15.26 | 15.41 | 15.27 | **15.29 (+0.79)** |
| PSNR only on pixels with α > 0.9 | 14.96 | 14.96 | 15.39 | 15.28 | 15.15 (+0.65) |
| α > 0.9 pixels + colour fit | 15.46 | 15.48 | 15.80 | 15.69 | 15.61 (+1.11) |
| fraction of pixels with α > 0.9 | 0.68 | 0.67 | 0.72 | 0.71 | 0.70 |
| uncovered pixels filled with the mean training colour | – | – | 14.44 | 14.33 | 14.39 (−0.11) |
| render blurred (3× box, r = 2 / 4 / 8 / 16 px) | – | – | 14.80/14.84/14.81/14.60 | 14.59/14.62/14.60/14.43 | +0.18 at r = 4 |
| test-time 6-DoF pose search on the held-out camera (raw PSNR) | 14.53 | 14.57 | 14.87 | 14.80 | 14.69 (+0.19) |
| training views, COLMAP pose, identity colour | 23.76 | 23.61 | 23.61 | 23.67 | 23.7 |
| training views, refined pose | 24.31 | 24.18 | 24.07 | 24.19 | 24.2 |
| training views, refined pose + colour fit | 27.24 | 27.15 | 27.34 | 27.41 | 27.3 |
| training views, trainer metric (learned appearance, mip) | 28.70 | 28.73 | 29.00 | 29.18 | 28.9 |

Per view (B s0): IMG_9014 15.60 (colour fit 16.89; covered 17.35; uncovered 13.41), IMG_9018 15.32 (15.75;
15.38; 15.14), IMG_9022 13.05 (13.59; 13.45; 12.04). The near-black share (render < 0.03 where the photo
> 0.1) is only 0.6–3.9 %: the held-out renders have **no large black holes or field-of-view gaps**.
Low-α pixels (α < 0.5) make up 7–16 %.

Readings:
* **Photometric**: the fitted gains are 0.54–0.99, mostly 0.55–0.9. This is the least-squares answer to a
  render whose structure is partly wrong (contrast is shrunk towards the mean), not an exposure offset.
  The appearance module therefore cannot recover this at evaluation time. Colour fit is worth +0.8 dB.
* **Coverage**: covered pixels are only 0.65 dB better than all pixels. Filling holes with a mean colour
  does not help (−0.1 dB on average; +1.3 dB on IMG_9014, −1 dB on the other two).
* **Pose**: on the training views a 0.25° / 0.5 % error costs 5 dB and 0.5° costs 7 dB. With the shift
  calibration above, this is the texture of the scene, not a sign of over-fitting. The held-out search
  finds almost no correction for IMG_9014 and IMG_9018 (≤ 0.06°, ≤ 0.25 %). For IMG_9022 it finds
  0.8–2.7° and 2.5–5.7 %, which brings +0.3 to +1.0 dB on that view. IMG_9022 (the far, wide view) may
  have a slightly inconsistent pose; it is not the main problem.
* **Blur**: low-passing the render adds only +0.2 dB. The renders are not dominated by high-frequency noise.
* The low training numbers without colour/pose correction (23.7 dB with the v1.2 features on, against
  31.2 dB with them off) come from evaluating with identity appearance and COLMAP poses. They say
  nothing about generalisation.

## 3. Genuine over-fitting

### 3a. Held-out PSNR during training (defaults, B, `--curve 500`)

| iteration | 0 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| held-out s0 | 12.28 | 14.69 | 14.65 | **14.72** | 14.56 | 14.35 | 14.48 | 14.50 | 14.45 | 14.46 | 14.66 |
| held-out s1 | 12.31 | **14.72** | 14.72 | 14.56 | 14.57 | 14.40 | 14.31 | 14.24 | 14.16 | 14.27 | 14.46 |
| SSIM s0 | 0.336 | 0.434 | 0.434 | **0.438** | 0.435 | 0.431 | 0.427 | 0.425 | 0.422 | 0.414 | 0.422 |
| training (trainer metric) s0 | 13.2 | 21.0 | 23.2 | 24.4 | 25.3 | 25.2 | 24.9 | 25.7 | 26.8 | 26.5 | 29.0 |

Dataset A gives the same shape: s0 peaks at 14.82 (1500), s1 at 14.72 (1000), then falls to
14.2–14.4 at 4000–4500. With every v1.2 feature off (A): 12.19 at iteration 0, 14.34 at 1000, then flat
at 14.0–14.1 up to 5000, while training PSNR climbs to 31.2. A run of only 300 iterations (A, p = 0.3
dropout smoke test) already reaches 14.62 held-out at 18.3 dB training PSNR.

### 3b. Iteration count (separate runs, schedule scaled to the length)

| iterations | s0 | s1 | mean PSNR | mean SSIM | training (trainer metric) |
|---|---|---|---|---|---|
| 1000 (A) | 14.85 | 14.46 | **14.66** | **0.421** | 21.9 |
| 2000 (A) | 14.62 | 14.59 | 14.61 | 0.421 | 24.2 |
| 5000 (A+B, 4 runs) | 14.42–14.66 | | 14.50 | 0.415 | 28.9 |
| 10000 (B, s0 only) | 14.46 | – | 14.46 | 0.409 | 36.0 (1.5 M Gaussians, 605 s) |

Early stopping at 1000–2000 iterations: about +0.1–0.15 dB and +0.006 SSIM on average, 5× faster.
This is within the noise for PSNR, but the in-run curves show the same small optimum early in every seed.

### 3c. Number of training views (always the same 3 held out; dataset A; spread evenly in name order)

| training views | set | s0 | s1 | mean PSNR / SSIM |
|---|---|---|---|---|
| 4 | 9012, 9016, 9020, 9024 | 12.21 | 12.10 | 12.15 / 0.267 |
| 6 | 9012, 9015, 9017, 9019, 9021, 9024 | 12.13 | 12.23 | 12.18 / 0.313 |
| 8 | 9012, 9013, 9016, 9017, 9019, 9020, 9023, 9024 | 13.86 | 14.08 | 13.97 / 0.377 |
| 10 | all | 14.42 | 14.47 | 14.45 / 0.410 |

With 4 or 6 views the model is no better than a flat image (12.36). The score depends on whether the
direct neighbours of a held-out view are in the training set (with 8 views, IMG_9014 loses neighbour 9015
and drops 2 dB). It is strongly non-linear, which is typical of the wide-baseline regime.

### 3d. Regularisation, depth prior, SH degree, resolution, background, dropout (5000 iterations)

| variant | dataset | s0 | s1 | mean PSNR | Δ vs defaults | mean SSIM | Δ SSIM |
|---|---|---|---|---|---|---|---|
| defaults | A | 14.42 | 14.47 | 14.45 | – | 0.410 | – |
| defaults | B | 14.66 | 14.46 | 14.56 | – | 0.420 | – |
| every v1.2 feature off | A | 14.09 | 14.04 | 14.07 | −0.38 | 0.389 | −0.021 |
| no depth prior and no dense init (SfM init) | A | 14.02 | 14.11 | 14.07 | −0.38 | 0.386 | −0.024 |
| Pearson depth weight 0 (dense init kept) | A | 14.45 | 14.47 | 14.46 | +0.01 | 0.410 | 0 |
| Pearson depth weight 0.2 (4×) | A | 14.34 | 14.00 | 14.17 | −0.28 | 0.409 | 0 |
| SH degree 0 | B | 14.32 | 14.41 | 14.37 | −0.19 | 0.414 | −0.006 |
| MCMC opacity + scale L1 0 (default 0.01) | B | 14.07 | 14.07 | 14.07 | **−0.49** | 0.407 | −0.013 |
| MCMC opacity + scale L1 0.05 (5×) | B | 14.93 | 14.33 | 14.63 | +0.07 | 0.437 | +0.017 |
| background = mean training colour (train + eval) | B | 15.05 | 14.32 | 14.69 | +0.13 | 0.420 | 0 |
| DropGaussian p 0 → 0.1 | B | 14.95 | 14.44 | 14.70 | +0.14 | 0.437 | +0.017 |
| DropGaussian p 0 → 0.3 | B | 15.20 | 14.41 | 14.81 | +0.25 | 0.445 | +0.025 |
| **train at half resolution (225×300), evaluate at full** | B | **15.46** | **15.34** | **15.40** | **+0.84** | **0.436** | **+0.016** |

Half resolution per view (s0 / s1): IMG_9014 16.45 / 16.39, IMG_9018 15.58 / 15.35, IMG_9022 14.36 / 14.28,
against 15.60 / 15.20, 15.32 / 15.16 and 13.05 / 13.01 with the defaults. All three views improve, and
IMG_9022 the most (+1.3 dB). Training views lose about 1 dB at full resolution (22.6 against 23.6 with
COLMAP poses) but stay at 28.5 dB on the trainer's own (half-resolution) metric.

Observations:
* The dense depth initialisation is worth about +0.4 dB. The Pearson loss weight itself does nothing at
  0.05 and hurts at 0.2. The depth L1 term is inactive for monocular depth (it is LiDAR only).
* Without depth (SfM init) the training PSNR stays at 19.6–19.9 dB, yet held-out is only 0.4 dB lower:
  held-out quality is almost independent of how well the training views are fitted.
* The MCMC opacity/scale L1 regularisers help (removing them costs 0.5 dB in both seeds). Five times the
  strength is neutral in PSNR and +0.017 SSIM.
* DropGaussian helps SSIM in both seeds (+0.017 / +0.025). Its PSNR gain is seed-dependent: +0.3 / +0.5
  on s0, −0.05 on s1. Not conclusive with 2 seeds.

## 4. DropGaussian implementation (code change, default off)

`TrainingConfiguration.dropout: GaussianDropoutSettings?` (nil = off; optional so a configuration without
it encodes byte-for-byte as before, so the manifest JSON and `OptimizerState.configurationHash` do not
change). When it is set, every training iteration leaves out each Gaussian with probability
p(t) = finalRate · t / T, using a per-iteration GPU hash (`dropGaussian` in `Common.metal`). The dropped
Gaussians are culled in `projectGaussians` (no tiles, so `projectBackward` gives them zero gradient). The
kept ones get opacity × 1/(1 − p) in the forward pass, and the backward applies the same factor to the
opacity-logit gradient and to the two mip paths (2D compensation and the 3D-filter scale term).
Evaluation, previews, snapshots and exports never drop. Files: `RasterSettings.dropRate/dropSeed`,
`SplatRenderer.makeMipUniforms` (`MipFilterUniforms.info.z/w`, previously unused), `Rasterizer.metal`,
`Backward.metal`, `GPUTrainer.advance()`.

Tests (`Tests/SplatMetalTests/DropoutMetalTests.swift`): GPU forward and backward with p = 0.3 against the
CPU reference on the reduced cloud (dropped Gaussians removed, opacity σ/(1 − p)), classic and mip, all
parameter gradients within 1e-4 + 1e-3·|cpu|. Dropped Gaussians get exactly zero gradient. p = 0 renders
bit-identically. The ramp and encoding behaviour are covered too. There is no separate gsplat golden or
finite-difference test: the kernels are unchanged at p = 0, and at p > 0 they are checked against the CPU
reference, which itself is finite-difference-verified. The CPU reference trainer does not implement
dropout.

## 5. Main finding: decomposition of the gap

| part | dB |
|---|---|
| training views, trainer metric | ≈ 28.9 |
| held-out, as reported | 14.5 |
| recoverable by per-view colour fit (photometric / contrast) | +0.8 |
| recoverable by held-out pose correction | +0.2 (only IMG_9022 is affected) |
| maximum attributable to low-coverage pixels | ≈ +0.65 (overlaps with the colour fit; +1.1 together) |
| **remaining: genuine generalisation failure across 28° / 0.5× baselines** | **≈ 12–13** |

So **about 1–1.3 dB (≈ 10 %) of the 14 dB gap is evaluation / photometric / coverage, and about 90 % is real
generalisation**. It is set by the capture geometry: 10 views on a full orbit, so each held-out view is
28° away. The optimisation reaches its held-out ceiling within the first few hundred iterations. Later
iterations, and most regularisers, only change how well the training views are memorised.

## 6. Ranked, evidence-backed changes

1. **Train at reduced resolution for sparse 360° captures** (e.g. half the long side, or a coarse-to-fine
   schedule): +0.84 dB / +0.016 SSIM, both seeds, every view. Also 2–4× cheaper. Not yet tested: more
   seeds, quarter resolution, coarse-to-fine, combination with dropout (these runs were queued and
   cancelled). This is the best candidate for a follow-up decision.
2. **Keep the MCMC opacity/scale regularisers on.** Removing them costs 0.5 dB in both seeds. A higher
   strength (0.05) is neutral in PSNR and +0.017 SSIM. Low priority.
3. **Keep the dense depth initialisation; the Pearson weight does not matter** (0 ≈ 0.05; 0.2 is worse).
   The +0.4 dB comes from the initialisation, not from the loss.
4. **Shorter training / early stopping at about 1000–2000 iterations for sparse captures:** held-out
   +0.1–0.15 dB and +0.006 SSIM on average, every seed peaks early, about 5× faster. Inside the PSNR noise.
5. **DropGaussian (p ramped to 0.3):** +0.025 SSIM in both seeds, PSNR +0.25 on average but
   seed-dependent. Implemented behind a flag, default off. It needs ≥ 4 seeds before any default change.
6. Not worth pursuing for this gap: background colour (+0.1, noise), SH degree 0 (−0.2), a higher depth
   weight (−0.3), switching v1.2 features off (−0.4), held-out blur (+0.2), and test-time pose refinement
   for 2 of 3 views.

The structural fix lies outside the optimiser: **more or closer views** (the held-out score rises 2.3 dB from
6 to 10 views and is still steep), or a stronger geometric prior than monocular depth (multi-view stereo
priors or learned novel-view priors). These match decision 14's experience that only depth-related
geometry gave clear gains.

## 7. Not determined / caveats

* With 2 seeds, every difference under about 0.5 dB is not conclusive. Only half resolution (+0.84, both
  seeds above every default run) and the MCMC regulariser removal (−0.49) clear that bar.
* The dataset A raw logs were lost in the reboot. Their numbers are transcribed from the logs read
  before the wipe.
* Sweeps not run: 12 of 13 training views (not possible with 3 fixed held-out views: at most 10 remain),
  quarter resolution, half resolution combined with dropout, and more seeds for dropout and half
  resolution. Iteration 10000 has one seed.
* The pose search is a coordinate descent from the COLMAP pose. A distant better optimum cannot be ruled
  out, but the small corrections found for 2 of 3 views argue against large pose errors.
* The perturbation test measures image texture as much as model quality (see the shift calibration).

## Addendum: 10000 iterations (B, seed 0)

The in-run curve (held-out every 1000 iterations): 12.28 (0), 14.82, **14.84** (2000), 14.75, 14.59, 14.53,
14.42, 14.47, 14.50, 14.47, 14.55 (10000). Final classic render: 14.46 / SSIM 0.409. Training PSNR keeps
rising from 23.2 to 36.0 dB and the cloud hits the 1.5 M Gaussian cap. The held-out score peaks at
2000 iterations and then stays at 14.5. Longer training only adds memorisation.

## Verification

`cd SplatKit && swift test -c release` on the Mac (Apple M5 Pro), with the dropout change and the new
`DropoutMetalTests`: every suite passed (59 + 96 + 370 + 43 + 502 tests, 0 failures).
