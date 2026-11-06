# Golden generator

Python reference for the Phase 2 differentiable rasterizer. It re-implements the Phase 1
forward pass in PyTorch (`reference.py`), cross-checks projection, covariance and spherical
harmonics against gsplat 1.5.3's own PyTorch reference, back-propagates a fixed random image
gradient with autograd, and writes JSON golden cases that the Swift tests load.

This directory is development tooling. It is never shipped in the app and never imported by
Swift. gsplat (Apache-2.0) is used only as a numerical reference; its code is not copied here.

## Files

- `reference.py`: PyTorch forward pass (`forward`, `forward_with_stats`), float32, gradient
  conventions from the Phase 2 plan (alpha clamp, color clamp, Jacobian clamp, constant gating).
- `generate.py`: builds the four cases (`single`, `multi`, `rotated`, `occluded`) with fixed
  seeds, runs the gsplat cross-check, writes `golden_<name>.json`.
- `requirements.txt`: package list (see setup below for the gsplat `--no-deps` step).

## Setup

Create a virtual environment in the repository root (ignored by git):

```sh
python3 -m venv .venv-golden
source .venv-golden/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install --no-deps gsplat==1.5.3
pip install numpy jaxtyping typing_extensions rich
```

`requirements.txt` lists the same packages for reference. gsplat must still be installed with
`--no-deps` so that it does not pull in the CUDA build of torch.

## Regenerate the fixtures

```sh
python Tools/golden/generate.py \
    --out SplatKit/Tests/SplatCoreTests/Fixtures SplatKit/Tests/SplatMetalTests/Fixtures
```

The generator prints a summary per case (alive Gaussians, Jacobian-clamped Gaussians,
clamped-alpha samples and pixels, terminated pixels, max gradient per parameter group, and
the gsplat cross-check differences). It exits non-zero if a cross-check difference exceeds
1e-5, if `occluded` has no clamped-alpha sample or no early-terminated pixel, or if `rotated`
has fewer than three Jacobian-clamped Gaussians.

Both fixture directories receive identical files. Each file must stay under 400 KB; the
generator enforces this.

### Phase 4 cases

`--only phase4` writes only the new cases: `golden_depth_single.json` and
`golden_depth_occluded.json` (rendered depth `RGB+D` on the Phase 2 scenes, with `renderedDepth`,
`renderedAlpha` and `depthGradient`), `golden_pearson.json` (Pearson and L1 depth loss in the
`depth` and `inverseDepth` domains) and `golden_anisotropy.json` (anisotropy hinge). The default
(`--only phase2`) still writes the four Phase 2 cases; `--only all` writes everything.

```sh
python Tools/golden/generate.py --only phase4 \
    --out SplatKit/Tests/SplatCoreTests/Fixtures SplatKit/Tests/SplatMetalTests/Fixtures
```

The depth cross-check uses gsplat's pure-PyTorch compositing (`gsplat.cuda._torch_impl.accumulate`)
with its geometry from gsplat. `gsplat.rendering.rasterization(render_mode="RGB+D")` needs the CUDA
extension, which this machine does not have. Because gsplat's compositing imports two nerfacc
primitives that are not installed (and must not be added), the generator installs in-process
equivalents with nerfacc's semantics when that import fails.

The Phase 2 fixtures (`golden_single/multi/rotated/occluded.json`) were generated with an older
torch. Regenerating them with the current torch is not expected to be byte-identical (float noise,
about 3e-6 relative), so they are kept as committed.
