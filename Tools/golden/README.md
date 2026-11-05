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
