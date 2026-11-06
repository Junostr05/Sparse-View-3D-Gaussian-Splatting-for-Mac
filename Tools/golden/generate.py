#!/usr/bin/env python3
"""Generates the golden forward/backward fixtures for the Swift test suites.

Usage:
    python Tools/golden/generate.py --out SplatKit/Tests/SplatCoreTests/Fixtures \
                                    SplatKit/Tests/SplatMetalTests/Fixtures

For every case it builds the Gaussians with fixed seeds, runs the PyTorch reference
(reference.py), cross-checks projection, covariance and SH against gsplat 1.5.3's
reference implementation, back-propagates a random upstream image gradient with
autograd, and writes golden_<name>.json. Python tooling only; never shipped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

from gsplat.cuda._torch_impl import (
    accumulate,
    _eval_sh_bases_fast,
    _persp_proj,
    _quat_scale_to_covar_preci,
    _world_to_cam,
)

import reference as ref
from reference import Camera, Params, Settings

CASES = ("single", "multi", "rotated", "occluded")
# Phase 4 cases (`--only phase4`): depth cases reuse the Phase 2 scenes, the other two are
# loss-only references. Output names are golden_<name>.json.
DEPTH_CASES = {"depth_single": "single", "depth_occluded": "occluded"}
PHASE4_NAMES = ("depth_single", "depth_occluded", "pearson", "anisotropy")
CROSS_CHECK_TOL = 1e-5
MAX_FILE_BYTES = 400_000
F32 = torch.float32


# ---------------------------------------------------------------------------
# Numeric helpers: parameters are snapped to 7 significant digits before anything
# is computed, so the JSON values are exactly the values the reference used.
# ---------------------------------------------------------------------------

def snap(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.array([float(f"{v:.7g}") for v in arr.reshape(-1)], dtype=np.float64).reshape(arr.shape)


def to_tensor(values, shape=None) -> torch.Tensor:
    arr = snap(values)
    t = torch.tensor(arr, dtype=F32)
    return t.reshape(shape) if shape is not None else t


def flat(t: torch.Tensor):
    return [float(f"{v:.7g}") for v in t.detach().to(torch.float64).reshape(-1).tolist()]


def uniform(gen, shape, lo, hi):
    return torch.rand(shape, generator=gen, dtype=torch.float64).numpy() * (hi - lo) + lo


def normal(gen, shape):
    return torch.randn(shape, generator=gen, dtype=torch.float64).numpy()


def rotation_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def rotation_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def make_camera(width, height, f, cx, cy, rot_world_to_cam: np.ndarray, t: np.ndarray) -> Camera:
    return Camera(
        width=width,
        height=height,
        fx=float(snap(f)),
        fy=float(snap(f)),
        cx=float(snap(cx)),
        cy=float(snap(cy)),
        rotation=to_tensor(rot_world_to_cam, (3, 3)),
        translation=to_tensor(t, (3,)),
    )


def screen_gaussians(gen, cam_np, mean2d, depth, sigma2d, opacity_logit, sh_degree,
                     quat_rng=None):
    """Builds Gaussians whose projected centres are `mean2d` at camera depth `depth`.

    `cam_np` = (width, height, f, cx, cy, R, t) in float64. Positions are computed in camera
    space and mapped to world space with p = Rᵀ (p_cam - t). Isotropic-ish scales are chosen so
    that the 2D standard deviation is about `sigma2d` pixels.
    Returns float64 arrays (positions, log_scales, quats, opacity_logits, sh_unused).
    """
    _, _, f, cx, cy, R, t = cam_np
    n = len(depth)
    x = (mean2d[:, 0] - cx) * depth / f
    y = (mean2d[:, 1] - cy) * depth / f
    p_cam = np.stack([x, y, depth], axis=-1)
    world = (p_cam - t) @ R  # Rᵀ (p_cam - t), row-wise
    aniso = uniform(gen, (n, 3), 0.8, 1.25)
    sigma3 = sigma2d[:, None] * depth[:, None] / f * aniso
    log_scales = np.log(sigma3)
    quats = normal(gen, (n, 4))
    return world, log_scales, quats, opacity_logit


def build_params(gen, cam_np, mean2d, depth, sigma2d, opacity_logit, sh_degree, sh_scale):
    world, log_scales, quats, logits = screen_gaussians(
        gen, cam_np, mean2d, depth, sigma2d, opacity_logit, sh_degree)
    n = len(depth)
    k = (sh_degree + 1) ** 2
    sh = normal(gen, (n, k, 3)) * sh_scale
    return Params(
        positions=to_tensor(world, (n, 3)),
        log_scales=to_tensor(log_scales, (n, 3)),
        rotations=to_tensor(quats, (n, 4)),
        opacity_logits=to_tensor(logits, (n,)),
        sh=to_tensor(sh, (n, k, 3)),
        sh_degree=sh_degree,
    )


def cam_numpy(width, height, f, cx, cy, rot, t):
    return (width, height, f, cx, cy, rot, t)


def case_single():
    gen = torch.Generator().manual_seed(1)
    W, H, f, cx, cy = 16, 16, 20.0, 8.0, 8.0
    rot, t = np.eye(3), np.zeros(3)
    cam_np = cam_numpy(W, H, f, cx, cy, rot, t)
    params = build_params(
        gen, cam_np,
        mean2d=np.array([[7.3, 9.1]]),
        depth=np.array([3.0]),
        sigma2d=np.array([3.0]),
        opacity_logit=np.array([0.8]),
        sh_degree=0,
        sh_scale=0.4,
    )
    cam = make_camera(W, H, f, cx, cy, rot, t)
    settings = Settings(sh_degree_limit=0, background=to_tensor([0.2, 0.3, 0.4], (3,)))
    return "single", cam, params, settings, gen


def case_multi():
    gen = torch.Generator().manual_seed(2)
    W, H, f, cx, cy = 40, 28, 30.0, 20.0, 14.0
    rot, t = np.eye(3), np.zeros(3)
    cam_np = cam_numpy(W, H, f, cx, cy, rot, t)
    n = 30
    mean2d = np.stack([uniform(gen, (n,), -4, W + 4), uniform(gen, (n,), -4, H + 4)], axis=-1)
    depth = uniform(gen, (n,), 1.5, 5.0)
    sigma2d = uniform(gen, (n,), 1.5, 8.0)
    logits = uniform(gen, (n,), -1.0, 2.5)
    params = build_params(gen, cam_np, mean2d, depth, sigma2d, logits, sh_degree=3, sh_scale=0.4)
    cam = make_camera(W, H, f, cx, cy, rot, t)
    # Degree 3 data with a limit of 2: bands 3 get no gradient.
    settings = Settings(sh_degree_limit=2, background=to_tensor([0.05, 0.1, 0.15], (3,)))
    return "multi", cam, params, settings, gen


def case_rotated():
    gen = torch.Generator().manual_seed(3)
    W, H, f, cx, cy = 32, 24, 28.0, 16.0, 12.0
    rot = rotation_x(20.0) @ rotation_y(-20.0)
    t = np.array([0.2, -0.15, 0.6])
    cam_np = cam_numpy(W, H, f, cx, cy, rot, t)

    n_in = 34
    mean_in = np.stack([uniform(gen, (n_in,), -2, W + 2), uniform(gen, (n_in,), -2, H + 2)], -1)
    depth_in = uniform(gen, (n_in,), 1.5, 4.5)
    sigma_in = uniform(gen, (n_in,), 1.5, 6.0)
    logit_in = uniform(gen, (n_in,), -1.0, 2.5)

    # Outliers beyond 1.3x the field of view (Jacobian clamp active) that still touch the image.
    # Their 2D sigma is large enough that the 3-sigma tile rect overlaps the image.
    mean_out = np.array([
        [cx + 1.45 * W / 2, uniform(gen, (), 0, H)],
        [cx - 1.45 * W / 2, uniform(gen, (), 0, H)],
        [cx + 1.45 * W / 2, uniform(gen, (), 0, H)],
        [cx - 1.45 * W / 2, uniform(gen, (), 0, H)],
        [uniform(gen, (), 0, W), cy + 1.5 * H / 2],
        [uniform(gen, (), 0, W), cy - 1.5 * H / 2],
    ])
    n_out = len(mean_out)
    depth_out = uniform(gen, (n_out,), 2.0, 3.0)
    sigma_out = np.full(n_out, 4.0)
    logit_out = uniform(gen, (n_out,), -1.0, 2.5)

    mean2d = np.concatenate([mean_in, mean_out], 0)
    depth = np.concatenate([depth_in, depth_out])
    sigma2d = np.concatenate([sigma_in, sigma_out])
    logits = np.concatenate([logit_in, logit_out])
    params = build_params(gen, cam_np, mean2d, depth, sigma2d, logits, sh_degree=2, sh_scale=0.4)
    cam = make_camera(W, H, f, cx, cy, rot, t)
    settings = Settings(sh_degree_limit=2, background=to_tensor([0.3, 0.1, 0.05], (3,)))
    return "rotated", cam, params, settings, gen


def case_occluded():
    gen = torch.Generator().manual_seed(4)
    W, H, f, cx, cy = 32, 32, 24.0, 16.0, 16.0
    rot, t = np.eye(3), np.zeros(3)
    cam_np = cam_numpy(W, H, f, cx, cy, rot, t)

    # Stacked on the optical axis in front of the rest of the scene, with opacity logit 12
    # (sigmoid ~ 0.999994): the first sample hits the 0.999 clamp with T = 1, and the second
    # one terminates the pixel before blending. The rest of the scene lies behind the stack.
    n_stack = 5
    mean_stack = np.tile([cx, cy], (n_stack, 1)).astype(np.float64)
    depth_stack = np.array([1.5, 1.9, 2.3, 2.7, 3.1])
    sigma_stack = np.full(n_stack, 30.0)
    logit_stack = np.full(n_stack, 12.0)

    n_rest = 55
    mean_rest = np.stack([uniform(gen, (n_rest,), -4, W + 4), uniform(gen, (n_rest,), -4, H + 4)], -1)
    depth_rest = uniform(gen, (n_rest,), 3.5, 6.0)
    sigma_rest = uniform(gen, (n_rest,), 2.0, 8.0)
    logit_rest = uniform(gen, (n_rest,), -1.0, 3.0)

    mean2d = np.concatenate([mean_stack, mean_rest], 0)
    depth = np.concatenate([depth_stack, depth_rest])
    sigma2d = np.concatenate([sigma_stack, sigma_rest])
    logits = np.concatenate([logit_stack, logit_rest])
    params = build_params(gen, cam_np, mean2d, depth, sigma2d, logits, sh_degree=1, sh_scale=0.4)
    cam = make_camera(W, H, f, cx, cy, rot, t)
    settings = Settings(sh_degree_limit=1, background=to_tensor([0.5, 0.25, 0.75], (3,)))
    return "occluded", cam, params, settings, gen


BUILDERS = {
    "single": case_single,
    "multi": case_multi,
    "rotated": case_rotated,
    "occluded": case_occluded,
}


# ---------------------------------------------------------------------------
# Cross-check against gsplat, forward and backward.
# ---------------------------------------------------------------------------

def cross_check(p: Params, cam: Camera, settings: Settings) -> dict:
    """Max abs differences between the reference and gsplat's torch implementation."""
    g = ref.geometry_for_check(p, cam, settings)
    in_front = g["z"] > ref.NEAR_PLANE

    with torch.no_grad():
        scales = torch.exp(p.log_scales)
        covars_w, _ = _quat_scale_to_covar_preci(
            p.rotations, scales, compute_covar=True, compute_preci=False)
        viewmat = torch.eye(4, dtype=F32)
        viewmat[:3, :3] = cam.rotation
        viewmat[:3, 3] = cam.translation
        means_c, covars_c = _world_to_cam(p.positions, covars_w, viewmat[None])
        Ks = torch.tensor([[[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]]],
                          dtype=F32)
        means2d, cov2d = _persp_proj(means_c, covars_c, Ks, cam.width, cam.height)
        means2d = means2d[0]
        cov2d = cov2d[0]

        eps = ref.EPS_2D
        d_mean = (g["mean2d"] - means2d).abs()[in_front].max().item()
        d_cov3d = (g["cov3d"] - covars_w).abs()[in_front].max().item()
        d_cov2d = max(
            (g["a"] - (cov2d[:, 0, 0] + eps)).abs()[in_front].max().item(),
            (g["b"] - cov2d[:, 0, 1]).abs()[in_front].max().item(),
            (g["c"] - (cov2d[:, 1, 1] + eps)).abs()[in_front].max().item(),
        )

        center = -(cam.rotation.T @ cam.translation)
        dirs = F.normalize(p.positions - center, p=2, dim=-1)
        degree = min(p.sh_degree, settings.sh_degree_limit)
        k_used = (degree + 1) ** 2
        bases = _eval_sh_bases_fast(k_used, dirs)
        gs_raw = (bases.unsqueeze(-1) * p.sh[:, :k_used, :]).sum(1)
        d_sh = (g["raw_sh"] - gs_raw).abs()[in_front].max().item()

    return {"mean2d": d_mean, "cov2d": d_cov2d, "cov3d": d_cov3d, "sh": d_sh}


# ---------------------------------------------------------------------------
# Forward + backward for one case.
# ---------------------------------------------------------------------------

GROUPS = ("positions", "logScales", "rotations", "opacityLogits", "sh")


def run_case(name, cam, params, settings, gen):
    H, W = cam.height, cam.width
    image_grad = (torch.rand((H, W, 3), generator=gen, dtype=torch.float64) * 2 - 1).to(F32)

    leaves = [t.detach().clone().requires_grad_(True) for t in params.tensors()]
    leaf_params = Params(
        positions=leaves[0], log_scales=leaves[1], rotations=leaves[2],
        opacity_logits=leaves[3], sh=leaves[4], sh_degree=params.sh_degree,
    )
    image, stats = ref.forward_with_stats(leaf_params, cam, settings)
    grads = torch.autograd.grad(image, leaves, grad_outputs=image_grad, allow_unused=True)
    grads = [torch.zeros_like(l) if g is None else g for l, g in zip(leaves, grads)]
    for gname, gt in zip(GROUPS, grads):
        if not torch.isfinite(gt).all():
            raise RuntimeError(f"[{name}] non-finite gradient in {gname}")
    if not torch.isfinite(image).all():
        raise RuntimeError(f"[{name}] non-finite image")

    cross = cross_check(params, cam, settings)
    return image.detach(), image_grad, grads, stats, cross


def golden_dict(name, cam, params, settings, image, image_grad, grads):
    H, W = cam.height, cam.width
    rot = cam.rotation.detach().to(torch.float64).reshape(-1).tolist()
    return {
        "name": name,
        "width": W,
        "height": H,
        "fx": float(cam.fx),
        "fy": float(cam.fy),
        "cx": float(cam.cx),
        "cy": float(cam.cy),
        "rotation": [float(f"{v:.7g}") for v in rot],
        "translation": flat(cam.translation),
        "background": flat(settings.background),
        "shDegree": params.sh_degree,
        "shDegreeLimit": settings.sh_degree_limit,
        "positions": flat(params.positions),
        "logScales": flat(params.log_scales),
        "rotations": flat(params.rotations),
        "opacityLogits": flat(params.opacity_logits),
        "sh": flat(params.sh),
        "image": flat(image),
        "imageGradient": flat(image_grad),
        "gradients": {gname: flat(gt) for gname, gt in zip(GROUPS, grads)},
    }


# ---------------------------------------------------------------------------
# Phase 4 cases: rendered depth (RGB+D), Pearson/L1 depth loss, anisotropy hinge.
# ---------------------------------------------------------------------------

def install_nerfacc_shim() -> None:
    """Supplies nerfacc's two primitives used by gsplat's pure-PyTorch compositing.

    gsplat.cuda._torch_impl.accumulate imports `accumulate_along_rays` and
    `render_weight_from_alpha` from nerfacc, which is not installed in the golden environment
    and must not be added. Only when that import fails, this installs in-process equivalents
    with nerfacc's semantics (samples sorted by ray, then by depth):
      render_weight_from_alpha: w_i = α_i · Π_{j<i} (1 − α_j) per ray, segmented log-cumsums
      accumulate_along_rays:    Σ_i w_i · v_i per ray (values None: Σ_i w_i)
    """
    try:
        import nerfacc  # noqa: F401
        return
    except ImportError:
        pass

    def render_weight_from_alpha(alphas, ray_indices, n_rays):
        a = alphas.to(torch.float64)
        log_keep = torch.log1p(-a)
        inclusive = torch.cumsum(log_keep, 0)
        exclusive = inclusive - log_keep
        new_segment = torch.ones_like(ray_indices, dtype=torch.bool)
        new_segment[1:] = ray_indices[1:] != ray_indices[:-1]
        segment_id = torch.cumsum(new_segment.long(), 0) - 1
        base = exclusive[new_segment][segment_id]
        weights = (a * torch.exp(exclusive - base)).to(alphas.dtype)
        trans = torch.exp(inclusive - base).to(alphas.dtype)
        return weights, trans

    def accumulate_along_rays(weights, values=None, ray_indices=None, n_rays=None):
        w = weights.to(torch.float64)
        if values is None:
            out = torch.zeros(n_rays, dtype=torch.float64).index_add_(0, ray_indices, w)
        else:
            v = values.to(torch.float64)
            out = torch.zeros(n_rays, v.shape[-1], dtype=torch.float64)
            out = out.index_add_(0, ray_indices, w[:, None] * v)
        return out.to(weights.dtype)

    shim = types.ModuleType("nerfacc")
    shim.render_weight_from_alpha = render_weight_from_alpha
    shim.accumulate_along_rays = accumulate_along_rays
    sys.modules["nerfacc"] = shim


def gsplat_depth_check(p: Params, cam: Camera, detail: dict) -> dict:
    """Composites the pairs chosen by reference.forward_detail with gsplat's own compositing.

    gsplat's rasterization(render_mode="RGB+D") needs the CUDA extension, which this machine
    does not have, so the cross-check uses gsplat's pure-PyTorch `accumulate` instead. Geometry
    comes from gsplat (world -> camera, 2D mean and covariance, camera-space z); opacity and the
    composited pair set come from the reference, whose colour gating is checked by the Phase 2
    images. Returns max |diff| for the accumulated depth (RGB+D channel 2) and alpha.
    """
    install_nerfacc_shim()
    H, W = cam.height, cam.width
    with torch.no_grad():
        scales = torch.exp(p.log_scales)
        covars_w, _ = _quat_scale_to_covar_preci(
            p.rotations, scales, compute_covar=True, compute_preci=False)
        viewmat = torch.eye(4, dtype=F32)
        viewmat[:3, :3] = cam.rotation
        viewmat[:3, 3] = cam.translation
        means_c, covars_c = _world_to_cam(p.positions, covars_w, viewmat[None])
        Ks = torch.tensor([[[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]]],
                          dtype=F32)
        means2d, cov2d = _persp_proj(means_c, covars_c, Ks, W, H)
        means2d = means2d[0]
        cov2d = cov2d[0]
        z = means_c[0, :, 2]

        eps = ref.EPS_2D
        a = cov2d[:, 0, 0] + eps
        b = cov2d[:, 0, 1]
        c = cov2d[:, 1, 1] + eps
        det = a * c - b * b
        conic = torch.stack([c / det, -b / det, a / det], -1)
        opacity = torch.sigmoid(p.opacity_logits)

        idx = detail["alive_index"]
        use = detail["use"]
        rows, cols = torch.nonzero(use, as_tuple=True)  # row-major: pixel, then depth order
        image_ids = torch.zeros_like(rows)
        renders, alphas = accumulate(
            means2d[idx][None],
            conic[idx][None],
            opacity[idx][None],
            z[idx][None, :, None],
            cols,
            rows,
            image_ids,
            W,
            H,
        )
        depth_gs = renders[0, :, :, 0]
        alpha_gs = alphas[0, :, :, 0]
        d_depth = (depth_gs - detail["depth_raw"].detach()).abs().max().item()
        d_alpha = (alpha_gs - detail["alpha"].detach()).abs().max().item()
    return {"depth": d_depth, "alpha": d_alpha}


def depth_golden_dict(name, cam, params, settings, out, image_grad, depth_grad, grads):
    golden = golden_dict(name, cam, params, settings, out["image"], image_grad, grads)
    golden["renderedDepth"] = flat(out["depth_raw"])
    golden["renderedAlpha"] = flat(out["alpha"])
    golden["depthGradient"] = flat(depth_grad)  # (H, W, 2) row-major: (dL/dDraw, dL/dA)
    return golden


def run_depth_case(name: str, base: str):
    """Rendered depth on a Phase 2 scene. Loss = Σ g·image + Σ gD·Draw + Σ gA·A (seeded)."""
    _, cam, params, settings, gen = BUILDERS[base]()
    H, W = cam.height, cam.width
    image_grad = (torch.rand((H, W, 3), generator=gen, dtype=torch.float64) * 2 - 1).to(F32)
    depth_grad = (torch.rand((H, W, 2), generator=gen, dtype=torch.float64) * 2 - 1).to(F32)

    leaves = [t.detach().clone().requires_grad_(True) for t in params.tensors()]
    leaf_params = Params(
        positions=leaves[0], log_scales=leaves[1], rotations=leaves[2],
        opacity_logits=leaves[3], sh=leaves[4], sh_degree=params.sh_degree,
    )
    out = ref.forward_detail(leaf_params, cam, settings)
    loss = ((out["image"] * image_grad).sum()
            + (out["depth_raw"] * depth_grad[..., 0]).sum()
            + (out["alpha"] * depth_grad[..., 1]).sum())
    grads = torch.autograd.grad(loss, leaves, allow_unused=True)
    grads = [torch.zeros_like(l) if g is None else g for l, g in zip(leaves, grads)]
    for gname, gt in zip(GROUPS, grads):
        if not torch.isfinite(gt).all():
            raise RuntimeError(f"[{name}] non-finite gradient in {gname}")
    for key in ("image", "depth_raw", "alpha"):
        if not torch.isfinite(out[key]).all():
            raise RuntimeError(f"[{name}] non-finite {key}")

    cross = gsplat_depth_check(params, cam, out)
    golden = depth_golden_dict(name, cam, params, settings, out, image_grad, depth_grad, grads)
    return golden, cross, out


def pearson_golden(failures: list) -> dict:
    """Task 6 depth loss references (two domains, same prior and mask, measured metric for L1)."""
    W, H = 12, 9
    lam_p, lam_l1, alpha_mask, near, minimum_pixels = 0.05, 0.1, 0.3, 0.01, 64
    gen = torch.Generator().manual_seed(5)
    n = W * H

    depth_true = uniform(gen, (n,), 1.2, 4.0)
    alpha_v = uniform(gen, (n,), 0.05, 1.0)            # about a third below the 0.3 mask
    draw_v = depth_true * alpha_v
    invalid = np.zeros(n, dtype=bool)
    invalid[uniform(gen, (10,), 0, n).astype(int)] = True
    noise_d = normal(gen, (n,)) * 0.1
    noise_i = normal(gen, (n,)) * 0.02
    metric_v = depth_true * 1.02 + normal(gen, (n,)) * 0.02
    metric_v[uniform(gen, (6,), 0, n).astype(int)] = 0.0  # invalid metric samples

    draw = to_tensor(draw_v, (n,)).to(torch.float64)
    alpha = to_tensor(alpha_v, (n,)).to(torch.float64)
    metric = to_tensor(metric_v, (n,)).to(torch.float64)
    target_depth = to_tensor(np.where(invalid, -1.0, 2.0 * depth_true + 0.5 + noise_d),
                             (n,)).to(torch.float64)
    target_inverse = to_tensor(np.where(invalid, 0.0, 1.5 / depth_true + noise_i),
                               (n,)).to(torch.float64)
    normalizer = ref.median_positive(metric)

    cases = []
    for domain, target in (("depth", target_depth), ("inverseDepth", target_inverse)):
        leaf_draw = draw.clone().requires_grad_(True)
        leaf_alpha = alpha.clone().requires_grad_(True)
        loss, rho, l1, count, degenerate = ref.pearson_depth_loss(
            leaf_draw, leaf_alpha, target, domain, metric, True,
            lam_p, lam_l1, alpha_mask, near, minimum_pixels, normalizer)
        if degenerate:
            failures.append(f"[pearson/{domain}] degenerate case; the golden must be non-degenerate")
            cases.append(None)
            continue
        g_draw, g_alpha = torch.autograd.grad(loss, [leaf_draw, leaf_alpha], allow_unused=True)
        g_draw = torch.zeros_like(draw) if g_draw is None else g_draw
        g_alpha = torch.zeros_like(alpha) if g_alpha is None else g_alpha
        print(f"[pearson/{domain}] valid pixels {count}/{n}, loss {float(loss.detach()):.6g}, "
              f"pearson {rho:.6g}, l1 {l1:.6g}, max |gradDraw| {g_draw.abs().max().item():.6g}, "
              f"max |gradA| {g_alpha.abs().max().item():.6g}")
        cases.append({
            "domain": domain,
            "target": flat(target),
            "loss": float(loss.detach()),
            "pearson": rho,
            "l1": l1,
            "validPixels": count,
            "degenerate": degenerate,
            "gradDraw": flat(g_draw),
            "gradA": flat(g_alpha),
        })

    return {
        "width": W,
        "height": H,
        "lambdaPearson": lam_p,
        "lambdaL1": lam_l1,
        "alphaMask": alpha_mask,
        "nearPlane": near,
        "minimumPixels": minimum_pixels,
        "isMeasuredMetric": True,
        "l1Normalizer": normalizer,
        "draw": flat(draw),
        "alpha": flat(alpha),
        "metricDepth": flat(metric),
        "cases": [c for c in cases if c is not None],
    }


def anisotropy_golden(failures: list) -> dict:
    """Task 7 anisotropy references: 16 Gaussians, the first 8 with ratio above 10."""
    gen = torch.Generator().manual_seed(6)
    n, lam, max_ratio = 16, 0.1, 10.0
    base = uniform(gen, (n, 3), -2.0, -0.5)
    raise_axis = np.argmax(uniform(gen, (n, 3), 0.0, 1.0), axis=1)
    spread = uniform(gen, (n,), 2.6, 3.4)
    # Above-ratio rows: one axis is lifted to (smallest other axis + spread), so it is the
    # maximum and s_max / s_min = exp(spread) > e^2.6 > 10. Other rows span at most 1.5.
    for i in range(n // 2):
        r = raise_axis[i]
        others = np.delete(base[i], r)
        base[i, r] = others.min() + spread[i]
    log_scales = to_tensor(base, (n, 3)).to(torch.float64).requires_grad_(True)

    loss = ref.anisotropy_loss(log_scales, lam, max_ratio)
    grad, = torch.autograd.grad(loss, [log_scales])
    ratio = torch.exp(log_scales.detach().max(dim=1).values - log_scales.detach().min(dim=1).values)
    above_count = int((ratio > max_ratio).sum())
    if above_count != n // 2:
        failures.append(f"[anisotropy] {above_count} Gaussians above the ratio, expected {n // 2}")
    print(f"[anisotropy] {n} Gaussians, {above_count} above ratio {max_ratio:g}, "
          f"loss {float(loss.detach()):.6g}, max |grad| {grad.abs().max().item():.6g}")
    return {
        "count": n,
        "weight": lam,
        "maxRatio": max_ratio,
        "logScales": flat(log_scales),
        "loss": float(loss.detach()),
        "gradLogScales": flat(grad),
    }


def phase4_outputs(failures: list) -> dict:
    outputs = {}
    for name, base in DEPTH_CASES.items():
        golden, cross, out = run_depth_case(name, base)
        stats = out["stats"]
        print(f"[{name}] {len(golden['positions']) // 3} Gaussians, "
              f"{golden['width']}x{golden['height']}, alive {stats['alive']}, "
              f"terminated pixels {stats['terminated_pixels']}")
        print("  gsplat depth cross-check max |diff|: " + ", ".join(
            f"{k}={v:.3g}" for k, v in cross.items()))
        for k, v in cross.items():
            if not v <= CROSS_CHECK_TOL:
                failures.append(f"[{name}] gsplat depth cross-check {k} diff {v:.3g} > {CROSS_CHECK_TOL}")
        outputs[name] = golden
    outputs["pearson"] = pearson_golden(failures)
    outputs["anisotropy"] = anisotropy_golden(failures)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", nargs="+", required=True, metavar="DIR",
                        help="one or more directories to write golden_<name>.json into")
    parser.add_argument("--only", choices=("phase2", "phase4", "all"), default="phase2",
                        help="phase2: the four Phase 2 cases (default, unchanged); "
                             "phase4: only the new Phase 4 cases; all: both")
    args = parser.parse_args()

    failures = []
    outputs = {}
    for name in (CASES if args.only in ("phase2", "all") else ()):
        _, cam, params, settings, gen = BUILDERS[name]()
        image, image_grad, grads, stats, cross = run_case(name, cam, params, settings, gen)

        print(f"[{name}] {params.positions.shape[0]} Gaussians, SH degree {params.sh_degree} "
              f"(limit {settings.sh_degree_limit}), {cam.width}x{cam.height}")
        print(f"  alive Gaussians: {stats['alive']}, Jacobian-clamped: {stats['jacobian_clamped']}")
        print(f"  clamped-alpha samples: {stats['clamped_alpha_samples']}, "
              f"clamped-alpha pixels: {stats['clamped_alpha_pixels']}, "
              f"terminated pixels: {stats['terminated_pixels']}")
        print("  max |grad|: " + ", ".join(
            f"{gname}={gt.abs().max().item():.6g}" for gname, gt in zip(GROUPS, grads)))
        print("  gsplat cross-check max |diff|: " + ", ".join(
            f"{k}={v:.3g}" for k, v in cross.items()))

        for k, v in cross.items():
            if not v <= CROSS_CHECK_TOL:
                failures.append(f"[{name}] gsplat cross-check {k} diff {v:.3g} > {CROSS_CHECK_TOL}")
        if name == "occluded":
            if stats["clamped_alpha_samples"] == 0:
                failures.append("[occluded] no clamped-alpha samples")
            if stats["terminated_pixels"] == 0:
                failures.append("[occluded] no early-terminated pixels")
        if name == "rotated" and stats["jacobian_clamped"] < 3:
            failures.append(f"[rotated] only {stats['jacobian_clamped']} Jacobian-clamped Gaussians (need >= 3)")

        outputs[name] = golden_dict(name, cam, params, settings, image, image_grad, grads)

    if args.only in ("phase4", "all"):
        outputs.update(phase4_outputs(failures))

    if failures:
        for msg in failures:
            print(f"FAILED: {msg}", file=sys.stderr)
        return 1

    for out_dir in args.out:
        os.makedirs(out_dir, exist_ok=True)
        for name, golden in outputs.items():
            text = json.dumps(golden, separators=(",", ":"))
            size = len(text.encode("utf-8"))
            if size >= MAX_FILE_BYTES:
                print(f"FAILED: {name} is {size} bytes (limit {MAX_FILE_BYTES})", file=sys.stderr)
                return 1
            path = os.path.join(out_dir, f"golden_{name}.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"wrote {path} ({size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
