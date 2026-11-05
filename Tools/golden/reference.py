"""PyTorch re-implementation of the Phase 1 forward pass (plan section "Shared math").

Reference only. Mirrors SplatKit/Sources/SplatCore/{Projection,CPURasterizer,
SphericalHarmonics,Matrix3}.swift and produces the golden images and gradients.
Never imported by Swift and never shipped.

Gradient conventions (plan "Global Constraints", identical in CPU, Metal and goldens):
- alpha = min(0.999, o*exp(-sigma)): gradients to o and sigma are 0 where the clamp is active.
- color = max(SH + 0.5, 0): gradient 0 per channel where the clamp is active.
- Jacobian clamp of x/z, y/z: the clamped coordinate has gradient 0 (mean2D is unclamped).
- Gating (culling, tile rect, sigma < 0, alpha < 1/255, early termination) is constant.
- SH bands above min(shDegree, shDegreeLimit) get gradient 0; background gets none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

TILE_SIZE = 16
NEAR_PLANE = 0.01
FAR_PLANE = 1e10
EPS_2D = 0.3
ALPHA_CLAMP = 0.999
ALPHA_MIN = 1.0 / 255.0
TRANSMITTANCE_MIN = 1e-4

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)

F32 = torch.float32


@dataclass
class Camera:
    """World-to-camera pinhole camera (OpenCV convention)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: torch.Tensor  # [3, 3], row-major, world -> camera
    translation: torch.Tensor  # [3]


@dataclass
class Settings:
    sh_degree_limit: int
    background: torch.Tensor  # [3]


@dataclass
class Params:
    """Gaussian parameters. Layouts match GaussianCloud in Swift."""

    positions: torch.Tensor  # [N, 3]
    log_scales: torch.Tensor  # [N, 3]
    rotations: torch.Tensor  # [N, 4], (w, x, y, z), raw (normalized inside)
    opacity_logits: torch.Tensor  # [N]
    sh: torch.Tensor  # [N, K, 3], K = (sh_degree + 1)^2, layout [gaussian][coefficient][channel]
    sh_degree: int

    def tensors(self):
        return [self.positions, self.log_scales, self.rotations, self.opacity_logits, self.sh]

    def select(self, idx: torch.Tensor) -> "Params":
        return Params(
            positions=self.positions.index_select(0, idx),
            log_scales=self.log_scales.index_select(0, idx),
            rotations=self.rotations.index_select(0, idx),
            opacity_logits=self.opacity_logits.index_select(0, idx),
            sh=self.sh.index_select(0, idx),
            sh_degree=self.sh_degree,
        )


def rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Matrix3.rotation(quaternion:) for [N, 4] (w, x, y, z), returned row-major [N, 3, 3].

    The quaternion is normalized; a zero quaternion yields the identity.
    """
    n2 = (q * q).sum(-1, keepdim=True)
    valid = n2 > 0
    safe_n2 = torch.where(valid, n2, torch.ones_like(n2))
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=F32)
    qn = torch.where(valid, q / safe_n2.sqrt(), identity)
    w, x, y, z = qn.unbind(-1)
    row0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1)
    row1 = torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1)
    row2 = torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
    return torch.stack([row0, row1, row2], -2)


def sh_basis(d: torch.Tensor, degree: int) -> torch.Tensor:
    """Real SH basis [N, (degree + 1)^2] for unit directions d [N, 3] (Phase 1 ordering)."""
    x, y, z = d.unbind(-1)
    terms = [torch.full_like(x, SH_C0)]
    if degree >= 1:
        terms += [-SH_C1 * y, SH_C1 * z, -SH_C1 * x]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        terms += [
            SH_C2[0] * xy,
            SH_C2[1] * yz,
            SH_C2[2] * (2 * zz - xx - yy),
            SH_C2[3] * xz,
            SH_C2[4] * (xx - yy),
        ]
    if degree >= 3:
        xx, yy, zz = x * x, y * y, z * z
        xy, xz = x * y, x * z
        terms += [
            SH_C3[0] * y * (3 * xx - yy),
            SH_C3[1] * xy * z,
            SH_C3[2] * y * (4 * zz - xx - yy),
            SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
            SH_C3[4] * x * (4 * zz - xx - yy),
            SH_C3[5] * z * (xx - yy),
            SH_C3[6] * x * (xx - 3 * yy),
        ]
    return torch.stack(terms, -1)


def _geometry(p: Params, cam: Camera, settings: Settings) -> Dict[str, torch.Tensor]:
    """Per-Gaussian projection and color. Differentiable; culling is applied by the caller."""
    R = cam.rotation
    t = cam.translation
    fx = torch.tensor(cam.fx, dtype=F32)
    fy = torch.tensor(cam.fy, dtype=F32)
    cx = torch.tensor(cam.cx, dtype=F32)
    cy = torch.tensor(cam.cy, dtype=F32)
    W = torch.tensor(float(cam.width), dtype=F32)
    H = torch.tensor(float(cam.height), dtype=F32)
    eps = torch.tensor(EPS_2D, dtype=F32)

    # World -> camera.
    pc = p.positions @ R.T + t
    x, y, z = pc.unbind(-1)

    # Σ3D = M Mᵀ with M = R_q · diag(scale).
    rq = rotation_matrix(p.rotations)
    scale = torch.exp(p.log_scales)
    M = rq * scale.unsqueeze(-2)
    cov3d = M @ M.transpose(-1, -2)

    # Screen-space mean (unclamped).
    mean_x = fx * x / z + cx
    mean_y = fy * y / z + cy

    # Jacobian with clamped x/z, y/z (limits at 1.3x the field of view).
    tan_fov_x = 0.5 * W / fx
    tan_fov_y = 0.5 * H / fy
    lim_x_pos = (W - cx) / fx + 0.3 * tan_fov_x
    lim_x_neg = cx / fx + 0.3 * tan_fov_x
    lim_y_pos = (H - cy) / fy + 0.3 * tan_fov_y
    lim_y_neg = cy / fy + 0.3 * tan_fov_y
    x_over_z = x / z
    y_over_z = y / z
    jac_x_clamped = (x_over_z < -lim_x_neg) | (x_over_z > lim_x_pos)
    jac_y_clamped = (y_over_z < -lim_y_neg) | (y_over_z > lim_y_pos)
    tx = z * torch.clamp(x_over_z, min=-lim_x_neg, max=lim_x_pos)
    ty = z * torch.clamp(y_over_z, min=-lim_y_neg, max=lim_y_pos)

    zero = torch.zeros_like(z)
    j0 = torch.stack([fx / z, zero, -fx * tx / (z * z)], -1)
    j1 = torch.stack([zero, fy / z, -fy * ty / (z * z)], -1)
    J = torch.stack([j0, j1], -2)  # [N, 2, 3]

    # Σ2D = (J R) Σ3D (J R)ᵀ + eps2D on the diagonal.
    T = J @ R
    S2 = T @ cov3d @ T.transpose(-1, -2)
    a = S2[..., 0, 0] + eps
    b = S2[..., 0, 1]
    c = S2[..., 1, 1] + eps
    det = a * c - b * b
    conic = torch.stack([c / det, -b / det, a / det], -1)

    # Color: SH of the view direction from the camera center, then max(. + 0.5, 0).
    center = -(R.T @ t)
    offset = p.positions - center
    dirs = offset / torch.sqrt((offset * offset).sum(-1, keepdim=True))
    degree = min(p.sh_degree, settings.sh_degree_limit)
    k_used = (degree + 1) * (degree + 1)
    basis = sh_basis(dirs, degree)  # [N, k_used]
    raw = (basis.unsqueeze(-1) * p.sh[:, :k_used, :]).sum(1)  # [N, 3]
    color = torch.clamp(raw + 0.5, min=0.0)

    opacity = torch.sigmoid(p.opacity_logits)

    return {
        "z": z,
        "mean2d": torch.stack([mean_x, mean_y], -1),
        "a": a,
        "b": b,
        "c": c,
        "det": det,
        "conic": conic,
        "cov3d": cov3d,
        "raw_sh": raw,
        "color": color,
        "opacity": opacity,
        "jac_x": jac_x_clamped,
        "jac_y": jac_y_clamped,
    }


def _cull(g: Dict[str, torch.Tensor], cam: Camera) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Alive mask, inclusive tile min and exclusive tile max. Runs on detached values."""
    ts = float(TILE_SIZE)
    tiles_x = float((cam.width + TILE_SIZE - 1) // TILE_SIZE)
    tiles_y = float((cam.height + TILE_SIZE - 1) // TILE_SIZE)
    a, b, c, det, z = (g[k].detach() for k in ("a", "b", "c", "det", "z"))
    mx, my = g["mean2d"].detach().unbind(-1)

    mid = 0.5 * (a + c)
    lam = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
    radius = torch.ceil(3 * torch.sqrt(lam))

    min_x = torch.clamp((mx - radius) / ts, min=0.0, max=tiles_x)
    max_x = torch.clamp((mx + radius + (ts - 1)) / ts, min=0.0, max=tiles_x)
    min_y = torch.clamp((my - radius) / ts, min=0.0, max=tiles_y)
    max_y = torch.clamp((my + radius + (ts - 1)) / ts, min=0.0, max=tiles_y)
    tile_min = torch.stack([torch.trunc(min_x), torch.trunc(min_y)], -1).long()
    tile_max = torch.stack([torch.trunc(max_x), torch.trunc(max_y)], -1).long()

    alive = (
        (z >= NEAR_PLANE)
        & (z <= FAR_PLANE)
        & (det > 0)
        & (radius > 0)
        & (tile_min[:, 0] < tile_max[:, 0])
        & (tile_min[:, 1] < tile_max[:, 1])
    )
    return alive, tile_min, tile_max


def forward_with_stats(p: Params, cam: Camera, settings: Settings):
    """Forward pass. Returns (image [H, W, 3], stats dict). Differentiable w.r.t. Params."""
    H, W = cam.height, cam.width
    bg = settings.background.to(F32)

    # Pass 1 (constant gating): culling and tile rects on detached values.
    with torch.no_grad():
        full = _geometry(p, cam, settings)
        alive, tile_min_full, tile_max_full = _cull(full, cam)
        alive_idx = torch.nonzero(alive).squeeze(1)
        jac_clamped_count = int((full["jac_x"] | full["jac_y"])[alive_idx].sum())

    stats = {
        "alive": int(alive_idx.numel()),
        "jacobian_clamped": jac_clamped_count,
        "clamped_alpha_samples": 0,
        "clamped_alpha_pixels": 0,
        "terminated_pixels": 0,
    }

    if alive_idx.numel() == 0:
        image = bg.expand(H, W, 3).clone()
        return image, stats

    # Pass 2: differentiable geometry of the surviving Gaussians.
    g = _geometry(p.select(alive_idx), cam, settings)
    tile_min = tile_min_full[alive_idx]
    tile_max = tile_max_full[alive_idx]

    # Front-to-back order: ascending depth, ties by ascending index (stable).
    order = torch.argsort(g["z"].detach(), stable=True)
    mean2d = g["mean2d"][order]
    conic = g["conic"][order]
    color = g["color"][order]
    opacity = g["opacity"][order]
    tile_min = tile_min[order]
    tile_max = tile_max[order]

    # Pixel grid, row-major, centers at (x + 0.5, y + 0.5).
    ys = torch.arange(H).repeat_interleave(W)
    xs = torch.arange(W).repeat(H)
    px = xs.to(F32) + 0.5
    py = ys.to(F32) + 0.5
    ptx = xs // TILE_SIZE
    pty = ys // TILE_SIZE

    # σ = 0.5 (a dx² + c dy²) + b dx dy with d = mean2D - p; [pixels, gaussians].
    dx = mean2d[:, 0].unsqueeze(0) - px.unsqueeze(1)
    dy = mean2d[:, 1].unsqueeze(0) - py.unsqueeze(1)
    sigma = 0.5 * (conic[:, 0] * dx * dx + conic[:, 2] * dy * dy) + conic[:, 1] * dx * dy

    with torch.no_grad():
        in_tile = (
            (ptx.unsqueeze(1) >= tile_min[:, 0].unsqueeze(0))
            & (ptx.unsqueeze(1) < tile_max[:, 0].unsqueeze(0))
            & (pty.unsqueeze(1) >= tile_min[:, 1].unsqueeze(0))
            & (pty.unsqueeze(1) < tile_max[:, 1].unsqueeze(0))
        )
        pre = in_tile & (sigma.detach() >= 0)

    # Double-where: keep invalid pairs finite so no NaN reaches the backward pass.
    sigma_s = torch.where(pre, sigma, torch.zeros_like(sigma))
    alpha_raw = opacity.unsqueeze(0) * torch.exp(-sigma_s)  # o * exp(-sigma), before the clamp

    with torch.no_grad():
        alpha_raw_d = alpha_raw.detach()
        keep = pre & (alpha_raw_d >= ALPHA_MIN)  # min(0.999, v) >= 1/255  <=>  v >= 1/255
        clamped = alpha_raw_d > ALPHA_CLAMP
        alpha_d = torch.clamp(alpha_raw_d, max=ALPHA_CLAMP)
        ones = torch.ones_like(alpha_d)
        fac_d = torch.where(keep, 1 - alpha_d, ones)
        cp_d = torch.cumprod(fac_d, dim=1)
        t_before_d = torch.cat([ones[:, :1], cp_d[:, :-1]], dim=1)
        next_t_d = t_before_d * (1 - alpha_d)
        # First kept sample whose nextT <= 1e-4 terminates the pixel; it and all later
        # samples are excluded.
        term = keep & (next_t_d <= TRANSMITTANCE_MIN)
        use = keep & (torch.cumsum(term.long(), dim=1) == 0)

        stats["clamped_alpha_samples"] = int((use & clamped).sum())
        stats["clamped_alpha_pixels"] = int((use & clamped).any(dim=1).sum())
        stats["terminated_pixels"] = int(term.any(dim=1).sum())

    # Differentiable transmittance over the used samples only.
    alpha = torch.clamp(alpha_raw, max=ALPHA_CLAMP)
    fac = torch.where(use, 1 - alpha, ones)
    cp = torch.cumprod(fac, dim=1)
    t_before = torch.cat([ones[:, :1], cp[:, :-1]], dim=1)
    weights = torch.where(use, alpha * t_before, torch.zeros_like(alpha))
    rgb = weights @ color  # [pixels, 3]
    t_final = cp[:, -1]
    image = (rgb + t_final.unsqueeze(1) * bg.unsqueeze(0)).reshape(H, W, 3)
    return image, stats


def forward(p: Params, cam: Camera, settings: Settings) -> torch.Tensor:
    image, _ = forward_with_stats(p, cam, settings)
    return image


def geometry_for_check(p: Params, cam: Camera, settings: Settings) -> Dict[str, torch.Tensor]:
    """Per-Gaussian geometry for every Gaussian (no culling), for cross-checks."""
    with torch.no_grad():
        return _geometry(p, cam, settings)
