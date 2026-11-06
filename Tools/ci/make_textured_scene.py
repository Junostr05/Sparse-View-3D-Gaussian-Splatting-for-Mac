#!/usr/bin/env python3
"""Renders a textured synthetic "room corner" (floor + two walls with multi-scale noise textures)
from an arc of cameras. Unlike Tools/synthetic/make_colmap_scene.py (smooth coloured sphere), this
scene has plenty of SIFT features, so it exercises a real COLMAP reconstruction.

Usage: make_textured_scene.py OUT_DIR [VIEWS]   (needs numpy and pillow; writes OUT_DIR/images/*.png)
"""
import os
import sys

import numpy as np
from PIL import Image

W, H, F = 800, 600, 700.0


def make_texture(rng, size=1024):
    tex = np.zeros((size, size, 3), np.float32)
    amp = 1.0
    for cells in (6, 12, 24, 48, 96, 192):
        small = rng.random((cells, cells, 3)).astype(np.float32)
        img = Image.fromarray((small * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
        tex += amp * np.asarray(img, np.float32) / 255.0
        amp *= 0.8
    tex -= tex.min()
    tex /= tex.max()
    return tex


def look_at(center, target):
    z = target - center
    z /= np.linalg.norm(z)
    down = np.array([0.0, -1.0, 0.0])
    y = down - np.dot(down, z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.stack([x, y, z])  # rows: camera axes in world coordinates (world -> camera)


def render(center, rot, textures):
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dirs_cam = np.stack([(u - W / 2) / F, (v - H / 2) / F, np.ones_like(u)], axis=-1)
    dirs = dirs_cam @ rot  # camera -> world (rot is orthonormal)
    best_t = np.full((H, W), np.inf)
    out = np.zeros((H, W, 3), np.float32)
    extent = 3.0
    # planes: (axis fixed at 0, texture index, the two free axes)
    planes = [(1, 0, (0, 2)), (2, 1, (0, 1)), (0, 2, (2, 1))]
    for axis, ti, (a, b) in planes:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -center[axis] / dirs[..., axis]
        pts = center + dirs * t[..., None]
        ok = (t > 0) & (pts[..., a] >= 0) & (pts[..., a] < extent) & (pts[..., b] >= 0) & (pts[..., b] < extent) & (t < best_t)
        if not ok.any():
            continue
        tex = textures[ti]
        n = tex.shape[0]
        iu = np.clip((pts[..., a] / extent * n).astype(int), 0, n - 1)
        iv = np.clip((pts[..., b] / extent * n).astype(int), 0, n - 1)
        out[ok] = tex[iv[ok], iu[ok]]
        best_t[ok] = t[ok]
    return out


def main():
    out_dir = sys.argv[1]
    views = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    rng = np.random.default_rng(7)
    textures = [make_texture(rng) for _ in range(3)]
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    target = np.array([0.9, 0.5, 0.9])
    for i in range(views):
        az = np.radians(25 + 40 * i / max(1, views - 1))  # 25..65 degrees around the corner diagonal
        center = np.array([0.9 + 2.2 * np.sin(az) + 0.4, 1.0, 0.9 + 2.2 * np.cos(az) + 0.4])
        rot = look_at(center, target)
        img = render(center, rot, textures)
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(os.path.join(out_dir, "images", "view_%02d.png" % i))
    print("wrote %d views to %s" % (views, out_dir))


if __name__ == "__main__":
    main()
