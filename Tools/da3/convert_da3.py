#!/usr/bin/env python3
"""Converts Depth Anything 3 (ByteDance, Apache-2.0) to one multi-function Core ML package.

    python Tools/da3/convert_da3.py <base|small> <output.mlpackage> [view counts, default 4,6,8,12,16]

Needs (in a throw-away venv, not part of the repo): Python 3.13, torch 2.7, coremltools 9, einops, omegaconf,
addict, safetensors, huggingface_hub, pillow, numpy<2.3; the DA3 python sources (`pip download depth-anything-3
--no-deps --ignore-requires-python`, unzip the wheel) on DA3_SRC; the weights (`depth-anything/DA3-BASE` or
`DA3-SMALL` from Hugging Face, Apache-2.0) in DA3_WEIGHTS/<name>/model.safetensors.

One function per (view count N, shape): `v<N>_<H>x<W>`, shapes 504x378 (portrait) and 378x504 (landscape).
Input `images` (1, N, 3, H, W) ImageNet-normalised RGB; outputs `depth` (1, N, H, W), `confidence` (1, N, H, W)
and `pose` (1, N, 9) = [T(3), quaternion xyzw(4), fov_h, fov_w], a camera-to-world pose in the OpenCV convention.
Weights are quantised to int8 and shared between the functions. Export-time changes: bicubic upsampling becomes
bilinear, and the rotary-position grid and the DPT head's UV grid are built from constants (Core ML has no op for
them); the reference-view reordering is switched off (`ref_view_strategy="first"`).
"""
import os, sys, shutil, time
sys.path.insert(0, os.environ.get("DA3_SRC", "/tmp/da3_probe/src"))
import numpy as np, torch, coremltools as ct
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction
from omegaconf import OmegaConf
from safetensors.torch import load_file
from depth_anything_3.cfg import create_object, load_config

def build(name, weights):
    net = create_object(load_config(f"depth_anything_3.configs.da3-{name}"))
    sd = load_file(weights)
    net.load_state_dict({(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}, strict=False)
    return net.eval(), None

# --- export patches: Core ML has no bicubic / cartesian_prod / dynamic meshgrid ops
_orig_interpolate = torch.nn.functional.interpolate
def _interp(*a, **k):
    if k.get("mode") == "bicubic": k["mode"] = "bilinear"
    return _orig_interpolate(*a, **k)
torch.nn.functional.interpolate = _interp
import depth_anything_3.model.dinov2.layers.rope as _rope
def _positions(self, batch_size, height, width, device):
    h, w = int(height), int(width)
    pos = torch.from_numpy(np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing="ij"), -1).reshape(-1, 2))
    return pos.view(1, h * w, 2).expand(int(batch_size), -1, -1).clone()
_rope.PositionGetter.__call__ = _positions
import depth_anything_3.model.utils.head_utils as _hu
def _uv(width, height, aspect_ratio=None, dtype=None, device=None):
    width, height = int(width), int(height)
    ar = float(width) / float(height) if aspect_ratio is None else float(aspect_ratio)
    diag = (ar ** 2 + 1.0) ** 0.5
    sx, sy = ar / diag, 1.0 / diag
    xs = np.linspace(-sx * (width - 1) / width, sx * (width - 1) / width, width)
    ys = np.linspace(-sy * (height - 1) / height, sy * (height - 1) / height, height)
    uu, vv = np.meshgrid(xs, ys, indexing="xy")
    return torch.from_numpy(np.stack([uu, vv], -1)).to(dtype or torch.float32)
_hu.create_uv_grid = _uv
import depth_anything_3.model.dualdpt as _dd; _dd.create_uv_grid = _uv
import depth_anything_3.model.dpt as _dp; _dp.create_uv_grid = _uv

name = sys.argv[1]; out_path = sys.argv[2]
counts = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "4,6,8,12,16").split(",")]
shapes = [(504, 378), (378, 504)]
net, _ = build(name, f"{os.environ.get('DA3_WEIGHTS', '/tmp/da3_weights')}/{name}/model.safetensors")

class Wrapper(torch.nn.Module):
    def __init__(self, net, H, W):
        super().__init__(); self.net = net; self.H = H; self.W = W
    def forward(self, x):
        feats, _ = self.net.backbone(x, cam_token=None, export_feat_layers=[], ref_view_strategy="first")
        out = self.net.head(feats, self.H, self.W, patch_start_idx=0)
        return out["depth"], out["depth_conf"], self.net.cam_dec(feats[-1][1])

tmp = os.path.dirname(out_path) + "/_parts"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
desc = MultiFunctionDescriptor()
first = None
for (H, W) in shapes:
    for N in counts:
        t0 = time.time()
        w = Wrapper(net, H, W).eval(); x = torch.randn(1, N, 3, H, W)
        traced = torch.jit.trace(w, x, check_trace=False)
        ml = ct.convert(traced, inputs=[ct.TensorType(name="images", shape=(1, N, 3, H, W))],
                        outputs=[ct.TensorType(name="depth"), ct.TensorType(name="confidence"), ct.TensorType(name="pose")],
                        compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15, convert_to="mlprogram")
        import coremltools.optimize as cto
        qcfg = cto.coreml.OptimizationConfig(global_config=cto.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel"))
        ml = cto.coreml.linear_quantize_weights(ml, qcfg)
        part = f"{tmp}/v{N}_{H}x{W}.mlpackage"; ml.save(part)
        fname = f"v{N}_{H}x{W}"
        desc.add_function(part, "main", fname)
        first = first or fname
        print("converted", fname, "%.0fs" % (time.time() - t0), flush=True)
desc.default_function_name = first
shutil.rmtree(out_path, ignore_errors=True)
save_multifunction(desc, out_path)
size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out_path) for f in fs)
print("saved", out_path, "%.0f MB" % (size / 1e6))
