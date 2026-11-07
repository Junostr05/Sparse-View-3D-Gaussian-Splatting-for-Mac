#!/usr/bin/env python3
"""Prepares the Hugging Face model repository for the converted Depth Anything 3 package.

    python3 Tools/da3/prepare_hf_repo.py <DepthAnything3BaseQ8.mlpackage> <output-dir> <hf-user/repo-name>

Copies the package into <output-dir>, writes the model card (README.md, with the Hugging Face metadata header)
and prints the `files:` array for `ModelDescriptor.depthAnything3Base` (relative path, SHA-256, size) plus the
upload command. Nothing is uploaded: publishing is done by the repository owner with their own account.
"""
import hashlib, os, shutil, sys

if len(sys.argv) != 4:
    sys.exit(__doc__)
package, out_dir, repo_id = sys.argv[1], sys.argv[2], sys.argv[3]
name = os.path.basename(package.rstrip("/"))
dest = os.path.join(out_dir, name)
if os.path.exists(dest):
    shutil.rmtree(dest)
os.makedirs(out_dir, exist_ok=True)
shutil.copytree(package, dest)

CARD = f"""---
license: apache-2.0
library_name: coreml
pipeline_tag: depth-estimation
base_model: depth-anything/DA3-BASE
tags:
  - coreml
  - depth-estimation
  - pose-estimation
  - gaussian-splatting
  - depth-anything-3
---

# Depth Anything 3 Base (Core ML, int8, pose-free)

A Core ML conversion of [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) Base
(`depth-anything/DA3-BASE`, ByteDance, Apache-2.0) for the macOS app *SparseSplat*. It estimates depth,
confidence and a camera pose for a set of photos **without a calibrated camera** and is used as a fallback when
Apple Photogrammetry cannot register the photos. It is less accurate than Apple Photogrammetry (about 0.8 dB
lower PSNR on the reference scene) and is not meant to replace it.

## Package

One multi-function `.mlpackage` (`{name}`), weights quantised to int8 and shared between the functions
(about 583 MB). One function per view count N in 4, 6, 8, 12, 16 and per shape, named `v<N>_<H>x<W>`;
shapes 504x378 (portrait) and 378x504 (landscape).

| | |
|---|---|
| Input `images` | `(1, N, 3, H, W)`, ImageNet-normalised RGB |
| Output `depth` | `(1, N, H, W)` |
| Output `confidence` | `(1, N, H, W)` |
| Output `pose` | `(1, N, 9)` = translation (3), quaternion xyzw (4), fov_h, fov_w; camera-to-world, OpenCV convention |

Export-time changes against the PyTorch model: bicubic upsampling became bilinear, the rotary-position grid and
the DPT head's UV grid are built from constants, and the reference-view reordering is switched off
(`ref_view_strategy="first"`). The conversion script is `Tools/da3/convert_da3.py` in the SparseSplat repository.

## Use

In SparseSplat: *Pose-Free Model, Import Package...* and pick the downloaded `{name}`. Or download it with

```bash
huggingface-cli download {repo_id} --local-dir ./{os.path.basename(repo_id)}
```

## Licence

The original weights and code are Apache-2.0 (ByteDance). This conversion is released under the same licence.
"""
open(os.path.join(out_dir, "README.md"), "w").write(CARD)

print("// ModelDescriptor.depthAnything3Base files:")
print("files: [")
for root, _, names in sorted(os.walk(dest)):
    for n in sorted(names):
        path = os.path.join(root, n)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        rel = os.path.relpath(path, dest).replace(os.sep, "/")
        print(f'    ModelFile(relativePath: "{rel}", sha256: "{h.hexdigest()}", sizeBytes: {os.path.getsize(path)}),')
print("]")
print(f"\n# Upload (owner account):\nhuggingface-cli upload {repo_id} {out_dir} . --repo-type model")
