# Synthetic COLMAP scene

`make_colmap_scene.py` writes a small synthetic dataset in COLMAP binary format, so the app
can be tried without real photos. It is pure Python 3 standard library (no numpy, no PIL).

```sh
python3 Tools/synthetic/make_colmap_scene.py OUTDIR [--views 4] [--size 160x120] \
    [--points 400] [--seed 1] [--ppm] \
    [--depth | --relative-depth] [--novel-views N]
```

Output layout (import `OUTDIR` with the app's COLMAP import):

```
OUTDIR/
  images/view_00.png ...      8-bit RGB PNG (plus view_NN.ppm with --ppm)
  sparse/0/cameras.bin        one PINHOLE camera shared by all views
  sparse/0/images.bin         one world->camera pose per view
  sparse/0/points3D.bin       object points + grey ground points
  depth/view_00.png.sdpt ...  only with --depth or --relative-depth (per-view depth)
  novel/images/novel_00.png   only with --novel-views N
  novel/sparse/0/...          COLMAP files for the novel views, empty points3D.bin
```

The scene is a coloured sphere of radius 1 at the origin (`--points` surface points, hue by
longitude) over a few grey ground points. Cameras sit on an elevated 110-degree arc at radius
3.5 and look at the origin. Intrinsics are `fx = fy = 1.3 * width` with the principal point at
the image centre. Conventions are OpenCV (x right, y down, z forward), with `p_cam = R p + t`.

### Depth maps (`--depth`, `--relative-depth`)

Each training view gets `depth/<image name>.sdpt` in the app's depth format (Phase 4 P2: a
`"SDPT"` header, then row-major little-endian Float32 values, where `<= 0` marks an invalid pixel).
The file name keeps the image's extension, so `view_00.png` gets `depth/view_00.png.sdpt`, the
same path the app's project uses.

- `--depth` writes **metric** depth (kind 0): the camera-space z of the front-most splat disc
  whose coverage at the pixel is at least 0.5. Pixels that no such disc covers are invalid.
- `--relative-depth` writes **relative inverse depth** (kind 2): `value = a / z + b` per view,
  with `a` in [0.5, 2] and `b` in [0, 0.5], drawn from a seeded stream (`seed + 1`). The script
  prints `a` and `b` for each view so that alignment can be checked by hand.

The two options are mutually exclusive. Neither changes the RGB images or the COLMAP files.

### Novel views (`--novel-views N`)

`novel/` holds ground-truth images and poses for held-out viewpoints, in the same COLMAP layout
as the training set, so it loads with the app's COLMAP loader. `points3D.bin` is empty. The
first views sit at azimuths halfway between consecutive training cameras. After those come the
quarter points (1/4 and 3/4 of each gap), and that list repeats if `N` is still larger. This
needs at least 2 training views.

This is the input for held-out evaluation (Phase 4, Task 21 benchmark).

### Verification

Verify an existing dataset without regenerating it:

```sh
python3 Tools/synthetic/make_colmap_scene.py OUTDIR --verify
```

Generation runs the same check at the end. It re-reads the `.bin` files, checks that most
points land inside each image, and checks that the object's pixel centroid is within 2 px of
the projected origin. It exits non-zero on failure. When the extra outputs exist, it also checks:

- Each `depth/*.sdpt` decodes, matches its image's size, and has valid pixels. For metric maps
  the sphere-centre pixel must lie on a sphere crossing: 2.5 (front) or 4.5 (back) within 0.1.
  The back crossing is accepted because the splat z-buffer can leave a hole in the front cap over
  the centre pixel, and then the back surface is the correct front-most covered disc.
- `novel/` re-reads, has an empty `points3D.bin`, and has one PNG per view at the camera size.

### Swift fixtures

`SplatKit/Tests/SplatIOTests/Fixtures/` holds two `.sdpt` files from this tool: view 00 of a
64x48 scene with two training views and seed 1, once with `--depth` and once with
`--relative-depth`. `SyntheticDepthFixtureTests` decodes them with `DepthMapCodec` and checks the
header, dimensions, valid-pixel count and value ranges. To regenerate them:

```sh
python3 Tools/synthetic/make_colmap_scene.py OUT --views 2 --size 64x48 --depth
cp OUT/depth/view_00.png.sdpt SplatKit/Tests/SplatIOTests/Fixtures/synthetic_depth_metric_view_00.sdpt
python3 Tools/synthetic/make_colmap_scene.py OUT2 --views 2 --size 64x48 --relative-depth
cp OUT2/depth/view_00.png.sdpt SplatKit/Tests/SplatIOTests/Fixtures/synthetic_depth_relative_view_00.sdpt
```

Each fixture is 12,312 bytes. Keep fixtures this small.
