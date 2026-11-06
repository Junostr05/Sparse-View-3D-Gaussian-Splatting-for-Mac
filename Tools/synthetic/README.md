# Synthetic COLMAP scene

`make_colmap_scene.py` writes a small synthetic dataset in COLMAP binary format, so the app
can be tried without real photos. It is pure Python 3 standard library (no numpy, no PIL).

```sh
python3 Tools/synthetic/make_colmap_scene.py OUTDIR [--views 4] [--size 160x120] \
    [--points 400] [--seed 1] [--ppm]
```

Output layout (import `OUTDIR` with the app's COLMAP import):

```
OUTDIR/
  images/view_00.png ...      8-bit RGB PNG (view_NN.ppm too with --ppm)
  sparse/0/cameras.bin        one PINHOLE camera shared by all views
  sparse/0/images.bin         one world->camera pose per view
  sparse/0/points3D.bin       object points + grey ground points
```

The scene is a coloured sphere of radius 1 at the origin (`--points` surface points, hue by
longitude) over a few grey ground points. Cameras sit on an elevated 110-degree arc at radius
3.5 and look at the origin. Intrinsics are `fx = fy = 1.3 * width` with the principal point at
the image centre. Conventions are OpenCV (x right, y down, z forward), with `p_cam = R p + t`.

Verify an existing dataset without regenerating it:

```sh
python3 Tools/synthetic/make_colmap_scene.py OUTDIR --verify
```

Generation runs the same check at the end. It re-reads the `.bin` files, checks that most
points land inside each image, and checks that the object's pixel centroid is within 2 px of
the projected origin. It exits non-zero on failure.
