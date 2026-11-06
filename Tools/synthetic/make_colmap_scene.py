#!/usr/bin/env python3
"""Synthetic COLMAP scene generator for trying the app without real photos.

Pure Python 3 standard library. Writes:

    OUTDIR/images/view_NN.png      8-bit RGB PNG (plus view_NN.ppm with --ppm)
    OUTDIR/sparse/0/cameras.bin    one PINHOLE camera shared by all views
    OUTDIR/sparse/0/images.bin     one world->camera pose per view
    OUTDIR/sparse/0/points3D.bin   scene points (coloured sphere + grey ground)

Conventions match the app's renderer: OpenCV camera (x right, y down, z forward),
p_cam = R * p_world + t, u = fx * x / z + cx, v = fy * y / z + cy. Pixel (i, j) has its
centre at (i + 0.5, j + 0.5). World up is +Y and the object is centred on the origin.

Usage:
    python3 Tools/synthetic/make_colmap_scene.py OUTDIR [--views 4] [--size 160x120]
                                                 [--points 400] [--seed 1] [--ppm]
    python3 Tools/synthetic/make_colmap_scene.py OUTDIR --verify
"""

import argparse
import colorsys
import math
import os
import random
import struct
import sys
import zlib

OBJECT_RADIUS = 1.0
GROUND_Y = -1.05
GROUND_RADIUS = 2.5
GROUND_COLOUR = (110, 104, 98)
CAMERA_DISTANCE = 3.5
CAMERA_ELEVATION = math.radians(15.0)
ARC_SPAN = math.radians(110.0)
FOCAL_SCALE = 1.3            # fx = fy = FOCAL_SCALE * width
SPLAT_K = 0.12               # splat radius in world units: radius_px = SPLAT_K * fx / z
BACKGROUND = (20.0, 20.0, 25.0)

SATURATION_THRESHOLD = 40    # max - min channel; separates coloured object pixels from grey ground
INSIDE_MIN_FRACTION = 0.8
CENTROID_TOLERANCE_PX = 2.0

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# MARK: - Vector helpers

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _normalize(a):
    return _scale(a, 1.0 / math.sqrt(_dot(a, a)))


# MARK: - Poses

class Pose(object):
    """World->camera pose. `rotation` rows are the camera axes (right, down, forward)."""

    def __init__(self, rotation, translation, centre=None):
        self.rotation = rotation
        self.translation = translation
        self.centre = centre


def look_at(centre):
    """Pose of a camera at `centre` looking at the origin, with +Y as world up."""
    forward = _normalize(_scale(centre, -1.0))
    up_negative = (0.0, -1.0, 0.0)  # camera y axis points down in the image
    down = _normalize(_sub(up_negative, _scale(forward, _dot(up_negative, forward))))
    right = _cross(down, forward)   # x = y x z keeps the frame right-handed
    rotation = (right, down, forward)
    translation = tuple(-_dot(row, centre) for row in rotation)
    return Pose(rotation, translation, centre)


def camera_poses(count):
    """`count` cameras on an elevated arc around the origin, spanning ARC_SPAN."""
    if count == 1:
        azimuths = [0.0]
    else:
        azimuths = [-ARC_SPAN / 2.0 + i * ARC_SPAN / (count - 1) for i in range(count)]
    poses = []
    for azimuth in azimuths:
        horizontal = CAMERA_DISTANCE * math.cos(CAMERA_ELEVATION)
        centre = (horizontal * math.sin(azimuth),
                  CAMERA_DISTANCE * math.sin(CAMERA_ELEVATION),
                  horizontal * math.cos(azimuth))
        poses.append(look_at(centre))
    return poses


def to_camera(pose, p):
    r = pose.rotation
    t = pose.translation
    return (_dot(r[0], p) + t[0], _dot(r[1], p) + t[1], _dot(r[2], p) + t[2])


def project(pose, intrinsics, p):
    """Returns continuous pixel coordinates (u, v) and camera-space depth z."""
    x, y, z = to_camera(pose, p)
    fx, fy, cx, cy = intrinsics
    return fx * x / z + cx, fy * y / z + cy, z


def rotation_to_quaternion(r):
    """World->camera rotation matrix to unit quaternion (qw, qx, qy, qz) with qw >= 0."""
    m00, m01, m02 = r[0]
    m10, m11, m12 = r[1]
    m20, m21, m22 = r[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    q = (w / n, x / n, y / n, z / n)
    if q[0] < 0.0:
        q = tuple(-c for c in q)
    return q


def quaternion_to_rotation(q):
    """Unit quaternion (qw, qx, qy, qz) to a world->camera rotation matrix."""
    n = math.sqrt(sum(c * c for c in q))
    w, x, y, z = (c / n for c in q)
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


# MARK: - Scene

def _colour(hue, value):
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, value)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def sphere_points(rng, count):
    """Points uniform on the surface of a sphere of radius OBJECT_RADIUS at the origin.
    Hue follows longitude and brightness follows latitude, so every side looks different.
    Each point is (x, y, z, r, g, b)."""
    points = []
    while len(points) < count:
        x, y, z = rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)
        length = math.sqrt(x * x + y * y + z * z)
        if length < 1e-6:
            continue
        x, y, z = (OBJECT_RADIUS * x / length, OBJECT_RADIUS * y / length,
                   OBJECT_RADIUS * z / length)
        hue = (math.atan2(z, x) / (2.0 * math.pi)) % 1.0
        value = 0.75 + 0.25 * (y / OBJECT_RADIUS + 1.0) / 2.0
        points.append((x, y, z) + _colour(hue, value))
    return points


def ground_points(rng, count):
    points = []
    for _ in range(count):
        radius = GROUND_RADIUS * math.sqrt(rng.random())
        angle = rng.uniform(0.0, 2.0 * math.pi)
        jitter = rng.randint(-6, 6)
        rgb = tuple(max(0, min(255, c + jitter)) for c in GROUND_COLOUR)
        points.append((radius * math.cos(angle), GROUND_Y, radius * math.sin(angle)) + rgb)
    return points


# MARK: - Rendering

def render(scene, pose, intrinsics, width, height):
    """Splats every point as an isotropic disc, back to front, over BACKGROUND.
    Returns the RGB raster as bytes (row-major, 3 bytes per pixel)."""
    fx, fy, cx, cy = intrinsics
    splats = []
    for (px, py, pz, r, g, b) in scene:
        xc, yc, zc = to_camera(pose, (px, py, pz))
        if zc <= 0.05:
            continue
        u = fx * xc / zc + cx
        v = fy * yc / zc + cy
        radius = SPLAT_K * fx / zc
        if u + radius < 0 or u - radius > width or v + radius < 0 or v - radius > height:
            continue
        splats.append((zc, u, v, radius, r, g, b))
    splats.sort(key=lambda s: -s[0])

    canvas = list(BACKGROUND) * (width * height)
    for _, u, v, radius, r, g, b in splats:
        x0 = max(0, int(math.floor(u - radius - 0.5)))
        x1 = min(width - 1, int(math.ceil(u + radius + 0.5)))
        y0 = max(0, int(math.floor(v - radius - 0.5)))
        y1 = min(height - 1, int(math.ceil(v + radius + 0.5)))
        for j in range(y0, y1 + 1):
            dy = j + 0.5 - v
            for i in range(x0, x1 + 1):
                dx = i + 0.5 - u
                coverage = radius + 0.5 - math.hypot(dx, dy)  # one pixel soft edge
                if coverage <= 0.0:
                    continue
                a = min(1.0, coverage)
                k = 3 * (j * width + i)
                canvas[k] = canvas[k] * (1.0 - a) + r * a
                canvas[k + 1] = canvas[k + 1] * (1.0 - a) + g * a
                canvas[k + 2] = canvas[k + 2] * (1.0 - a) + b * a
    return bytes(min(255, max(0, int(c + 0.5))) for c in canvas)


def png_bytes(width, height, rgb):
    stride = width * 3
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type None
        raw.extend(rgb[y * stride:(y + 1) * stride])

    def chunk(tag, body):
        return (struct.pack(">I", len(body)) + tag + body
                + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (PNG_SIGNATURE + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))


def read_png(path):
    """Reads an 8-bit RGB PNG written by png_bytes. Checks every chunk CRC; only filter type 0
    (what this tool writes) is accepted. Returns (width, height, rgb bytes)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != PNG_SIGNATURE:
        raise ValueError("not a PNG: " + path)
    offset = 8
    width = height = None
    idat = []
    while True:
        (length,) = struct.unpack_from(">I", data, offset)
        tag = data[offset + 4:offset + 8]
        body = data[offset + 8:offset + 8 + length]
        (crc,) = struct.unpack_from(">I", data, offset + 8 + length)
        if zlib.crc32(tag + body) & 0xFFFFFFFF != crc:
            raise ValueError("bad CRC in " + path)
        if tag == b"IHDR":
            width, height, depth, colour, _, _, _ = struct.unpack(">IIBBBBB", body)
            if depth != 8 or colour != 2:
                raise ValueError("expected 8-bit RGB PNG: " + path)
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
        offset += 12 + length
    if width is None:
        raise ValueError("missing IHDR: " + path)
    raw = zlib.decompress(b"".join(idat))
    stride = width * 3
    if len(raw) != height * (stride + 1):
        raise ValueError("unexpected PNG payload size: " + path)
    pixels = bytearray()
    for y in range(height):
        row = y * (stride + 1)
        if raw[row] != 0:
            raise ValueError("unsupported PNG filter in " + path)
        pixels.extend(raw[row + 1:row + 1 + stride])
    return width, height, bytes(pixels)


def ppm_bytes(width, height, rgb):
    return b"P6\n%d %d\n255\n" % (width, height) + rgb


# MARK: - COLMAP binary writer and reader

def write_colmap(sparse_dir, poses, names, width, height, intrinsics, scene):
    fx, fy, cx, cy = intrinsics
    os.makedirs(sparse_dir, exist_ok=True)
    with open(os.path.join(sparse_dir, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<IiQQdddd", 1, 1, width, height, fx, fy, cx, cy))  # PINHOLE
    with open(os.path.join(sparse_dir, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(poses)))
        for index, (pose, name) in enumerate(zip(poses, names), start=1):
            qw, qx, qy, qz = rotation_to_quaternion(pose.rotation)
            tx, ty, tz = pose.translation
            f.write(struct.pack("<I4d3dI", index, qw, qx, qy, qz, tx, ty, tz, 1))
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # no 2D observations
    with open(os.path.join(sparse_dir, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(scene)))
        for index, (x, y, z, r, g, b) in enumerate(scene, start=1):
            f.write(struct.pack("<Q3d3BdQ", index, x, y, z, r, g, b, 0.5, 0))


def read_colmap(sparse_dir):
    """Parses the three binary files with struct. Every file must be consumed exactly."""
    def load(name):
        with open(os.path.join(sparse_dir, name), "rb") as f:
            return f.read()

    data = load("cameras.bin")
    offset = 0
    (count,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    cameras = {}
    for _ in range(count):
        camera_id, model = struct.unpack_from("<Ii", data, offset)
        offset += 8
        width, height = struct.unpack_from("<QQ", data, offset)
        offset += 16
        if model != 1:
            raise ValueError("expected PINHOLE (model 1), found %d" % model)
        fx, fy, cx, cy = struct.unpack_from("<4d", data, offset)
        offset += 32
        cameras[camera_id] = {"width": width, "height": height, "intrinsics": (fx, fy, cx, cy)}
    if offset != len(data):
        raise ValueError("trailing bytes in cameras.bin")

    data = load("images.bin")
    offset = 0
    (count,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    images = []
    for _ in range(count):
        (image_id,) = struct.unpack_from("<I", data, offset)
        offset += 4
        qvec = struct.unpack_from("<4d", data, offset)
        offset += 32
        tvec = struct.unpack_from("<3d", data, offset)
        offset += 24
        (camera_id,) = struct.unpack_from("<I", data, offset)
        offset += 4
        end = data.index(b"\x00", offset)
        name = data[offset:end].decode("utf-8")
        offset = end + 1
        (points2d,) = struct.unpack_from("<Q", data, offset)
        offset += 8 + points2d * 24
        images.append({"id": image_id, "name": name, "qvec": qvec, "tvec": tvec,
                       "camera_id": camera_id})
    if offset != len(data):
        raise ValueError("trailing bytes in images.bin")

    data = load("points3D.bin")
    offset = 0
    (count,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    points = []
    for _ in range(count):
        offset += 8  # point id
        x, y, z = struct.unpack_from("<3d", data, offset)
        offset += 24
        r, g, b = struct.unpack_from("<3B", data, offset)
        offset += 3
        offset += 8  # reprojection error
        (track,) = struct.unpack_from("<Q", data, offset)
        offset += 8 + track * 8
        points.append((x, y, z, r, g, b))
    if offset != len(data):
        raise ValueError("trailing bytes in points3D.bin")

    return {"cameras": cameras, "images": images, "points": points}


# MARK: - Verification

def verify(outdir, out=sys.stdout):
    """Re-reads the written dataset and checks, for every view, that (a) most scene points land
    inside the stored image and (b) the centroid of the object's pixels is within
    CENTROID_TOLERANCE_PX of the projected object centre (origin). Object pixels are the
    saturated ones: ground points are grey and would otherwise pull the centroid."""
    sparse_dir = os.path.join(outdir, "sparse", "0")
    images_dir = os.path.join(outdir, "images")
    try:
        colmap = read_colmap(sparse_dir)
    except (OSError, ValueError, struct.error) as error:
        print("FAIL: cannot read COLMAP files: %s" % error, file=out)
        return False

    points = colmap["points"]
    if not colmap["images"] or not points:
        print("FAIL: no views or no points", file=out)
        return False

    all_ok = True
    for image in colmap["images"]:
        camera = colmap["cameras"].get(image["camera_id"])
        if camera is None:
            print("FAIL: %s references missing camera %d" % (image["name"], image["camera_id"]),
                  file=out)
            all_ok = False
            continue
        try:
            width, height, pixels = read_png(os.path.join(images_dir, image["name"]))
        except (OSError, ValueError, zlib.error, struct.error) as error:
            print("FAIL: %s: %s" % (image["name"], error), file=out)
            all_ok = False
            continue
        if (width, height) != (camera["width"], camera["height"]):
            print("FAIL: %s is %dx%d but camera is %dx%d" % (
                image["name"], width, height, camera["width"], camera["height"]), file=out)
            all_ok = False
            continue

        pose = Pose(quaternion_to_rotation(image["qvec"]), image["tvec"])
        intrinsics = camera["intrinsics"]
        inside = 0
        for (x, y, z, _, _, _) in points:
            u, v, depth = project(pose, intrinsics, (x, y, z))
            if depth > 0 and 0 <= u < width and 0 <= v < height:
                inside += 1
        inside_fraction = inside / float(len(points))

        object_count = 0
        sum_u = 0.0
        sum_v = 0.0
        for j in range(height):
            for i in range(width):
                k = 3 * (j * width + i)
                r, g, b = pixels[k], pixels[k + 1], pixels[k + 2]
                if max(r, g, b) - min(r, g, b) > SATURATION_THRESHOLD:
                    object_count += 1
                    sum_u += i + 0.5
                    sum_v += j + 0.5

        centre_u, centre_v, _ = project(pose, intrinsics, (0.0, 0.0, 0.0))
        if object_count == 0:
            error_px = float("inf")
            centroid_text = "none"
        else:
            cu = sum_u / object_count
            cv = sum_v / object_count
            error_px = math.hypot(cu - centre_u, cv - centre_v)
            centroid_text = "(%.2f,%.2f)" % (cu, cv)

        ok = (inside_fraction >= INSIDE_MIN_FRACTION
              and error_px <= CENTROID_TOLERANCE_PX)
        all_ok = all_ok and ok
        print("%s: points_inside=%.2f object_px=%d centroid=%s centre=(%.2f,%.2f) err=%.2fpx %s"
              % (image["name"], inside_fraction, object_count, centroid_text,
                 centre_u, centre_v, error_px, "OK" if ok else "FAIL"), file=out)
    return all_ok


# MARK: - Generation

def generate(outdir, views, width, height, points, seed, write_ppm):
    rng = random.Random(seed)
    object_points = sphere_points(rng, points)
    ground = ground_points(rng, max(12, points // 8))
    scene = object_points + ground

    intrinsics = (FOCAL_SCALE * width, FOCAL_SCALE * width, width / 2.0, height / 2.0)
    poses = camera_poses(views)
    images_dir = os.path.join(outdir, "images")
    os.makedirs(images_dir, exist_ok=True)

    names = []
    for index, pose in enumerate(poses):
        name = "view_%02d.png" % index
        rgb = render(scene, pose, intrinsics, width, height)
        with open(os.path.join(images_dir, name), "wb") as f:
            f.write(png_bytes(width, height, rgb))
        if write_ppm:
            stem = os.path.splitext(name)[0]
            with open(os.path.join(images_dir, stem + ".ppm"), "wb") as f:
                f.write(ppm_bytes(width, height, rgb))
        names.append(name)

    write_colmap(os.path.join(outdir, "sparse", "0"), poses, names,
                 width, height, intrinsics, scene)
    return len(object_points), len(ground)


def parse_size(text):
    parts = text.lower().split("x")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError("size must look like 160x120")
    width, height = int(parts[0]), int(parts[1])
    if width < 16 or height < 16:
        raise argparse.ArgumentTypeError("size must be at least 16x16")
    return width, height


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate a synthetic COLMAP dataset.")
    parser.add_argument("outdir", help="output directory")
    parser.add_argument("--views", type=int, default=4, help="number of cameras (default 4)")
    parser.add_argument("--size", type=parse_size, default=(160, 120),
                        help="image size WxH (default 160x120)")
    parser.add_argument("--points", type=int, default=400,
                        help="object surface points (default 400)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default 1)")
    parser.add_argument("--ppm", action="store_true", help="also write images/view_NN.ppm")
    parser.add_argument("--verify", action="store_true",
                        help="only verify an existing OUTDIR, do not generate")
    args = parser.parse_args(argv)

    if args.verify:
        return 0 if verify(args.outdir) else 1

    if args.views < 1:
        parser.error("--views must be at least 1")
    if args.points < 10:
        parser.error("--points must be at least 10")

    width, height = args.size
    object_count, ground_count = generate(args.outdir, args.views, width, height,
                                          args.points, args.seed, args.ppm)
    print("wrote %d views at %dx%d to %s: %d object + %d ground points"
          % (args.views, width, height, args.outdir, object_count, ground_count))
    return 0 if verify(args.outdir) else 1


if __name__ == "__main__":
    sys.exit(main())
