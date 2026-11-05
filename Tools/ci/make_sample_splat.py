#!/usr/bin/env python3
"""Writes a deterministic synthetic .splat file for CI app smoke tests.

Usage: make_sample_splat.py OUTPUT [COUNT]

Record layout (32 bytes, little-endian, matches AntimatterSplatReader):
  position  3 x float32
  scale     3 x float32 (linear)
  RGBA      4 x uint8
  rotation  4 x uint8 (w, x, y, z), decoded as (b - 128) / 128
"""
import colorsys
import math
import random
import struct
import sys


def clamp_byte(value):
    return max(0, min(255, int(round(value))))


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    output = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    rng = random.Random(1234)

    records = bytearray()
    for _ in range(count):
        # Uniform point inside a sphere of radius 1.
        while True:
            x, y, z = (rng.uniform(-1.0, 1.0) for _ in range(3))
            if x * x + y * y + z * z <= 1.0:
                break
        scale = [rng.uniform(0.02, 0.06) for _ in range(3)]
        red, green, blue = colorsys.hsv_to_rgb(rng.random(), 0.8, 0.95)
        alpha = rng.uniform(0.5, 0.95)
        rgba = bytes([clamp_byte(red * 255), clamp_byte(green * 255),
                      clamp_byte(blue * 255), clamp_byte(alpha * 255)])

        # Random unit quaternion (w, x, y, z).
        quat = [rng.gauss(0.0, 1.0) for _ in range(4)]
        norm = math.sqrt(sum(v * v for v in quat)) or 1.0
        rot = bytes(clamp_byte(v / norm * 128 + 128) for v in quat)

        records += struct.pack("<3f3f", x, y, z, *scale) + rgba + rot

    with open(output, "wb") as handle:
        handle.write(records)
    print(f"wrote {count} splats ({len(records)} bytes) to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
