#!/usr/bin/env bash
# COLMAP bundling spike 2 (docs/superpowers/plans/2026-11-06-phase-4-addendum-input-pipeline.md,
# "P5 result"): build a minimal, permissively licensed COLMAP from source on a macOS arm64 runner and
# run it on a textured synthetic scene. Never aborts; everything lands in $REPORT_DIR for the artifact.
set -uo pipefail

export HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_ENV_HINTS=1
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPORT_DIR=${REPORT_DIR:-${RUNNER_TEMP:-/tmp}/colmap-min-report}
WORK_DIR=${WORK_DIR:-${RUNNER_TEMP:-/tmp}/colmap-min-work}
COLMAP_REF=${COLMAP_REF:-4.2.1}
CERES_REF=${CERES_REF:-2.2.0}
PREFIX=$WORK_DIR/prefix
LOGS=$REPORT_DIR/logs
REPORT=$REPORT_DIR/report.md
mkdir -p "$LOGS" "$WORK_DIR" "$PREFIX"
: >"$REPORT"

md() { printf '%s\n' "$*" >>"$REPORT"; }
step() { # step NAME CMD...  -> runs, logs, records exit code
  local name=$1; shift
  local t0=$SECONDS
  "$@" >"$LOGS/$name.log" 2>&1 </dev/null
  local rc=$?
  md "- **$name**: exit $rc in $((SECONDS - t0)) s (log: logs/$name.log)"
  if [ $rc -ne 0 ]; then md '  ```'; tail -n 25 "$LOGS/$name.log" | cut -c1-220 | sed 's/^/  /' >>"$REPORT"; md '  ```'; fi
  return $rc
}

md '# COLMAP minimal source build'
md
md "colmap ref \`$COLMAP_REF\`, ceres ref \`$CERES_REF\`, runner: $(sw_vers -productVersion) $(uname -m), $(xcodebuild -version | head -1)"
md
md '## Build'

# Permissive-only build dependencies. No Qt, ffmpeg, SuiteSparse, CGAL, ONNX, CUDA.
step brew_install brew install cmake ninja eigen boost glog gflags sqlite flann metis libomp ccache pkgconf openimageio || true

BREW=$(brew --prefix)
ok=1
step clone_ceres git clone --depth 1 --branch "$CERES_REF" https://github.com/ceres-solver/ceres-solver.git "$WORK_DIR/ceres" || ok=0
if [ $ok = 1 ]; then
  step configure_ceres cmake -S "$WORK_DIR/ceres" -B "$WORK_DIR/ceres-build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
    -DSUITESPARSE=OFF -DCXSPARSE=OFF -DEIGENSPARSE=ON -DACCELERATESPARSE=ON -DLAPACK=ON \
    -DCMAKE_PREFIX_PATH="$BREW" || ok=0
fi
[ $ok = 1 ] && { step build_ceres cmake --build "$WORK_DIR/ceres-build" -j 4 || ok=0; }
[ $ok = 1 ] && { step install_ceres cmake --install "$WORK_DIR/ceres-build" || ok=0; }

step clone_colmap git clone --depth 1 --branch "$COLMAP_REF" https://github.com/colmap/colmap.git "$WORK_DIR/colmap" || ok=0
if [ -d "$WORK_DIR/colmap" ]; then
  grep -nE '^option\(|^set\(.*CACHE' "$WORK_DIR/colmap/CMakeLists.txt" | cut -c1-200 >"$LOGS/colmap_options.txt"
  md; md '### COLMAP CMake options'; md '```'; cat "$LOGS/colmap_options.txt" >>"$REPORT"; md '```'
fi

if [ $ok = 1 ]; then
  step configure_colmap cmake -S "$WORK_DIR/colmap" -B "$WORK_DIR/colmap-build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX/colmap" \
    -DGUI_ENABLED=OFF -DCUDA_ENABLED=OFF -DONNX_ENABLED=OFF -DCGAL_ENABLED=OFF -DOPENGL_ENABLED=OFF \
    -DTESTS_ENABLED=OFF -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_PREFIX_PATH="$PREFIX;$BREW" -DCMAKE_INSTALL_RPATH='@loader_path/../lib' || ok=0
fi
[ $ok = 1 ] && { step build_colmap cmake --build "$WORK_DIR/colmap-build" -j 4 || ok=0; }
[ $ok = 1 ] && { step install_colmap cmake --install "$WORK_DIR/colmap-build" || ok=0; }

COLMAP=$PREFIX/colmap/bin/colmap
md; md '## Result'
if [ -x "$COLMAP" ]; then
  md "- binary: \`$COLMAP\`"
  md '```'; otool -L "$COLMAP" | cut -c1-200 >>"$REPORT"; md '```'
  # Non-system dependency closure (breadth-first over otool -L).
  python3 -I - "$COLMAP" "$REPORT_DIR/closure.md" <<'PY'
import subprocess, sys, os
start, out = sys.argv[1], sys.argv[2]
seen, queue, rows = set(), [start], []
while queue:
    f = queue.pop()
    f = os.path.realpath(f)
    if f in seen: continue
    seen.add(f)
    rows.append((f, os.path.getsize(f)))
    o = subprocess.run(["otool", "-L", f], capture_output=True, text=True).stdout.splitlines()[1:]
    for line in o:
        p = line.strip().split(" (")[0]
        if p.startswith(("/usr/lib/", "/System/")): continue
        if p.startswith("@rpath/") or p.startswith("@loader_path/"):
            name = p.split("/", 1)[1]
            for root in (os.path.dirname(f), os.path.join(os.path.dirname(f), "../lib"), "/opt/homebrew/lib"):
                cand = os.path.join(root, name)
                if os.path.exists(cand): queue.append(cand); break
        elif os.path.exists(p): queue.append(p)
total = sum(s for _, s in rows)
with open(out, "w") as fh:
    fh.write("closure files: %d, total %.1f MB\n\n" % (len(rows), total / 1e6))
    for f, s in sorted(rows):
        fh.write("%8.2f MB  %s\n" % (s / 1e6, f))
PY
  md '### Dependency closure'; md '```'; cat "$REPORT_DIR/closure.md" | cut -c1-200 >>"$REPORT"; md '```'

  md; md '## Headless run on a textured synthetic scene'
  python3 -m pip install --quiet --break-system-packages numpy pillow >"$LOGS/pip.log" 2>&1 || pip3 install --quiet numpy pillow >>"$LOGS/pip.log" 2>&1
  step make_scene python3 "$SCRIPT_DIR/make_textured_scene.py" "$WORK_DIR/scene" 6
  DB=$WORK_DIR/scene/database.db
  step feature_extractor "$COLMAP" feature_extractor --database_path "$DB" --image_path "$WORK_DIR/scene/images" \
      --ImageReader.single_camera 1 --FeatureExtraction.use_gpu 0
  step exhaustive_matcher "$COLMAP" exhaustive_matcher --database_path "$DB" --FeatureMatching.use_gpu 0
  mkdir -p "$WORK_DIR/scene/sparse"
  step mapper "$COLMAP" mapper --database_path "$DB" --image_path "$WORK_DIR/scene/images" --output_path "$WORK_DIR/scene/sparse"
  step model_analyzer "$COLMAP" model_analyzer --path "$WORK_DIR/scene/sparse/0"
  grep -E 'Registered images|Points|Mean reprojection' "$LOGS/model_analyzer.log" | sed 's/^[^]]*\] //' >>"$REPORT" || true
  ls -la "$WORK_DIR/scene/sparse/0" >>"$REPORT" 2>&1
else
  md "- colmap binary was not built"
fi
exit 0
