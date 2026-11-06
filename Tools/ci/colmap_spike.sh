#!/usr/bin/env bash
# COLMAP bundling feasibility spike (plan: docs/superpowers/plans/2026-11-06-phase-4-addendum-input-pipeline.md,
# risks 1 and 2, task P5). Runs on a macOS arm64 runner.
#
# Every probe records its command, exit status, wall time and the tail of its output in
# $REPORT_DIR/report.md; full logs go to $REPORT_DIR/logs. The script never aborts on a failing
# probe, so the report is complete even when COLMAP does not install or does not run.
#
# Optional overrides: REPORT_DIR, WORK_DIR, SIZE_LIMIT_MB (closure size budget, default 250).
set -uo pipefail

export HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_ENV_HINTS=1 QT_QPA_PLATFORM=offscreen

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
REPORT_DIR=${REPORT_DIR:-${RUNNER_TEMP:-/tmp}/colmap-report}
WORK_DIR=${WORK_DIR:-${RUNNER_TEMP:-/tmp}/colmap-work}
SIZE_LIMIT_MB=${SIZE_LIMIT_MB:-250}

LOG_DIR=$REPORT_DIR/logs
DATA_DIR=$REPORT_DIR/data
REPORT=$REPORT_DIR/report.md
STATUS=$DATA_DIR/status.tsv

rm -rf "$LOG_DIR" "$DATA_DIR"
mkdir -p "$LOG_DIR" "$DATA_DIR" "$WORK_DIR"
: >"$REPORT"
: >"$STATUS"

# ---- helpers ---------------------------------------------------------------

md() { printf '%s\n' "$*" >>"$REPORT"; }
section() { md; md "## $1"; md; }
now() { python3 -c 'import time; print("%.3f" % time.time())'; }

# Stock macOS has no GNU timeout; fall back to perl's alarm(), which exits with 142 on timeout.
run_limited() {
  local secs=$1
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$secs" "$@"
  else
    perl -e 'alarm shift; exec @ARGV or die "exec failed: $!\n"' "$secs" "$@"
  fi
}

# probe SLUG TIMEOUT_S TAIL_LINES TITLE -- CMD [ARG...]
probe() {
  local slug=$1 secs=$2 tail_n=$3 title=$4
  local log t0 t1 rc dur note cmd_str lines
  shift 4
  [ "${1:-}" = "--" ] && shift
  log=$LOG_DIR/$slug.log
  t0=$(now)
  run_limited "$secs" "$@" >"$log" 2>&1 </dev/null
  rc=$?
  t1=$(now)
  dur=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.1f", b - a }')
  case $rc in
    124 | 142) note=" (stopped by the ${secs} s time limit)" ;;
    *) note="" ;;
  esac
  lines=$(wc -l <"$log" | tr -d ' ')
  cmd_str="$*"
  printf '%s\t%s\t%s\n' "$slug" "$rc" "$dur" >>"$STATUS"

  md "### $title"
  md
  md "- Command: \`$cmd_str\`"
  md "- Exit status: $rc$note"
  md "- Wall time: ${dur} s"
  md "- Full log: \`logs/$slug.log\` ($lines lines)"
  md
  if [ -s "$log" ]; then
    md "Last $tail_n lines of output:"
    md
    md '````text'
    tail -n "$tail_n" "$log" >>"$REPORT"
    md '````'
  else
    md "(no output)"
  fi
  md
}

# write_py NAME  (python source on stdin) -> $WORK_DIR/NAME.py
write_py() { cat >"$WORK_DIR/$1.py"; }

# pyblock SLUG TITLE ARGS...  runs $WORK_DIR/SLUG.py with python3 -I; its stdout (markdown) goes into the report.
pyblock() {
  local slug=$1 title=$2 rc
  shift 2
  run_limited 600 python3 -I "$WORK_DIR/$slug.py" "$@" >"$DATA_DIR/$slug.md" 2>"$LOG_DIR/$slug.log" </dev/null
  rc=$?
  printf '%s\t%s\t%s\n' "py_$slug" "$rc" "-" >>"$STATUS"
  md "### $title"
  md
  md "Helper exit status: $rc (stderr: \`logs/$slug.log\`)"
  md
  cat "$DATA_DIR/$slug.md" >>"$REPORT"
  md
}

# pick_opt COMMAND NEW_NAME OLD_NAME -> "--<name>.use_gpu" as printed by the command's --help
pick_opt() {
  local help
  help=$("$COLMAP_BIN" "$1" --help 2>&1 || true)
  if grep -q -- "--$2\\.use_gpu" <<<"$help"; then
    echo "--$2.use_gpu"
  elif grep -q -- "--$3\\.use_gpu" <<<"$help"; then
    echo "--$3.use_gpu"
  else
    echo "--$3.use_gpu"
  fi
}

# ---- report header ---------------------------------------------------------

md "# COLMAP bundling feasibility spike"
md
md "- Generated (UTC): $(date -u '+%Y-%m-%d %H:%M:%S')"
md "- Repository commit: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
md "- Probe time limits: install 1800 s, COLMAP steps 900 s, other probes 10-300 s."
md "- Read section 8 (Verdict) first. Each probe shows its exit status and the tail of its output; full logs are in \`logs/\`."

# ---- 1. Environment --------------------------------------------------------

section "1. Environment"
probe env_sw_vers 30 40 "sw_vers" -- sw_vers
probe env_uname 10 5 "uname -m" -- uname -m
probe env_xcode 60 20 "Xcode version (xcodebuild -version)" -- xcodebuild -version
probe env_brew 60 10 "Homebrew version" -- brew --version
probe env_python 30 5 "python3 version" -- python3 --version

# ---- 2. Install ------------------------------------------------------------

section "2. Install"
probe install 1800 60 "brew install colmap (timed; time limit 1800 s)" -- brew install colmap
probe install_versions 60 10 "brew list --versions colmap" -- brew list --versions colmap

COLMAP_PREFIX=$(brew --prefix colmap 2>/dev/null || true)
BREW_ROOT=$(brew --prefix 2>/dev/null || true)
COLMAP_BIN="$COLMAP_PREFIX/bin/colmap"
export COLMAP_BIN
COLMAP_REAL=$(python3 -I -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$COLMAP_BIN" 2>/dev/null || true)
export COLMAP_REAL
md "Resolved: brew prefix \`$COLMAP_PREFIX\`; binary \`$COLMAP_BIN\` -> real path \`$COLMAP_REAL\`."
if [ -x "$COLMAP_REAL" ]; then
  md "Binary present: yes."
else
  md "Binary present: **no** (install failed or the prefix differs). Dependent sections are skipped or show failures."
fi

# ---- 3. Version and capabilities -------------------------------------------

section "3. Version and capabilities"
probe cap_help 60 40 "colmap --help (first 60 lines; exit status is that of the pipeline)" -- bash -c '"$COLMAP_BIN" --help 2>&1 | head -n 60'
probe cap_h 60 20 "colmap -h (first 40 lines)" -- bash -c '"$COLMAP_BIN" -h 2>&1 | head -n 40'
probe cap_help_cmd 60 20 "colmap help (first 40 lines)" -- bash -c '"$COLMAP_BIN" help 2>&1 | head -n 40'
probe cap_fx_gpu 60 40 "feature_extractor --help, GPU/CUDA options (grep exit 1 = none)" -- bash -c '"$COLMAP_BIN" feature_extractor --help 2>&1 | grep -i -E "gpu|cuda"'
probe cap_match_gpu 60 40 "exhaustive_matcher --help, GPU/CUDA options (grep exit 1 = none)" -- bash -c '"$COLMAP_BIN" exhaustive_matcher --help 2>&1 | grep -i -E "gpu|cuda"'
probe cap_linkage 30 20 "Linkage to Qt (GUI) or CUDA libraries (grep exit 1 = none)" -- bash -c 'otool -L "$COLMAP_REAL" 2>&1 | grep -i -E "qt|cuda|cublas|cudart"'

FX_GPU_OPT=$(pick_opt feature_extractor SiftExtraction FeatureExtraction)
MATCH_GPU_OPT=$(pick_opt exhaustive_matcher SiftMatching FeatureMatching)
md "CPU-only option names chosen from the --help output: \`$FX_GPU_OPT\` (extraction), \`$MATCH_GPU_OPT\` (matching)."

# ---- 4. Dependency closure -------------------------------------------------

section "4. Dependency closure"
if [ -x "$COLMAP_REAL" ]; then
  probe otool_L_binary 30 40 "otool -L on the colmap binary (raw)" -- otool -L "$COLMAP_REAL"
  write_py closure <<'PY'
#!/usr/bin/env python3
# Dependency closure of a Mach-O executable, following the references reported by otool -L.
# argv: BINARY JSON_OUT BREW_ROOT. Writes JSON to JSON_OUT and a markdown section to stdout.
import json
import os
import re
import subprocess
import sys

BINARY_REAL = os.path.realpath(sys.argv[1])
JSON_OUT = sys.argv[2]
BREW_ROOT = sys.argv[3]
SYSTEM_PREFIXES = ("/usr/lib/", "/System/")
MB = 1024.0 * 1024.0


def run(args):
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
        return proc.returncode, proc.stdout + proc.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def load_commands(path):
    return run(["otool", "-l", path])[1].splitlines()


def rpaths_from(cmds):
    found = []
    for i, line in enumerate(cmds):
        if "cmd LC_RPATH" in line:
            for nxt in cmds[i + 1:i + 5]:
                m = re.match(r"\s*path (.+?) \(offset", nxt)
                if m:
                    found.append(m.group(1))
                    break
    return found


def minos_from(cmds):
    found = set()
    for i, line in enumerate(cmds):
        if "cmd LC_BUILD_VERSION" in line or "cmd LC_VERSION_MIN_MACOSX" in line:
            for nxt in cmds[i + 1:i + 6]:
                m = re.match(r"\s*(?:minos|version) (\S+)", nxt)
                if m:
                    found.add(m.group(1))
                    break
    return sorted(found)


def deps_and_install_name(path):
    deps = []
    for line in run(["otool", "-L", path])[1].splitlines()[1:]:
        m = re.match(r"\s*(\S+) \(compatibility", line)
        if m:
            deps.append(m.group(1))
    lines = run(["otool", "-D", path])[1].splitlines()
    own = lines[1].strip() if len(lines) > 1 else None
    return deps, own


def resolve(dep, loader, rpaths):
    loader_dir = os.path.dirname(loader)
    exe_dir = os.path.dirname(BINARY_REAL)
    if dep.startswith("@rpath/"):
        tail = dep[len("@rpath/"):]
        bases = [rp.replace("@loader_path", loader_dir).replace("@executable_path", exe_dir)
                 for rp in rpaths]
        bases.append(os.path.join(BREW_ROOT, "lib"))  # fallback search, Homebrew lib dir
        candidates = [os.path.join(b, tail) for b in bases]
    elif dep.startswith("@loader_path/"):
        candidates = [os.path.join(loader_dir, dep[len("@loader_path/"):])]
    elif dep.startswith("@executable_path/"):
        candidates = [os.path.join(exe_dir, dep[len("@executable_path/"):])]
    else:
        candidates = [dep]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.realpath(candidate)
    return None


def main():
    cmds_cache = {BINARY_REAL: load_commands(BINARY_REAL)}
    exe_rpaths = rpaths_from(cmds_cache[BINARY_REAL])
    order = [BINARY_REAL]
    seen = {BINARY_REAL}
    unresolved = []
    system_deps = set()
    i = 0
    while i < len(order):
        cur = order[i]
        i += 1
        rpaths = rpaths_from(cmds_cache[cur]) + exe_rpaths
        deps, own = deps_and_install_name(cur)
        for dep in deps:
            if dep == own:
                continue
            if dep.startswith(SYSTEM_PREFIXES):
                system_deps.add(dep)
                continue
            real = resolve(dep, cur, rpaths)
            if real is None:
                unresolved.append({"dep": dep, "from": cur})
                continue
            if real not in seen:
                seen.add(real)
                order.append(real)
                cmds_cache[real] = load_commands(real)

    records = []
    total = 0
    for path in order:
        size = os.path.getsize(path)
        total += size
        rc, archs = run(["lipo", "-archs", path])
        cellar = re.search(r"/Cellar/([^/]+)/([^/]+)/", path)
        records.append({
            "path": path,
            "kind": "executable" if path == BINARY_REAL else "library",
            "size_bytes": size,
            "archs": archs.strip() if rc == 0 else "?",
            "minos": minos_from(cmds_cache[path]),
            "formula": cellar.group(1) if cellar else None,
            "formula_version": cellar.group(2) if cellar else None,
        })
    formulas = sorted({r["formula"] for r in records if r["formula"]})
    non_formula = [r["path"] for r in records if not r["formula"]]
    result = {
        "binary": BINARY_REAL,
        "entries": records,
        "total_bytes": total,
        "total_mb": round(total / MB, 1),
        "system_deps": sorted(system_deps),
        "unresolved": unresolved,
        "formulas": formulas,
        "non_formula_paths": non_formula,
    }
    with open(JSON_OUT, "w") as handle:
        json.dump(result, handle, indent=2)

    print(f"Binary: `{BINARY_REAL}`")
    print()
    print(f"**Binary + dependency closure: {result['total_mb']} MB** in {len(records)} files "
          f"({len(formulas)} Homebrew formulas). System libraries from /usr/lib and /System skipped: "
          f"{len(system_deps)}. Unresolved references: {len(unresolved)}.")
    print()
    print("| # | file | formula | size (MB) | archs | min macOS (LC_BUILD_VERSION / LC_VERSION_MIN_MACOSX) |")
    print("|---|---|---|---|---|---|")
    for n, rec in enumerate(records):
        formula = rec["formula"] or "(not a formula)"
        version = rec["formula_version"] or ""
        minos = ", ".join(rec["minos"]) or "?"
        size_mb = rec["size_bytes"] / MB
        print(f"| {n} | `{rec['path']}` | {formula} {version} | {size_mb:.2f} | {rec['archs']} | {minos} |")
    print()
    print("System dependencies skipped by rule:")
    for dep in sorted(system_deps):
        print(f"- `{dep}`")
    if not system_deps:
        print("- none")
    print()
    print("Unresolved references (not found on disk or in the Homebrew lib dir):")
    for item in unresolved:
        print(f"- `{item['dep']}` referenced from `{item['from']}`")
    if not unresolved:
        print("- none")


if __name__ == "__main__":
    main()
PY
  pyblock closure "Binary and dylib closure" "$COLMAP_REAL" "$DATA_DIR/closure.json" "$BREW_ROOT"
  python3 -I -c '
import json, sys
data = json.load(open(sys.argv[1]))
libs = [e for e in data["entries"] if e["kind"] == "library"]
for e in sorted(libs, key=lambda e: -e["size_bytes"])[:3]:
    print(e["path"])
' "$DATA_DIR/closure.json" >"$DATA_DIR/sign_samples.txt" 2>"$LOG_DIR/sign_samples.err"
else
  md "Skipped: no executable at \`$COLMAP_BIN\`."
fi

# ---- 5. Licenses -----------------------------------------------------------

section "5. Licenses"
brew deps --formula --include-build=false colmap >"$DATA_DIR/deps.txt" 2>"$LOG_DIR/brew_deps.log"
deps_rc=$?
printf '%s\t%s\t%s\n' "brew_deps" "$deps_rc" "-" >>"$STATUS"
md "\`brew deps --formula --include-build=false colmap\`: exit $deps_rc, $(grep -c . "$DATA_DIR/deps.txt") formulas (stderr in \`logs/brew_deps.log\`)."
md
write_py licenses <<'PY'
#!/usr/bin/env python3
# License check: colmap, every formula from brew deps, and every formula found in the closure.
# argv: DEPS_FILE CLOSURE_JSON OUT_JSON. Markdown table to stdout.
import json
import os
import re
import subprocess
import sys

DEPS_FILE, CLOSURE_JSON, OUT_JSON = sys.argv[1:4]

# Licenses accepted without review (case-insensitive).
PERMISSIVE = {
    "mit", "mit-0", "bsd-2-clause", "bsd-3-clause", "bsd-4-clause", "0bsd", "apache-2.0", "isc",
    "zlib", "unlicense", "cc0-1.0", "bsl-1.0", "boost-1.0", "psf-2.0", "python-2.0", "x11",
    "hpnd", "curl", "bzip2-1.0.6",
}
# Strong copyleft or non-commercial terms. LGPL is deliberately not here: it is flagged CHECK only.
HARD = re.compile(r"(?<!l)gpl|fipl|sspl|-nc\b", re.I)
STOPWORDS = {"and", "or", "with", "any", "all", "of"}


def as_text(lic):
    return lic if isinstance(lic, str) else json.dumps(lic, sort_keys=True)


def tokens(text):
    return [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9.+\-]*", text) if t.lower() not in STOPWORDS]


def alternatives(lic):
    if isinstance(lic, str):
        return re.split(r"\s+OR\s+", lic, flags=re.I)
    if isinstance(lic, dict) and isinstance(lic.get("any_of"), list):
        return [as_text(x) for x in lic["any_of"]]
    return [as_text(lic)]


def is_permissive(text):
    toks = tokens(text)
    return bool(toks) and all(t.lower() in PERMISSIVE for t in toks)


def classify(lic):
    """Return (status, reason, copyleft_only)."""
    if not lic:
        return "CHECK", "missing license metadata", False
    text = as_text(lic)
    if is_permissive(text):
        return "OK", "permissive: " + ", ".join(tokens(text)), False
    bad = [t for t in tokens(text) if t.lower() not in PERMISSIVE]
    alt_ok = any(is_permissive(a) for a in alternatives(lic))
    hard = bool(HARD.search(text)) and not alt_ok
    if hard:
        reason = "copyleft-only (no permissive option): " + ", ".join(bad)
    elif alt_ok:
        reason = "dual license with a permissive option; confirm the option used: " + ", ".join(bad)
    else:
        reason = "not on the permissive allowlist: " + ", ".join(bad)
    return "CHECK", reason, hard


def brew_info(names):
    info = {}
    if names:
        try:
            proc = subprocess.run(["brew", "info", "--json=v2", "--formula", *names],
                                  capture_output=True, text=True, timeout=600)
            if proc.returncode == 0:
                for formula in json.loads(proc.stdout).get("formulae", []):
                    info[formula.get("name")] = formula
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    for name in names:
        if name in info:
            continue
        try:
            proc = subprocess.run(["brew", "info", "--json=v2", "--formula", name],
                                  capture_output=True, text=True, timeout=300)
            formulae = json.loads(proc.stdout).get("formulae", [])
            if formulae:
                info[name] = formulae[0]
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    return info


def main():
    deps = []
    if os.path.exists(DEPS_FILE):
        with open(DEPS_FILE) as handle:
            deps = [line.strip() for line in handle if line.strip()]
    closure_formulas = []
    if os.path.exists(CLOSURE_JSON):
        with open(CLOSURE_JSON) as handle:
            closure_formulas = json.load(handle)["formulas"]
    names = sorted(set(deps) | set(closure_formulas) | {"colmap"})
    info = brew_info(names)

    records = []
    for name in names:
        formula = info.get(name)
        if formula is None:
            version, lic = "?", None
            status, reason, hard = "CHECK", "formula metadata not found (brew info failed)", False
        else:
            version = (formula.get("versions") or {}).get("stable") or "?"
            lic = formula.get("license")
            status, reason, hard = classify(lic)
        records.append({
            "name": name,
            "version": version,
            "license": lic,
            "status": status,
            "reason": reason,
            "copyleft_only": hard,
            "in_deps": name in deps or name == "colmap",
            "in_closure": name in closure_formulas,
        })
    with open(OUT_JSON, "w") as handle:
        json.dump(records, handle, indent=2)

    check_all = [r for r in records if r["status"] != "OK"]
    check_closure = [r for r in check_all if r["in_closure"]]
    print(f"Formulas checked: {len(records)} (colmap, brew deps, closure extras). "
          f"**CHECK** overall: {len(check_all)}; **CHECK** inside the closure: {len(check_closure)}.")
    print()
    print("| formula | version | license (as reported by Homebrew) | status | in closure | reason |")
    print("|---|---|---|---|---|---|")
    for r in records:
        status = "OK" if r["status"] == "OK" else "**CHECK**"
        lic = json.dumps(r["license"]) if not isinstance(r["license"], str) else (r["license"] or "(none)")
        closure_flag = "yes" if r["in_closure"] else "no"
        print(f"| {r['name']} | {r['version']} | {lic} | {status} | {closure_flag} | {r['reason']} |")
    print()
    print("License strings are Homebrew formula metadata, not a legal review.")


if __name__ == "__main__":
    main()
PY
pyblock licenses "License table" "$DATA_DIR/deps.txt" "$DATA_DIR/closure.json" "$DATA_DIR/licenses.json"

# ---- 6. Code-signing facts -------------------------------------------------

section "6. Code-signing facts"
if [ -x "$COLMAP_REAL" ]; then
  probe sign_colmap 60 40 "codesign -dv --verbose=2: colmap binary" -- codesign -dv --verbose=2 "$COLMAP_REAL"
  probe rpath_colmap 30 40 "LC_RPATH entries of the colmap binary (grep exit 1 = none)" -- bash -c 'otool -l "$COLMAP_REAL" | grep -A2 LC_RPATH'
  sample_no=0
  while IFS= read -r lib; do
    [ -n "$lib" ] || continue
    sample_no=$((sample_no + 1))
    probe "sign_dylib_$sample_no" 60 40 "codesign -dv --verbose=2: sample dylib $sample_no (\`$lib\`)" -- codesign -dv --verbose=2 "$lib"
  done <"$DATA_DIR/sign_samples.txt"
  if [ "$sample_no" -eq 0 ]; then md "No sample dylibs (closure not computed)."; fi
else
  md "Skipped: no executable at \`$COLMAP_BIN\`."
fi

# ---- 7. Headless run -------------------------------------------------------

section "7. Headless run (explicit CPU-only pipeline)"
md "Not used: \`colmap automatic_reconstructor\` (it needs GUI-style setup). The synthetic sphere is texture-poor, so SIFT and the mapper may legitimately find little. The question is whether the pipeline runs headless and what it prints when it fails. Views: 4."
md
SCENE=$WORK_DIR/scene
DB=$WORK_DIR/db.db
IMAGES=$SCENE/images
SPARSE=$WORK_DIR/sparse
rm -rf "$SCENE" "$DB" "$SPARSE" "$WORK_DIR/sparse_txt" "$WORK_DIR/points.ply"
mkdir -p "$SPARSE" "$WORK_DIR/sparse_txt"

probe synth_scene 300 10 "Synthetic scene (4 views, 640x480, 800 points)" -- python3 "$REPO_ROOT/Tools/synthetic/make_colmap_scene.py" "$SCENE" --views 4 --size 640x480 --points 800
probe headless_feat 900 40 "colmap feature_extractor (CPU SIFT)" -- "$COLMAP_BIN" feature_extractor --database_path "$DB" --image_path "$IMAGES" "$FX_GPU_OPT" 0 --ImageReader.single_camera 1
probe headless_match 900 40 "colmap exhaustive_matcher (CPU)" -- "$COLMAP_BIN" exhaustive_matcher --database_path "$DB" "$MATCH_GPU_OPT" 0
probe headless_mapper 900 40 "colmap mapper" -- "$COLMAP_BIN" mapper --database_path "$DB" --image_path "$IMAGES" --output_path "$SPARSE"
probe headless_convert_txt 300 40 "colmap model_converter to TXT (spec variant: --output_type TXT)" -- "$COLMAP_BIN" model_converter --input_path "$SPARSE/0" --output_path "$WORK_DIR/sparse_txt" --output_type TXT
probe headless_convert_ply 300 40 "colmap model_converter to PLY (spec's 'ply' output path with --output_type PLY)" -- "$COLMAP_BIN" model_converter --input_path "$SPARSE/0" --output_path "$WORK_DIR/points.ply" --output_type PLY
probe headless_analyzer 300 40 "colmap model_analyzer" -- "$COLMAP_BIN" model_analyzer --path "$SPARSE/0"

write_py registered <<'PY'
#!/usr/bin/env python3
# Registered-image count from images.bin (uint64 header) and model_analyzer output; scans headless logs for loader errors.
# argv: MODEL_DIR ANALYZER_LOG LOG_DIR OUT_JSON
import glob
import json
import os
import re
import struct
import sys

MODEL_DIR, ANALYZER_LOG, LOG_DIR, OUT_JSON = sys.argv[1:5]
LOADER_ERROR = re.compile(r"dyld|Library not loaded|Symbol not found|Bad CPU type|image not found", re.I)


def registered_from_bin(path):
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
    except OSError:
        return None
    return struct.unpack("<Q", head)[0] if len(head) == 8 else None


def registered_from_analyzer(path):
    try:
        with open(path, errors="replace") as handle:
            text = handle.read()
    except OSError:
        return None
    match = re.search(r"Registered images:\s*(\d+)", text, re.I)
    return int(match.group(1)) if match else None


def main():
    reg_bin = registered_from_bin(os.path.join(MODEL_DIR, "images.bin"))
    reg_analyzer = registered_from_analyzer(ANALYZER_LOG)
    loader_hits = []
    for log in sorted(glob.glob(os.path.join(LOG_DIR, "headless_*.log"))):
        with open(log, errors="replace") as handle:
            if LOADER_ERROR.search(handle.read()):
                loader_hits.append(os.path.basename(log))
    result = {"registered_bin": reg_bin, "registered_analyzer": reg_analyzer, "loader_errors": loader_hits}
    with open(OUT_JSON, "w") as handle:
        json.dump(result, handle, indent=2)

    def show(value):
        return "not written / not parsed" if value is None else str(value)

    print("| check | value |")
    print("|---|---|")
    print(f"| images.bin header (registered images) | {show(reg_bin)} |")
    print(f"| model_analyzer \"Registered images\" | {show(reg_analyzer)} |")
    print(f"| loader errors (dyld and similar) in headless logs | {', '.join(loader_hits) or 'none'} |")


if __name__ == "__main__":
    main()
PY
pyblock registered "Registered images and loader errors" "$SPARSE/0" "$LOG_DIR/headless_analyzer.log" "$LOG_DIR" "$DATA_DIR/headless.json"

# ---- 8. Verdict ------------------------------------------------------------

section "8. Verdict"
write_py verdict <<'PY'
#!/usr/bin/env python3
# Verdict from the other sections' JSON and status.tsv. argv: DATA_DIR SIZE_LIMIT_MB. Markdown to stdout.
import json
import os
import sys

DATA_DIR, LIMIT = sys.argv[1], float(sys.argv[2])
HEADLESS = ["headless_feat", "headless_match", "headless_mapper",
            "headless_convert_txt", "headless_convert_ply", "headless_analyzer"]


def load(name, default):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return default
    with open(path) as handle:
        return json.load(handle)


closure = load("closure.json", None)
licenses = load("licenses.json", [])
headless = load("headless.json", {})
status = {}
if os.path.exists(os.path.join(DATA_DIR, "status.tsv")):
    with open(os.path.join(DATA_DIR, "status.tsv")) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                status[parts[0]] = parts[1]

no_go = []
unclear = []
closure_check = []
closure_hard = []
other_check = []
if status.get("install") != "0":
    unclear.append(f"brew install colmap exited {status.get('install')}; there is no closure to assess")
elif closure is None:
    unclear.append("the colmap binary was not found under the brew prefix")
else:
    for rec in licenses:
        if rec["status"] == "OK":
            continue
        if rec["in_closure"]:
            (closure_hard if rec["copyleft_only"] else closure_check).append(f"{rec['name']} ({rec['license']})")
        else:
            other_check.append(rec["name"])
    loader_errors = headless.get("loader_errors") or []
    if closure_hard:
        no_go.append("copyleft-only license in the closure: " + ", ".join(closure_hard))
    if loader_errors:
        no_go.append("colmap does not load headless (loader errors in " + ", ".join(loader_errors) + ")")
    if not no_go:
        if closure_check:
            unclear.append("license review needed for closure formulas: " + ", ".join(closure_check))
        if closure["non_formula_paths"]:
            unclear.append(f"{len(closure['non_formula_paths'])} closure file(s) not from a Homebrew formula")
        if closure["unresolved"]:
            unclear.append(f"{len(closure['unresolved'])} dylib reference(s) could not be resolved")
        if closure["total_mb"] > LIMIT:
            unclear.append(f"closure is {closure['total_mb']} MB, above the {LIMIT:g} MB budget")
        if status.get("synth_scene") != "0":
            unclear.append("synthetic scene generation failed")
        failed = [s for s in HEADLESS if status.get(s) != "0"]
        if failed:
            if status.get("headless_feat") == "0" and status.get("headless_match") == "0" \
                    and status.get("headless_mapper") != "0":
                unclear.append("feature extraction and matching ran, but the mapper did not finish "
                               f"(exit {status.get('headless_mapper')}) on the texture-poor synthetic "
                               "sphere; this is a data problem until a textured run says otherwise")
            else:
                unclear.append("headless step(s) failed: "
                               + ", ".join(f"{s} (exit {status.get(s)})" for s in failed)
                               + "; see the logs")
        else:
            counts = [x for x in (headless.get("registered_bin"), headless.get("registered_analyzer"))
                      if x is not None]
            registered = max(counts) if counts else 0
            if registered < 3:
                unclear.append(f"all headless commands exited 0 but only {registered} image(s) registered")

if no_go:
    verdict = "NO-GO"
elif unclear:
    verdict = "UNCLEAR"
else:
    verdict = "GO"


def cell(value):
    return "n/a" if value is None else str(value)


print("### Verdict inputs")
print()
print("| input | value |")
print("|---|---|")
if closure is not None:
    print(f"| closure size | {closure['total_mb']} MB (budget {LIMIT:g} MB) |")
    print(f"| closure files / Homebrew formulas | {len(closure['entries'])} / {len(closure['formulas'])} |")
print(f"| **CHECK** licenses inside the closure | {len(closure_check) + len(closure_hard)} |")
print(f"| copyleft-only licenses inside the closure | {len(closure_hard)} |")
print(f"| **CHECK** licenses outside the closure (info only) | {len(other_check)} |")
print("| exit codes: install, synth, feat, match, mapper, convert txt, convert ply, analyzer | "
      + ", ".join(cell(status.get(s)) for s in
                   ["install", "synth_scene", "headless_feat", "headless_match", "headless_mapper",
                    "headless_convert_txt", "headless_convert_ply", "headless_analyzer"]) + " |")
print(f"| registered images (images.bin / model_analyzer) | "
      f"{cell(headless.get('registered_bin'))} / {cell(headless.get('registered_analyzer'))} |")
print(f"| loader errors | {', '.join(headless.get('loader_errors') or []) or 'none'} |")
print()
print(f"**Verdict: {verdict}**")
print()
reasons = no_go or unclear
if reasons:
    for reason in reasons:
        print(f"- {reason}")
else:
    print("- Closure licenses all on the permissive allowlist, closure within budget, and every headless "
          "command exited 0 with at least 3 images registered.")
if verdict == "GO":
    print("- Caveat: GO is for feasibility only. The synthetic sphere is not a quality test; a textured "
          "dataset run is still needed before P6.")
print()
print("Rules: NO-GO if a copyleft-only (GPL/FIPL/NC-type) formula is in the closure or colmap does not load. "
      "UNCLEAR if install failed, any CHECK license or non-formula file is in the closure, a dylib is "
      "unresolved, the closure is over budget, or a headless step failed or registered too few images. "
      "GO only when none of these apply.")
PY
python3 -I "$WORK_DIR/verdict.py" "$DATA_DIR" "$SIZE_LIMIT_MB" >>"$REPORT" 2>"$LOG_DIR/verdict.log"
verdict_rc=$?
if [ "$verdict_rc" -ne 0 ]; then
  md "Verdict helper failed (exit $verdict_rc; see \`logs/verdict.log\`). **Verdict: UNCLEAR**"
fi

echo "Report written to $REPORT"
grep -E '^\*\*Verdict' "$REPORT" || echo "Verdict line missing; see $REPORT"
exit 0
