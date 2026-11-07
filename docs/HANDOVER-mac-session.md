# Handover: cloud session -> Mac session (2026-11-06)

A Claude Code session that runs **on the user's Mac** continues the work of the cloud session. The Mac can do what the cloud container cannot: build with Xcode, run the app, run Metal, check Hugging Face, try Apple Photogrammetry.

Read first, in this order: `CLAUDE.md` (rules, routing, German replies), `docs/ARCHITECTURE.md` (source of truth), `docs/superpowers/plans/2026-11-06-phase-4-addendum-input-pipeline.md` (decisions A1-A10, tasks P1-P9, "P5 result"), then the main Phase 4 plan next to it.

## Where things stand (branch `claude/claude-md-setup-skills-4g76ld`)

- Phases 1-3 done and green on CI (viewer, differentiable rasterizer, training, `.gsproj` project window with Train button).
- Phase 4 interview done; input pipeline comes first. Landed and tested (Linux; macOS parts compiled and tested on CI):
  - P1 frame sharpness + selection (`FrameSelectionMode.count(6)` / `.all`)
  - P2 `VideoFrameExtractor`, `PhotoFolderImporter` (SplatIO/Capture, macOS-only, CI-tested)
  - P3 manifest provenance (`inputKind`, `frameProvenance`)
  - P4 `PoseProvider` protocol, `PoseProviderChain`, `COLMAPWriter`, `ProjectBundle.importEstimatedPoses`
  - P8 model store: pure-Swift SHA-256, `ModelCatalog`, `ModelStore`, `URLSessionModelDownloader` (verified with a real download on the Mac)
  - `TrainingPreset` Fast / Standard / Quality
- The app was **never run by the user on real data**. A COLMAP dataset of the user's own photos (`table-scene`, 13 views) was produced in the cloud session and sent as a zip; the user has not reported the result of "Import COLMAP..." + Train yet.

## Open work, in priority order

**Update (Mac session, 2026-11-06, end):** items 1, 3, 4 and 5 are done; item 2 (P6) is dropped by user decision (see the addendum, "Decisions of 2026-11-06"). Phase 4 tasks 1-4, 6-10, 12 and 13 are merged; 11, 14 and 15 were in flight. Depth alignment goes into `SplatDepth`.

1. **Manual app check on this Mac** (Xcode 27.x): build, launch, "Import COLMAP..." with a COLMAP folder (`Tools/synthetic/make_colmap_scene.py` makes one; the user's `table-scene` zip may be in Downloads), press Train, watch the viewport. Report/fix problems.
2. **P6 bundled COLMAP** (decision A4: keep the App Sandbox, ship COLMAP as helper via XPC). Spike 1 (`Tools/ci/colmap_spike.sh`) result: the Homebrew build is a no-go (GPL closure: Qt, ffmpeg, x264/x265, SuiteSparse). Spike 2 (`Tools/ci/colmap_minimal_build.sh`, workflow `colmap-minimal-build.yml`) builds Ceres without SuiteSparse and COLMAP without GUI/CUDA/ONNX/CGAL; CI run 2 got through Ceres (builds in ~6 min without SuiteSparse) and failed in COLMAP configure on OpenMP (`libomp` is keg-only; fixed with `-DOpenMP_ROOT`, re-run pending on CI). Run 3 then failed in COLMAP configure on CHOLMOD (a hard requirement; its supernodal module is GPL): `Tools/ci/patch_colmap_no_cholmod.py` patches COLMAP to use only Eigen SimplicialLDLT, and the build now also turns off MVS, LSD (AGPL) and downloads. On the Mac the same script can be run locally (`bash Tools/ci/colmap_minimal_build.sh`) and iterated quickly. Goal: a minimal permissive build, closure size and license list, headless run on `Tools/ci/make_textured_scene.py` output (needs numpy, pillow). Then re-sign with our identity, `@rpath` relink, XPC helper; Opus review for the XPC/sandbox design.
3. **P8 depth model**: done (pinned in `ModelCatalog.depthAnythingV2Small`, entitlement added, real download verified outside the sandbox). Results are in the addendum, "P8 result". Still to check: the download inside the sandboxed app, once P9 has a UI for it.
4. **P7 Apple Photogrammetry provider**: `PhotogrammetrySession` poses; it returns poses but no intrinsics, and our photo import strips EXIF. Decide how to keep the focal length (store 35 mm-equivalent focal length in `FrameProvenance` at import) before building it.
5. **P9 pipeline UI** (app target): entry points (video / photo folder / COLMAP / ARKit), frame strip with swap/remove, frame-count setting (default 6) plus "use all frames", sampling interval, step list Frames -> Poses -> Depth -> Train, presets. Use the `liquid-glass:liquid-glass` and `swiftui-macos-skill:swiftui-macos` skills; run `scripts/check_glass.py --target 26.0` on changed UI files.
6. Then the main Phase 4 plan tasks 1-22 (depth channel in the kernels, Pearson loss, dense depth init, pseudo-views, regularizers, leave-one-out, benchmark). Task 20 uses P8's model location, Task 22 is merged with P9. Plan deviations that still need the user's approval: isolation pruning on the CPU instead of GPU grid hash; scale/shift alignment in SplatCore instead of SplatDepth.
7. Fill `docs/ARCHITECTURE.md` decision rows for A1-A8 as tasks land; update the status line in `CLAUDE.md`.

## Working rules that matter on the Mac

- Answer in German; code, identifiers, comments and commit messages in English.
- Routing (CLAUDE.md): Sonnet orchestrates, reviews, tests and commits; Haiku workers do the bulk coding (`Agent` with `model: "haiku"`, worktree isolation, "do not commit"); Opus only for the hardest design/review. Only the orchestrator commits and pushes.
- Checkpoint: commit + push after every reviewed task, at least every ~30 min. No PR unless asked.
- Never claim a Metal/Xcode result that was not actually run. On the Mac you *can* run it: `cd SplatKit && swift test`, then `xcodegen generate` and `xcodebuild -project SparseSplat.xcodeproj -scheme SparseSplat -destination 'platform=macOS' build`.
- Type-checker rule: explicitly typed `let` steps and loops in numeric code and test fixtures (Xcode rejects what the Linux toolchain accepts).
- Licensing: never port code from `graphdeco-inria/gaussian-splatting`; only permissive components in the bundle; no new third-party dependencies without asking.
- Changing a recorded decision requires asking the user (use `anthropic-skills:grill-me`).
- The cloud session may still push to the same branch (it has CI checks scheduled, e.g. the spike 2 re-run). `git pull --rebase` before pushing.

## Update 2026-11-07 (end of the Mac session)

Items 1–7 above are done: the pose stage is Apple Photogrammetry plus the existing COLMAP import (decision of 2026-11-06), the pipeline UI (P9) and Phase 4 tasks 1–22b are merged, CI is green. Measurements and the decision on training defaults (depth prior, dense init, anisotropy, isolation pruning on; scale cap, pseudo-views, early stopping off) are in `docs/superpowers/plans/2026-11-06-phase-4-addendum-input-pipeline.md`. Verified since: the held-out readout in the running app, and the per-view visibility test was measured and not built (see the addendum). Still open: why the depth prior loses at 4 synthetic views (pseudo-views are explained: their warped depth target is too noisy, see the addendum), the viewer's default up-axis for Apple-pose projects, and the first run of the benchmark workflow on GitHub.
