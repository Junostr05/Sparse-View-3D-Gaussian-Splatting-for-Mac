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

Items 1–7 above are done: the pose stage is Apple Photogrammetry plus the existing COLMAP import (decision of 2026-11-06), the pipeline UI (P9) and Phase 4 tasks 1–22b are merged, CI is green. Measurements and the decision on training defaults (depth prior, dense init, anisotropy, isolation pruning on; scale cap, pseudo-views, early stopping off) are in `docs/superpowers/plans/2026-11-06-phase-4-addendum-input-pipeline.md`. Verified since: the held-out readout in the running app, and the per-view visibility test was measured and not built (see the addendum). Still open: the Phase 4 quality target (at least baseline + 1 dB at 3 and 4 views) is not met at 4 views (the 4-view result is seed noise around a slightly negative mean, see the addendum; pseudo-views are explained: their warped depth target is too noisy), and the first run of the benchmark workflow on GitHub.

Viewer up axis (2026-11-07): Apple Photogrammetry and ARKit keep a +Y-up world (measured: mean camera up vector +Y, cameras above the point cloud), COLMAP uses -Y up. `UpAxis.forInputKind` sets the viewer's start value from the project's input kind, and the project inspector has an "Up Axis" picker under "Viewport". Checked in the app on the 13 table photos: the trained scene now stands upright.

## Viewer controls (2026-11-07)

The viewer uses `ViewCamera` (SplatCore: turntable camera with roll, Blender views, flight, key mapping; `OrbitCamera` stays as the reference for the gesture directions and the framing). A project opens in the view of its first estimated camera and the camera list comes from `ProjectBundle.cameraPoses` (no image decoding).

- Mouse: middle drag orbits, Shift+middle pans, Ctrl+middle zooms; wheel zooms, Ctrl+wheel pans horizontally, Shift+wheel vertically. Left drag orbits (Option pans), right drag pans.
- Trackpad: two-finger scroll orbits, Shift pans, Ctrl or Command zooms, pinch zooms.
- Keys (top row or numpad): 1/3/7 front/right/top, Ctrl+1/3/7 back/left/bottom (the whole scene is framed, unlike Blender, because a camera view's pivot can lie inside the scene); 4/6/8/2 orbit by 15 degrees, with Ctrl they pan; 0 returns to the camera view; `[` and `]` previous/next camera; `.` or Home frames the scene; Space plays or pauses the camera flight.
- Inspector "Viewport": Up Axis, View menu, Camera picker, "Fly through" slider and Play/Pause. Reduce Motion makes flights jump.
- A keyboard button (toolbar of the .ply window, "Viewport" section of the project inspector) opens a popover with all shortcuts (`ViewerShortcutsView`).
- Not done: orthographic view (key 5; the rasterizer is perspective only), selection-based framing.

## Editing and export (Phase 5, 2026-11-07)

Edit mode (paintbrush in the toolbar, project window and .ply/.splat/.spz viewer window): left drag paints a selection by projected centre (x-ray, brush radius in points, Shift subtracts), Delete/X deletes, A / Option+A / Ctrl+I select all / clear / invert, Cmd+Z / Shift+Cmd+Z undo / redo, plus a glass control bar with the same actions. Deletions are non-destructive: a project stores `edits/deleted.bin` (+ `edits` in the manifest) next to the checkpoint; a new checkpoint or import drops them. In the viewer window edits live in memory and are saved by exporting. Export (toolbar) writes `.ply`, `.splat` (DC colour only) or `.spz` (v2, gzip, SPZ's own RUB frame; v3/v4 files cannot be read) with the deletions applied.

Checked in the running app: painting, delete, undo/redo, delete-all and restore, persistence across save and reopen (27 901 deleted Gaussians restored), export sheet. Not checked end to end: the saved export files in other viewers (only round-trip unit tests), the Save panel with a typed path (an automation artefact), pasteboard and trackpad gestures. The brush has an optional Front Only mode (key F, toggle in the glass bar). Known limit: the brush projects every Gaussian per mouse event (fine at 757 k Gaussians on this Mac, not measured on slower ones). Exporting drops non-finite Gaussians (a stored checkpoint can contain a few); the exports were verified against an independent reader (`.splat` exact, `.spz` v2 layout, positions within one quantisation step, RUB frame).

Gestures verified with synthetic events (2026-11-07): middle drag orbits, Shift+middle pans, Ctrl+middle zooms (down = out), wheel zooms, trackpad-style pixel scrolling orbits, and Front Only selected 8 074 of 9 350 Gaussians for the same stroke. Note for automation: a synthetic drag needs the delta fields set, and the first click on an inactive window only activates it.

## Release (2026-11-07)

`Tools/release/make_dmg.sh` builds the Release app (arm64 only; the default Release build would also compile x86_64, which `SplatDepth` does not build for) and packs it into `build/SparseSplat-<version>.dmg` (ad-hoc signed, runs on this Mac; checked: the Release app from the mounted DMG renders a project). Decision 13 (notarized DMG via GitHub Releases) additionally needs things only the owner can supply: a Developer ID Application certificate in the keychain and a notarytool keychain profile, then `Tools/release/make_dmg.sh --sign "Developer ID Application: …" --notarize <profile>` (untested, no credentials here). A GitHub release workflow is not written because it would need those secrets in the repository settings.

## Phase 6: pose-free fallback (2026-11-07)

Depth Anything 3 (Apache-2.0) is a `PoseProvider` (`SplatDepth/DepthAnything3PoseProvider.swift`; maths in `SplatIO/DepthAnything3Math.swift`) behind Apple Photogrammetry. The pipeline's Poses step has a "Method" picker (Automatic / Apple Photogrammetry / Depth Anything 3) and a "Pose-Free Model" section with "Import Package…" and "Remove". Automatic falls back to DA3 only when its package is installed. DA3 projects store `poseMethod` in the manifest and start the viewer with -Y up (the DA3 world is the first camera's OpenCV frame).

The Core ML package is built with `Tools/da3/convert_da3.py` (one function per view count 4/6/8/12/16 and orientation, int8 weights, about 580 MB, bilinear instead of bicubic upsampling, reference-view reordering off) and is **not published**: it has no download URL (`ModelDescriptor.depthAnything3Base` stays "not configured") until the owner uploads it, for example as a GitHub release, and fills in `baseURL` and the file hashes. A local copy of the package for testing was imported into this Mac's app container (`Application Support/SparseSplat/Models/depth-anything-3-base/1/`, about 580 MB, remove it with the Remove button).

Measured (13 table photos, 3 held-out views, 5000 iterations, no Phase 4 features): held-out PSNR/SSIM Apple 14.2/0.34, COLMAP 14.4/0.38, DA3 Base 13.4/0.23 (Swift provider: 13.3–13.6/0.23–0.24), DA3 Small 12.9/0.23. DA3 cameras are 2.0 % (centres) and 1.3° (orientation) from COLMAP. Findings: rotated inputs degrade a lot (the model expects upright images); the camera frame needs no correction (a depth-consistency test picks the identity); too few or too strict SfM-like points lower the score, so the provider keeps the confident half of every view. Not tested: a capture that Apple Photogrammetry really cannot register (the case this fallback is for), landscape photos, video frames.
