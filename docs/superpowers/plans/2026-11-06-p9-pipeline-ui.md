# P9 — Input pipeline UI (app target)

Part of `2026-11-06-phase-4-addendum-input-pipeline.md` (decisions A1–A8, task P9). The library pieces it uses already exist: `PhotoFolderImporter`, `VideoFrameExtractor` (`scoreCandidates` / `frames(from:atRequestedTimes:)`), `FrameSelectionPlan`, `PoseProviderChain`, `ApplePhotogrammetryPoseProvider`, `ProjectBundle.importEstimatedPoses`, `TrainingPreset`, `ModelStore` / `URLSessionModelDownloader` / `ModelCatalog.depthAnythingV2Small`.

Skills for UI work: `swiftui-macos-skill:swiftui-macos` and `liquid-glass:liquid-glass`; run `scripts/check_glass.py --target 26.0` from the latter on every changed UI file.

## What the user can do

1. **Start.** In an empty project the inspector offers four entry points: *Photos…* (a folder), *Video…* (a movie file), *COLMAP…*, *ARKit Capture…*. COLMAP and ARKit work as today (they carry poses and skip the pose stage).
2. **Frames.** Photos or a video produce *candidates*. The detail area shows them as a thumbnail grid (the *frame strip*); selected frames have an accent border and a check mark, a click toggles a frame. The inspector holds the selection settings:
   - mode: **Best N** (stepper, default 6, 1…candidate count) or **Use all frames** (toggle). Photo folders start on *all*, videos on *best 6* (decision A2).
   - video only: **sampling interval** (0.1 s, 0.25 s default, 0.5 s, 1 s). Changing it re-scores the video.
   - a line "N of M frames selected" and the `FrameSelectionPlan.warning` as plain text (too few / many views / none).
   Manual toggles stay until the mode or interval changes (documented in the UI: "Changing the mode resets manual changes").
3. **Poses.** The single prominent action in this state is **Estimate Poses**. For a video it first decodes the selected frames at full size; then it runs `PoseProviderChain(providers: [ApplePhotogrammetryPoseProvider()])`. Status text shows the running step and, afterwards, either "13 of 13 images registered (Apple Photogrammetry)" or the failure reason (`PoseEstimationFailure`) with the hint to import a COLMAP folder instead. On success the poses are written with `ProjectBundle.importEstimatedPoses(frames:result:inputKind:into:)` (`inputKind` `.video` or `.photos`), the project adopts them, the frame strip disappears and the normal project view (image list, training) takes over.
4. **Depth.** A row shows the depth model state from `ModelStore.status(for: ModelCatalog.depthAnythingV2Small)` (not downloaded / ready / damaged) with a *Download (50 MB)* button, a progress bar and the error text when offline. The model is **not used by training yet** (Phase 4 tasks 20+): say so in the row ("Used for depth guidance in a later update"). The store lives in the app container, `Application Support/SparseSplat/Models`.
5. **Train.** As today (Train / Pause / Stop). A **preset** picker (Fast / Standard / Quality) sits above it and writes `TrainingPreset.apply(to: &manifest.training)` into the project; it is disabled while training. The picker also shows for COLMAP/ARKit projects.
6. A compact **step list** at the top of the inspector: Frames → Poses → Depth → Train, each with a status symbol (pending / running / done / failed / skipped) derived from the pipeline state. COLMAP/ARKit projects show Frames and Poses as *skipped* ("Poses come from the capture").

## State and structure

New folder `App/Pipeline/`:
- `PipelineModel.swift` — `@Observable`, owns everything pending: the source (`.photos(folder)` or `.video(url)`), the candidates (`[CapturedFrame]` for photos; `[VideoCandidate]` for video), the `FrameSelectionPlan`, the sampling interval, the step states, status/error text, the depth model state and its download progress. Not persisted. Heavy work in `@concurrent` functions of a `nonisolated enum` (same pattern as `ProjectImport` in `App/Project/ProjectModel.swift`); results come back to the main actor.
- `FrameStripView.swift` — the grid (`LazyVGrid`, adaptive ≥ 140 pt, thumbnails from PNG data via ImageIO thumbnails (decode off the main actor and cache in the model), caption with time or file name). Content layer: **no glass**. Every cell has an accessibility label ("Frame 3 at 1.25 s, selected") and the toggle trait.
- `PipelineInspector.swift` — the sections described above; replaces the import section of `ProjectInspector` when the project has no images or the pipeline has pending frames.
- Edit `ProjectModel.swift`, `ProjectWindow.swift`, `ProjectInspector.swift` as needed to host the above; keep their existing behaviour (COLMAP/ARKit import, training, checkpoints).

**Sandbox access.** `fileImporter` URLs are security-scoped. For a video, decoding happens later than the import click, so the model must hold `startAccessingSecurityScopedResource()` from selection until the pipeline is discarded or finished, and release it then (also when the window closes). Photo folders are read fully at import time (`PhotoFolderImporter.load(folder:maxSide:)`, 1600 px, PNG), so the scope can end right after.

**Concurrency.** The app is `@MainActor` by default with `NonisolatedNonsendingByDefault`: put blocking decode/IO/Photogrammetry calls in `@concurrent` functions. Cancel the running task when the user starts a different import or closes the pipeline. Never block the main actor.

**Cleanup.** *Discard* (a bordered button in the Frames section) drops the pending frames and returns to the empty state.

## Liquid Glass

Follow `CLAUDE.md`: system components only (`Form`, `.inspector`, `Picker`, `Stepper`, `Toggle`, `ProgressView`, `.buttonStyle(.glassProminent)` for the one primary action of each state, default or `.bordered` for the rest). No custom `glassEffect`. The strip and thumbnails are content: no glass. Text on controls uses `.primary`. Every icon-only control gets an accessibility label. Respect Reduce Motion for any animation (prefer none).

## Verification

- `xcodegen generate` then `xcodebuild -project SparseSplat.xcodeproj -scheme SparseSplat -destination 'platform=macOS' build` must succeed without new warnings.
- `scripts/check_glass.py --target 26.0` on every changed Swift UI file.
- `cd SplatKit && swift test` stays green (the app target has no unit tests).
- Manual flow on this Mac (orchestrator): Photos… with the 13 table photos -> frame strip -> Estimate Poses -> train; Video… with a short phone clip; Download the depth model inside the sandbox.
