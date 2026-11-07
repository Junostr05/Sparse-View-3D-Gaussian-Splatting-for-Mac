# CLAUDE.md

Guidance for Claude Code when working in this repository.

# Claude Code Agent Configuration (v5.5 Pipeline)

## Subagent Routing Rules
- **Orchestrator Mode:** Enabled
- **Primary Conductor / Reviewer:** Claude 5.5 Opus (`claude-opus-5-5`)
- **Task Workers & Subagents:** Claude 5.5 Sonnet / Claude 5.5 Haiku

## Multi-Agent Architecture (5.5 Family)
- **Claude 5.5 Opus:** Strategic Orchestrator. Keeps full contextual judgment. Acts as the exclusive final code reviewer. Never writes raw boilerplate.
- **Claude 5.5 Sonnet:** Core Builder. Handles advanced implementation, refactoring, and general script generation.
- **Claude 5.5 Haiku:** Scout & Runner. Executes unit tests, runs terminal lookups, checks dependencies, and generates documentation.

## Execution Workflow (GitHub / Local App)
1. **Decomposition:** Opus 5.5 analyzes incoming GitHub issues/PR tasks and splits them into distinct sub-tasks.
2. **Delegation:** High-volume or mechanical steps are routed to Haiku 5.5 (tests/scouts). Structural code writing is routed to Sonnet 5.5.
3. **Context Isolation:** Subagents process tasks independently in their mini-contexts to optimize token usage.
4. **Strict Review Loop:** Subagents are forbidden from committing code directly. All output must be returned to Opus 5.5.
5. **Approval & Gatekeeping:** Opus 5.5 cross-checks the code against repo standards. It will either issue precise correction guidelines back to the 5.5 workers or approve and execute the final commit.

### How this maps to Claude Code
- **Routing rule (user decision 2026-11-05, overrides the roles above):**
  - **Sonnet is the main model.** It orchestrates, reviews every worker result, runs the tests (locally on Linux, via CI for macOS/Metal), and is the only one that commits and pushes. If a result is wrong or a test fails, Sonnet sends it back to the worker with precise correction notes (or fixes small issues itself).
  - **Haiku does the bulk of the coding.** All implementation work goes to Haiku workers by default.
  - **Opus is consulted only for the most demanding tasks**, e.g. architecture, data structures, GPU memory layouts or a hard design trade-off — not for routine coding or review.
- Spawn workers with the `Agent` tool and set `model: "haiku"` (default for implementation) or `model: "opus"` (architecture-level questions only).
- Every worker prompt must be self-contained (goal, files, constraints, expected output) and must say: *do not commit or push — return the diff/result.*
- Only the orchestrator runs `git commit` / `git push`, after reviewing the worker output against the standards below.

---

# Project

**Sparse-View 3D Gaussian Splatting for Mac** — a native macOS app that reconstructs a 3D Gaussian Splatting scene from only a few input images and renders/inspects it interactively on Apple Silicon.

> Status: Phases 1 (viewer), 2 (differentiable rasterizer: CPU + Metal backward, verified against PyTorch goldens and finite differences) and 3 (training: COLMAP/ARKit import, CPU references, Metal Adam/loss/densification, GPUTrainer/TrainingSession, `.gsproj` project window with Train button and live preview) done and green on CI (197 library tests on a real Metal device + app build). Exercised on a real COLMAP dataset (13 iPhone photos) in the app on a Mac: import, Train and live preview work (headless: 757 k Gaussians, 32 dB mean PSNR on the training views after 10 000 iterations). Phase 4 (sparse-view quality) is implemented and merged (input pipeline, depth prior, dense init, regularizers, pseudo-views, leave-one-out, pipeline UI; CI green): measured on a synthetic benchmark and on the 13 real photos, only the depth prior gives a clear gain (+0.7 dB held-out), so defaults are depth prior + dense init + anisotropy + isolation pruning on, and scale cap, pseudo-views and early stopping off (decision 14, revised 2026-11-07); open questions are in the addendum. Phase 5 (editing and export) is implemented: brush selection, non-destructive deletion with undo/redo and persistence in the project, export to `.ply`/`.splat`/`.spz`, plus the Blender-style viewer with camera views (see `docs/HANDOVER-mac-session.md`); plans live in `docs/superpowers/plans/`. **`docs/ARCHITECTURE.md` is the source of truth** for all decisions (decision log, modules, phases, defaults). Read it before any implementation work. Changing a recorded decision requires asking the user first (use the `grill-me` skill).

## Tech stack (target)
- **Platform:** macOS 26+ (macOS 27 APIs gated with `#available`) on Apple Silicon. Baseline M1 / 8 GB with an adaptive Gaussian budget; M3+ extras (`MTLGPUFamily.apple9`, hardware ray tracing) enabled at runtime, always with a base-tier fallback.
- **Language:** Swift 6.2, strict concurrency, `@MainActor` default isolation. Check whether `NonisolatedNonsendingByDefault` (SE-0461) is enabled before reasoning about where `nonisolated async` code runs.
- **UI:** SwiftUI with Apple's Liquid Glass design. AppKit (`NSViewRepresentable`, `NSWindow`) only where SwiftUI has no equivalent — e.g. hosting an `MTKView` for the 3D viewport.
- **Rendering / compute:** pure Metal compute — hand-written forward/backward kernels, GPU radix sort, Adam, losses. No autodiff framework. The viewer reuses the training forward rasterizer.
- **App model:** document-based (`DocumentGroup`, `.gsproj` package).
- **Layout:** local Swift package with modules `SplatCore`, `SplatMetal`, `SplatIO`, `SplatDepth`, `SplatTraining` + app target via XcodeGen (`project.yml`; never commit an `.xcodeproj`).
- **Dependencies:** no third-party frameworks without asking. Apple's open-source Swift packages (`swift-collections`, `swift-algorithms`, `swift-async-algorithms`) are allowed.

## Skills to use
- `swiftui-macos-skill:swiftui-macos` — for any SwiftUI/macOS code: observation, concurrency, performance, AppKit bridging, multi-window.
- `liquid-glass:liquid-glass` — for any UI chrome, toolbars, inspectors, sheets, buttons. Run its linter (`scripts/check_glass.py --target 26.0`) on changed Swift UI files.
- `anthropic-skills:grill-me` — before any significant design or architecture decision, interview the user one question at a time.

## UI rules (Liquid Glass)
- The **3D viewport, image thumbnails and point-cloud previews are the content layer** — never put `glassEffect` on them.
- Glass is only for the **functional layer**: toolbars, floating viewport controls (orbit/zoom/reset, render-mode switch), inspector chrome.
- Prefer system components (`NavigationSplitView`, `.toolbar`, `.inspector`, `.buttonStyle(.glass)` / `.glassProminent`) — they are already glass; don't re-glass them.
- Floating custom controls over the viewport go in one `GlassEffectContainer`. Use `.regular` by default; `.clear` only over bright rendered content with a dimming layer.
- Text on glass uses `.primary`; carry hierarchy with size and weight.

## Swift conventions
- `@Observable` models; `@State` owns them at the root; pass down via environment or plain properties.
- No `DispatchQueue` for new code — use structured concurrency. Long-running training/optimization runs off the main actor and reports progress via `AsyncStream`.
- Keep heavy GPU state (buffers, pipelines) out of observed properties (`@ObservationIgnored`) to avoid view-update cascades.
- **Type-checker timeouts:** Xcode's Swift type checker rejects expressions that the Linux toolchain accepts (seen with long mixed SIMD/Float arithmetic and with `map { SIMD3(Float(...) / 255, ...) }` closures). Write numeric code and test fixtures with explicitly typed `let` steps and loops; a green Linux `swift test` does not prove the macOS build compiles.
- Accessibility: every icon-only control gets an accessibility label; respect Reduce Motion for viewport animations.

## Gradient correctness (non-negotiable)
- Every backward kernel is checked three ways: against the Swift CPU reference, against gsplat goldens (`Tools/golden/`), and with finite differences.
- No backward kernel is merged without these tests. A subtly wrong gradient does not crash — it silently trains worse.

## Build & verification
- Swift 6.2 for Linux can be installed in the cloud container with `Tools/install-swift-linux.sh` (idempotent; it prints the `PATH` export line to use).
- `SplatCore` and `SplatIO` build and test on Linux: `cd SplatKit && swift test`.
- `SplatMetal` and the app are macOS-only — verified by the CI `macos` job or locally on a Mac. Shaders are compiled at runtime, so `swift test` also works on macOS without the Xcode Metal compiler.
- App build: `xcodegen generate`, then `xcodebuild -project SparseSplat.xcodeproj -scheme SparseSplat -destination 'platform=macOS' build`.
- Library modules (`SplatCore`, `SplatIO`, `SplatMetal`) use nonisolated default isolation; only the app target uses `@MainActor` default isolation. Never set `defaultIsolation(MainActor.self)` on a library target.
- Never claim a macOS/Metal build or test passed unless CI or a Mac actually ran it.
- What *can* be checked in the cloud: the Liquid Glass linter (Python), Python tooling/scripts (golden generation), docs.

## Licensing (hard rules)
- Repo is Apache-2.0.
- **Never port code from `graphdeco-inria/gaussian-splatting`** or its rasterizer (non-commercial license). gsplat (Apache-2.0) is the allowed code reference; implement FSGS/SparseGS-style methods from the papers.
- Only commercially usable models in the app bundle (Depth Anything V2 Small). Non-commercial models (MASt3R, larger Depth Anything) are optional user downloads, never shipped.

## Git
- Never commit secrets, datasets, trained scenes (`*.ply`, `*.splat`) or large binaries — keep them out of git.
- Clear, descriptive commit messages. Do not open pull requests unless asked.
- **Checkpoint regularly (user request 2026-11-05):** commit and push to the working branch after every reviewed and merged task, and at the latest every ~30 minutes of work (use a `WIP:` commit if a task is half done). Worker output only exists in ephemeral worktrees (they were lost once on a session restart): merge or WIP-commit it as soon as it returns, never let several finished tasks pile up uncommitted. Before pausing or ending a session, make sure `git status` is clean and the remote head equals the local head.

## Communication
- The user writes in German — answer in German. Code, identifiers, comments and commit messages in English.

## Phases
Called *Phase 1…6* — never "M1…", which collides with chip names. 1 Viewer → 2 Differentiable rasterizer → 3 Training → 4 Sparse-view quality → 5 Editing & export → 6 Pose-free. Details in `docs/ARCHITECTURE.md`.
