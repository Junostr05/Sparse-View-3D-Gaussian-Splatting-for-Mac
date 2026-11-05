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
- Spawn workers with the `Agent` tool and set `model: "sonnet"` (builder) or `model: "haiku"` (scout/runner).
- Every worker prompt must be self-contained (goal, files, constraints, expected output) and must say: *do not commit or push — return the diff/result.*
- Only the orchestrator runs `git commit` / `git push`, after reviewing the worker output against the standards below.

---

# Project

**Sparse-View 3D Gaussian Splatting for Mac** — a native macOS app that reconstructs a 3D Gaussian Splatting scene from only a few input images and renders/inspects it interactively on Apple Silicon.

> Status: repository just initialized. Architecture, training backend and data pipeline are **not decided yet** — see *Open decisions*. Do not assume a structure; resolve it with the user first (use the `grill-me` skill).

## Tech stack (target)
- **Platform:** macOS 26+ on Apple Silicon.
- **Language:** Swift 6.2, strict concurrency, `@MainActor` default isolation. Check whether `NonisolatedNonsendingByDefault` (SE-0461) is enabled before reasoning about where `nonisolated async` code runs.
- **UI:** SwiftUI with Apple's Liquid Glass design. AppKit (`NSViewRepresentable`, `NSWindow`) only where SwiftUI has no equivalent — e.g. hosting an `MTKView` for the 3D viewport.
- **Rendering / compute:** Metal (splat rasterization, sorting, optimization kernels).
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
- Accessibility: every icon-only control gets an accessibility label; respect Reduce Motion for viewport animations.

## Build & verification
- The cloud Claude Code container runs **Linux without Swift/Xcode/Metal**. Swift code cannot be compiled or run there — say so explicitly instead of claiming a build passed.
- What *can* be checked in the cloud: the Liquid Glass linter (Python), Python tooling/scripts, docs.
- Real builds and tests happen locally on a Mac (`xcodebuild` / Xcode). Build commands will be added here once the project structure exists.

## Git
- Never commit secrets, datasets, trained scenes (`*.ply`, `*.splat`) or large binaries — keep them out of git.
- Clear, descriptive commit messages. Do not open pull requests unless asked.

## Communication
- The user writes in German — answer in German. Code, identifiers, comments and commit messages in English.

## Open decisions
- [ ] Training backend: native Metal kernels vs. MLX vs. Python/PyTorch-MPS pipeline driven by the app.
- [ ] Camera pose estimation for sparse views (own SfM, COLMAP import, learned pose prior).
- [ ] Sparse-view regularization approach (depth priors, monocular depth model, etc.).
- [ ] Project layout: Xcode project vs. XcodeGen vs. Swift Package with app target.
- [ ] Scene file formats for import/export (`.ply`, `.splat`, own format).
- [ ] Minimum deployment target (26 only, or 27 APIs gated).
