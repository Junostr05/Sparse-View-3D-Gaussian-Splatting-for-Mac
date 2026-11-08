# sparsesplat: command-line tool

`sparsesplat` trains, continues and exports `.gsproj` projects without the app. It is a
macOS command-line tool built from the same project (`project.yml`, target `sparsesplat`).
It is not sandboxed, so it can read and write any path the user can.

Build it with XcodeGen and Xcode:

    xcodegen generate
    xcodebuild -project SparseSplat.xcodeproj -scheme sparsesplat -destination 'platform=macOS' build

The binary is written to the build products folder (`TARGET_BUILD_DIR`, shown by
`xcodebuild -scheme sparsesplat -showBuildSettings`). Copy it somewhere on your `PATH`.

## Commands

    sparsesplat train <project.gsproj> [--preset fast|standard|quality] [--iterations N]
                      [--continue] [--depth-model <file.mlpackage>] [--json]
    sparsesplat export <source> --output <file> [--format ply|splat|spz] [--json]
    sparsesplat info <project.gsproj> [--json]
    sparsesplat import-colmap <folder> --output <project.gsproj> [--json]
    sparsesplat --help | sparsesplat <command> --help | sparsesplat --version

### train

Trains the project headless and writes the result back into the package.

- Without `--continue` the run starts from the original poses (the stored refined poses are dropped), as
  "Retrain from Scratch" does in the app.
- With `--continue` the run takes the stored checkpoint (minus its stored deletions) and runs one round of
  `--iterations` iterations (default: the round length of the continuation settings, 5 000). The stored refined
  poses are kept and refined further.
- `--preset` replaces the stored preset (image size, iterations and schedules). `--iterations` sets the
  iterations of the run, or of the one round of `--continue`.
- `--depth-model <file.mlpackage>` copies a Core ML depth model into the tool's model store before the run, the
  same way the app's Depth Model section imports one (see below).

The package is written only after the run finished. A failed run leaves the package unchanged. Ctrl-C stops the
run after the current iteration: the last completed model is stored and the tool exits with code 130. A second
Ctrl-C ends the process at once, and then nothing is stored.

Progress goes to standard output, one line per logged step, in a stable format:

    progress iteration=1200/5000 loss=0.031 gaussians=412000 ips=9.2

`iteration` and the total are the iterations of this run (for `--continue`, the round), not the cumulative count.
A missing loss prints as `-`. The last line reports the stored checkpoint:

    Checkpoint stored: iteration 10200, 757864 Gaussians, PSNR 28.63 dB, 2 checkpoints in the package.

Warnings (for example "Depth Anything V2 Small is not installed") go to standard error.

### export

Exports the latest checkpoint of a `.gsproj` package, without its stored deletions, or a `.ply` checkpoint file.

- `--format` is `ply` (standard 3DGS, full SH), `splat` (antimatter15 `.splat`, DC colour only) or `spz`
  (compressed, version 2). When omitted it is taken from the output file extension.
- Gaussians with NaN or infinite values are dropped in every format, as the app does.

### info

Prints the photo count, input kind, pose source, the stored preset, the number of checkpoints, the latest
checkpoint (iteration, Gaussians, PSNR of the training views, held-out view if any), the stored deletions and
whether refined poses exist.

### import-colmap

Creates a package from a COLMAP folder that contains `sparse/0/` (`cameras.bin`, `images.bin`, optionally
`points3D.bin`) and `images/`. The package gets the default training preset. An existing package at the output
path is replaced.

## JSON output

`--json` prints one JSON object per line (JSON Lines, sorted keys):

- `train` writes `{"event":"progress", ...}` for each logged step and one `{"event":"result", ...}` at the end.
- `export`, `import-colmap` and `info` write one object (`info` writes the summary object itself).
- A failed command with `--json` writes `{"event":"result","ok":false,"message":...}` to standard output and exits
  with code 1. Usage errors always print to standard error.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Failure: the package could not be read or written, training failed, export failed |
| 64 | Usage error (unknown command or option, missing or invalid value) |
| 130 | Training stopped with Ctrl-C; the last completed model was stored |

## Depth models and the model store

The tool keeps its models in `~/Library/Application Support/SparseSplat/Models`. The sandboxed app keeps its
own models inside its container, so the two stores are separate. Without a depth model the depth prior is skipped
and the run warns. Pass `--depth-model <file.mlpackage>` once to import a converted model into the tool's store;
later runs find it there. The tool does not download models: use the app's Depth Model section to download one and
then import the package for the tool.

## Examples

    sparsesplat info Scene.gsproj
    sparsesplat import-colmap ~/captures/scene -o Scene.gsproj
    sparsesplat train Scene.gsproj --preset quality
    sparsesplat train Scene.gsproj --continue --iterations 2000 --json
    sparsesplat export Scene.gsproj --output Scene.spz
    sparsesplat export Scene.gsproj --format ply -o Scene.ply

Work on a copy of a project when you experiment: `train` and `--continue` replace the package's checkpoint.

## Implementation

- Argument parsing, help text and report formatting: `SplatKit/Sources/SplatIO/CLI/` (tested on Linux and macOS,
  `SplatKit/Tests/SplatIOTests/CLIArgumentsTests.swift`).
- Training: `HeadlessTraining` in `SplatKit/Sources/SplatTraining/HeadlessTraining.swift`. The app's training queue
  (`App/Queue/`) runs the same function, so only one implementation exists.
- Package files: `ProjectPackageFiles` (read and atomic write of a `.gsproj` folder).
- Entry point: `CLI/SparseSplatCLI.swift`.
