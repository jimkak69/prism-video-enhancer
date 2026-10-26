# Prism Video Enhancer

An independent, from-scratch Rust implementation of the architecture documented in
`REPORT.md` (Topaz Video AI 6.0.4; kept outside this repository): a GUI and CLI that plan jobs as FFmpeg
command lines, plus a native engine that replaces Topaz's proprietary `tvai_*` FFmpeg filters.

Prism contains no Topaz code, model weights, assets, branding or binaries. It has no licensing or
telemetry code, and it never contacts a network service except when you ask it to download an
openly licensed model.

## What it does

| Feature | Prism | How |
|---|---|---|
| Upscaling and enhancement | ✓ | Real-ESRGAN networks (ONNX, GPU) or the native **Classic** model; sliders for revert compression, recover details, sharpen, reduce noise, dehalo, anti-alias/deblur, add noise and recover original detail; Auto, Relative and Manual parameter modes; focus fix; two-stage enhancement |
| Frame interpolation and slow motion | ✓ | Native **Flow** model (bidirectional hierarchical block motion, occlusion-aware warping, scene-cut and duplicate-frame handling), **Blend**, or **RIFE** v4.26 on the GPU (one-click download) |
| Stabilization | ✓ | Two passes: camera-motion analysis, then auto-crop or full-frame transform (libvidstab) |
| Deinterlacing, inverse telecine | ✓ | bwdif / fieldmatch + decimate |
| SDR → HDR10 / HLG | ✓ | Native inverse tone mapping to BT.2020 PQ/HLG; output tagged and auto-upgraded to a 10-bit encoder |
| Motion deblur, film grain | ✓ | Native kernels |
| Trim, crop, rotate, square pixels, 3D LUT | ✓ | FFmpeg filters in the decode chain |
| Export | ✓ | x264, x265 (8/10-bit), SVT-AV1, VP9, ProRes, DNxHR, FFV1, NVENC, AMF, QSV, PNG/TIFF/JPEG/EXR/DPX sequences; quality or bitrate rate control; audio copy/convert/none; subtitles copied into MKV |
| Preview | ✓ | Live single-frame enhancement while paused; "Render N s preview" clip for the full pipeline; split wipe and side-by-side with synchronized zoom and pan |
| Queue | ✓ | Parallel workers, priorities, **pause/resume without killing FFmpeg**, cancel with cleanup, error diagnosis with advice |
| Presets, projects | ✓ | Built-in and user presets (JSON); comparison tabs; project autosave |
| Benchmark | ✓ | `prism bench` measures every stage against FFmpeg's closest built-in filter |
| Cloud rendering, login/licensing, telemetry, OFX/AE plug-ins, alpha channel, multi-GPU | ✗ | Out of scope (see "Limitations") |

## Install

Packages are built into `dist/` by the scripts in `packaging/` (see "Packaging" below).

| System | Package | Notes |
|---|---|---|
| Windows 10/11 x64 | `PrismVideoEnhancer-<version>-windows-x64-setup.exe` | Per-user install, no administrator rights: Start menu and desktop shortcuts, an entry in Settings › Apps, optional `prism` on PATH. Uninstall from Settings › Apps (settings and models are kept unless you tick the box). A portable `.zip` is built too. |
| Ubuntu 20.04+, Debian 11+, Mint, Pop!_OS | `prism-video-enhancer_<version>_amd64.deb` | `sudo apt install ./prism-video-enhancer_<version>_amd64.deb` installs FFmpeg as a dependency; the app is in `/opt/prism-video-enhancer`, with `prism` and `prism-app` on PATH and a menu entry. |
| Other Linux x86_64 (Fedora, Arch, …) | `prism-video-enhancer-<version>-linux-x86_64.tar.gz` | Unpack and run `./install.sh` (per user, under `~/.local`); `./install.sh --uninstall` removes it. Install FFmpeg from your distribution. |
| macOS 11+ (Apple silicon and Intel) | `PrismVideoEnhancer-<version>-macos-universal.dmg` | Built on a Mac with `packaging/macos/build.sh`. Drag the app to Applications; `brew install ffmpeg`. Unless the build was signed with a Developer ID, open it the first time with right-click › Open. |

FFmpeg 4.4 or newer is required on every system (tested with 4.4, 8.1 and 9.0). Prism finds it on
`PATH`, next to its own executable, in winget, Chocolatey, Scoop and Homebrew locations, or wherever
`PRISM_FFMPEG` points.

## Build

Requirements: a recent stable Rust toolchain and FFmpeg (see above).

```powershell
cd replica
# Only for the x86_64-pc-windows-gnu Rust toolchain (rustup's default on machines without
# Visual Studio). It installs a small dlltool replacement and points the GNU target at it in
# your user-level %USERPROFILE%\.cargo\config.toml; the MSVC toolchain needs no setup.
powershell -ExecutionPolicy Bypass -File tools\setup-gnu-toolchain.ps1
cargo build --release
```

This produces `target/release/prism-app.exe` (GUI) and `target/release/prism.exe` (CLI).

The build targets **x86-64-v3** (AVX2 + FMA; Intel Haswell, AMD Zen or newer), set in
`.cargo/config.toml`, because the image kernels rely on auto-vectorisation. Remove that block to
run on older CPUs at reduced speed.

Why the GNU setup step exists: rustup's GNU toolchain ships `dlltool.exe` without the GNU assembler
it calls, so crates that use `raw-dylib` imports (the `windows-sys` family under winit, wgpu and
clap) fail to link. `tools/dlltool-shim` writes the GNU-style import libraries itself. It must use
the GNU format: MSVC-style short import libraries link without error, but leave some imports unbound
at run time when several crates import from the same DLL. rustc needs an absolute path to it, so
the setting goes in the user-level Cargo config (copied shim: `%USERPROFILE%\.cargo\bin\prism-dlltool.exe`),
never in the project's committed `.cargo/config.toml`.

### Building on GitHub

`.github/workflows/build.yml` builds and smoke-tests all three platforms on GitHub's hosted
runners, including the macOS `.dmg` on a Mac runner. Start it from the repository's Actions tab
(Run workflow) and download the packages from the run's Artifacts; pushing a tag such as `v0.1.0`
also publishes them as a GitHub Release. Hosted runners are free for public repositories; private
repositories use the account's monthly minutes, and macOS minutes count ten times.

### Packaging

| Output | Command | Where it runs |
|---|---|---|
| Windows installer and portable zip | `powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1` | Windows |
| Linux `.deb` and `.tar.gz` | `bash packaging/linux/build.sh`, or from Windows `powershell -ExecutionPolicy Bypass -File packaging\linux\docker-build.ps1` (Docker Desktop; builds on Debian 11 so the binaries need only glibc 2.31) | Linux, or Windows with Docker |
| macOS `.dmg` (universal) | `bash packaging/macos/build.sh` (optional `SIGN_IDENTITY`, `NOTARY_PROFILE`) | macOS with Xcode command line tools |

All three use the `dist` Cargo profile (release, no debug info, stripped). The Linux and macOS
packages bundle Microsoft's ONNX Runtime 1.20.1 (MIT): the CPU build on Linux (`ORT_FLAVOR=gpu`
bundles the CUDA 12 build instead) and the universal build with Core ML on macOS. The Windows
package relies on the ONNX Runtime with DirectML that Windows 11 ships.

The Windows installer (`crates/prism-installer`) is Prism's own: a small egui program that carries
the app as a zip archive appended to itself, so no installer toolkit is needed. `prism-setup.exe
--silent [--dir D] [--no-desktop] [--add-to-path]` installs without a window, and the installed
`uninstall.exe --uninstall --silent` removes exactly what `install.json` records. Adding to PATH
keeps the existing value's registry type and `%VARIABLE%` references.

## Desktop app

The window follows the layout documented in `REPORT.md` §3.3, with Prism's own green-accented
theme, drawn icons and naming:

- **Menu row** (File, Edit, View, Process, Help) with the standard shortcuts: Ctrl+I import,
  I/O mark in/out, Shift+1/2/3 view modes, Ctrl+0/1/2/4 zoom, Ctrl+P render preview, Ctrl+E export,
  Ctrl+Shift+P/E show the FFmpeg commands. The window is frameless, with its own window buttons.
- **Header**: project breadcrumb (switch, create, open projects), full-screen preview, single,
  split and side-by-side views, Presets, Input settings (rotation, deinterlacing, telecine, pixel
  aspect, colour overrides) and a panel toggle.
- **Comparison tabs**: each tab holds its own settings. In split and side-by-side views the left
  half picks what to compare against (Original or any tab); the right half picks the tab being
  edited. Each half shows a breadcrumb of active filters and a tab menu.
- **Viewer**: live processing of the paused frame (an "Unprocessed" badge shows until it is ready),
  scroll zoom and pan, a draggable split divider, press-and-hold for the original in single view,
  and an interactive crop tool.
- **Transport and timeline**: editable timecode, play and step controls, zoom presets, a live-render
  switch, the Render button with a length menu, a ruler, rendered-preview bars, a filmstrip, in/out
  handles and timeline zoom.
- **Right pane**: Adjustments (Enhancement with output resolution, video type, a model card
  carousel or list, sliders, focus fix, grain and parameters; second enhancement; frame
  interpolation; stabilization; motion deblur; SDR to HDR; grain) and Codec settings, plus Quick
  export and Export as.
- **Panels and dialogs**: Sources, Export queue (pause, resume, stop, reveal, error details),
  Presets (custom and recommended, favourites), Preferences (processing device, preview, parallel
  exports, folders), Model manager and About.

## GPU inference and models

Networks run through ONNX Runtime, loaded at run time. Search order: `PRISM_ORT_DYLIB`, the
library next to the executable (or in `runtime/`), then the copy that Windows 11 ships in System32
(ONNX Runtime 1.17 with DirectML) or the usual library folders on Linux and macOS. So on Windows 11,
any DirectX 12 GPU works with nothing extra installed. Providers by system: DirectML or CUDA on
Windows, CUDA on Linux (with a CUDA build of ONNX Runtime), Core ML (GPU and Neural Engine) on
macOS, and the CPU everywhere. `prism devices` shows what was found.

Install the openly licensed Real-ESRGAN weights (BSD-3-Clause). Prism downloads the published
`.pth` files and converts them to ONNX (FP32 and FP16) without PyTorch:

```powershell
prism models fetch resr-general-v3      # General (real-world footage)
prism models fetch resr-anime-v3        # Animation, fastest network
prism models fetch rife                 # RIFE v4.26 frame interpolation (GPU)
prism models fetch all                  # also General Soft and the heavy RRDB "Detail" models
prism models import my_weights.pth --id resr-general-v3   # convert a local checkpoint
prism models import my_rife.onnx --id rife                # install a ready ONNX file
```

RIFE comes from the ONNX exports that vs-mlrt publishes (`rife_v4.26.7z`, 20 MB; MIT-licensed
Practical-RIFE weights). Prism downloads the archive, extracts the `rife_v2` export (7-channel
input, internal padding) with Windows' built-in `tar` (or 7-Zip/bsdtar elsewhere), and installs it
as `rife.onnx`. The GUI offers Download and Import buttons for any network that is not installed,
in the model picker and in File › Model manager. Any other single-input image-to-image ONNX file
placed in the models folder appears as a custom model; an optional `<name>.json` sidecar sets its
name, kind and scale.

Models live in `%APPDATA%\Prism Video Enhancer\models`, in a `models` folder next to the executable,
or wherever `PRISM_MODEL_DIR` points.

The first time a network runs on a GPU, Prism times a few tile sizes and caches the fastest
(`tile-cache.json` in the data folder). Throughput collapses once tiles no longer fit in free GPU
memory, and how much is free depends on what else is running.

## CLI

```text
prism probe clip.mp4
prism plan clip.mp4 --upscale 4k --model resr-general-v3      # show passes and FFmpeg commands
prism run clip.mp4 --upscale 4k --model resr-general-v3 --encoder hevc-nvenc
prism run clip.mp4 --upscale 2x --denoise 40 --sharpen 20 --recover 10     # Classic, manual sliders
prism run clip.mp4 --fps 60                                   # motion-compensated 60 fps
prism run clip.mp4 --slowmo 4 --fi-model prism-flow
prism run clip.mp4 --stabilize fullframe --smoothness 70
prism run old.vob --deinterlace double --upscale 1080p --encoder prores-hq
prism run clip.mp4 --hdr pq --encoder h265-x265-10
prism run clip.mp4 --trim 00:00:10:00-00:00:20:00 --audio aac:192 -o out.mkv
prism run clip.mp4 --preset "Upscale to 4K"
prism preview clip.mp4 --at 12.5 --duration 2 --upscale 2x
prism batch jobs.json --workers 2
prism encoders --probe       # which hardware encoders actually work here
prism bench                  # stage throughput vs FFmpeg's equivalents
```

`prism run --help` lists every option. A batch file is a JSON array of
`{"input", "preset"?, "settings"?, "output"?, "priority"?}` objects; `settings` is the same
`JobSettings` JSON that presets store.

## Architecture

```
crates/
  prism-core     domain model and planning; no video data
    media        ffprobe JSON → VideoInfo (colour, rotation, VFR, audio/subtitle streams)
    settings     JobSettings: every filter's state (serde; presets and projects store it)
    models       catalog: taxonomy, scales, licences, availability, recommendations
    plan         settings + VideoInfo → passes (FFmpeg-only or decode → stages → encode)
    encoders     encoder/audio tables, rate control, container rules
    progress     frame-count progress, 0.9/0.1 FPS smoothing, ETA; FFmpeg stats parser
    errors       FFmpeg/engine message → category + advice
    presets, project, naming, timecode, paths
  prism-engine   the frame pipeline (replaces tvai_* filters + tvai/videoai/aiengine DLLs)
    pipeline     reader → unpack → stage threads → pack → writer, bounded channels, buffer reuse
    frame        planar f32 RGB frames, pooled buffers, YUV 4:2:0/RGB ⇄ f32 conversion
    enhance      two-stage enhancement (pre-ops + upscale | output-resolution post-ops)
    interp       motion estimation, warping, rate-conversion scheduler
    ai/          ONNX Runtime wrapper, tile planning and blending, .pth → ONNX converter
    ops/         resampling, filters, noise/grain, colour/HDR, frame analysis
    process      child processes: suspend/resume (pause), cancel, stderr capture
    runner       runs a plan's passes, progress events, cleanup
    queue        export queue with workers and priorities
  prism-cli      `prism` command
  prism-app      `prism-app` GUI (egui)
```

Data flow for a job with Rust stages:

```
ffmpeg (decode, trim, rotate, deinterlace, crop, stabilize, LUT)
   │ raw yuv420p / yuv420p10le / gbrp16le on stdout
   ▼
unpack → [motion deblur] → [interpolate] → [enhance: pre-ops + network/Lanczos] → [enhance post]
       → [HDR] → [grain] → pack            (each step its own thread; GPU and CPU work overlap)
   │ raw frames on stdin
   ▼
ffmpeg (fps change, pad/crop, colour tags, encoder; audio, subtitles and metadata from the source)
```

Design decisions that matter for speed:

- **Overlap everything.** Decoding, each stage, packing and encoding run concurrently. Enhancement
  is split so the network (GPU) and the output-resolution post-processing (CPU) work on
  neighbouring frames at the same time.
- **Use FFmpeg where it is fastest.** Work FFmpeg's 8-bit SIMD filters do well (frame blending,
  plain Lanczos resize, deinterlacing, stabilization) stays in FFmpeg. Rust stages run only for
  work FFmpeg cannot do or does slowly. Jobs with no Rust stage are a single FFmpeg command.
- **Minimal conversions.** 4:2:0 sources travel as YUV and are converted in vectorised Rust; 10-bit
  stays 10-bit; RGB is used only when an encoder needs 4:2:2/4:4:4 or RGB.
- **GPU sessions are compiled once.** All tiles of a job have one shape. A replicated border makes
  frame edges independent of tiling, and sessions are cached between previews and exports.
- **Pause without losing work.** The queue suspends FFmpeg processes in place
  (`NtSuspendProcess` / `SIGSTOP`); Topaz kills FFmpeg and concatenates segments on resume.

## Performance

`prism bench` decodes the same clip for every scenario and discards the output (FFmpeg's null
muxer), so the figures measure processing, not encoding. Prism fps excludes model loading. The
reference column runs FFmpeg's closest built-in filter on the same clip.

Measured on the development machine (Ryzen 7 6800H, 16 threads; RTX 3060 Laptop; FFmpeg 9.0.2;
generated 1080p30 clip, 120 frames). Other applications were running, so absolute numbers vary by
±30% between runs. Decode alone ran at 143 fps in this run and 208 fps in a quieter one.

| Scenario | Prism | FFmpeg reference | Ratio |
|---|---:|---|---:|
| Motion-compensated interpolation 30→60 fps (Flow) | 34.6 fps | `minterpolate` mci/aobmc/bidir: 3.3 fps | **10.6x** (13.2x in a quieter run) |
| Enhance 2x: denoise + deblock + sharpen (Classic) | 6.7 fps | `nlmeans`, `deblock`, Lanczos, `cas`: 2.8 fps | **2.4x** |
| Plain 2x resize (routed to FFmpeg) | 122 fps | Lanczos `scale`: 130 fps | 0.94x |
| Blend interpolation 30→60 fps (routed to FFmpeg) | 247 fps | `framerate`: 301 fps | 0.82x |
| Classic 2x, Lanczos + anti-ringing (1080p → 4K) | 18.2 fps | plain Lanczos: 129 fps | 0.14x |
| Classic 4x, Lanczos + anti-ringing (1080p → 8K) | 7.6 fps | plain Lanczos: 92 fps | 0.08x |
| SDR → HDR10 inverse tone mapping | 41.7 fps | (none) | |
| Film grain | 60.5 fps | `noise` (simpler, not luma-weighted): 155 fps | 0.39x |

How to read the table:

- Prism is much faster where FFmpeg's equivalent is slow or single-threaded (motion-compensated
  interpolation, non-local-means denoising). That is where the engine adds value.
- For trivial per-pixel work, FFmpeg's 8-bit YUV SIMD filters beat any f32 RGB pipeline: each pixel
  carries 12 bytes instead of 1.5. Prism therefore routes plain resizes and frame blending to FFmpeg
  (≈1x). Classic upscaling with anti-ringing stays in Rust; at 4K output it is bound by memory
  bandwidth (four concurrent 100 MB-per-frame passes).

**Networks (Real-ESRGAN on DirectML, System32 ONNX Runtime 1.17, FP16):**

| Input → output | General (32 convs) | Animation (16 convs) |
|---|---:|---:|
| 960x540 → 3840x2160 | 1.3 fps | 1.7 fps |
| 1920x1080 → 7680x4320 (4x model, 24 frames) | 0.3 fps | 0.4 fps |

These were measured while the laptop GPU was held at 307 MHz (power state P5). NVIDIA's
counters showed long software thermal slowdown and power capping, so this is about a fifth of the
GPU's normal clock. Earlier in the same session, at normal clocks, the Animation network
processed a 672x400 tile in 48 ms (5.6 Mpx/s). The 512x512 tile measured during the benchmark ran
at 1.5 Mpx/s, about 3.7x slower. Expect proportionally higher network throughput on an unthrottled
GPU. Run `prism models retune` after the power state changes so tile sizes are re-measured.

### Compared with Topaz Video AI

There is **no measured head-to-head comparison**. Topaz was not run for benchmarking (see
REPORT.md §0.1: the installed copy is not genuinely licensed), and its models differ from
Prism's, so equal-quality timings are not possible. REPORT.md §12 has a single Topaz data point,
a cold-start 2x run of `prob-4` on 320x180 at about 7 fps including TensorRT engine loading, which
is not comparable to anything above.

What can be said from the architecture:

- **Likely faster in Prism:** classical restoration and motion-compensated interpolation run
  in-engine, multithreaded, while decode, CPU stages, GPU inference and encode overlap. Pausing
  suspends processes instead of killing FFmpeg and concatenating segments on resume. Jobs without
  network stages never pay for an f32 pipeline.
- **Likely faster in Topaz:** network inference on NVIDIA GPUs. Topaz ships TensorRT engines
  prebuilt per GPU, tile size and precision; Prism runs portable ONNX through DirectML, which is
  generally slower than TensorRT on the same GPU. A newer ONNX Runtime with the CUDA or TensorRT
  execution provider (`PRISM_ORT_DYLIB`) is the route to closing that gap; it was not available
  on this machine to test.

Reproduce with `prism bench` (add `--input your.mp4`, `--size`, `--only <text>`, or `--json results.json`).

## Testing

- `cargo test` runs about 70 unit tests. They cover planning (geometry, passes, routing, audio
  rules, HDR upgrades), conversions, kernels, the interpolation scheduler, the checkpoint reader,
  and a **numerical check of the ONNX converter**: a converted network run in ONNX Runtime is
  compared with a reference forward pass.
- `crates/prism-engine/tests/queue_control.rs` runs real FFmpeg jobs through the queue. It checks
  that pausing freezes progress, that resuming completes the job, and that cancelling removes
  partial output.
- `cargo test -p prism-engine --release kernel_bench -- --ignored --nocapture` prints kernel
  timings; `inference_bench` (with `PRISM_BENCH_MODEL`) times a network on the GPU.
- An end-to-end smoke run of 21 CLI scenarios passed on the development machine (Windows 11,
  Ryzen 7 6800H, RTX 3060 Laptop, FFmpeg 9.0.2): Classic and network upscales, stabilization,
  double-rate deinterlacing, HDR10 tags, ProRes, PNG sequences, two-stage enhancement, trim with
  Opus audio, slow motion, blend routing, a 4K NVENC letterbox, rotate and crop, preview, a batch
  with two workers, missing-model errors and encoder probing.

## Limitations

- **Quality is not Topaz's.** The networks are public Real-ESRGAN models, not Topaz's proprietary
  ones. Classic, Flow, HDR, deblur and Auto-parameter estimation are hand-written algorithms, not
  learned models. No side-by-side quality comparison with Topaz was made.
- RIFE was checked on one synthetic clip (30 → 60 fps against the real 60 fps frames): it placed
  moving objects without the ghosting Flow and Blend show there. The adapter also accepts separate
  `img0`/`img1`/`timestep` inputs, which no tested export used.
- The live single-frame preview skips stabilization and interpolation, which need neighbouring
  frames or an analysis pass; "Render preview" covers them.
- Variable-frame-rate sources are resampled to a constant rate. Alpha channels are dropped (with a
  warning). Rolling-shutter correction is not available.
- On the GNU toolchain the build depends on the dlltool replacement described above.
- **macOS is built but not tested.** The code type-checks for `aarch64-apple-darwin`, and
  `packaging/macos/build.sh` produces the `.dmg`, but no Mac was available to run either. The
  Core ML provider, the native title bar and the bundled runtime are unverified there.
- Linux was tested in containers: the `.deb` on Ubuntu 22.04 (FFmpeg 4.4) and the `.tar.gz` on
  Fedora 44 (FFmpeg 8.1), including RIFE and Real-ESRGAN on the CPU, and the GUI on a virtual X
  display with software rendering. No Linux GPU (CUDA) run was made. FFmpeg 4.3 (Debian 11) should
  work but was not tested.
- The x86-64-v3 build target means Intel Macs from before 2013 and pre-Haswell PCs are not
  supported by the packages. Windows 10 has no ONNX Runtime in System32: network models need an
  `onnxruntime.dll` placed next to `prism-app.exe` there.
- Figures above come from one laptop whose GPU was shared and throttled; they are not a general
  performance claim.
