# GPUBench — Full Review Report
Date: 2026-07-19
Scope: features & accuracy, TUI/CLI, GUI, cross-platform support, installers/packaging
Repo: /home/naoki/Development/GPUBench (hybrid C++17/CMake core + Rust workspace: gpubench-sys/core/gui)

====================================================================
1. EXECUTIVE SUMMARY
====================================================================
GPUBench is an ambitious multi-backend (Vulkan 1.4 / OpenCL 1.2 / ROCm-HIP)
GPU benchmark covering FP64..FP4, INT8/INT4, memory/cache bandwidth and a
full ray-tracing suite. The C++ production path is solid engineering: sound
timing methodology, correct FP32 FLOP accounting, a polished ANSI result
formatter, and sensible dynamic backend loading. The Rust layer (CLI + iced
GUI) is significantly behind: the CLI is effectively non-functional, the GUI
has a release-breaking kernel-path bug and misleading backend labels, and the
Rust Vulkan reimplementation produces invalid numbers. Cross-platform claims
overshoot reality — macOS is shipped by CI but is almost certainly broken
(Vulkan 1.4 requested unconditionally; no Apple-specific code paths), and
Linux has no CI at all. Installers exist for all three OSes but each has at
least one correctness gap.

====================================================================
2. FEATURES & ACCURACY
====================================================================
Feature inventory (C++ production path, cpp_src/benchmarks/):
- Compute: FP64, FP32, FP16, BF16, FP8 (native+emulated), FP6, FP4, INT8, INT4
- Memory: device bandwidth (read/write/read-write), system memory BW + latency,
  cache bandwidth (L0/L1/L2/L3 variants) + cache latency
- Ray tracing: 8 suites (coherent, divergence, any-hit, payload, procedural,
  incoherent, material divergence, AS build) — Vulkan only
- Backends loaded dynamically at runtime (optional deps) — good design

Timing methodology (C++): SOUND
- BenchmarkRunner.cpp:390-434: wall-clock (chrono) brackets Run()+waitIdle(),
  loops until 2500 ms total; score = ops*invocations/total_time.
- Kernel compilation and buffer setup excluded from timing (Setup phase).
- Minor: no dedicated warmup iteration (negligible over 2.5 s amortization);
  fresh command buffer + fence per dispatch adds per-iteration overhead —
  negligible for compute, can be several % for short memory kernels
  (biases reported GB/s slightly LOW, which is the safe direction).

FLOPS math (C++ FP32): CORRECT
- fp32.comp: 32 vec4 FMAs/iter = 256 FLOPs/iter/thread x 16384 iters;
  Fp32Bench.cpp:75 counts exactly that over 8192 WG x 64 threads. FMA=2 FLOPs
  (standard). Push-constant multiplier 1.0001f defeats constant folding.
  All 32 accumulators reduced and written to SSBO — DCE-safe. No result
  validation (write-back is only an optimization barrier).

Accuracy DEFECTS found:
A. MemBandwidthBench.cpp:203-204 uses ONE byte formula
   (WG*threads*512*32) for Read, Write AND ReadWrite configs. A single byte
   count cannot be correct for all three modes; at least one mode's GB/s is
   misreported (likely ReadWrite under/over-counted by 2x).
B. Rust path (gpubench-core/src/benchmarks/*) is invalid:
   - vulkan.rs dispatch() records cmd_dispatch TWICE (lines ~834, ~850) inside
     one timed submit — every kernel does 2x work, counted once.
   - fp32.rs:89 counts 8192 workgroups but dispatches 1024 — 8x overcount.
     Combined with the double-dispatch, reported TFLOPS ~4x inflated.
   - membw.rs creates 1 buffer binding but membw.comp declares 2 SSBOs —
     unbound descriptor, validation failure / UB.
   - Buffer allocation inside the timed region (fp32.rs:75-76).
C. No correctness validation anywhere: no checksum/readback/NaN check.
   A silently miscompiled kernel yields plausible numbers.
D. DeviceInfo capability flags hardcoded true (vulkan.rs:112-114) instead of
   queried — feature gates are cosmetic in the Rust path.
E. PERFORMANCE_ANALYSIS.md reports FP64 "103% of theoretical" (RX 6900 XT)
   treated as "excellent" — >100% of theoretical should trigger scrutiny
   (boost clocks vs. spec, or op-count error), not celebration.

VERDICT: C++ FP-family compute numbers are credible within a few percent.
Memory bandwidth numbers need per-mode byte accounting fixed before trusting.
All Rust-path numbers are invalid (WIP code). Nothing is validated.

====================================================================
3. TUI / CLI
====================================================================
There is no interactive TUI (no ratatui/ncurses/FTXUI anywhere). "TUI" =
line-based stdout with ANSI colors + progress bar.

C++ CLI (CLI11, main.cpp) — GOOD, with gaps:
+ Clean flags: -b/-d/-k/--list-benchmarks/--list-devices/--list-backends/
  --verbose/--debug/--version; verified --help and listing work.
+ ResultFormatter.cpp produces a genuinely nice grouped, unit-aware
  (TFLOPS/GB/s/ns/GRays/s) colored report with per-backend columns.
- Bogus benchmark name silently runs nothing, exits 0 (verified).
- Substring matching: `-b performance` unintentionally matches many;
  MemBandwidthBench names itself "Performance" — ambiguous in --list-benchmarks.
- Out-of-range device index warns but exits 0.
- --list-backends reports compile-time "Supported" even with no driver/GPU.
- Auto mode creates only ONE backend (Vulkan>OpenCL>ROCm); devices on other
  backends invisible; no runtime fallback if Vulkan compiled but throws.
- Vulkan kernel-compile progress bar is dead code (contradictory verbose
  guards, VulkanContext.cpp:1148 vs 1155). No progress during the 2.5 s
  timing loops in non-verbose mode.
- NO machine-readable output (no --output json/csv) from either CLI — a big
  gap for a benchmark tool. JSON export exists only in the GUI.

Rust CLI (clap derive, gpubench-core) — NON-FUNCTIONAL:
- main.rs:7-16 ignores argv entirely; hardcodes benchmarks=["All"],
  device=0, backend=Vulkan. `--help` launches a benchmark (verified).
- The clap Cli struct (lib.rs:24-55) is never parsed; run_cli() is a stub
  ("Implementation pending").
- Hardcoded "All" matches nothing in the C++ matcher → prints
  "Total results: 0", exits 0 (verified). Default invocation is a no-op.
- Doesn't set MESA_VK_IGNORE_CONFORMANCE_WARNING → radv warning spam.
- FFI exceptions (e.g. no GPU) cross cxx boundary → std::terminate/abort
  (RunnerAPI.cpp:70-107 has no try/catch; bridge returns no Result).

====================================================================
4. GUI (iced 0.12, gpubench-gui, ~1141 LOC)
====================================================================
Architecture: statically links the C++ core via cxx bridge (no CLI scraping)
— good. Benchmarks run on spawn_blocking with mpsc progress + 50 ms Tick —
UI never blocks. Full benchmark coverage incl. ray tracing, group toggles,
tooltips per metric, dark custom theme.

DEFECTS (ranked):
1. RELEASE-BREAKING: main.rs:262 bakes GPUBENCH_KERNEL_PATH at COMPILE time
   to the dev tree (CARGO_MANIFEST_DIR/../kernels). Packaged GUI looks for
   kernels at the build machine's path (CI: D:\a\GPUBench\...), overriding the
   installed share/gpubench/kernels. Broken for every end user. Also stomps a
   user-set env var.
2. "ROCm" backend pill actually filters api=="opencl" (main.rs:275,359) —
   misleading/broken. JSON export hardcodes compute_api:"Vulkan" (main.rs:561)
   and omits sys-mem results it already collected.
3. No error state: spawn_blocking JoinError discarded (main.rs:482-492);
   failures silently show "Complete" with UNSUPPORTED metrics. File-write
   errors ignored; device parse falls back to 0 silently.
4. Progress counter keys on res.component → multiple Ray suites share
   "Ray Tracing" → N/M undercounts until completion.
5. Branding: window title "BenchmarkX 2026 - Pro Edition" / "BENCHMARK X PRO"
   while everything else is GPUBench — unreconciled rename.
6. Code health: 25 hand-managed f32 metric fields with copy-paste resets
   (~100 lines a HashMap/struct would eliminate); duplicated AppState/GPUBenchApp
   state kept in sync manually; dead cpu_bw field; test_tooltip.rs is an
   unwired leftover; iced 0.12 is dated (0.13 redesigned the APIs used).
7. rfd sync save dialog blocks UI while open; Linux rfd needs GTK3/portal dev
   packages — undocumented.

====================================================================
5. CROSS-PLATFORM SUPPORT
====================================================================
Claimed: Linux + Windows (README). Shipped by CI: Windows + macOS. Reality:

LINUX — best supported. Vulkan/OpenCL/ROCm all build; RADV-tested. BUT:
- No Linux CI job at all. Linux DEB/RPM/TGZ are built manually on the dev
  machine (RELEASING.md) — regressions ship undetected.
- No non-release CI either (only tag-triggered release.yml; no PR checks,
  no tests workflow).

WINDOWS — CI works but fragile:
- Workflow never installs/pins Rust, yet CMake ALL-target builds the Rust
  workspace and install() REQUIRES gpubench-gui.exe (NSIS menu links point at
  it). Works only because windows-latest happens to ship Rust. If missing:
  CMake warns, cpack fails on missing file. Add dtolnay/rust-toolchain.
- MinGW support documented (mingw-toolchain.cmake, WINDOWS_MINGW_COMPATIBILITY.md)
  but CI uses MSVC default — two toolchains, only one exercised.
- glslc fallback glob is Windows-hardcoded ("C:/VulkanSDK/*/Bin/glslc.exe").

MACOS — shipped but likely BROKEN:
- VulkanContext.cpp:55 requests VK_API_VERSION_1_4 unconditionally. MoltenVK
  (the only Vulkan on macOS) does not support 1.4 → instance creation fails.
  No enumerateInstanceVersion fallback path exists.
- Ray-tracing suite needs VK_KHR_acceleration_structure etc. — unsupported by
  MoltenVK. Native FP8/INT4 shader extensions likewise unavailable.
- getExecutableDir() (KernelPath.cpp) has _WIN32 and __linux__ branches only;
  macOS returns empty → exe-relative kernel lookup silently dead.
- ROCm backend is Linux-only (fine), OpenCL on macOS is deprecated 1.2.
- DMG is a raw DragNDrop of binaries — no .app bundle, no signing/notarization;
  Gatekeeper will block the CLI/GUI for typical users.
- README says "Linux and Windows" while CI ships macOS installers — pick one.

BUILD-SYSTEM COUPLING:
- Two-way recursion: CMake ALL-target runs `cargo build --workspace`;
  gpubench-sys/build.rs runs a nested CMake configure+build of gpubench_lib.
  Works today (nested build targets gpubench_lib only) but doubles configure
  cost and is one edge away from a loop. add_dependencies(gpubench,
  gpubench_rust_gui) means every C++ edit triggers a full Rust rebuild.
- cmake_minimum 3.16 + C++17 is conservative and fine.

====================================================================
6. INSTALLERS & PACKAGING
====================================================================
Windows: CPack ZIP + NSIS (uninstall-before-install, PATH option, menu links,
icon, help/issue URLs) — mature. RunAllBenchmarks.bat convenience included.
MinGW runtime DLLs bundled (OPTIONAL). Gaps: GUI exe assumed present (see 5);
no vcredist handling (commented out); unsigned binaries (SmartScreen prompts).

Linux: DEB + RPM + TGZ via CPack, dynamic backend loading keeps hard deps
minimal — good philosophy. Gaps:
- DEB declares ZERO dependencies (CPACK_DEBIAN_PACKAGE_DEPENDS "") while RPM
  requires vulkan-loader/glibc/libstdc++ — inconsistent; DEB install on a
  Vulkan-less system fails at runtime instead of at install.
- RPM_AUTOREQ disabled to dodge a 2GB ROCm requirement (documented,
  reasonable), but manual packaging (env -i wrapper, gh upload) means no
  reproducibility guarantee.
- Kernels installed to share/gpubench/kernels; C++ KernelPath searches
  cwd ./kernels BEFORE env var and exe-relative paths — an installed binary
  run from a dir containing "kernels" silently uses the wrong kernels.

macOS: DMG + TGZ only; no .app, no notarization; see Section 5 — the package
wraps a binary that can't initialize Vulkan.

Versioning: PROJECT 1.1.0 + manual BUILD_NUMBER in CMakeLists — works, but
tag/version drift risk (RELEASING.md requires manual BUILD_NUMBER bumps).

====================================================================
7. RECOMMENDATIONS (priority order)
====================================================================
P0 — correctness:
1. Fix MemBandwidthBench per-mode byte accounting (read vs write vs read+write).
2. Fix GUI kernel-path resolution: runtime lookup (user env var → exe-relative
   share/gpubench/kernels → dev fallback); delete the compile-time set_var.
3. Gate or fix the Rust benchmark path: remove double-dispatch in vulkan.rs,
   align dispatch size with op counting, bind both SSBOs in membw, and mark
   Rust numbers as experimental until validated. Wire up the clap CLI or
   remove the binary — shipping a no-op default is worse than none.
4. Add a result-validation pass (checksum/readback) to at least one compute
   and one memory benchmark; treat FP64 ">100% of theoretical" claims as bugs
   until explained.

P1 — platform truth:
5. macOS: either drop it from CI/README, or add enumerateInstanceVersion-based
   apiVersion negotiation (1.2/1.3 fallback), skip RT/native-FP8/INT4 suites
   under MoltenVK, add the __APPLE__ branch to getExecutableDir, and produce
   a signed/notarized .app. Until then the DMG is a broken artifact.
6. Windows CI: add dtolnay/rust-toolchain; make CMake FATAL_ERROR on missing
   cargo for packaging builds; decide MSVC vs MinGW and exercise one in CI.
7. Add a Linux CI job (at minimum: build + --list-benchmarks smoke test) and
   a PR-triggered workflow; move Linux packaging into CI for reproducibility.

P2 — UX & robustness:
8. Add --output json|csv to the C++ CLI (fields already exist); fix GUI JSON
   hardcoded backend + include sys-mem results.
9. Warn + non-zero exit when no benchmark matched / device out of range;
   make isAvailable() a real runtime probe; wrap RunnerAPI enumeration in
   try/catch so FFI can't abort the GUI.
10. GUI: fix ROCm/OpenCL label, add AppState::Error, fix progress counter
    (key on component+name), collapse the 25 metric fields into a struct/map,
    delete test_tooltip.rs, reconcile "BenchmarkX Pro" branding, consider
    iced 0.13 before building more UI.
11. Fix dead Vulkan progress bar (VulkanContext.cpp:1155) and add a non-verbose
    spinner/ETA for timing loops.
12. Packaging consistency: give DEB the same minimal deps as RPM; reorder
    KernelPath (exe-relative installed path before cwd dev path); align tag
    with BUILD_NUMBER automatically.

====================================================================
8. OVERALL ASSESSMENT
====================================================================
Core C++ benchmark engine: B (sound methodology, one real math bug, no validation)
C++ CLI:                        B (polished output, weak validation of inputs,
                                   no machine-readable export)
Rust CLI:                       D (non-functional stub shipped as a binary)
GUI:                            C (good architecture and threading; release-
                                   breaking path bug, misleading labels, no
                                   error handling)
Cross-platform:                 C- (Linux solid, Windows fragile, macOS broken
                                   but shipped)
Installers:                     B- (NSIS good; DEB/RPM inconsistent and manual;
                                   DMG wraps a non-working binary)
Trustworthiness of numbers:     C++ compute: yes, within a few %. C++ memory
                                BW: not until per-mode bytes fixed. Rust: no.
