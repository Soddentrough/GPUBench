# GPUBench — Full Project Review

- **Date:** 2026-10-08
- **Codebase state:** `master` @ `b1d5982` **plus 41 uncommitted working-tree changes** (see C-8)
- **Review environment:** AMD Radeon 8060S Graphics (gfx1151, PCI deviceID `0x1586`), Mesa RADV 26.2.3, Vulkan 1.4.354, ROCm 10.0 / HIP 7.15, Fedora 44, single GPU (`-d 0`)
- **Project scale:** ~35.5k lines first-party C++, ~13k lines shader/HIP/OpenCL kernel source, ~5.5k lines Python tooling, ~5.4k lines documentation
- **Measurement caveat:** a `llama-server` instance was resident on the iGPU at 96 % `gpu_busy_percent` for the entire review. All throughput figures below are therefore **contaminated** and are used only as evidence of internal consistency, never as hardware claims. Correctness, labelling and static findings are unaffected.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [What is genuinely strong](#2-what-is-genuinely-strong)
3. [The central problem: measurement validity](#3-the-central-problem-measurement-validity)
4. [Critical bugs](#4-critical-bugs)
5. [Minor bugs](#5-minor-bugs)
6. [Optimizations](#6-optimizations)
7. [Nice-to-have features](#7-nice-to-have-features)
8. [Build system and tooling currency](#8-build-system-and-tooling-currency)
9. [Recommended sequencing](#9-recommended-sequencing)
10. [Prior-audit verification matrix](#10-prior-audit-verification-matrix)
11. [Appendix: verification notes](#11-appendix-verification-notes)

---

## 1. Executive summary

GPUBench is a cross-backend (Vulkan 1.4 / OpenCL / ROCm-HIP, all `dlopen`'d at runtime) GPU microbenchmark suite whose differentiating content is **ray-scheduling architecture comparison** — monolithic megakernel vs. device-generated-command wavefront compaction vs. SER — across four scenes, delivered through a Unicode-card CLI and a Dear ImGui "Workstation Profiler" GUI.

**No rewrite is needed.** The architecture (backend facade → benchmark plugins → runner → formatter) is sound and extensible, first-party code compiles warning-clean, GPU timestamps are now real on all three backends, and the bit-exact parity gate genuinely works.

The problem is not structure. It is **measurement validity and data integrity**. The tool currently publishes headline results that are unverifiable or outright wrong, and nothing in the system can detect it:

- A **113× "Wavefront Scheduling Speedup"** produced by dividing a traversal-and-shading megakernel by a traversal-and-classify-only DGC pass, both reporting identical `operations`.
- **FP32 and all three Dual-Issue FP32 configs fail their own NaN/Inf validation on every run**, and the harness discards that signal unless `--verbose` happens to be set.
- The **`DeviceDatabase` PCI device IDs are wrong**, including for the reviewer's own machine — the Strix Halo GPU entry is registered under the *NPU's* device ID, so the table never matches real hardware.
- The **README's sample CLI transcript contains rows the binary cannot produce**, including L1/L2/L3 cache latency benchmarks that are commented out of the registry.
- The **GUI prints invented VGPR counts, SIMD utilisation, per-pass times and BVH steps/ray** as if measured.

For a benchmarking product, whose deliverable *is* its numbers, these are first-order defects. The remedy is a focused one-to-two-month programme: a work-equivalence contract between compared configs, load-bearing validation, hardware-counter instrumentation in place of wall-clock inference, provenance stamps on every published number, and CI that refuses to ship when any of those regress.

---

## 2. What is genuinely strong

These should be preserved deliberately, because they are the parts most likely to be eroded by future feature work.

| Area | Assessment |
| :--- | :--- |
| **UNSUPPORTED-with-reason policy** | Excellent. `IBenchmark::SupportLimitation` distinguishes `kHardware` / `kApi` / `kToolchain` and surfaces a human-readable reason. Verified live: FP6, FP8, FP4, INT4, BF16 and all SER configs report correctly-attributed UNSUPPORTED rows rather than silently falling back. |
| **GPU timestamps on all three backends** | Real. `VkQueryPool` + `vkCmdWriteTimestamp` (`VulkanContext.cpp:1022-1080`), `clEnqueueMarkerWithWaitList` + `clGetEventProfilingInfo` (`OpenCLContext.cpp:732-803`), HIP events (`ROCmContext.cpp:684-719`). C-4 from the 2026-10-06 review is genuinely fixed. |
| **Bit-exact parity gate** | Real and enforced (`RaySchedulingBench.cpp:2025-2028`, `check-parity` CMake target). On-disk artifacts confirm PSNR 120 dB, MAE 0.0000, 0 discrepant pixels over 8,294,400 px. This is the project's most valuable and most novel asset. |
| **`dlopen`'d backends** | Clean design. RPM/DEB correctly declare only `vulkan-loader`, libc/libstdc++ and SDL3, leaving OpenCL/ROCm optional at runtime. |
| **C++23, single GUI stack** | The Rust/Iced purge was correct and is complete across CMake, CI, packaging and desktop entries. |
| **Warning posture** | `-Wall -Wextra` plus `-fstack-protector-strong`, `_FORTIFY_SOURCE=3`, full RELRO and `noexecstack`. First-party code compiles **warning-clean**; the only suppressed classes are `unused-parameter` and Vulkan C `missing-field-initializers` (see §8). |
| **Documentation tone** | Marketing language has been systematically removed. No "world-class" / "best-in-class" / "revolutionary" style claims remain. |
| **Profiling tooling** | `scripts/capture_gpu_profiles.py`, `profile_registers.py`, `validate_rt_microbench.py` and `docs/PROFILING_GUIDE.md` constitute publication-grade supporting tooling — it just is not wired into CI. |

---

## 3. The central problem: measurement validity

### 3.1 Evidence

`./build/gpubench -d 0 -b rayscheduling -s indoor -r 720p`:

```text
│ Path Tracing (16 SPP) (Megakernel)  │ Vulkan │   6.05 MRays/s │ [Baseline]  │
│ Path Tracing (16 SPP) (DGC)         │ Vulkan │ 416.73 MRays/s │ └──> 68.87x │

╭─ Executive Performance Summary & Architectural Takeaways ──────────────────────────╮
│ • Wavefront Scheduling Speedup : 113.07x in PBR Ray Tracing (1,301.9 vs 11.5 MRays/s) │
```

113× and 68× between two implementations of the same algorithm on the same silicon is not physically credible. It is an apples-to-oranges comparison, and it is the figure the tool promotes into its own executive summary.

### 3.2 Root cause

`RaySchedulingBench::Run()` configs 17 and 18 (`RaySchedulingBench.cpp:1428-1442`):

| Config | What it actually dispatches | Reported `operations` |
| :--- | :--- | :--- |
| 17 — "Primary Rays (Compute Megakernel)" | full megakernel: **traverse + shade** | `rayCount` |
| 18 — "Primary Rays (Wavefront - DGC)" | `kernelReset` + `kernelClassify` only: **traverse + bin into queues, never shade** | `rayCount` |

Both report identical `operations` (`GetResult`, `RaySchedulingBench.cpp:2483`: `r.operations = rayCount` for everything that is not path tracing). `ResultFormatter.cpp:356-367` then matches these two rows **by substring on their names** and divides them, printing the quotient as "Wavefront Scheduling Speedup". The same asymmetry drives the 68× path-tracing figure and the 12.2× shadow and 15.2× incoherent-GI rows.

### 3.3 Why nothing catches it

`BenchmarkRunner.cpp:1112-1115`:

```cpp
bool isValid = bench->ValidateResults(i);
if (!isValid && verbose) {
  std::cerr << " [WARNING] Result validation failed for " << bench_name << std::endl;
}
```

`isValid` is never used again. It does not affect the reported `SUCCESS` status, the JSON payload, or the exit code. Most validators are `return true;` stubs (`InShaderIndirectBench`, `LdsBankConflictBench`, `CacheLatencyCurveBench`) or a `!isnan && !isinf` check at best.

Live proof that this matters:

```text
[TIMING FP32] single_run_ms: 107.002, iterations: 2, total_time_ms: 217.027
 [WARNING] Result validation failed for FP32
│ FP32 │ Vulkan │ 20.26 TFLOPS │        ← reported as SUCCESS
```

FP32 and all three Dual-Issue FP32 configs fail their own NaN/Inf check **on every run**. The cause is analytic: `shaders/fp32.comp` uses `a[i] = fma(m, a[(i+1)&31], a[i])`, whose update matrix has dominant eigenvalue `1 + m = 1.999`. Over 16 384 iterations magnitude grows as `1.999^16384`; every lane saturates to `inf` within roughly 130 iterations. The benchmark times `inf` arithmetic and reports a score the harness itself flagged invalid.

### 3.4 The fix — this is the architectural change worth making

Introduce a **work-equivalence contract** rather than continuing to patch individual config pairs:

1. Each benchmark declares, per config, a `WorkContract { tracedRays, hitShadeEvaluations, bounces }` derived from **measured** GPU counters (a counter buffer written by the kernel), not from the `rayCount` push constant.
2. `BenchmarkRunner` refuses to compute a speedup ratio between configs whose work contracts differ beyond tolerance, rendering `NOT COMPARABLE (traversal-only vs traverse+shade)` instead.
3. `ValidateResults` becomes load-bearing: failure ⇒ status `INVALID`, excluded from speedups, non-zero exit under `--strict`.
4. Add a `--self-test` mode asserting every published A/B pair lands inside an expected ratio envelope. `scripts/verify_benchmarks.py` half-implements this already, but it is not in CI and its envelopes are far too loose to catch 113×.

---

## 4. Critical bugs

### C-1. Executive summary and Ray-Scheduling speedups compare non-equivalent work

**Where:** `RaySchedulingBench.cpp:1428-1442` (configs 17/18), `:2483` (op accounting), `ResultFormatter.cpp:356-367, 935-975`.

As described in §3. **Highest-priority defect in the project.**

Secondary defect in the same code: `megakernelPBRRate` and `dgcPBRRate` are plain assignments inside the result loop, so with `-s all` the summary silently reports only whichever scene ran last rather than any stated scene.

**Fix:** work contracts + explicit `GetBaselineConfigIndex(config)` on `IBenchmark`, replacing formatter-side substring guessing.

### C-2. FP32 / Dual-Issue FP32 saturate to `inf`; validation failure is discarded

**Where:** `shaders/fp32.comp`, `cpp_src/benchmarks/Fp32Bench.cpp:83-92`, `BenchmarkRunner.cpp:1112-1115`.

The recurrence diverges by construction (eigenvalue `1 + m`); the ring cannot be stabilised by sign changes either, because a 32-element ring has `ω = -1` among its roots.

**Fix:** independent self-chains — `a[i] = fma(m, a[i], c_i)` with `|m| < 1` converges to `c/(1-m)`, keeps 32 independent register-operand FMA chains, and stays finite. Additionally make validation failure non-silent (status `INVALID`, non-zero exit under `--strict`).

### C-3. GUI displays fabricated microarchitectural metrics

**Where:** `cpp_src/gui/GuiApp.cpp:5548-5620`, `:6011-6040`, `:6060-6200`.

The parity banner is now honest — `NOT MEASURED` / `PASS` / `FAIL` with live PSNR and max-delta (`:5796-5818`). The remaining static table is not:

| Location | Fabricated content | Presentation |
| :--- | :--- | :--- |
| `:6032-6039` | `VGPR Pressure: 128 VGPRs` / `64 VGPRs`; `SIMD Utilization: 68.2% (Divergent Wavefronts)` / `94.7%` | bright orange/green text, **no** qualifier, computed by nothing |
| `:6017`, `:6035` | `Reference Target: 185.40 MRays/s (201.2 FPS)` | labelled "Reference Target", but the reference is a **720 p** figure (`"Primary Rays: 921,600 (1280x720)"`) shown beside a **4K** live measurement |
| `:6118` | `BVH Traversal: 44.8 steps/ray` (and per-scene variants) | drawn as canvas HUD overlay; nothing measures it |
| `:6105-6117`, `:6180-6195` | Pipeline-Pass mode: `Time: 2.03 ms`, `Throughput: 4,085.6 MRays/s`, `Effective FPS: 492.6` for all 7 passes | **no qualifier at all**; rendered in the pass banner and every pill tooltip |

**Fix:** delete these fields, or drive them from a versioned `baselines/*.json` asset stamped with device + driver + date, rendered explicitly as `"Reference (R9700, mesa 26.2, 2026-09-28): …"`.

### C-4. `DeviceDatabase` device IDs are wrong — including for this machine

**Where:** `cpp_src/core/DeviceDatabase.cpp:23-260`, cross-checked against `/usr/share/hwdata/pci.ids`.

| DB entry | DB claims | `pci.ids` truth |
| :--- | :--- | :--- |
| `0x7448` | "AMD Radeon AI PRO R9700", gfx1201 **RDNA 4** | **Navi 31 Radeon Pro W7900 (RDNA 3)** |
| `0x7449` | RX 9070 XT, RDNA 4 | Navi 31 Radeon Pro W7800 48 GB |
| `0x744A` | RX 9070, RDNA 4 | Navi 31 Radeon Pro W7900 Dual Slot |
| `0x17F0` | "AMD Radeon 8060S Graphics" | **Strix/Krackan/Strix Halo NPU** — not a GPU |
| `0x15BF` / `0x15C8` | Radeon 890M / 880M (gfx1150) | Phoenix1 / Phoenix2 (RDNA 3) |
| `0x7460` | RX 7800 XT | Navi 32 **Radeon PRO V710** |
| `0x740F` | Instinct MI300X, gfx942 | **Aldebaran MI210, gfx90a** |

Correct identifiers: RDNA 4 Navi 48 is `0x7550` (RX 9070 family) and `0x7551` (Radeon AI PRO R9700); Navi 44 is `0x7590`; Strix Halo GPU is `0x1586`; MI300X is `0x74A1`.

**Verified consequences.** This machine reports `device_id: 0x1586`, matching no row, so lookup falls through to name-pattern synthesis and the device is labelled **`gfx1150 (RDNA 3.5)`** where `rocm-smi` reports **`gfx1151`**, with all theoretical peaks silently zeroed (so "% of Boost Peak" fields vanish from JSON). Conversely a real Radeon Pro W7900 would be labelled `gfx1201 (RDNA 4)` and scored against an RDNA 4 box peak.

**Fix:** regenerate the table from `pci.ids`, add a CI job that diffs it against `hwdata/pci.ids`, and prefer driver-reported `VkPhysicalDeviceProperties` / `VK_EXT_physical_device_drm` over marketing-name pattern matching.

### C-5. Theoretical-peak table is internally inconsistent and partly wrong

**Where:** `DeviceDatabase.cpp:24-160`.

- R9700 and RX 9070 XT carry **identical** `48.66 TFLOPS / 300.8 GIS/s / 1203.2 TIS/s`. The R9700 is a cut-down Navi 48 (54 CU) and cannot match the 9070 XT (64 CU). `48.66 = 64 CU × 256 FLOP/clk × 2.97 GHz` — i.e. the 9070-XT figure is sitting on the R9700 row.
- The 8060S entry lists 23.56 TFLOPS = `40 CU × 256 × 2.3 GHz` (a *game* clock) while the AMD discrete rows above it use boost clocks. Measured 20.26 TFLOPS under contention is consistent with a ~29.7 TFLOPS boost peak, so the table understates by ~20 %.
- Navi 48 Infinity Cache is listed as 64 MB; the published RDNA 4 figure is 48 MB. **Verify against AMD documentation before shipping.**
- `theoreticalBoxGis = 4 × theoreticalTriangleGis` is a modelling assumption presented as a spec constant. State the assumption.

### C-6. `InShaderIndirectBench` rewards doing less work

**Where:** `InShaderIndirectBench.cpp:160-162` — `GetResult()` returns `kTotalItems` unconditionally, so the metric is items/s over a *fixed nominal* item count that includes pruned items.

Live output:

```text
│ 1 % Active                                   │ Vulkan │ 39,955.85 MItems/s │ └──> 1.68x (+68.2%)  │
│ In-Shader Zero-Dispatch Pruning (0 % Active) │ Vulkan │ 50,618.38 MItems/s │ └──> 2.13x (+113.0%) │
```

A pruning benchmark whose headline number rises as work falls is a category error. **Fix:** report wall-time (ms) or speedup-vs-fixed-grid; if items/s is retained, count only *active* items.

### C-7. Cancelled runs inflate throughput

**Where:** `BenchmarkRunner.cpp:1078-1090`.

`total_invocations = iterations;` is assigned **before** the timed loop, which can `break` on `cancelToken`. GPU timestamps then measure N actual dispatches while `operations` uses the full planned count. The GUI exposes an Abort button, so this is reachable in normal use.

**Fix:** assign `total_invocations` from the actual loop counter after the loop.

### C-8. Uncommitted working tree contains a source-tree-pollution regression

Committed HEAD does **not** contain this; the dirty tree does:

```cmake
set(MIRROR_SHADER "${CMAKE_CURRENT_SOURCE_DIR}/kernels/vulkan/${SHADER_NAME}")
COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADER} ${MIRROR_SHADER}
```

`kernels/vulkan/` (118 tracked files) is a **build-generated mirror** of `shaders/` (99 tracked files). Confirmed byte-identical copies, and confirmed that a plain `cmake --build build` leaves `kernels/vulkan/{fp16,fp32,fp64}.comp` modified in `git status`. This is the same defect already fixed for the ROCm half in the 2026-10-06 audit (C-3 there).

**Do not commit this.** **Fix:** `git rm -r --cached kernels/vulkan`, delete the mirror commands, add `kernels/vulkan/` to `.gitignore`, and let `KernelPath.cpp` resolve build/install directories (it already does).

### C-9. Latent wrong `VkStructureType` fallbacks

**Where:** `VulkanContext.cpp:727-735`.

```cpp
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT            ((VkStructureType)1000521001)  // real: 1000567000
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES_KHR  ((VkStructureType)1000528001)  // real: 1000528000
```

Currently dead code — both the vendored 1.4.344 headers and the system 1.4.341 headers define these symbols, and `1000521001` corresponds to no structure at all. The moment anything builds against older headers it chains **bogus sTypes into `vkCreateDevice`**, which is undefined behaviour that validation layers would only catch if enabled.

**Fix:** delete all three `#ifndef` fallbacks and the hand-rolled `struct VkPhysicalDeviceFloat8FeaturesEXT` / `…FloatControls2…` duplicates; the vendored headers are already first in the include path.

### C-10. README sample CLI transcript is not reproducible from the binary

Rows present in `README.md` that exist nowhere in the code (verified by grep across `cpp_src/`):

| README row | Code hits |
| :--- | :--- |
| `Hardware BVH8 Box Peak Rate` | 0 |
| `Hardware Triangle Peak Rate` | 0 |
| `VRAM Read (256 threads/group)` | 0 |
| `Primary rays (coherent)` | 0 |
| `L1 Cache Latency` / `L2` / `L3` | registered only inside `/* */` blocks at `BenchmarkRunner.cpp:389-401` |

The Features list also still claims "L0/L1/L2/L3 Cache latency", and lists FP6 among supported data types although `Fp6Bench::Setup` throws unconditionally and no `fp6.comp` exists.

For a product whose deliverable is numbers, a fabricated sample transcript is the highest-visibility integrity defect in the repository. **Fix:** regenerate the transcript from a real run **on stated hardware**, or delete it.

### C-11. Third-party asset licensing is unaddressed

`assets/models/sponza.glb` (25.8 MB) is Crytek Sponza, distributed under a research/non-commercial licence. The project is MIT and ships `.rpm`, `.deb`, `.tar.gz` and NSIS packages. There is **no `THIRD_PARTY_NOTICES` file at all**, so the required copyright notices for Dear ImGui, ImPlot, cgltf, stb, CLI11 (BSD-3), SDL3 (zlib) and the Vulkan headers are not redistributed.

**Fix:** add `THIRD_PARTY_NOTICES.md`, ship it in all packages, and replace Sponza with a permissively-licensed equivalent (CC0 or glTF-Sample-Assets).

---

## 5. Minor bugs

1. **Dual-Issue labels do not match the kernels.** `dual_issue_ilp4.comp` implements a 2-value ping-pong, not the "sequential dependency prevents dual-issuing" single-issue baseline its description claims; `dual_issue_ilp8.comp` has 4 chains, not the advertised "8 chains". The `ilpN` suffix denotes *ops per iteration*, not chain count. Consequence: the reported "1.43× dual-issue speedup" is really ILP scaling, and on RDNA 3.5 — which has no FP32 co-issue — the tool still prints `FP32 (FP32+FP32) → 1.43x`. Worse, the "peak dual-issue saturation" figure (13.06 TFLOPS) is **below** the independent FP32 benchmark (20.26 TFLOPS) because the kernel shapes differ. Fix: compare like-for-like kernels, rename the suite to "ILP Scaling & Datapath Concurrency", or gate the dual-issue claim on an architecture check.
2. **Two benchmarks, two answers for the same quantity.** `L0 Cache Latency` = 34.6 ns; `Cache Latency Curve (16 KB)` = 62.0 ns. Same working set, 1.8× apart, no explanation. The curve is also non-monotonic (4 MB → 212.5 ns, 8 MB → 199.2 ns) because there is no repetition or median.
3. **`"L0 cache"` does not exist on AMD.** The 16 KB pointer chase measures GL1/L1 on RDNA. Document the vendor mapping or rename.
4. **Telemetry indexes DRM cards as Vulkan device indices.** `TelemetryWorker.cpp:63-70` enumerates `/sys/class/drm/cardN` and stores `deviceIndex = cardIdx`, which `updateTelemetrySelection()` then matches against the *Vulkan* device index. On a hybrid laptop (card0 = iGPU, card1 = dGPU; Vulkan enumeration order may differ) telemetry is attributed to the wrong GPU. `DeviceInfo::pcieBusId` already exists — match on it.
5. **Telemetry is Linux-only.** No `_WIN32`/`__APPLE__` path anywhere in `TelemetryWorker.cpp` or `HardwareTelemetry.cpp`, while the README presents Windows as first-class and `release.yml` builds macOS DMGs. The HUD is dead on both platforms.
6. **`-r auto` is not adaptive.** `main.cpp:296` maps `auto` straight to 3840×2160; there is no VRAM-based tier-down. This contradicts the README ("default: auto; 4K UHD on >=16GB VRAM") and `TODO.md` ("automatic graceful tier down to 1440p/1080p for lower VRAM cards"). A 6 GB card will attempt 4K.
7. **README ↔ GUI scene-stat contradictions.** Forest: README claims "1,001,280 triangles … 512×512 terrain … 850 trees"; the code builds a **256×256** grid (`AAAForestScene.h:117`) with 350 pines + 180 birches = **530** trees, summing to ~1,007,280 triangles; the GUI says "1,050,000+". Outdoor: README says 57,216 (correct — matches `PRIM_TREES_END`, `OutdoorLandscapeScene.h:18`); the GUI says "150,000+". Compute scene statistics at build time and print them; never hand-maintain them in two places.
8. **Speedup claims disagree three ways** for the same scene: README 2.23× (indoor), GUI static table 2.80×, measured 3.26×. None state hardware, driver, resolution or date. Every published number needs a provenance stamp.
9. **Version drift.** CMake, README and AppStream metainfo all say `1.0.0`; `TODO.md` describes work "Completed in v1.2.0" and targets "v1.3.0". `gui/main.cpp:52` and `GuiApp.cpp:32` carry `#define GPUBENCH_VERSION "1.0.0"` fallbacks — make that `#error` instead of silently misreporting.
10. **Unverified cross-vendor guarantees in the whitepapers.** `RDNA3_RAY_TRACING_ARCHITECTURE.md:535` ("guaranteed to … route to a distinct L2 cache bank across AMD, NVIDIA and Intel") and `RDNA4_RAY_TRACING_ARCHITECTURE.md:323` ("Keeping payloads ≤ 16 bytes guarantees that multi-million ray queues fit entirely within the monolithic L2 cache") — the latter is simply false: residency depends on queue bytes versus L2 capacity, not payload width. Soften to observed behaviour with evidence, or delete.
11. **Stale audits in `docs/`.** `FULL_PROJECT_REVIEW_2026.md` grades subsystems C-/F and describes a "dual GUI reality" and a Rust stack that no longer exist. Move to `docs/archive/` with a superseded banner.
12. **`VK_KHR_portability_enumeration` requested unconditionally** (`VulkanContext.cpp:147`) rather than only when present. Harmless, but emits a loader warning on strict conformant implementations.
13. **GUI requests `VK_API_VERSION_1_3`** (`gui/VulkanContext.cpp:176`) while the core negotiates 1.4 and the README claims "Vulkan 1.4+".
14. **`ImGuiConfigFlags_DockingEnable` set but no dockspace ever created** — `GuiApp.cpp` contains zero `DockSpace`/`DockBuilder` calls. Risk without benefit. Separately, `style.WindowBorderSize = 0.0f` while dozens of `BeginChild(..., true)` call sites request borders, so those borders are invisible.
15. **Mixed GLSL versions.** 11 of 51 `.comp` files are `#version 450`, the rest 460 — contradicting `VERSION_REQUIREMENTS.md` ("All Vulkan shaders in the shaders/ directory target GLSL version 460").
16. **16 orphan shaders** are compiled every build and shipped in every package with no C++ referencing them: `fp4_native.comp`, `fp8_native.comp`, `fp8_emulated.comp`, `int4_native.comp`, `membw_512.comp`, `rt_path_tracing.comp`, `rayincoherent.{rgen,rchit,rmiss}`, `raymatdiv.{rgen,rmiss}`, `raymatdiv_mat{0..3}.rchit`, `raydiv_pipeline_ser.rgen`.
17. **Deprecated CLI aliases retained** (`--output`, `--output-file`, hidden via `->group("")`). The project states no backwards-compatibility requirement — delete them.
18. **`--quiet` suppresses the report, not just progress.** `gpubench -g compute -q` printed only the version banner.
19. **`.gitignore` has an unconditional `*.txt`** (with a single `!CMakeLists.txt` exception). Any future `requirements.txt` or `NOTICE.txt` is silently untracked.
20. **Stray artifact at repo root:** `VP_VULKANINFO_AMD_Radeon_8060S_Graphics_(RADV_STRIX_HALO)_26_2_3.json` (265 KB, untracked). Clean up and ignore.
21. **Silent error path in `VulkanContext::setKernelArg`** (`:2000-2008`): when a value argument lands on a descriptor slot it prints to `std::cerr` and **returns** rather than throwing, unlike every other failure in that class.
22. **Push-constant packing assumes 4-byte scalars.** `offset = (arg_index - numBufferDescriptors) * 4` (`:2011`) ignores GLSL member alignment. Silent corruption if a shader ever declares a `uint64_t` or `vec2` push constant.
23. **Wave32 is forced for all compute pipelines** (`:1858-1865`) while `DeviceInfo::subgroupSize` still reports 64 for this device — a silent mismatch between reported and actual execution width. Document it or surface the effective subgroup size.

---

## 6. Optimizations

### O-1. Use the hardware counters that are already available

`VK_KHR_performance_query` **is exposed by RADV on this machine** (verified via `vulkaninfo`) and is unused. So is `VK_EXT_calibrated_timestamps`.

The tool currently *infers* microarchitectural behaviour from wall-clock and then publishes the inference as a metric — which is precisely how the 113× and "1.43× dual-issue" artefacts were born. Add an optional instrumentation layer reporting, alongside each result:

- `VALU busy %`, `SIMD waves occupied`, `VGPRs/wave`, `LDS bank conflicts`, memory read/write counts from `VK_KHR_performance_query` (RADV), `rocprofiler-sdk` / AMD GPU Metrics, or RGP;
- `VK_EXT_calibrated_timestamps` to correlate GPU timestamps with the host clock so results align with `amd-smi` power/clock telemetry.

This converts the dual-issue, LDS-bank-conflict, divergence and occupancy claims from *inferred* to *measured*, and is the single change that most improves the credibility of the project's differentiating content.

### O-2. Eliminate per-dispatch command-buffer recording in the timed loop

`VulkanContext::dispatch()` (`:2017-2097`) performs `vkBeginCommandBuffer` → record → `vkEndCommandBuffer` → `vkQueueSubmit` for **every single dispatch**, and the timestamp bracket spans the whole loop — so CPU submission cost (~10–30 µs/dispatch) lands *inside* the measured GPU interval whenever a kernel is short. With `target_duration_ms = 250` and up to 5000 iterations this is material for every microbenchmark.

**Fix:** for compute microbenchmarks, record N iterations into one (or a few) pre-recorded command buffers, or use `VK_EXT_nested_command_buffer` (available here). Expect measurable gains on FP32/FP16/INT8 and every short RT config.

### O-3. Statistics and environment rigour

Currently mean-only, single window, no environment capture — and the tool reports scores with the iGPU at 96 % busy. It did exactly that during this review.

- Report **min / median / p95** per benchmark, not mean.
- Pre-run guard: sample `gpu_busy_percent` and CPU load; warn, and refuse without `--force`.
- Record in JSON: clocks during the run, background busy %, calibrated timestamp offset, and the `--self-test` envelopes.
- Add **energy (J/TLOP)** — power is already sampled in the GUI; wiring it into the CLI is a genuine differentiator, especially on APUs.

### O-4. Close the backend-parity gap deliberately, and explain it

Same-hardware FP32: **Vulkan 20.26 vs ROCm 16.58 TFLOPS** (−18 %). The kernels are now structurally identical (32 `float4` accumulators, 16 384 iterations, 256 ops/iter), so the gap is Mesa ACO versus LLVM codegen — a legitimate *finding*, but the tool should say so. Emit per-backend VGPR/spill counts under `--verbose` (RGA, `-Rpass-analysis=kernel-resource-usage`, or ACO stats via `scripts/profile_registers.py`) so a 20 % delta is attributable rather than mysterious.

### O-5. Shader and pipeline build hygiene

- `glslc -O` is used; add an explicit `spirv-opt -O` pass and, more importantly, **check in pre-compiled SPIR-V for release packages** so behaviour does not depend on the build host's glslang version.
- Replace `--offload-arch=native` (`CMakeLists.txt:221`) with an explicit list (`gfx1151,gfx1201,gfx1100,gfx1030,…`). `native` makes release artefacts host-specific and silently drops support on every other card.
- Add `CONFIGURE_DEPENDS` to `file(GLOB SHADERS …)` (the HIP glob already has it).

### O-6. Adopt VMA — or test the hand-rolled suballocator

The 64 MB staging pool plus first-fit block suballocator (`VulkanContext.cpp:1092+`) is a real improvement over raw `vkAllocateMemory` churn. But it is ~300 lines of untested allocator logic with manual `new`/`delete` handles (`:1357`, `:1770`, …, 20+ sites) behind `void*` typedefs. Either adopt VMA — which also yields `VK_EXT_memory_priority`, defragmentation and budget queries for free — or add CTest coverage for split / coalesce / alignment / exhaustion. Given the stated "no legacy, cutting edge" posture, VMA is the better call.

### O-7. Modernise the two remaining legacy API paths

- `PixelFillRateBench` uses `VkRenderPass` + `vkCmdBeginRenderPass`. `VK_KHR_dynamic_rendering` is core in 1.4 and available here — switch, and add a `VK_EXT_mesh_shader` variant (also available) so the raster suite covers the modern geometry path.
- **Sync2 is probed but unused:** `vkCmdPipelineBarrier2KHR` is loaded (`:968-972`) while **31 legacy `vkCmdPipelineBarrier` call sites remain** across core and the RT benchmarks. Either use it or stop paying for the probe.
- Timeline semaphores (`VK_SEMAPHORE_TYPE_TIMELINE`, core since 1.2) would tighten the multi-dispatch DGC sequences, which are still fence-based.

### O-8. Data-type width choices worth revisiting

- Compute kernels use `float`/`vec4` throughout. On RDNA 3.5/4 the packed `f16_vec2`/`bf16_vec2` path is the one that reveals true peak. FP16 vector reaches 24.7 TFLOPS against a ~2× FP32 ceiling — dump ISA to confirm whether packed ops are actually being generated.
- `RayRawTraversalBench` reports GIS/s using a **hard-coded** `× 64` "box tests per ray" (`ResultFormatter.cpp:1643`, `RayRawTraversalBench.cpp:521`). That is an assumption, not a measurement; label it and make the divisor explicit in the output.
- `rayCount = 64'000'000` is commented "to maximally saturate 64 CUs (128 SIMD32s)" on a 40-CU part. Scale from the queried CU count.

---

## 7. Nice-to-have features

1. **`--self-test` / conformance mode** — the single most valuable new feature (§3.4). Assert every A/B pair and every absolute number against a versioned envelope; exit non-zero on violation. This is what makes the project citable.
2. **Real test suite.** No `enable_testing()`, no CTest, no gtest. Highly testable pure logic exists: JSON round-trip (the hand-rolled parser/writer in `ResultImporter.cpp` is ~1100 bespoke lines — either replace with nlohmann/json or cover it properly), formatter width arithmetic with CJK/emoji, baseline-tree selection, PSNR/MAE math, suballocator split/coalesce, `KernelPath` resolution.
3. **CI that actually gates.** Current CI builds Linux and Windows, runs smoke tests with `|| true`, sets no `-Werror`, runs no tests, and `check-parity` exists but **CI never runs it**. Add: `-Werror` on first-party code; clang-tidy; a self-hosted GPU job running `verify_benchmarks.py` + `check-parity`; a `pci.ids` consistency check for `DeviceDatabase`.
4. **CSV export** and a **composite index** — the `TODO.md` leaderboard depends on the latter.
5. **Windows/macOS telemetry** (ADL / `amd-smi` on Windows, IOKit on macOS) so the HUD is honest on all advertised platforms — or state Linux-only in the README.
6. **SOTA backlog worth planning for:** `VK_KHR_performance_query` (§O-1); `VK_EXT_shader_object`, which would delete most of the pipeline/descriptor boilerplate the `void*` facade exists to paper over; `VK_EXT_mesh_shader`; `VK_EXT_opacity_micromap`; 2:4 structured sparsity; bindless / descriptor indexing; async-compute multi-queue overlap (the context currently requests **one** queue family and never exercises a second); rasterization-order views; OpenCL 3.1 SPIR-V ingestion (already in `TODO.md` — would let OpenCL share the Vulkan `.spv` pipeline and unlock native FP8 there).
7. **Methodology references to align with.** The field has already solved several of these problems: Yan et al., *"Benchmarking the Memory Hierarchy of Modern GPUs"* and *"Triad, too slow on GPUs"* (PPoPP '20) bear directly on the membw/cache benchmarks; Pullini et al., *"Understanding the AMD RDNA architecture"* (IEEE TPDS) is the correct methodology for the dual-issue/VOPD claims; Cederman & Tirtha's GPU **Roofline** model is a natural presentation layer for GPUBench's data. Adopting their controls — cache invalidation between passes, explicit warmup counts, clock pinning — would materially raise the rigour of the suite.
8. **Clock governance.** DPM warmup exists (good). Add documented, optionally-automated clock locking via `amd-smi`, and stamp every result with the achieved clocks.
9. **GUI polish.** No font is bundled; the loader falls back to system Adwaita/DejaVu/Noto and finally to ImGui's embedded bitmap font, which looks dated when upscaled on HiDPI. Bundle a font as an asset. Also: 123 `PushStyleColor` calls and 353 inline `ImVec4` literals scattered through `GuiApp.cpp` mean the well-designed `setupDarkTheme` is largely bypassed — centralise a semantic palette (e.g. `theme::accentCyan`) so the look is adjustable in one place.

---

## 8. Build system and tooling currency

| Item | Status |
| :--- | :--- |
| Hard-coded `/opt/rocm/core-10.0` in 8 places (`CMakeLists.txt:87-201`) | ❌ **Broken today** — `/opt/rocm/core-10.1` is installed on this machine and is ignored. Use `find_package(hip)` / `/opt/rocm` / `ROCM_PATH`. |
| SDL3 pinned to `release-3.2.8` via FetchContent | ❌ Stale — system SDL3 here is **3.4.16**. Bump the fallback pin. |
| Vendored Vulkan headers 1.4.344 shadowing system 1.4.341 | ⚠️ Unnecessary; also vendors `vulkan-1.lib` / `.dll` binaries in-tree. Prefer `find_package(Vulkan)` and delete `external/vulkan/`. |
| Dear ImGui 1.91.3-WIP, ImPlot 0.16 | ⚠️ Current-ish. ImGui 1.92 unified docking into master and reworked per-monitor DPI; worth a planned bump. |
| Warning flags | ✅ `-Wall -Wextra` + stack-protector + `_FORTIFY_SOURCE=3` + full RELRO + `noexecstack`. First-party code is clean. |
| Two blanket suppressions | ⚠️ `-Wno-unused-parameter` and `-Wno-missing-field-initializers` currently mask ~hundreds of diagnostics. The latter is fixable properly with **C++23 designated initialisers** for Vulkan structs, which would also remove the ability of `-Wno-…` to hide a genuinely uninitialised `VkAccelerationStructureBuildGeometryInfoKHR::mode`. |
| `-Werror` | ❌ Absent. Add in CI with a short allowlist. |
| C++23 | ✅ Core and GUI both on 23. |
| `IComputeContext` `void*` handles + 25 `dynamic_cast` bypasses | ⚠️ Still present; the root cause of much RT fragility and the reason RT benchmarks cannot run on non-Vulkan backends. |
| clang-tidy / sanitiser builds | ❌ Absent. An ASan/UBSan CI job over the non-GPU logic would be cheap and valuable. |

---

## 9. Recommended sequencing

| Horizon | Work |
| :--- | :--- |
| **This week — integrity (days, not weeks)** | C-1 work-equivalence guard on all speedups · C-2 fix FP32 recurrence and make `ValidateResults` load-bearing · C-3 strip fabricated GUI fields (VGPR / SIMD % / pass times / steps-per-ray) · C-4 regenerate `DeviceDatabase` IDs from `pci.ids` · C-10 regenerate or delete the README CLI transcript · C-8 delete the source-tree mirror and untrack `kernels/vulkan/` · C-9 delete the bogus sType fallbacks · C-6 metric fix · C-7 cancellation fix · C-11 third-party notices and Sponza replacement · commit or discard the 41-file dirty tree |
| **Next 2–4 weeks — rigour** | O-3 statistics, load guard and clock metadata · O-1 `VK_KHR_performance_query` instrumentation layer · O-2 command-buffer batching · O-4 per-backend parity attribution in `--verbose` · N-2 CTest unit tests · N-3 CI gates (`-Werror`, parity job, `pci.ids` check) · O-6 VMA · §5.1/5.2/5.7 label and metric corrections |
| **Backlog** | O-5 explicit `--offload-arch` and checked-in SPIR-V · O-7 dynamic rendering, Sync2 call-site migration, timeline semaphores · typed Vulkan RT interface to retire the 25 `dynamic_cast` bypasses · Windows/macOS telemetry · SOTA list (§7.6) · composite index and leaderboard |

---

## 10. Prior-audit verification matrix

Verification of the 2026-10-06 review's findings against the current tree.

| Prior finding | Status at `b1d5982` |
| :--- | :--- |
| C-1 GUI displays fabricated parity results | ⚠️ **Partially fixed** — parity banner is now honest (`NOT MEASURED`/`PASS`/`FAIL` with live PSNR). Static `SceneMetadata` and `PassInfo` tables still fabricate VGPRs, SIMD %, per-pass times and BVH steps/ray. See C-3. |
| C-2 "VRAM Round-Trip Bandwidth" mislabelled | ✅ **Fixed** — config 28 renamed to "Queue Compaction Throughput". |
| C-3 Build artifacts committed / source-tree pollution | ⚠️ **Regressed** — ROCm half fixed (`.co` and duplicate `.hip` untracked), but the Vulkan half is **re-introduced uncommitted** as a source-tree mirror. See C-8. |
| C-4 Cross-backend timing not comparable | ✅ **Fixed** — GPU timestamps on Vulkan, OpenCL and ROCm. |
| C-5 FP6 empty stub | ✅ **Fixed** — reports UNSUPPORTED with an NVIDIA-only reason. README still lists FP6 as a supported data type. |
| C-6 Zero compiler warning flags | ✅ **Fixed** — `-Wall -Wextra` plus hardening flags. `-Werror` still absent. |
| M-1 `"GB GDDR"` label on unified memory | ✅ **Fixed** — now "81 GB Unified Memory". |
| M-2 R9700 peaks hardcoded in JSON export | ⚠️ **Partially fixed** — routed through `DeviceDatabase`, but the table itself is wrong (C-4, C-5) and never matches this machine. |
| M-3 Cache sizes hardcoded by device-name substring | ⚠️ **Partially fixed** — centralised into one table; the table's device IDs are wrong. |
| M-4 Feature flags hardcoded `true` | ✅ **Fixed** — real driver queries. |
| M-5 README claims vs reality (cache latency, RayPayload, scene FPS) | ❌ **Still open** — see C-10, §5.7, §5.8. |
| M-6 GUI version strings hardcoded / GUI pinned to C++17 | ⚠️ **Partially fixed** — C++23 now; `GPUBENCH_VERSION "1.0.0"` fallback defines remain. |
| M-7 `--quiet` not exposed | ⚠️ **Fixed but over-broad** — `-q` now exists yet also suppresses the report. |
| M-8 `hip_kernels/fp64.hip` single-accumulator RAW chain | ✅ **Fixed** (FP64 Vulkan/ROCm parity addressed). |
| M-9 `hip_kernels/int4.hip` dummy | ✅ **Fixed** — deleted. |
| M-11 Shader reflection filename fallback / magic sType `#define`s | ❌ **Still open** — and the fallback values are wrong. See C-9. |
| M-12 `dynamic_cast<VulkanContext*>` in benchmarks | ❌ **Still open** — 25 sites. |
| M-13 "L0 cache" naming | ❌ **Still open**. |
| M-14 Telemetry Linux-only | ❌ **Still open**, plus a new device-index conflation bug (§5.4). |
| O-1 BF16 unmeasurable on every backend | ❌ **Still open** — correctly reports UNSUPPORTED, but no native path implemented. |
| O-2 VMA / staging pool | ✅ **Fixed** — persistent staging pool and block suballocator (untested; see O-6). |
| O-3 Load guard + measurement metadata | ❌ **Still open** — demonstrated live during this review. |
| O-4 Clock governance | ❌ **Still open**. |
| O-5 Timeline semaphores | ❌ **Still open**. |
| O-6 ROCm kernel parity | ✅ **Substantially fixed** — FP32 kernels now structurally identical; residual 18 % delta is compiler codegen (see O-4). |
| O-7 `spirv-opt -O3` / explicit `--offload-arch` | ❌ **Still open** — `glslc -O` only; `--offload-arch=native`. |
| O-8 Energy efficiency metric | ❌ **Still open**. |
| N-1 Delete Rust/Iced GUI | ✅ **Fixed** — complete across CMake, CI, packaging and desktop entries. |
| N-2 Real test suite | ❌ **Still open** — no CTest, no CI test job, `check-parity` never run by CI. |

---

## 11. Appendix: verification notes

- `git log --oneline | wc -l` → 78 commits, 2026-09-20 → 2026-10-07. Working tree: 41 modified/deleted/untracked paths.
- **Fresh strict compile.** All first-party translation units compiled with `g++ -std=c++23 -fsyntax-only -Wall -Wextra` (suppressing only `unused-parameter` and `missing-field-initializers`): **zero remaining warnings**. Removing both suppressions yields ~hundreds of diagnostics, all in the two suppressed classes.
- **Device identity.** `rocm-smi --showproductname` → `Card Model: 0x1586`, `GFX Version: gfx1151`. GPUBench JSON export → `"device_id": "0x1586"`. `/usr/share/hwdata/pci.ids` → `1586 Strix Halo [Radeon Graphics / Radeon 8050S Graphics / Radeon 8060S Graphics]`; `17f0 Strix/Krackan/Strix Halo Neural Processing Unit`; `7448 Navi 31 [Radeon Pro W7900]`; `7551 Navi 48 [Radeon AI PRO R9700]`; `7550 Navi 48 [Radeon RX 9070/9070 XT/9070 GRE]`; `74a1 Aqua Vanjaram [Instinct MI300X]`; `740f Aldebaran/MI200 [Instinct MI210]`.
- **`VkStructureType` values.** Vendored `external/vulkan/Include/vulkan/vulkan_core.h`: `…SHADER_FLOAT8_FEATURES_EXT = 1000567000`, `…SHADER_FLOAT_CONTROLS_2_FEATURES = 1000528000`, `…INVOCATION_REORDER_FEATURES_NV = 1000490000`. `1000521001` appears nowhere in the header.
- **Runtime runs performed** (all under 96 % background GPU load; directional only): `--list-devices`, `--list-backends`, `-b FP32` (Vulkan 20.26 / ROCm 16.58 TFLOPS), `-g compute` (full), `-g memory`, `-b rayscheduling -s indoor -r 720p`, `-b FP6`.
- **Validation failures observed:** `FP32`, `Dual-Issue (Standard FP32)`, `Dual-Issue FP32 (Partial Co-Issue)`, `Dual-Issue FP32 (FP32+FP32)` — 4 of 4 FP32-family configs, every run.
- **Extension availability on this device** (`vulkaninfo`): `VK_KHR_performance_query` ✔, `VK_EXT_calibrated_timestamps` ✔, `VK_KHR_synchronization2` ✔, `VK_KHR_dynamic_rendering` ✔, `VK_KHR_dynamic_rendering_local_read` ✔, `VK_EXT_mesh_shader` ✔, `VK_EXT_nested_command_buffer` ✔, `VK_EXT_device_generated_commands` ✔. None of the first four are used by the codebase.
- **Duplicate shader trees.** `shaders/` = 93 files, `kernels/vulkan/` = 93 files; `diff` confirms `fp32.comp`, `fp16.comp`, `membw_128.comp` byte-identical. Both tracked in git (99 and 118 files including non-`.comp`).
- **Orphan shaders** (no C++ reference, extension-agnostic grep): 16 files listed in §5.16.
- **Scene statistics.** `AAAForestScene.h`: 131 072 (terrain, `grid_n = 256`) + 2 048 (water) + 450 000 (350 pines) + 92 160 (180 birches) + 224 000 (3 500 ferns) + 40 000 (2 500 shrubs) + 20 000 (180 nurse logs) + 48 000 (600 boulders) ≈ **1 007 280 triangles**, 530 trees, 8 materials. `OutdoorLandscapeScene.h`: `PRIM_TREES_END = 57216` total primitives, `grid_n = 128`.
- **Parity artifacts read:** `renders/render_forest_profile.json`, `renders/render_indoor_profile.json`, `renders/render_indoor_pt16_profile.json` — all report `psnr: 120.0000`, `mae: 0.0000`, `diff_pixels: 0`, `exact_pct: 100.0000`. The README's bit-exact parity claim is genuinely backed; note the enforced gate is looser (`PSNR ≥ 45 dB`, ≤ 0.01 % discrepant pixels, `RaySchedulingBench.cpp:2025-2028`).
