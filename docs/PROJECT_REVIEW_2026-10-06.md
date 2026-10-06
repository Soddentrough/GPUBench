# GPUBench — Full Project Review

- **Date:** 2026-10-06
- **Codebase state:** `master` @ `9ef4e3b` (fully in sync with `origin/master`; no `main` branch exists)
- **Review environment:** AMD Ryzen AI MAX+ 395 / Radeon 8060S (RADV STRIX_HALO, RDNA 3.5), Fedora 44, single GPU (`-d 0`)
- **Caveat:** A 27B `llama-server` instance was running on the same iGPU/unified memory during this review. All runtime measurements taken on this machine are therefore contaminated and were treated as *directional only*; findings rest primarily on static analysis, toolchain verification, and cross-referencing the two prior audits.

---

## 1. Executive Summary

GPUBench is a cross-platform (Linux/Windows/macOS), cross-backend (Vulkan 1.4 / OpenCL / ROCm-HIP) GPU microbenchmark suite covering compute (FP64→INT4), memory/cache hierarchy, ROP fill rate, and — its flagship — hardware ray tracing scheduling architectures (Megakernel vs DGC vs SER) across four 3D scenes, delivered through a Unicode-card CLI (JSON export/import, multi-run comparison) and a Dear ImGui/ImPlot "Workstation Profiler" GUI with live telemetry and a ray-tracing parity viewport.

The project is ~16 days old (62 commits, 2026-09-20 → 2026-10-06), agent-assisted, and fast-moving. Two prior audits exist in-repo (`AUDIT_REPORT.md`, 2026-09-30; `docs/FULL_PROJECT_REVIEW_2026.md`, 2026-10-05). The latest commit (`9ef4e3b`, "upgrade to C++23, resolve audit issues…") claims to resolve those audits. **This review verified that claim item by item: it is substantially true.** The project is in good shape and improving quickly, but it carries a set of *integrity* problems — fabricated GUI parity data, mislabeled metrics, advertised-but-disabled benchmarks, and hardcoded single-chip constants leaking into generic output — that a benchmarking tool cannot afford, because its product *is* its numbers.

**No rewrite is needed.** The recommended path is: (1) fix the integrity defects (days of work), (2) consolidate device-specific magic numbers into one validated table, (3) pick one GUI stack and delete the other, (4) finish measurement rigor (backend event timing, load guards, clock metadata), (5) integrate the existing verification scripts into CI.

### Prior-audit verification matrix

| Prior audit finding | Status at `9ef4e3b` |
| :--- | :--- |
| No GPU hardware timestamps (P0) | ✅ **Fixed for Vulkan** — `VkQueryPool` + `vkCmdWriteTimestamp` (`VulkanContext.cpp:989–1119`), consumed by `BenchmarkRunner` with clock-ramp warmup and 250 ms target duration. ❌ **ROCm & OpenCL still CPU wall-clock** (no `hipEvent*` / `clGetEventProfilingInfo` loaded) |
| OpenCL `CL_MEM_USE_HOST_PTR` PCIe pinning | ✅ Fixed (`CL_MEM_COPY_HOST_PTR`, `OpenCLContext.cpp:486–492`) |
| membw 512 B uncoalesced thread stride | ✅ Fixed (contiguous wave stores in `shaders/membw_*.comp`) |
| FP64 single-accumulator RAW chain | ✅ Fixed in Vulkan (`shaders/fp64.comp`, 8 accumulators) — ❌ **still broken in `hip_kernels/fp64.hip`** (single `val = val*c1+c2` × 2048) |
| RayIntersect non-opaque flags / ×64 ray inflation | ✅ Fixed (`VK_GEOMETRY_OPAQUE_BIT_KHR` at `RayIntersectBench.cpp:117,156`, `gl_RayFlagsOpaqueEXT` in `rt_benchmark.comp:32`, workgroup-reduced atomic) |
| RayASBuild 10×/5× under-report; RayPayload 2× under-report | ✅ Fixed (`ops*iters` at `RayASBuildBench.cpp:575`; `rayCount*2` at `RayPayloadBench.cpp:304`) |
| Legacy Vulkan 1.0 barriers only | ⚠️ Partial — sync2 probed and available, but **21 legacy `vkCmdPipelineBarrier` call sites remain** across core + RT benchmarks |
| No Wave32 request (RADV defaults Wave64) | ✅ Fixed (`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`, `VulkanContext.cpp:1681–1685`) |
| Unconditional ANSI colors; hardcoded 128-col boxes | ✅ Fixed (`isatty` + `NO_COLOR`, `TIOCGWINSZ` clamped 80–128, `ResultFormatter.cpp:29–51,361`) |
| `MESA_VK_IGNORE_CONFORMANCE_WARNING` suppression | ✅ Removed (only `MESA_VK_TRACE=rra` for opt-in RRA capture remains) |
| Validation layer without debug utils | ✅ Fixed (`vkCreateDebugUtilsMessengerEXT` registered) |
| C++17 pin | ✅ Core is C++23 — ❌ GUI target still pinned to C++17 |
| SER dummy rgen shader | ✅ Now a real shader (`shaders/rt_scheduling_ser.rgen`) |
| Megakernel bounce gating on `dumpRenders` | ✅ Inverted intentionally (`maxBounces = dumpRenders ? 2 : pc.bounces`, `rt_scheduling_traditional_megakernel.comp:1054`) |
| VMA absence / 4096-allocation limit / staging churn | ❌ **Still open** (in `TODO.md`) |
| GUI mock data leak (P0) | ❌ **Still open** — see Critical #1 |
| Dual GUI (Rust iced vs C++ ImGui) | ✅ **RESOLVED** — Purged unreleased Rust/Iced stack; standardized on C++23 Dear ImGui (`cpp_src/gui`) across CMake, packages, and desktop entries |
| Zero warning flags | ❌ **Still open** — no `-Wall/-Wextra/-Werror` anywhere |

---

## 2. Critical Bugs

### C-1. GUI displays fabricated parity results

**Where:** `cpp_src/gui/GuiApp.cpp:4132–4310` (`renderRayTracingViewport`).

The static `SceneMetadata` table hardcodes per-scene "reference" values: `"185.40 MRays/s (201.2 FPS)"`, `"523.80 MRays/s (568.3 FPS)"`, `"2.82x (+182.5%)"`, VGPR counts `128/64`, `"120.0 dB (BIT-EXACT)"`, `"0.000"` max delta, and `"BVH Traversal: 44.8 steps/ray"`. The parity banner **unconditionally prints `[PARITY: PASS]` in green** and shows the hardcoded PSNR/max-delta regardless of whether any parity check has run. Live values are computed (`dynamicTechAScore`/`dynamicSpeedup` from `m_allResults`) but the speedup falls back to the hardcoded string when no results exist, and the static table is displayed alongside the dynamic values.

For a benchmarking product, a permanently-green fake "PASS" is a credibility-destroying defect. This was P0 in the prior review and is unfixed.

**Fix:** bind the banner to actual parity results (the engine already computes PSNR/diff-pixels in `RaySchedulingBench.cpp:2001–2089` and can surface them); show "NOT MEASURED" until a parity run exists; delete or clearly label the static reference table (and reconcile it with the README — see B-5).

### C-2. "VRAM Round-Trip Bandwidth" is a mislabeled benchmark

**Where:** `cpp_src/benchmarks/RaySchedulingBench.cpp:1570–1577` (config 28), `GetResult` at 2459–2463.

Config 28 dispatches the *same classify kernel* as config 26 (queue compaction — which performs BVH traversal + hit classification + atomics) and then computes GB/s from an assumed 64 B/ray. On this LPDDR5X-8000 APU it reported **764 GB/s — physically impossible** for the DRAM fabric (~256 GB/s theoretical); the queue working set is L2-resident, so the metric actually measures L2/compaction throughput. Label ≠ function, and the number is not verifiable as claimed.

**Fix:** either implement a true VRAM streaming benchmark for the queues (buffer larger than L2, dependent read-modify-write) or rename to "Queue Compaction Throughput" and report records/s like config 26.

### C-3. Build artifacts committed to git + source-tree pollution

**Where:** `kernels/rocm/*.co` (23 tracked binaries), `CMakeLists.txt` POST_BUILD copy, `kernels/rocm/*.hip` duplicates.

- 23 machine-specific ROCm code objects (compiled `--offload-arch=native`) are tracked in git and were re-committed in `9ef4e3b`.
- A `POST_BUILD` step copies `build/kernels` **into the source tree**, so every build dirties the working tree (verified: `git status` shows all 23 `.co` files modified on a clean checkout after one build).
- `kernels/rocm/*.hip` are verbatim duplicates of `hip_kernels/*.hip` (verified by diff).

**Fix:** `git rm --cached kernels/rocm/*.co kernels/rocm/*.hip`, delete the POST_BUILD source-tree copy (runtime lookup already prefers exe-relative and install paths via `KernelPath.cpp`), keep `hip_kernels/` as the single HIP source, and add `kernels/rocm/` to `.gitignore` (the `*.co` pattern is ineffective on already-tracked files).

### C-4. Cross-backend timing is not comparable

**Where:** `cpp_src/core/BenchmarkRunner.cpp:963–995`, `cpp_src/core/ROCmContext.cpp`, `cpp_src/core/OpenCLContext.cpp`.

Vulkan uses GPU timestamps; `hasGpuTiming()` is implemented only by `VulkanContext`. ROCm and OpenCL fall back to CPU wall-clock around dispatch + `waitIdle()`. The ROCm dlopen layer never loads `hipEventCreate/hipEventRecord/hipEventElapsedTime` and the OpenCL layer never creates events for `clGetEventProfilingInfo` — both are trivial additions to the existing function-pointer tables. Until then, side-by-side backend tables mix measurement methodologies (driver submission + fence latency included for two of three backends).

### C-5. FP6 benchmark is an empty stub

**Where:** `cpp_src/benchmarks/Fp6Bench.cpp` (all of `Setup/Run/Teardown` are empty: "Implementation will be added in a future step"), no `fp6.comp` exists, `info.fp6Support = false` hardcoded (`VulkanContext.cpp:550`).

The README lists FP6 under "Comprehensive Compute Data Types". FP6 is NVIDIA-only (`SPV_NV_float6`); there is no AMD path. The honest fix — matching how INT4 is handled — is to report UNSUPPORTED with a capability reason and remove FP6 from the feature list, not ship a stub that silently produces nothing.

### C-6. Zero compiler warning flags

**Where:** `CMakeLists.txt` (no `-Wall -Wextra -Werror`, no MSVC `/W4`, no clang-tidy).

For ~120k lines of first-party code this is a hygiene hole; it was flagged in the prior review and remains open. Enable `-Wall -Wextra` for local builds and `-Werror` in CI (with a short allowlist if legacy third-party files complain — they currently don't, since imgui/implot compile clean).

---

## 3. Minor Bugs

1. **Hardcoded `"GB GDDR"` VRAM label** — `BenchmarkRunner.cpp:589,597`. Prints "81 GB GDDR" for the Strix Halo's unified LPDDR5X. `dedicatedVramBytes` is already collected in `DeviceInfo`; label shared vs dedicated memory correctly ("unified memory" on APUs).
2. **R9700 peaks hardcoded into JSON export for all devices** — `ResultFormatter.cpp:1561–1571`: `300.8 GIS/s` / `1203.2 GIS/s` compute `pct_theoretical_peak` for *any* GPU, labeled "R9700 Ref" on non-R9700 hardware. Make it a per-architecture table or drop the field.
3. **Cache sizes hardcoded by device-name substring, duplicated 3×** — `VulkanContext.cpp:421–435` and `599–615`, `OpenCLContext.cpp:286–300`: `"gfx12"→8 MB/64 MB`, `"strix"→2 MB/32 MB`, else `4 MB/32 MB`. Fragile string matching, values unverified against AMD architecture documentation, and wrong values silently skew cache-latency buffer sizing. Consolidate into one per-`deviceID` table; verify the Strix Halo (gfx1151) L2/L3 numbers.
4. **Feature flags hardcoded `true` instead of queried** — **RESOLVED (2026-10-06)**: Replaced hardcoded values in `VulkanContext.cpp` with true driver queries: `shaderInt8` from `VkPhysicalDeviceShaderFloat16Int8Features`, `cooperativeMatrix` from `VkPhysicalDeviceCooperativeMatrixFeaturesKHR`, bfloat16 via extension check (`VK_KHR_shader_bfloat16`/`VK_EXT_shader_bfloat16`), and `structuredSparsitySupport` via `VK_NV_cooperative_matrix2`.
5. **README claims vs reality:**
   - "L0/L1/L2/L3 Cache latency" — **L1/L2/L3 latency and all cache-bandwidth benchmarks are commented out** (`BenchmarkRunner.cpp:360–400`: "temporarily disabled due to memory prefetcher & measurement volatility" / compiler DCE). Only the 16 KB "L0" chase + the latency *curve* actually run.
   - CLI help footer advertises **RayPayload** — it is commented out of `discoverBenchmarks()` (`BenchmarkRunner.cpp:336`).
   - README showroom numbers (DGC 101.3 FPS / 1.76×) **contradict** the GUI static table (568.3 FPS / 2.82×) for the same 720p scene — one is stale; pick one source of truth.
   - "BF16, FP8, FP4, INT4" listed as supported data types, but on RDNA 3.5 all four report UNSUPPORTED, and **BF16 is UNSUPPORTED on every backend on every machine** (see O-1).
6. **GUI version strings hardcoded** — `"v1.0.0"` in `GuiApp.cpp:1100`, `gui/main.cpp:55,126` instead of `GPUBENCH_VERSION`; GUI target pinned to C++17 while core is C++23.
7. **`--quiet` not exposed in CLI** although `BenchmarkRunner::setQuiet` exists (used by `RunnerAPI`); "Running Benchmarks (0 workloads)" also prints even when unsupported rows follow.
8. **`hip_kernels/fp64.hip`** is still a single-accumulator RAW chain (measures dependency latency, not throughput) — inconsistent with the fixed 8-accumulator Vulkan shader; cross-backend FP64 numbers are not comparable.
9. **`hip_kernels/int4.hip` is a dummy** — **RESOLVED (2026-10-06)**: Dead dummy work kernel deleted from repository (`git rm hip_kernels/int4.hip`).
10. **CMake hygiene:** `file(GLOB … CONFIGURE_DEPENDS)` added for HIP sources to avoid stale build dependencies; unreleased Rust options purged; machine-specific binary `.co` build copies removed from source tree.
11. **Shader reflection fallback is still filename-based** — `file_name.find("rt_")` at `VulkanContext.cpp:1619` (after the SPIR-V opcode scan); magic `sType` `#define` fallbacks at `VulkanContext.cpp:761–769`.
12. **`dynamic_cast<VulkanContext*>` in 10 benchmark files** — the `IComputeContext` void* facade is bypassed everywhere ray tracing is involved. It works, but it is the root of much fragility and the reason RT benchmarks cannot run on non-Vulkan backends.
13. **"L0 cache" naming** — AMD has no L0; the 16 KB pointer-chase measures GL1/L1 on RDNA. Rename or document the vendor mapping.
14. **Telemetry is Linux-only** — `TelemetryWorker.cpp` reads `/sys/class/drm`; `HardwareTelemetry.cpp` likewise. The "real-time GPU telemetry" GUI feature is dead on Windows/macOS, which the README presents as first-class platforms.

---

## 4. Optimizations

### O-1. BF16 is unmeasurable on every backend — and the blocking assumption is fixable

Verified against the installed toolchains on this system:

- Current `glslc` (shaderc 2026.1, glslang 301b4ede) compiles `GL_EXT_bfloat16` **storage** but still fails bfloat16 **arithmetic** (`fma` on `bfloat16_t` → conversion errors). The project's "glslc lacks bfloat16" comment is therefore accurate *today* but is a moving target.
- ROCm 10.0's `hip_bfloat16` operators are **FP32-emulated** (`float(a) + float(b)` in `/opt/rocm/core-10.0/include/hip/amd_detail/amd_hip_bfloat16.h`) — the project's ROCm comment is verified correct.
- OpenCL C has no native bfloat16 arithmetic — correct.

So the UNSUPPORTED stance is defensible now, but the fix is available and this is the single biggest hole in the "comprehensive data types" claim:

- **HIP:** inline-asm kernel using native packed BF16 ops (`v_pk_fma_f32`-class / `__builtin_amdgcn` intrinsics) — RDNA 3.5/4 have real BF16 datapaths.
- **Vulkan:** Slang (native `bfloat` type) → SPIR-V, ingested via the existing `createKernel` path; or track glslang's improving `GL_EXT_bfloat16` support.

### O-2. VMA (or at minimum a staging-buffer pool) — **RESOLVED (2026-10-06)**

- **Persistent 64 MB Staging Pool**: Implemented persistent host-visible mapped staging buffer, command buffer, and fence in `VulkanContext`, eliminating repeated per-chunk allocation/mapping/destruction in `writeBuffer` and `readBuffer`.
- **Block Memory Suballocator**: Implemented a 64 MB block memory suballocator in `VulkanContext` for device buffers $\le 32$ MB with first-fit aligned chunk splitting and free chunk coalescing. `RayASBuildBench` (5,000 BLAS buffers) now operates well within driver allocation budgets, eliminating raw `vkAllocateMemory` churn. Dedicated allocations are used only for buffers $> 32$ MB.

### O-3. System-load guard + measurement metadata

A benchmarking tool should detect and record conditions. Today, results run (and are reported) silently under heavy background load — exactly this machine's state with the LLM server on the same iGPU. Concretely:

- Pre-run: sample GPU busy% (telemetry code already exists) and CPU load; warn (or refuse with `--force`) above a threshold.
- Record in JSON: clocks during the run, background busy%, driver version (present), and **min/median/p95** instead of mean-only.

### O-4. Clock governance

Warmup for DPM ramp already exists (good). Add documented (and optionally automated via `amd-smi`) clock locking. Published whitepaper numbers should be reproducible; DPM variance on APUs is large.

### O-5. Timeline semaphores

DGC sequence submission is fence-based; timeline semaphores (`VK_SEMAPHORE_TYPE_TIMELINE`, core since 1.2) would tighten multi-dispatch DGC paths and remove per-frame fence recycling.

### O-6. ROCm kernel parity

ROCm membw read @128 threads measured ~25–30% below Vulkan on the same hardware (directional, load-contaminated). For a cross-backend tool, kernel parity per benchmark is a first-class requirement: same accumulator counts, same dispatch shapes. Several already diverge (e.g. FP16: 128 vs 256 ops/iter between backends; FP64: 1 vs 8 accumulators).

### O-7. Shader/kernel build optimization

- `spirv-opt -O3` pass after `glslc -O` for Vulkan shaders.
- Compile release `.co` for an explicit architecture list (`gfx1151,gfx1201,…`) instead of `--offload-arch=native`.

### O-8. Energy efficiency metric

Power is already sampled in the GUI telemetry; wire joules/TLOP into CLI results. This differentiates GPUBench from every mainstream GPU benchmark, especially on APUs where unified-memory power behavior matters.

### O-9. Tie the big speedup claims to profiling evidence

"22.99× in 16 SPP stress" and "8.05× incoherent GI" DGC speedups are large enough to warrant published ISA evidence (megakernel VGPR spills vs DGC micro-kernel occupancy). The tooling exists (`scripts/capture_gpu_profiles.py`, `scripts/profile_registers.py`, RGA/RGP workflows in `docs/PROFILING_GUIDE.md`) but the headline numbers are not tied to it in the docs. (Not verifiable on this box under LLM load.)

---

## 5. Nice-to-Have Features

1. **Pick one GUI and delete the other.** ✅ **RESOLVED:** Formally deleted the unreleased Rust/Iced stack (`gpubench-gui/`, `gpubench-core/`, `gpubench-sys/`, `Cargo.toml`, `Cargo.lock`), removed Rust CI jobs, and consolidated 100% onto the pure C++23 Dear ImGui + ImPlot + SDL3/Vulkan workstation frontend (`cpp_src/gui`). Halved CI build times and eliminated 5.6 GB of build cache.
2. **Real test suite.** Today: no CTest/gtest; the Rust `test_*.rs` bins are manual debug tools; the de-facto regression suite is `scripts/verify_benchmarks.py` (genuinely good — per-architecture baselines, cross-backend ±10% parity, logical invariants) but it is not wired into CI. Add: CTest unit tests for pure logic (JSON round-trip, group expansion, formatter width math, PSNR/parity math), `cargo clippy`/`fmt` in CI, clang-tidy, and a GPU CI job (self-hosted) running the existing `check-parity` CMake target — the parity gate exists but **CI never runs it** (CI runners are GPU-less; smoke tests use `|| true`).
3. **Composite score / index** across the suite for quick comparisons (the planned leaderboard in `TODO.md` depends on it).
4. **CSV export** (JSON exists; CSV is the sysadmin default).
5. **SOTA gaps worth planning:** 2:4 structured sparsity (CDNA4/Blackwell); rasterization-order views (Vulkan 1.4, relevant to the ROP suite); bindless/descriptor-indexing throughput; async-compute multi-queue overlap; OpenCL 3.1 SPIR-V ingestion (already in `TODO.md` — would let OpenCL share the Vulkan `.spv` pipeline and unlock native FP8 there); mesh-shader path for the raster suite.
6. **Windows/macOS telemetry** (ADL/`amd-smi` on Windows, IOKit on macOS) so the HUD is honest on all advertised platforms.
7. **Archive the two audit docs** into `docs/archive/` once their issues close, to keep the root uncluttered.

---

## 6. Strategic Assessment

**What is working:**

- The core design (thin backend facade + benchmark plugins + runner + formatter) is functional and extensible; the recent audit-fix cycle demonstrates the project absorbs hard changes well.
- The differentiating content — DGC vs megakernel vs SER across four scenes with **real** bit-exact parity gating (`PSNR ≥ 45 dB`, ≤0.01% discrepant pixels, four render pairs, `--verify-parity`, `check-parity` target) — is novel and well executed.
- Timing methodology is 80% there: GPU timestamps (Vulkan), DPM warmup, 250 ms target windows, cancellation tokens, UNSUPPORTED-with-reason reporting (a deliberate anti-silent-fallback policy visible in `bcfa36d`).
- Library choices are current: Vulkan headers 1.4.344 (slightly stale), imgui 1.91.3, implot 0.16, CLI11 2.3.2, SDL3, C++23 core, Rust 2024 edition.
- Documentation is deep and mostly factual; the whitepapers (RDNA 3/4, DGC, dual-issue, BVH traversal audit) are publication-grade in scope.

**What is misaligned with the stated goals:**

1. **Integrity.** Fake green "PARITY: PASS", an impossible 764 GB/s "VRAM" label, advertised-but-disabled benchmarks (FP6 stub, cache latency/bandwidth, RayPayload), and R9700-specific peaks in generic output all undercut the product's core value. Fix group 2 (Critical) first — most are days, not weeks.
2. **Device-specific data as code.** Cache sizes, theoretical peaks, memory-type labels, and feature flags are scattered magic numbers keyed on device-name substrings. This project will be run on many GPUs; consolidate into one validated per-`deviceID` database (querying the driver where possible).
3. **Two GUIs, one product.** Keep C++ ImGui (better integrated, smaller, current) and delete the Rust stack.
4. **Measurement rigor.** Finish with ROCm/OpenCL event timing, load guards, clock metadata, and min/median/p95 reporting — that is what separates a hobby benchmark from a citable one.
5. **Documentation drift.** The development testbed moved from R9700/Threadripper to Strix Halo (`AGENTS.md`), but whitepapers, the profiling guide, README examples, and `verify_benchmarks.py` baselines remain R9700-centric. Each published number should state which silicon it was measured on; several "Boost Peak" percentages are only meaningful on the R9700.

### Recommended sequencing

| Horizon | Items |
| :--- | :--- |
| **Quick wins (this week)** | C-1 (GUI mock parity), C-2 (rename/fix config 28), C-3 (git/build hygiene), C-6 (warning flags), M-1…M-7 (labels, README, CLI footer, version strings, `--quiet`), O-9 (docs), N-7 (archive audits) |
| **Sprints (2–4 weeks)** | C-4 (ROCm/OpenCL event timing), O-2 (VMA/staging pool), O-1 (native BF16 path), O-3 (load guard + JSON metadata), N-1 (delete Rust GUI), N-2 (CTest + CI GPU parity gate) |
| **Backlog** | O-5 (timeline semaphores), O-8 (energy metrics), M-12 (typed Vulkan RT interface), SOTA additions (sparsity, ROV, bindless, OpenCL 3.1), composite score, leaderboard (existing TODO) |

---

## 7. Appendix: Verification Notes

- `git fetch origin` → local `master` == `origin/master` (0/0); remote HEAD `9ef4e3b`; only remote tag `v1.0.0` (points at `fecd244`).
- Runtime smoke runs performed on this box (contaminated by background LLM load; directional only): `--list-devices`, `--list-backends` (all 3 backends available), FP32/FP64/INT8 Vulkan, membw Vulkan+ROCm, rayscheduling showroom 720p, FP8/FP4/FP6/INT4/BF16 (all UNSUPPORTED on RDNA 3.5 as designed).
- Toolchain probes: `glslc` bfloat16 storage OK / arithmetic fails (shaderc 2026.1); ROCm 10.0 `hip_bfloat16` operators FP32-emulated (source-verified); HIP 7.15 / clang 23.
- Cross-file checks: `kernels/rocm/*.hip` ≡ `hip_kernels/*.hip` (diff-identical); 23 `.co` files tracked; GUI static table vs README FPS numbers contradictory; `Bf16Bench` history (`e8c87b2` → `3d5d666` re-enable → `bcfa36d` re-disable) confirms the BF16 stance oscillated before settling on UNSUPPORTED.
