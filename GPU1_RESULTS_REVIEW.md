# GPU1 (AMD Radeon AI PRO R9700, device index 1) — Full Suite Results Review
Date: 2026-07-19
Command: `gpubench -d 1 --output json --output-file gpu1_full_suite.json`
Result: exit 0, 57 results, all on device_index=1 (Vulkan) + Host CPU (System backend).
Artifacts: build/gpu1_full_suite.json, build/gpu1_full_suite.log

====================================================================
1. WHAT LOOKS HEALTHY
====================================================================
- FP32 45.6 TFLOPS = ~95% of R9700 theoretical (~47.8). Excellent.
- FP16 matrix 204.7 TFLOPS vs 191 spec (dense) — slightly above spec via boost. Plausible.
- INT8 matrix 391 TOPS vs 383 spec — plausible with boost.
- FP64 0.81 TFLOPS — RDNA4 FP64 is 1/64 rate (~0.75 theoretical). Plausible.
- Memory read BW 625.5 GB/s = 98% of 640 GB/s spec. Excellent.
- R/W combined 563.9 GB/s with 2x byte accounting = 88% of spec. Plausible.
- Cache latency: L0 30.9 / L1 65.2 / L2 83.6 / L3 154.8 ns — monotonic, sane.
  (NOTE: in JSON, 'operations' holds pointer-chase step counts for these;
  ns must be derived as time_ms/ops. The TUI displays it correctly.)
- Ray coherence scaling behaves: RayDivergence 45.8 GRays/s (100% coherent)
  → ~37 (0%); RayAnyHit 37.3 → 21.7 GRays/s as any-hit load rises.
- AS build/update rates plausible (BLAS build 50.7 MTris/s, update 117.9).
- System memory: MT ~51.5 GB/s, 1T read 18.2, latency 135 ns — sane for DDR5.

====================================================================
2. ISSUES FOUND (ranked)
====================================================================
A. RayPathThroughput scaling is INVERTED (likely op-counting bug)
   2 bounces: 7048 MRays/s, 4: 10599, 8: 16419 MRays/s.
   More bounces must be SLOWER per primary ray, not 2.3x faster.
   The counter likely accumulates secondary rays (or normalizes per-bounce),
   inflating throughput with depth. Also inconsistent with every other RT
   suite (~33-45 GRays/s coherent scale). Investigate RayPathTracingBench
   op accounting.

B. RayMaterialDivergence looks ~25x too slow
   Coherent (1 shader): 1.83 GRays/s, Divergent (4 shaders): 1.49 GRays/s.
   Other coherent RT suites do 33-45 GRays/s. A coherent single-shader test
   should be in the same range. Suspect dispatch-size or op-count error
   (e.g. counting only a fraction of launched rays).

C. FP8 matrix and INT4 matrix at HALF their rated rate
   FP8 matrix 207 TFLOPS vs 383 spec; INT4 matrix 390 TOPS vs 766 spec.
   Both measure almost exactly the FP16/INT8 matrix rate (1.00-1.01x).
   Likely the cooperative-matrix shaders use the same K-dim (16) for
   sub-word types instead of 32, or RADV doesn't dual-rate these layouts.
   Either the shader is under-utilizing the hardware (benchmark bug) or the
   numbers should be labeled as such.

D. FP16/BF16 vector only ~1.13x FP32 — packed-rate not achieved
   51.3 / 50.3 TFLOPS vs expected ~2x FP32 (~90+). fp16.comp likely isn't
   issuing packed f16 ops (or RADV scalarizes). Under-reports RDNA4 vector
   FP16/BF16 by ~1.75x.

E. "Write" bandwidth reporting is misleading (side-effect of the per-mode
   byte-accounting fix)
   Write 281.9 GB/s vs Read 625.5 — looks like writes are 2x slower, but
   the Vulkan/ROCm membw kernels issue loads UNCONDITIONALLY (mode is a
   runtime arg), so write-mode bus traffic is actually read+write (~564
   GB/s, matching R/W). We now report logical store bytes only.
   Options: (1) make the shader truly write-only (gate loads like the
   OpenCL kernel does), or (2) relabel the metric to clarify it is logical
   store bandwidth, or (3) count actual traffic on Vulkan/ROCm.

F. FP4 is listed by --list-benchmarks but NEVER runs
   No fp4*.comp shader is compiled (compilation log jumps int4 -> membw),
   no FP4 result, no error, exit 0. Silent gap: either the bench is
   registered unconditionally but inert, or native FP4 unsupported and the
   emulated path never engages. Should print UNSUPPORTED like other skipped
   metrics.

G. FP8 (Vector) reports is_emulated=false in JSON, but the compilation log
   shows fp8_emulated.comp being built — possible mislabeling of the
   emulated flag (verify which shader FP8 vector actually ran).

H. JSON export nits
   - System-backend results carry device_index 4294967295 (UINT32_MAX
     sentinel leaked into output). Should be null or omitted.
   - Latency benchmarks store step counts in 'operations' with metric 'ns'
     but no computed ns value; consumers must know to derive time/ops.
     Consider emitting the derived value as its own field.
   - RayPayload 16B/128B/256B all ~33 GRays/s (payload size has no effect) —
     possibly legitimate (traversal-bound), but worth a sanity check that
     payload size is actually plumbed into the shader.

I. Suite completeness notes
   - Cache BANDWIDTH benchmarks (L0/L1/L2/L3 bandwidth) are not registered
     at all (only latency) — kernels exist in shaders/ (cachebw_*.comp).
   - INT4 Vector and FP6 are absent from --list-benchmarks (Fp6Bench.cpp
     exists in tree). Confirm intended gating.
   - membw_128.comp compiled 3x in a row (41%/38%/35% logs show repeats) —
     per-config recompilation; consider caching compiled pipelines.

====================================================================
3. VERDICT
====================================================================
Core compute (FP32/FP16-matrix/INT8-matrix/FP64), read bandwidth, cache
latency, divergence/any-hit RT suites: trustworthy.
Do NOT cite: RayPathTracing (inverted), RayMaterialDivergence (~25x low),
FP8/INT4 matrix (half-rate), FP16/BF16 vector (half packed rate),
Write bandwidth (logical vs actual traffic).
Reporting bugs to fix: FP4 silent skip, is_emulated flag, JSON device_index
sentinel, latency value derivation.
