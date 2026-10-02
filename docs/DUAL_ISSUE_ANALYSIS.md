# Dual-Issue & Datapath Concurrency Architectural Analysis

## 1. Executive Overview

Modern GPU microarchitectures have evolved beyond simple SIMD vector execution pipelines by introducing dual-issue and concurrent arithmetic execution datapaths:
- **AMD RDNA 3 (Navi 3x / GFX11)**: Introduced static **VOPD (Vector Operations Dual-issue)** 64-bit instruction pairing within each SIMD32 unit.
- **AMD RDNA 4 (Navi 4x / GFX12)**: Overhauled the dual-issue pipeline with streamlined dual-ALU vector execution, removing restrictive legacy VOPD FMA pairing rules while requiring bank-aware register allocation.
- **NVIDIA Turing, Ampere, Ada Lovelace, and Blackwell**:
  - *Turing (TU10x)*: Decoupled FP32 and INT32 execution units capable of concurrent $1\times \text{FP32} + 1\times \text{INT32}$ execution per warp cycle.
  - *Ampere (GA10x) & Ada Lovelace (AD10x)*: Dual FP32 datapath design where Datapath 0 handles FP32 or INT32, and Datapath 1 handles FP32 only (yielding $2\times \text{FP32}$ dual-issue peak, or $1\times \text{FP32} + 1\times \text{INT32}$ concurrent execution).

**GPUBench** provides the **Dual-Issue Efficiency Benchmark Suite** (`--benchmark dualissue` or `-b dualissue`) to systematically evaluate instruction-level parallelism (ILP) scaling, dual-issue saturation, and mixed arithmetic concurrency across AMD and NVIDIA GPUs.

---

## 2. Microarchitectural Comparison

| Feature / Metric | AMD RDNA 3 (GFX11) | AMD RDNA 4 (GFX12) | NVIDIA Ampere / Ada Lovelace |
| :--- | :--- | :--- | :--- |
| **Dual-Issue Mechanism** | Static 64-bit **VOPD** opcode pairing (`v_dual_*`) | Dynamic / compiler-scheduled dual-issue SIMD32 | Dynamic warp scheduler dual-dispatch (Scoreboard) |
| **FP32 Issue Rate** | Up to 2 FMAs / cycle / SIMD32 | Up to 2 FMAs / cycle / SIMD32 | Up to 2 FMAs / cycle / SM sub-partition |
| **INT32 Issue Rate** | Up to 1–2 ops / cycle (dependent on opcode pairing) | Dedicated integer ALUs | 1 op / cycle / sub-partition (Datapath 0 only) |
| **Mixed FP32 + INT32** | Supported for specific bitwise/shift pairs (`v_dual_lshlrev_b32`, `v_dual_and_b32`) | Supported via dual-ALU scheduling | **Full concurrency**: $1\times \text{FP32} + 1\times \text{INT32}$ simultaneously |
| **Primary Failure Modes** | **VGPR Bank Conflicts**: Shared read ports across even/odd register banks; literal constant limitations | **Register Allocation**: Requires bank-aware allocation (Mesa ACO succeeds; standard LLVM `hipcc` falls back to 1-issue) | **Instruction-Level Parallelism (ILP)**: Insufficient independent instructions stalls dual-dispatch |

---

## 3. The 7-Phase Dual-Issue Benchmark Suite

The suite evaluates hardware through 7 distinct configurations structured symmetrically across floating-point, integer, and mixed datapaths:

### Config 0: `Standard FP32`
- **Workload**: Ping-pong dependent FP32 accumulator chains (`val0` and `val1`).
- **Architectural Purpose**: Establishes the true single-issue FP32 baseline (1 instruction per cycle). By enforcing strict RAW dependency between accumulator pairs while providing sufficient vector width to saturate the single-issue ALU pipeline, it prevents compilers (such as Mesa ACO) from prematurely co-issuing dual instructions, ensuring an honest 1-issue baseline.

### Config 1: `Dual-Issue FP32 (Partial Co-Issue)`
- **Workload**: 8 independent FP32 accumulator chains (`val0..val7`).
- **Architectural Purpose**: Evaluates the transition threshold where warp schedulers (NVIDIA) and compiler pairers (AMD) have enough independent instructions to co-issue 2 operations per cycle across the pipeline window.

### Config 2: `Dual-Issue FP32 (FP32+FP32)`
- **Workload**: 16 independent FP32 accumulator chains (`val0..val15`).
- **Architectural Purpose**: Saturates the dual-issue silicon ceiling (2 FP32 instructions per cycle). Measures the peak theoretical speedup (up to 2.0x over Standard FP32) when both execution pipelines fire simultaneously.

### Config 3: `Standard INT32`
- **Workload**: 4 independent INT32 accumulator chains (`u0..u3`).
- **Architectural Purpose**: Establishes the single-issue INT32 integer baseline (1 instruction per cycle).

### Config 4: `Dual-Issue INT32 (Partial Co-Issue)`
- **Workload**: 8 independent INT32 accumulator chains (`u0..u7`).
- **Architectural Purpose**: Tests the transition threshold for integer co-issuing.

### Config 5: `Dual-Issue INT32 (INT32+INT32)`
- **Workload**: 16 independent INT32 accumulator chains (`u0..u15`).
- **Architectural Purpose**: Tests whether the GPU microarchitecture possesses dual integer ALUs. On architectures with only one integer pipeline (e.g. NVIDIA Ampere/Ada), this remains at ~1.0x over Standard INT32.

### Config 6: `Dual-Issue Mixed (FP32+INT32)`
- **Workload**: Perfectly interleaved 8 FP32 FMAs and 8 INT32 operations.
- **Architectural Purpose**: Tests concurrent execution across decoupled floating-point and integer execution units (simultaneous $1\times \text{FP32} + 1\times \text{INT32}$ per cycle).

---

## 4. Empirical Case Study: AMD Radeon 8060S (Strix Halo / GFX1151) & R9700 (GFX1201)

Running `./build/gpubench -b Dual-Issue -d 0 -k vulkan,rocm,opencl` produces cross-backend telemetry demonstrating dual-issue scaling once single-issue baseline skew and mixed-mode baselines are properly calibrated:

```text
╭─ Dual-Issue ─────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Workload                                       │ Backend  │             Throughput │ Details / Speedup                       │
├────────────────────────────────────────────────┼──────────┼────────────────────────┼─────────────────────────────────────────┤
│ Standard FP32                                  │ OpenCL   │           11.27 TFLOPS │ [Baseline]                              │
│                                                │ ROCm     │           12.10 TFLOPS │ [Baseline]                              │
│                                                │ Vulkan   │           17.34 TFLOPS │ [Baseline]                              │
│ Dual-Issue FP32 (Partial Co-Issue)             │ OpenCL   │           12.31 TFLOPS │ └──> 1.09x (+9.3%)                      │
│                                                │ ROCm     │           13.74 TFLOPS │ └──> 1.14x (+13.5%)                     │
│                                                │ Vulkan   │           24.70 TFLOPS │ └──> 1.42x (+42.5%)                     │
│ Dual-Issue FP32 (FP32+FP32)                    │ OpenCL   │           12.13 TFLOPS │ └──> 1.08x (+7.6%)                      │
│                                                │ ROCm     │           12.99 TFLOPS │ └──> 1.07x (+7.3%)                      │
│                                                │ Vulkan   │           24.63 TFLOPS │ └──> 1.42x (+42.0%)                     │
│ Standard INT32                                 │ OpenCL   │              3.32 TOPS │ [Baseline]                              │
│                                                │ ROCm     │             13.68 TOPS │ [Baseline]                              │
│                                                │ Vulkan   │              2.89 TOPS │ [Baseline]                              │
│ Dual-Issue INT32 (Partial Co-Issue)            │ OpenCL   │              3.27 TOPS │ └──> 0.99x (-1.5%)                      │
│                                                │ ROCm     │             13.77 TOPS │ └──> 1.01x (+0.7%)                      │
│                                                │ Vulkan   │              2.90 TOPS │ └──> 1.00x (+0.3%)                      │
│ Dual-Issue INT32 (INT32+INT32)                 │ OpenCL   │              3.38 TOPS │ └──> 1.02x (+1.8%)                      │
│                                                │ ROCm     │             13.43 TOPS │ └──> 0.98x (-1.8%)                      │
│                                                │ Vulkan   │              2.89 TOPS │ └──> 1.00x (-0.1%)                      │
│ Dual-Issue Mixed (FP32+INT32)                  │ OpenCL   │              5.11 TOPS │ └──> 1.00x (-0.4%)                      │
│                                                │ ROCm     │             23.14 TOPS │ └──> 1.80x (+80.2%)                     │
│                                                │ Vulkan   │              4.92 TOPS │ └──> 0.99x (-0.7%)                      │
╰────────────────────────────────────────────────┴──────────┴────────────────────────┴─────────────────────────────────────────╯
```

### Key Takeaways & Microarchitectural Insights

1. **Harmonic Baseline for Mixed Datapaths (Config 6)**:
   - Config 6 contains a 50/50 mix of independent FP32 FMAs and INT32 operations. In serialized single-issue execution, the total execution time is $T_{\text{serial}} = 2 t_{\text{FP32}} + 2 t_{\text{INT32}}$, meaning the theoretical un-co-issued baseline is the **harmonic mean** of Config 0 and Config 3:
     $$\text{Baseline}_{\text{Mixed}} = \frac{2}{\frac{1}{\text{FP32 Baseline}} + \frac{1}{\text{INT32 Baseline}}}$$
   - Comparing against this composite baseline reveals the true concurrency speedup of executing FP32 and INT32 simultaneously across independent execution units (+80.2% on ROCm, and +74.8% on Radeon AI PRO R9700).

2. **True Single-Issue Baseline Calibration**:
   - By eliminating accumulator independence in `Standard FP32` through ping-pong RAW hazards, the baseline under Vulkan drops from the skew of 24.54 TFLOPS down to **17.34 TFLOPS** (matching physical single-issue ALU capacity at boost clock).
   - `Dual-Issue FP32` now clearly demonstrates an honest **1.42x – 1.50x (+42% to +50%) dual-issue speedup** on Vulkan with Mesa RADV/ACO!

3. **Vulkan vs ROCm INT32 Throughput Disparity**:
   - In GLSL (`shaders/dual_issue_int32_4.comp`) and OpenCL (`kernels/opencl/dual_issue.cl`), vector arithmetic `(u * mu) + cu` lowers to `v_mul_lo_u32` followed by `v_add_nc_u32`. On AMD RDNA, `v_mul_lo_u32` is not single-cycle full-rate (requiring 4–6 cycles throughput), bounding performance to ~3–5 TOPS.
   - In ROCm HIP (`hip_kernels/dual_issue.hip`), `umult` is scalar, allowing `amdclang++` / LLVM to fuse the expression into a single hardware instruction: `v_mad_co_u64_u32` with an SGPR scalar operand, achieving dedicated full hardware throughput (~14–27 TOPS).

4. **ROCm (`hipcc`/LLVM) Pure FP32 vs Mixed Concurrency**:
   - In ROCm HIP (`run_dual_issue_ilp16`), 16 `float4` accumulators plus constant offsets consume >128 VGPRs per thread. LLVM's `SIFormVOPD` pass enforces strict register bank conflict rules for VOPD opcode pairing. Under this extreme register pressure, LLVM fails to find valid VOPD pairings and wave occupancy drops, causing pure FP32 dual-issue to plateau.
   - In contrast, when FP32 is interleaved with INT32 (`run_dual_issue_mixed`), instructions target different execution pipelines without VOPD bank conflicts, allowing LLVM to co-schedule dual instructions effortlessly (**+75% to +80% concurrency speedup**).

---

## 5. Live Telemetry & Profiling Hints

### AMD: Live ACO Compiler Telemetry (`RADV_DEBUG=shaderstats`)
On Linux AMD systems using the Mesa RADV driver, prefix execution with `RADV_DEBUG=shaderstats` to inspect register pressure, latency, and VOPD instruction counts during pipeline construction:
```bash
RADV_DEBUG=shaderstats ./build/gpubench -b dualissue -d 0
```
Key metrics in the output:
- `Latency`: Total estimated instruction cycle latency of the unrolled loop.
- `Inverse Throughput`: Cycles required to issue the unrolled instruction sequence.
- `VGPRs`: Vector register count allocated for the configuration ($12 \to 24 \to 36 \to 72$).
- `VOPD`: On RDNA 3 (GFX11) and RDNA 4 (GFX12), reports the exact count of 64-bit dual-issue instructions emitted.

### AMD: Static ISA Disassembly via RGA
Use the Radeon GPU Analyzer directly to inspect disassembled vector instructions:
```bash
/opt/RadeonDeveloperToolSuite-2026-05-28-1806/rga \
    -s vk-spv-offline \
    --isa /tmp/dual_issue_isa.txt \
    -c gfx1201 \
    build/kernels/vulkan/dual_issue_ilp16.comp.spv
```

### NVIDIA: Profiling Dual-Issue via Nsight Compute (`ncu`)
On NVIDIA systems, run GPUBench under Nsight Compute to measure dual-issue warp cycles and datapath activity:
```bash
ncu --metrics \
  sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
  sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,\
  smsp__issue_active.avg.pct_of_peak_sustained_active \
  ./build/gpubench -b dualissue
```
- `sm__pipe_fma_cycles_active`: Measures primary and secondary FP32 datapath utilization.
- `sm__pipe_alu_cycles_active`: Measures integer datapath activity during Configs 3–5 (INT32) and Config 6 (Mixed Concurrent FP32+INT32).
- `smsp__issue_active`: Shows the percentage of cycles where the warp scheduler successfully issued 2 instructions simultaneously.
