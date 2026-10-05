# Dual-Issue & Datapath Concurrency Architectural Analysis

## 1. Executive Overview

Modern GPU microarchitectures have evolved beyond simple SIMD vector execution pipelines by introducing dual-issue and concurrent arithmetic execution datapaths:
- **AMD RDNA 3 (Navi 3x / GFX11)**: Introduced static **VOPD (Vector Operations Dual-issue)** 64-bit instruction pairing within each SIMD32 unit.
- **AMD RDNA 4 (Navi 4x / GFX12)**: Continues and refines 64-bit **VOPD** dual-issue execution (`v_dual_*`), paired with Dynamic VGPR allocation and improved compiler scheduling to maximize dual-FMA throughput.
- **NVIDIA Turing, Ampere, Ada Lovelace, and Blackwell**:
  - *Turing (TU10x)*: Decoupled FP32 and INT32 execution units capable of concurrent $1\times \text{FP32} + 1\times \text{INT32}$ execution per warp cycle.
  - *Ampere (GA10x) & Ada Lovelace (AD10x)*: Dual FP32 datapath design where Datapath 0 handles FP32 or INT32, and Datapath 1 handles FP32 only (yielding $2\times \text{FP32}$ dual-issue peak, or $1\times \text{FP32} + 1\times \text{INT32}$ concurrent execution).

**GPUBench** provides the **Dual-Issue Efficiency Benchmark Suite** (`--benchmark dualissue` or `-b dualissue`) to systematically evaluate instruction-level parallelism (ILP) scaling, dual-issue saturation, and mixed arithmetic concurrency across AMD and NVIDIA GPUs.

---

## 2. Microarchitectural Comparison

| Feature / Metric | AMD RDNA 3 (GFX11) | AMD RDNA 4 (GFX12) | NVIDIA Ampere / Ada Lovelace |
| :--- | :--- | :--- | :--- |
| **Dual-Issue Mechanism** | Static 64-bit **VOPD** opcode pairing (`v_dual_*`) | 64-bit **VOPD** opcode pairing (`v_dual_*`) with Dynamic VGPRs | Dynamic warp scheduler dual-dispatch (Scoreboard) |
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
- **Architectural Meaning**: Moderate Instruction-Level Parallelism (ILP). While superscalar schedulers and compiler pairers have enough independent instructions to form dual-issue pairs, the 8-chain depth is insufficient to hide instruction latency across every clock cycle. The pipeline exhibits intermittent data-dependency bubbles, alternating between 2-issue and 1-issue cycles.
- **What It Measures**: Measures realistic dual-issue scaling efficiency under moderate ILP typical of compiled game shaders and compute kernels (which rarely possess 16 completely independent operations queued back-to-back), demonstrating how much co-issue speedup hardware achieves before peak synthetic saturation.

### Config 2: `Dual-Issue FP32 (FP32+FP32)`
- **Workload**: 16 independent FP32 accumulator chains (`val0..val15`).
- **Architectural Meaning & Measurement**: Peak Instruction-Level Parallelism (16 chains). Saturates the dual-issue silicon ceiling (2 FP32 instructions per cycle), measuring maximum theoretical speedup (up to 2.0x+ over Standard FP32) when both execution pipelines fire continuously.

### Config 3: `Standard INT32`
- **Workload**: 4 independent INT32 accumulator chains (`u0..u3`).
- **Architectural Meaning & Measurement**: Establishes the single-issue INT32 integer baseline (1 instruction per cycle; single integer datapath).

### Config 4: `Dual-Issue INT32 (Partial Co-Issue)`
- **Workload**: 8 independent INT32 accumulator chains (`u0..u7`).
- **Architectural Meaning**: Moderate integer Instruction-Level Parallelism (8 chains).
- **What It Measures**: Tests whether the integer execution pipeline can achieve any dual-issue speedup under moderate instruction parallelism. On architectures with only one integer ALU per SIMD (such as RDNA 3/3.5), throughput remains identical to Standard INT32 (~1.00x), proving that integer execution is strictly datapath-limited rather than latency-bound.

### Config 5: `Dual-Issue INT32 (INT32+INT32)`
- **Workload**: 16 independent INT32 accumulator chains (`u0..u15`).
- **Architectural Purpose**: Tests whether the GPU microarchitecture possesses dual integer ALUs. On architectures with only one integer pipeline (e.g. NVIDIA Ampere/Ada), this remains at ~1.0x over Standard INT32.

### Config 6: `Dual-Issue Mixed (FP32+INT32)`
- **Workload**: Perfectly interleaved 8 FP32 FMAs and 8 INT32 operations.
- **Architectural Purpose**: Tests concurrent execution across decoupled floating-point and integer execution units (simultaneous $1\times \text{FP32} + 1\times \text{INT32}$ per cycle).

---

## 4. Empirical Case Study: AMD Radeon 8060S (Strix Halo / GFX1151) & R9700 (GFX1201)

Running `./build/gpubench -b Dual-Issue -d 0 -k vulkan,rocm,opencl` produces cross-backend telemetry demonstrating dual-issue scaling once single-issue baseline skew and compiler constant promotion are eliminated:

```text
  [Compute]
  ╭─ Dual-Issue ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
  │ Workload                                       │ Backend  │             Throughput │ Details / Speedup                       │
  ├────────────────────────────────────────────────┼──────────┼────────────────────────┼─────────────────────────────────────────┤
  │ Standard FP32                                  │ OpenCL   │           12.63 TFLOPS │ [Baseline]                              │
  │                                                │ ROCm     │           12.34 TFLOPS │ [Baseline]                              │
  │                                                │ Vulkan   │           17.40 TFLOPS │ [Baseline]                              │
  │ Dual-Issue FP32 (Partial Co-Issue)             │ OpenCL   │           22.16 TFLOPS │ └──> 1.75x (+75.5%)                     │
  │                                                │ ROCm     │           21.69 TFLOPS │ └──> 1.76x (+75.8%)                     │
  │                                                │ Vulkan   │           24.41 TFLOPS │ └──> 1.40x (+40.3%)                     │
  │ Dual-Issue FP32 (FP32+FP32)                    │ OpenCL   │           22.82 TFLOPS │ └──> 1.81x (+80.7%)                     │
  │                                                │ ROCm     │           30.37 TFLOPS │ └──> 2.46x (+146.0%)                    │
  │                                                │ Vulkan   │           25.01 TFLOPS │ └──> 1.44x (+43.8%)                     │
  │ Standard INT32                                 │ OpenCL   │              9.95 TOPS │ [Baseline]                              │
  │                                                │ ROCm     │             13.65 TOPS │ [Baseline]                              │
  │                                                │ Vulkan   │             13.63 TOPS │ [Baseline]                              │
  │ Dual-Issue INT32 (Partial Co-Issue)            │ OpenCL   │              9.95 TOPS │ └──> 1.00x (+0.1%)                      │
  │                                                │ ROCm     │             13.48 TOPS │ └──> 0.99x (-1.2%)                      │
  │                                                │ Vulkan   │             13.29 TOPS │ └──> 0.98x (-2.5%)                      │
  │ Dual-Issue INT32 (INT32+INT32)                 │ OpenCL   │              9.71 TOPS │ └──> 0.98x (-2.4%)                      │
  │                                                │ ROCm     │             13.73 TOPS │ └──> 1.01x (+0.6%)                      │
  │                                                │ Vulkan   │             13.86 TOPS │ └──> 1.02x (+1.7%)                      │
  │ Dual-Issue Mixed (FP32+INT32)                  │ OpenCL   │             17.09 TOPS │ └──> 1.72x (+71.8%)                     │
  │                                                │ ROCm     │             23.03 TOPS │ └──> 1.69x (+68.8%)                     │
  │                                                │ Vulkan   │             16.44 TOPS │ └──> 1.21x (+20.7%)                     │
  ╰────────────────────────────────────────────────┴──────────┴────────────────────────┴─────────────────────────────────────────╯
```

### Key Takeaways & Microarchitectural Insights

1. **Monotonic FP32 Dual-Issue Ladder**:
   - **True Single-Issue Baseline Calibration**: Enforcing sequential ping-pong dependency (`val0 = fma(val1, m, val0); val1 = fma(val0, m, val1)`) establishes a genuine single-issue execution rate of **12.3–17.4 TFLOPS** across all three backends.
   - **Interleaved Pair Architecture**: By coupling pairs of accumulators without compile-time constants, LLVM is prevented from:
     - Promoted execution to the Scalar ALU (`s_fmamk_f32`).
     - Loop elimination via Scalar Evolution (SCEV closed-form arithmetic).
     - Falling back to non-coissuable `v_fmaak_f32` instructions.
   - Disassembly confirms 100% hardware VOPD dual-issue instruction generation (`v_dual_fmac_f32 :: v_dual_fmac_f32`).
   - Under ROCm HIP, peak FP32 throughput reaches **30.37 TFLOPS** (**2.46x speedup** / +146%), while Vulkan sustains **25.01 TFLOPS** (+43.8%) and OpenCL reaches **22.82 TFLOPS** (+80.7%).

2. **Integer Silicon Ceiling (Single Datapath Limit)**:
   - RDNA 3 and RDNA 3.5 contain exactly one 32-bit integer ALU pipeline per SIMD32 unit.
   - Standard, Partial Co-Issue, and Full INT32 benchmarks all hit the identical silicon limit of **~13.6–13.8 TOPS** in ROCm and Vulkan (and **~9.95 TOPS** in OpenCL), showing an entirely flat **0.98x–1.02x** scaling curve.
   - This proves conclusively that integer instructions cannot dual-issue with integer instructions on this architecture.

3. **Mixed FP32 + INT32 Concurrency & Harmonic Baseline**:
   - Config 6 contains a 50/50 mix of independent FP32 FMAs and INT32 operations. In serialized single-issue execution, the total execution time is $T_{\text{serial}} = 2 t_{\text{FP32}} + 2 t_{\text{INT32}}$, meaning the theoretical un-co-issued baseline is the **harmonic mean** of Config 0 and Config 3:
     $$\text{Baseline}_{\text{Mixed}} = \frac{2}{\frac{1}{\text{FP32 Baseline}} + \frac{1}{\text{INT32 Baseline}}}$$
   - Because RDNA 3/3.5 provides independent floating-point and integer execution pipelines, interleaving FP32 FMAs with INT32 operations allows both datapaths to fire simultaneously.
   - Dual-Issue Mixed reaches **23.03 TOPS** under ROCm (**+68.8% speedup**), **17.09 TOPS** under OpenCL (**+71.8%**), and **16.44 TOPS** under Vulkan (**+20.7%**).

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
