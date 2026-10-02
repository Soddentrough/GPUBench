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

## 3. The 6-Phase Dual-Issue Benchmark Suite

The suite evaluates hardware through 6 distinct configurations designed to map out the execution curve:

### Config 0: `Single-Issue Baseline (ILP-1)`
- **Workload**: A strictly serialized FMA chain ($y = \text{fma}(y, m, c)$).
- **Architectural Purpose**: Measures execution latency under zero instruction-level parallelism. Modern GPU ALU pipelines take 4–5 clock cycles to complete an FMA; with $\text{ILP} = 1$, each operation must wait for writeback before the next can issue, exposing the latency-bound baseline throughput.

### Config 1: `ILP-4 (Latency-Bound Single-Issue)`
- **Workload**: 4 independent accumulator chains.
- **Architectural Purpose**: Hides the 4-cycle arithmetic pipeline latency. For a single-issue ALU, 4 independent operations in flight are sufficient to achieve 100% single-issue saturation. However, 4 chains are insufficient to sustain dual-issue execution without starvation.

### Config 2: `ILP-8 (Dual-Issue Transition Threshold)`
- **Workload**: 8 independent accumulator chains.
- **Architectural Purpose**: Evaluates the inflection point where warp schedulers (NVIDIA) and compiler pairers (AMD) have enough independent instructions to co-issue 2 operations per cycle across the 4-cycle pipeline window ($2 \text{ ops/cycle} \times 4 \text{ cycles} = 8 \text{ ops}$).

### Config 3: `ILP-16 (Peak Dual-Issue Saturated FP32)`
- **Workload**: 16 independent accumulator chains (64 scalar accumulators in flight per thread).
- **Architectural Purpose**: Saturates the dual-issue silicon ceiling. At this ILP level, bank conflicts and dependency latencies are fully hidden, driving modern GPUs to their advertised peak boost FLOPS (e.g. **~49.5 TFLOPS** on the AMD Radeon AI PRO R9700).

### Config 4: `Concurrent FP32 + INT32 (50/50 Dual-Issue)`
- **Workload**: Perfectly interleaved 8 FP32 FMAs and 8 INT32 operations.
- **Architectural Purpose**: Tests the secondary datapath's flexibility:
  - On **NVIDIA Ampere/Ada/Blackwell**, Datapath 0 executes INT32 while Datapath 1 executes FP32 concurrently, demonstrating zero-overhead concurrent address/indexing computation and math.
  - On **AMD RDNA 3/4**, measures how effectively the compiler and execution units handle mixed floating-point and integer pipelines.

### Config 5: `Pure INT32 (Single Datapath Ceiling)`
- **Workload**: 16 independent INT32 accumulator chains.
- **Architectural Purpose**: Measures the integer execution ceiling:
  - On **NVIDIA**, only Datapath 0 contains integer ALUs, cutting peak issue rate by $50\%$ relative to pure FP32.
  - On **AMD RDNA 4**, measures the throughput of the native integer pipeline.

---

## 4. Empirical Case Study: AMD Radeon AI PRO R9700 (Navi 48 / GFX1201)

Running `gpubench -b dualissue -k vulkan,rocm -d 0` produces real-world telemetry illustrating the dual-issue compiler disparity:

```text
╭─ Dual-Issue ─────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Workload                                       │ Backend  │             Throughput │ Details / Speedup                       │
├────────────────────────────────────────────────┼──────────┼────────────────────────┼─────────────────────────────────────────┤
│ Single-Issue Baseline (ILP-1)                  │ ROCm     │           22.46 TFLOPS │ [Baseline]                              │
│                                                │ Vulkan   │           41.62 TFLOPS │ [Baseline]                              │
│ ILP-4 (Latency-Bound Single-Issue)             │ ROCm     │           25.60 TFLOPS │ └──> 1.14x (+14.0%)                     │
│                                                │ Vulkan   │           48.27 TFLOPS │ └──> 2.15x (+114.9%)                    │
│ ILP-8 (Dual-Issue Transition Threshold)        │ ROCm     │           26.01 TFLOPS │ └──> 1.16x (+15.8%)                     │
│                                                │ Vulkan   │           48.83 TFLOPS │ └──> 2.17x (+117.4%)                    │
│ ILP-16 (Peak Dual-Issue Saturated FP32)        │ ROCm     │           24.86 TFLOPS │ └──> 1.11x (+10.7%)                     │
│                                                │ Vulkan   │           49.26 TFLOPS │ └──> 2.19x (+119.3%)                    │
│ Concurrent FP32 + INT32 (50/50 Dual-Issue)     │ ROCm     │             48.01 TOPS │ └──> 2.14x (+113.7%)                    │
│                                                │ Vulkan   │              9.87 TOPS │ └──> 0.44x (-56.1%)                     │
│ Pure INT32 (Single Datapath Ceiling)           │ ROCm     │             27.68 TOPS │ └──> 1.23x (+23.2%)                     │
│                                                │ Vulkan   │              5.55 TOPS │ └──> 0.25x (-75.3%)                     │
╰────────────────────────────────────────────────┴──────────┴────────────────────────┴─────────────────────────────────────────╯
```

### Key Takeaways
1. **Vulkan (Mesa ACO) Dual-Issue Saturation**:
   - Vulkan scales from $41.62 \text{ TFLOPS}$ at ILP-1 to **$49.26 \text{ TFLOPS}$** at ILP-16 ($+119.3\%$ over ROCm baseline), saturating the full dual-issue capability of the 64 CUs at Boost clock.
2. **ROCm (`hipcc`/LLVM) Limitation**:
   - ROCm tops out at **$24.86\text{--}26.01 \text{ TFLOPS}$** on FP32 regardless of ILP, because standard LLVM lacks the bank-aware register allocation necessary to emit dual-issue pairs on GFX1201.
3. **ROCm Concurrent Concurrency**:
   - Under ROCm, mixing FP32 and INT32 achieves **$48.01 \text{ TOPS}$**, demonstrating that LLVM successfully co-schedules mixed arithmetic across separate execution pipes even when pure FP32 dual-issue is suppressed.

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
- `VOPD`: On RDNA 3 (GFX11), reports the exact count of 64-bit dual-issue instructions emitted.

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
- `sm__pipe_alu_cycles_active`: Measures integer datapath activity during Config 4 (Concurrent) and Config 5 (INT32).
- `smsp__issue_active`: Shows the percentage of cycles where the warp scheduler successfully issued 2 instructions simultaneously.
