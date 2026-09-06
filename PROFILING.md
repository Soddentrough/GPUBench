# Performance Profiling & Hardware Pathway Verification Guide

**Target Architecture**: AMD RDNA 4 (GFX1201 / AMD Radeon AI PRO R9700 / Navi 4x)  
**Target Driver**: Mesa RADV 26.1.8 / Vulkan 1.4 / SPIR-V 1.4  
**Host Platform**: AMD Threadripper 3750X (64GB RAM), Fedora 44  
**Primary Target Device**: GPU 1 (`-d 1`). *(Note: GPU 0 is strictly reserved for external host display and background tasks)*.

---

## 1. Overview & Objectives

This document serves as the standard operational guide for profiling, verifying, and validating compute, ray tracing, and path tracing pipelines in GPUBench.

Specifically, it details how to:
1. **Verify Native Execution Pathways**: Prove that GPU workloads labeled as **Device-Generated Commands (Work Lists / DGC)** execute autonomously on the GPU Command Processor (CP) via `vkCmdExecuteGeneratedCommandsEXT`, rather than falling back to host-recorded CPU dispatches.
2. **Inspect Hardware Command Processor Packets**: Trace indirect buffers, DMA prefetch, pipeline register updates, and hardware predication using `RADV_DEBUG=dumpibs`.
3. **Analyze Shader ISA & Register Pressure**: Disassemble SPIR-V compute kernels targeting GFX1201 to evaluate VGPR/SGPR allocations, instruction sizes, scratch spills, and theoretical SIMD wave occupancy using **Radeon GPU Analyzer (RGA)**.
4. **Capture Live Compiler Telemetry**: Inspect ACO (AMD Compiler) live stats, including subgroup occupancy and dual-issue VOPD vector instructions, using `RADV_DEBUG=shaderstats`.
5. **Tune Wavefront Scheduling**: Evaluate Wave32 vs Wave64 trade-offs using `RADV_PERFTEST`.
6. **Ensure Analytical & Visual Parity**: Verify bit-exact parity (0 discrepant pixels, PSNR = 120 dB) against ground-truth megakernels.

---

## 2. Pre-Flight System Checks & SMI Telemetry

Before launching GPU-intensive benchmarks or profiling sessions, verify GPU 1 utilization, memory availability, and temperatures to prevent thermal throttling or VRAM exhaustion.

### 2.1 Querying GPU 1 with `amd-smi`
Always use the exact absolute paths:
- `/home/naoki/.local/bin/amd-smi`
- `/opt/rocm/core-10.0/bin/amd-smi`

```bash
# Query GPU 1 utilization, power, and VRAM capacity
/home/naoki/.local/bin/amd-smi metric --gpu 1
```

**Key Parameters to Verify:**
- `FREE_VRAM`: Ensure sufficient capacity (GFX1201 has 32,624 MB total).
- `GFX_ACTIVITY`: Ensure baseline is near 0% before starting.
- `HOTSPOT` / `EDGE` Temperature: Ensure < 60°C before running intensive path tracing.
- `THROTTLE_STATUS`: Must report `UNTHROTTLED`.

---

## 3. Profiling Toolchain & Environment Variables

### 3.1 Tool Suite Installation
The AMD Radeon Developer Tool Suite is located at:
```bash
/opt/RadeonDeveloperToolSuite-2026-05-28-1806/
```
Key binaries available:
- `rga`: Radeon GPU Analyzer (offline/online ISA disassembly and resource analysis).
- `RadeonGPUProfiler`: RGP GUI trace visualizer.
- `RadeonRaytracingAnalyzer`: RRA GUI ray tracing BVH visualizer.
- `RadeonDeveloperServiceCLI`: Headless daemon for developer connections.

> [!NOTE]
> Binaries located outside the repository workspace require execution outside the standard sandbox (`BypassSandbox: true`).

### 3.2 Mesa RADV Driver Variables

| Variable | Values / Flags | Description |
|---|---|---|
| `RADV_DEBUG` | `shaderstats` | Prints compiler pipeline statistics (VGPRs, SGPRs, LDS, Code Size, Waves/SIMD, VOPD). |
| `RADV_DEBUG` | `dumpibs` | Dumps hardware command processor instruction packets (`INDIRECT_BUFFER`, `DISPATCH`, `SET_PREDICATION`). |
| `RADV_DEBUG` | `cs` | Logs compute shader pipeline submissions. |
| `RADV_PERFTEST` | `cswave32` | Forces compute shaders to execute in Wave32 mode. |
| `RADV_PERFTEST` | `rtwave64` | Forces ray tracing shaders to execute in Wave64 mode. |
| `MESA_VK_TRACE` | `rgp` or `rra` | Generates RGP execution trace or RRA ray tracing trace. |
| `MESA_VK_TRACE_PER_SUBMIT` | `1` | Forces trace capture on queue submit (essential for headless compute). |

### 3.3 GPUBench Native Profiling Flags

| CLI Flag | Purpose |
|---|---|
| `-d 1` | Targets GPU 1 (AMD Radeon AI PRO R9700). |
| `-c <index>` | Runs an isolated benchmark configuration (0 to 29). |
| `-s <scene>` | Selects scene: `showroom`, `indoor` (Sponza GLTF), `outdoor`, `forest`. |
| `--profile-snapshot` | Sets 1 warmup pass and 1 timed submit for clean profiler traces. |
| `--no-dump` | Bypasses 4K frame dumping and image verification for fast throughput runs. |
| `--dump-renders` | Executes full analytical and visual verification (generates diff heatmaps & PSNR). |
| `--rra` | Enables RRA trace capture mode. |

---

## 4. Layer 1: Offline Shader ISA & Register Analysis (RGA)

The **Radeon GPU Analyzer (RGA)** compiles SPIR-V shaders directly for target AMD ASICs and generates detailed assembly disassemblies and hardware resource consumption CSV files.

### 4.1 Running RGA on Compute SPIR-V
Target ASIC for AMD Radeon AI PRO R9700 is **`gfx1201`**:

```bash
/opt/RadeonDeveloperToolSuite-2026-05-28-1806/rga \
    -s vk-spv-offline \
    -c gfx1201 \
    --isa scratch/rga/resolve_isa.txt \
    -a scratch/rga/resolve_stats.csv \
    --comp build/kernels/vulkan/rt_scheduling_resolve.comp.spv
```

### 4.2 GFX1201 Hardware Resource Profile

Compilation statistics for all `rt_scheduling` compute kernels targeting GFX1201:

| Shader Kernel | VGPRs | SGPRs | LDS (Bytes) | Binary Size | VGPR Spills | SGPR Spills | SIMD Wave Occupancy |
|---|---|---|---|---|---|---|---|
| `rt_scheduling_reset.comp` | **4** | 9 | 0 | 168 B | 0 | 0 | **16 waves / SIMD (Peak)** |
| `rt_scheduling_resolve.comp` (DGC Resolve) | **19** | 34 | 0 | 2,068 B | 0 | 0 | **16 waves / SIMD (Peak)** |
| `rt_scheduling_worklist_shadow.comp` | **34** | 43 | 2,048 | 3,176 B | 0 | 0 | **16 waves / SIMD (Peak)** |
| `rt_scheduling_worklist_bounce.comp` | **49** | 56 | 4,096 | 5,064 B | 0 | 0 | **16 waves / SIMD (Peak)** |
| `rt_scheduling_worklist_classify.comp` | **85** | 106 | 4,096 | 37,396 B | 0 | 0 | **5 waves / SIMD** |
| `rt_scheduling_worklist_material.comp` | **92** | 92 | 2,048 | 209,004 B | 0 | 0 | **5 waves / SIMD** |
| **Traditional Megakernel** (`rt_scheduling_traditional.comp`) | **97** | **106 (Max)** | 4,096 | **266,364 B** | 0 | 0 | **2–3 waves / SIMD** |

### 4.3 Architectural Insights from Register Analysis
1. **The Megakernel Occupancy Cliff**:
   - The monolithic traditional megakernel requires **97 VGPRs and all 106 available SGPRs**.
   - Because each Compute Unit has a fixed physical register file, this forces the hardware scheduler to throttle down to only **2–3 active waves per SIMD**.
   - When divergent rays cause high-latency BVH traversal stalls (cache misses), the SIMD unit sits idle because there are not enough in-flight waves to hide memory latency.
2. **Micro-kernel Latency Hiding**:
   - The Work List / DGC pipeline splits execution into specialized micro-kernels.
   - `resolve` uses only **19 VGPRs**, `shadow` uses **34 VGPRs**, and `bounce` uses **49 VGPRs**.
   - This unlocks the **maximum hardware occupancy of 16 waves per SIMD**, ensuring continuous arithmetic execution while other waves wait for memory fetches.
3. **L1 Instruction Cache (L1I) Footprint**:
   - The monolithic megakernel is **266.3 KB**, far exceeding the L1I cache size and causing constant instruction cache misses.
   - The DGC bounce micro-kernel is **5.0 KB** and the shadow kernel is **3.1 KB**, allowing both to fit entirely inside high-speed L1I cache.

---

## 5. Layer 2: Live Driver Telemetry & Dual-Issue VOPD (`RADV_DEBUG=shaderstats`)

Executing with `RADV_DEBUG=shaderstats` provides telemetry directly from the Mesa ACO compiler during pipeline construction.

### 5.1 Running with Shader Stats
```bash
RADV_DEBUG=shaderstats ./build/gpubench \
    --benchmark rayscheduling \
    --scene indoor \
    -c 2 \
    -d 1 \
    --no-dump \
    --profile-snapshot
```

### 5.2 Key ACO Telemetry Output
```text
Compute Shader:
*** SHADER STATS ***
Driver pipeline hash: 16048176347099467867
SGPRs: 108
VGPRs: 96
Spilled SGPRs: 0
Spilled VGPRs: 0
Code size: 1736
LDS size: 0
Scratch size: 0
Subgroups per SIMD: 16
VALU: 54
SALU: 84
VMEM: 42
SMEM: 30
VOPD: 6
Pre-Sched SGPRs: 13
Pre-Sched VGPRs: 11
********************
```

### 5.3 Interpreting Live Metrics
- **Zero Spills (`Scratch size: 0`)**: Confirms all state fits entirely within registers without spilling to slow VRAM scratch memory.
- **RDNA 4 Dual-Issue VOPD (Vector Dual-issue Operations)**:
  - GFX1201 features dual-issue SIMD units capable of executing two vector ALU instructions in parallel per clock.
  - ACO automatically identifies dual-issue pairs:
    - Material Shading Kernels: up to **6,298 VOPD instructions** dual-issued.
    - Classify Kernel: **361 VOPD instructions**.
    - Bounce Traversal Kernel: **89 VOPD instructions**.
    - Shadow Traversal Kernel: **41 VOPD instructions**.
- **Pre-Scheduling Pressure**: In `rt_scheduling_resolve.comp`, pre-scheduling pressure is only **11 VGPRs and 13 SGPRs**, verifying zero register allocation bottlenecks in the DGC resolve stage.

---

## 6. Layer 3: Hardware Command Processor Packet Inspection (`RADV_DEBUG=dumpibs`)

To confirm that `vkCmdExecuteGeneratedCommandsEXT` executes natively on the GPU Command Processor without CPU intervention, inspect the command buffer packets.

### 6.1 Running Hardware Packet Capture
```bash
RADV_DEBUG=dumpibs ./build/gpubench \
    --benchmark rayscheduling \
    --scene indoor \
    -c 2 \
    -d 1 \
    --no-dump \
    --profile-snapshot > scratch/dumpibs_config2.log 2>&1
```

### 6.2 Key Hardware Packet Types
During execution, look for these packets emitted by the GPU Command Processor:
1. **`c0023f00 INDIRECT_BUFFER`**:
   - The GPU CP fetches and executes an indirect command buffer generated in GPU memory.
   ```text
   c0023f00 INDIRECT_BUFFER:
   00015000         IB_BASE_LO <- 0x00015000
   ffff8001         IB_BASE_HI <- 0xffff8001
   00000150         IB_CONTROL <- 336 (0x00000150)
   ```
2. **`c0022000 SET_PREDICATION`**:
   - Sets hardware condition evaluation. The CP evaluates whether to execute or skip subsequent dispatches based on GPU memory flags:
   ```text
   c0022000 SET_PREDICATION:
   00000000         PRED_BOOL <- DRAW_IF_NOT_VISIBLE_OR_OVERFLOW
   ```
3. **`c0031503 DISPATCH_DIRECT(predicated)`**:
   - Dispatches compute waves only if the predicate condition holds (i.e., non-empty queue count). If empty, the CP skips the dispatch in 0 cycles:
   ```text
   c0031503 DISPATCH_DIRECT(shader_type=compute)(predicated):
   00002045         COMPUTE_DISPATCH_INITIATOR <- 8261 (0x00002045)
   ```
4. **`c0027600 SET_SH_REG` / `c021ba04 SET_SH_REG_PAIRS`**:
   - Direct hardware shader register writes for pipeline switches and push constant updates.

### 6.3 Command Processor Packet Verification Across All 8 DGC Tests

| Config Index | Benchmark Test Name | `INDIRECT_BUFFER` Packets | Predicated Dispatches | Register Sets (`SET_SH_REG`) | Memory Barriers (`ACQUIRE`/`REL`) |
|---|---|---|---|---|---|
| **Config 2** | Material Shading - Work Lists (DGC) | 4 | 2 | 10 | 6 / 2 |
| **Config 6** | Full Scene Path Tracing (1 SPP) - Work Lists (DGC) | 6 | 4 | 24 | 18 / 2 |
| **Config 10** | Incoherent Ray Tracing - Work Lists (DGC) | 4 | 2 | 16 | 10 / 2 |
| **Config 14** | Total Scene Render - Work Lists (DGC) | 4 | 2 | 18 | 12 / 2 |
| **Config 22** | Full Scene Ray Tracing (PBR - Morton Z) - Work Lists (DGC) | 4 | 2 | 18 | 12 / 2 |
| **Config 25** | Directional Shadows - Work Lists (Wavefront Compaction) | 4 | 2 | 18 | 12 / 2 |
| **Config 27** | Directional Shadows - Multi-Light Directional Binning (DGC) | 4 | 2 | 18 | 12 / 2 |
| **Config 29** | Full Scene Path Tracing (16 SPP) - Work Lists (DGC) | 96 | 64 | 384 | 288 / 32 |

---

## 7. Layer 4: Wavefront Size Optimization (`RADV_PERFTEST`)

RDNA architectures support both Wave32 and Wave64 execution modes.

### 7.1 Comparing Wavefront Modes
To test the impact of wavefront size on path tracing throughput:
```bash
# Default (Native Wave32)
./build/gpubench --benchmark rayscheduling --scene indoor -c 6 -d 1 --no-dump --profile-snapshot

# Enforce Wave32
RADV_PERFTEST=cswave32 ./build/gpubench --benchmark rayscheduling --scene indoor -c 6 -d 1 --no-dump --profile-snapshot

# Enforce Wave64
RADV_PERFTEST=rtwave64 ./build/gpubench --benchmark rayscheduling --scene indoor -c 6 -d 1 --no-dump --profile-snapshot
```

### 7.2 Measured Performance Impact on GFX1201

| Mode | Throughput | Frame Rate | Relative Speed |
|---|---|---|---|
| **Default (Wave32)** | **1,369.50 MRays/s** | **165.1 FPS** | **100.0% (Optimal)** |
| **`RADV_PERFTEST=cswave32`** | 1,345.32 MRays/s | 162.2 FPS | 98.2% |
| **`RADV_PERFTEST=rtwave64`** | 1,299.59 MRays/s | 156.7 FPS | 94.9% (-5.1%) |

**Why Wave32 is Superior on RDNA 4 for Ray Tracing:**
- Wave64 groups 64 threads together. Under divergent ray conditions, branch divergence within the 64-thread wavefront is significantly worse than in a 32-thread wavefront.
- Wave64 doubles the register requirement per wave, cutting maximum wave occupancy in half and impairing the GPU's ability to hide BVH traversal latency.

---

## 8. Layer 5: End-to-End Performance Benchmarking & Parity Testing

### 8.1 Performance Results (GPU 1: AMD Radeon AI PRO R9700)
Tested across all 8 configurations against baseline monolithic megakernels at 4K resolution (3840x2160, 8.29M primary rays) in the Sponza Atrium scene:

| Benchmark Test | Megakernel Config | DGC Config | Megakernel Throughput | DGC Throughput | Measured Speedup | Parity Match |
|---|---|---|---|---|---|---|
| **Material Shading** (Pure Shading Micro-kernel) | Config 0 | Config 2 | 232.50 MHits/s | **2,405.22 MHits/s** | **10.35x** | 100.00% (PSNR 120 dB) |
| **Full Scene Path Tracing** (1 SPP, 2 Bounces) | Config 4 | Config 6 | 116.81 MRays/s | **1,348.01 MRays/s** | **11.54x** | 100.00% (PSNR 120 dB) |
| **Incoherent Ray Tracing** (Directional Binning) | Config 8 | Config 10 | 203.53 MRays/s | **1,807.70 MRays/s** | **8.88x** | 100.00% (PSNR 120 dB) |
| **Total Scene Render** (Material Sorting) | Config 12 | Config 14 | 101.90 MRays/s | **306.51 MRays/s** | **3.01x** | 100.00% (PSNR 120 dB) |
| **Full Scene Ray Tracing** (PBR - Morton Z-Curve) | Config 21 | Config 22 | 113.47 MRays/s | **308.89 MRays/s** | **2.72x** | 100.00% (PSNR 120 dB) |
| **Directional Shadows** (Wavefront Compaction) | Config 23 | Config 25 | 455.28 MRays/s | **2,257.62 MRays/s** | **4.96x** | 100.00% (PSNR 120 dB) |
| **Directional Shadows** (Multi-Light Binning) | Config 23 | Config 27 | 310.26 MRays/s | **2,331.30 MRays/s** | **7.51x** | 100.00% (PSNR 120 dB) |
| **Full Scene Path Tracing** (16 SPP Multi-Bounce) | Config 28 | Config 29 | 13.41 MRays/s | **101.24 MRays/s** | **7.55x** | 100.00% (PSNR 120 dB) |

### 8.2 Visual & Analytical Parity Verification
When `--dump-renders` is enabled, GPUBench compares the rendered 4K framebuffer outputs pixel-by-pixel between the Megakernel and Work Lists / DGC:
- **Max Color Delta**: 0.000000 (0 / 255)
- **Mean Absolute Error (MAE)**: 0.000000
- **Root Mean Squared Error (RMSE)**: 0.000000
- **Peak Signal-to-Noise Ratio (PSNR)**: 120.00 dB (numerical bit-exact limit)
- **Bit-Exact Pixels**: 8,294,400 / 8,294,400 (100.00%)
- **Discrepant Pixels (>1 LSB)**: 0 / 8,294,400 (PARITY VERIFIED)

---

## 9. Profiling Recipes & Quick-Start Playbook

### Recipe 1: Fast Throughput Check of a Single Configuration
To run a fast 2-second benchmark without image disk writes:
```bash
./build/gpubench \
    --benchmark rayscheduling \
    --scene indoor \
    -c 6 \
    -d 1 \
    --no-dump \
    --profile-snapshot
```

### Recipe 2: Run Offline RGA Analysis on All Shaders
```bash
OUTDIR="scratch/rga_analysis"
mkdir -p "$OUTDIR"

for shader in rt_scheduling_resolve rt_scheduling_worklist_bounce rt_scheduling_worklist_shadow; do
    /opt/RadeonDeveloperToolSuite-2026-05-28-1806/rga \
        -s vk-spv-offline \
        -c gfx1201 \
        --isa "${OUTDIR}/${shader}_isa.txt" \
        -a "${OUTDIR}/${shader}_stats.csv" \
        --comp "build/kernels/vulkan/${shader}.comp.spv"
done
```

### Recipe 3: Verify Hardware Indirect Packets for Any Config
```bash
RADV_DEBUG=dumpibs ./build/gpubench \
    --benchmark rayscheduling \
    --scene indoor \
    -c <CONFIG_ID> \
    -d 1 \
    --no-dump \
    --profile-snapshot 2>&1 | grep -E "INDIRECT_BUFFER|DISPATCH_DIRECT|SET_PREDICATION"
```

### Recipe 4: Python Script to Benchmark All 8 DGC vs Baseline Pairs
```python
import subprocess, re

configs = [(2, 0), (6, 4), (10, 8), (14, 12), (22, 21), (25, 23), (27, 23), (29, 28)]

for dgc, base in configs:
    for cfg in [base, dgc]:
        cmd = ["./build/gpubench", "--benchmark", "rayscheduling", "--scene", "indoor", "-c", str(cfg), "-d", "1", "--no-dump", "--profile-snapshot"]
        out = subprocess.run(cmd, stdout=subprocess.PIPE, text=True).stdout
        for line in out.splitlines():
            clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
            if "RayScheduling (Indoor Atrium)" in clean and "Vulkan |" in clean:
                print(f"Config {cfg:2d}: {clean.split('Vulkan |')[1].strip()}")
```

---

## 10. Rules of Thumb & Best Practices

1. **Always Target GPU 1 (`-d 1`)**: Never run benchmarks on GPU 0 on this system.
2. **Watch for ANSI Codes**: When parsing benchmark outputs in automation scripts, always strip ANSI escape sequences (`re.sub(r'\x1b\[[0-9;]*m', '', text)`).
3. **Use `--no-dump` for Pure Profiling**: Full image dumping and difference generation adds ~15–30 seconds per run; use `--no-dump` when doing rapid iterative profiling or hardware packet dumping.
4. **Inspect SGPR Budget First**: On RDNA, the SGPR budget is 106 registers. If a monolithic kernel approaches 106 SGPRs, wave occupancy collapses. Splitting into DGC micro-kernels is the primary architectural solution.
5. **Verify Zero Spills**: Check that `Scratch size: 0` in ACO shader stats. Any spilling to scratch memory severely degrades ray tracing throughput.
