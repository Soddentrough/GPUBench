# Ray Scheduling Architectures & Hardware Profiling

## 1. Overview & Motivation

Real-time ray tracing in modern graphics pipelines encounters two fundamental hardware bottlenecks that traditional compute shading pipelines were not designed for:

1. **Spatial Ray Divergence**: Neighboring pixels in screen space cast rays that diverge widely after the first specular or diffuse bounce. As ray directions scatter, memory accesses into bounding volume hierarchies (BVH) lose cache locality, resulting in high L1/L2 texture cache miss rates.
2. **Material Shading Divergence**: Real-world production scenes feature heterogeneous material archetypes with vastly different computational weights and register footprints (e.g. dispersive Cauchy glass, subsurface scattering, multi-lobe car paint, and procedural rust).

In a **Traditional Megakernel** approach, ray traversal, intersection, material classification, shading evaluation, and ray bouncing are packed into a single monolithic compute shader. This design suffers from the **convoy effect**:
- **Register Spilling & Reduced Occupancy**: The compiler must allocate vector registers (VGPRs) for the union of all material branches, drastically cutting active wavefront occupancy on SIMD execution units.
- **SIMD Lane Starvation**: When threads in a 32-lane wave diverge across different material branches, non-matching lanes are masked off and idle while matching lanes execute, serializing ALU execution.
- **Ray Death Inefficiencies**: In multi-bounce path tracing, terminated rays (via Russian roulette or sky misses) must remain allocated in the wave until all other rays in the wave terminate.

GPUBench implements and contrasts modern ray scheduling paradigms to decouple these stages:
- **Traditional Megakernel**: Monolithic compute dispatch (`vkCmdDispatch`).
- **Work Lists / Device-Generated Commands (DGC)**: Decoupled micro-kernels where hit records are classified into uniform queues and executed via GPU-driven indirect dispatches (`vkCmdDispatchIndirect`).
- **Active-Ray Compaction**: Atomic queue compaction repacking surviving path-tracing rays into dense wavefronts after every bounce.
- **Shader Execution Reordering (SER)**: Hardware ray reordering (`VK_KHR_ray_tracing_reorder`).

---

## 2. Dual-Scenario Morphology

To evaluate how geometric occlusion and material distribution dictate scheduling efficiency, GPUBench provides two distinct procedural scenarios:

### 2.1 Complex Indoor Atrium / Cathedral (`-s indoor`)
- **Geometry**: $35,272$ triangles representing an enclosed architectural showroom with central pedestals, a Suzanne centerpiece, flanking fluted columns, and arched cathedral vaulting.
- **Optical Confinement**: $0\%$ sky escape. Every secondary ray hits walls, pillars, or centerpiece sculptures.
- **Materials**: 8 heterogeneous production-grade BSDF archetypes:
  1. *Clearcoat Car Paint*: Dual GGX specular lobes + Voronoi procedural flake glints.
  2. *Jade / Marble SSS*: Multi-channel exponential subsurface scattering profiles.
  3. *Cauchy Glass*: Snell's law refraction, total internal reflection (TIR), and chromatic dispersion.
  4. *Anisotropic Velvet Sheen*: Charlie micro-fiber grazing sheen with dual-axis roughness.
  5. *Weathered Bronze Rust*: Multi-octave Fractal Brownian Motion (FBM) noise.
  6. *Terrazzo Stone Floor*: Composite mineral aggregate matrix.
  7. *Polished Gold*: Complex Fresnel conductor optics.
  8. *Chrome Mirror*: Ideal Dirac delta specular reflection.
- **Path Depth**: 4 diffuse/specular bounces with Russian roulette termination ($40\%$ absorption probability per bounce $\ge 2$).

### 2.2 Open-World Outdoor Landscape (`-s outdoor`)
- **Geometry**: $57,216$ triangles spanning an expansive $>2,000\text{m}$ terrain, procedural mountain heightfields, river water plane, and instanced conifer pine foliage clusters.
- **Optical Confinement**: High sky escape ($>60\%$ of secondary rays escape into the sky dome).
- **Lighting & Atmosphere**: Directional solar lighting, hard sun shadows, and analytic Rayleigh-Mie aerial perspective haze.
- **Materials**: Terrain rock, soil, grass, water, and conifer foliage with dominant Cook-Torrance diffuse and GGX specular evaluations.

---

## 3. Microarchitectural Analysis & Profiling Results

Profiling benchmarks were performed on **AMD Radeon AI PRO R9700 (GFX1201 / RDNA 4)** using Vulkan 1.4 at $1920 \times 1080$ ($2,073,600$ primary rays).

### 3.1 Performance Comparison Table

| Pipeline Stage / Workload | Indoor Atrium: Traditional | Indoor Atrium: Work Lists | Speedup | Outdoor Landscape: Traditional | Outdoor Landscape: Work Lists | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Material Shading** | 3,422.78 MHits/s | **16,414.84 MHits/s** | **4.80x** | 29,153.69 MHits/s | **34,586.99 MHits/s** | **1.19x** |
| **Path Tracing (4 Bounces)** | 612.82 MRays/s | **1,224.96 MRays/s** | **2.00x** | 1,191.50 MRays/s | **1,860.34 MRays/s** | **1.56x** |
| **Incoherent Secondary Rays** | 1,318.47 MRays/s | **2,523.65 MRays/s** | **1.91x** | 1,984.89 MRays/s | **3,707.89 MRays/s** | **1.87x** |
| **Primary RT (Linear Baseline)** | 2,354.17 MRays/s | **3,621.79 MRays/s** | **1.54x** | 4,347.57 MRays/s | **6,603.76 MRays/s** | **1.52x** |
| **Primary RT (Morton 8x4)** | 2,519.78 MRays/s | **3,695.92 MRays/s** | **1.47x** | 4,628.61 MRays/s | **6,827.38 MRays/s** | **1.48x** |
| **Raw BVH Traversal (Linear)** | 3,825.39 MRays/s | 3,825.39 MRays/s | 1.00x | 4,759.34 MRays/s | 4,759.34 MRays/s | 1.00x |
| **Raw BVH Traversal (Tiled 8x4)** | 4,092.16 MRays/s | 4,092.16 MRays/s | 1.07x | 5,117.65 MRays/s | 5,117.65 MRays/s | 1.08x |
| **Raw BVH Traversal (Morton 8x4)**| 4,068.49 MRays/s | 4,068.49 MRays/s | 1.06x | 5,073.04 MRays/s | 5,073.04 MRays/s | 1.07x |
| **Queue Compaction Overhead** | - | 27,885.45 MRec/s | ($0.07\text{ ms}$) | - | 27,960.41 MRec/s | ($0.07\text{ ms}$) |

### 3.2 Low-Level ISA Disassembly (RGA GFX1201 Compiler)

Compiling the compute pipelines using the Radeon GPU Analyzer (`rga`) for `gfx1201` targets illustrates the root cause of the performance delta:

| Shader Kernel | VGPRs | SGPRs | LDS Allocation | Total Instructions | VALU | SALU | Branches (`s_cbranch`) | Wait / Stalls (`s_wait`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`rt_scheduling_traditional.comp`** (Monolithic) | **68** | **81** | 2,560 B | **8,025** | 4,288 | 1,439 | **333** | **1,869** |
| **`rt_scheduling_worklist_classify.comp`** | **46** (-32%) | **53** | 4,608 B | **1,526** (-81%) | 724 | 296 | **81** | **381** |
| **`rt_scheduling_worklist_material.comp`** | **67** | **52** | 4,096 B | **2,652** (-67%) | 1,533 | 376 | **113** | **609** |
| **`rt_scheduling_worklist_bounce.comp`** | **35** (-49%) | **26** | 4,608 B | **434** (-95%) | 196 | 105 | **23** | **93** |

- **VGPR Footprint**: Monolithic megakernels require 68 VGPRs, limiting wave occupancy on RDNA 4 SIMDs to $\le 4$ active waves per SIMD. The decoupled bounce kernel drops VGPR consumption to 35, allowing up to 8 active waves per SIMD.
- **Instruction Bloat**: The megakernel compiles to 8,025 instructions with 333 branch tests. Specialized material dispatches eliminate $67\%$ of total instructions and $66\%$ of divergent branch evaluations.
- **Pipeline Stalls**: In the megakernel, 1,869 wait cycles are spent synchronizing divergent lane masks. Work lists reduce wait stalls to 609 ($-67\%$).

### 3.3 Radeon GPU Profiler (RGP) Thread Trace Findings

Thread trace captures (`MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=1`) reveal:
- **Trace Footprint**: The megakernel generates **$20.6\text{ MB}$** of trace tokens per submit in material shading, compared to only **$5.6\text{ MB}$** for Work Lists ($3.7\times$ reduction).
- **Execution Timeline**: The megakernel dispatch requires **$351.2\ \mu\text{s}$** across all CUs. Work Lists complete all 8 material dispatches in **$73.4\ \mu\text{s}$** total ($4.78\times$ faster), reducing GPU execution latency by **$79.1\%$**.

---

## 4. Analytical Parity & Visual Quality Verification

To ensure that performance optimizations do not compromise image fidelity or introduce numerical divergence:

1. **Bit-Exact Parity**:
   - Primary ray outputs between Traditional Megakernel and Work Lists / DGC achieve **$120.00\text{ dB}$ PSNR**, $0.000000$ RMSE, and **0 discrepant pixels** across all $2,073,600$ pixels.
2. **Ground-Truth Comparison**:
   - Renders are cross-compared against an offline Blender Cycles reference render with full multi-bounce path tracing and multiple importance sampling (MIS).

---

## 5. Profiling & Benchmarking Workflow

### 5.1 Command-Line Usage

```bash
# Benchmark Indoor Atrium (Default)
gpubench -d 1 -b RayScheduling

# Benchmark Outdoor Landscape
gpubench -d 1 -b RayScheduling -s outdoor

# Benchmark Both Scenes Sequentially
gpubench -d 1 -b RayScheduling -s all

# Dump 1080p PPM/PNG frames and parity diff heatmaps to renders/
gpubench -d 1 -b RayScheduling -s indoor --dump-frames

# Run an isolated sub-workload (e.g. Config 4: Path Tracing Megakernel)
gpubench -d 1 -b RayScheduling -s indoor -c 4

# Run single-submit profiling snapshot mode for profiler attachment
gpubench -d 1 -b RayScheduling -s indoor -c 2 --profile-snapshot
```

### 5.2 Automated Profiling Script

GPUBench includes a turnkey Python orchestration script that executes all 16 scene and paradigm combinations, captures binary RGP thread traces via Mesa RADV, queries live `amd-smi` hardware telemetry, and parses GFX1201 ISA metrics:

```bash
python3 scripts/capture_gpu_profiles.py
```

Profile outputs:
- Binary RGP traces: `profiles/*.rgp`
- Consolidated hardware metrics: `profiles/gpu_profiling_snapshots.json`

To inspect the captured thread traces in AMD's desktop GUI:
```bash
/opt/RadeonDeveloperToolSuite-2026-05-28-1806/RadeonGPUProfiler profiles/indoor_material_traditional.rgp &
/opt/RadeonDeveloperToolSuite-2026-05-28-1806/RadeonGPUProfiler profiles/indoor_material_worklist.rgp &
```
