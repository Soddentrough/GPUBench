# Ray Scheduling Architectures, Hardware Occupancy & Profiling

## 1. Overview & Motivation

Real-time ray tracing in modern graphics pipelines encounters two fundamental hardware bottlenecks that traditional compute shading pipelines were not designed for:

1. **Spatial Ray Divergence**: Neighboring pixels in screen space cast rays that diverge widely after the first specular or diffuse bounce. As ray directions scatter, memory accesses into bounding volume hierarchies (BVH) lose cache locality, resulting in high L1/L2 texture cache miss rates.
2. **Material Shading Divergence**: Real-world production scenes feature heterogeneous material archetypes with vastly different computational weights and register footprints (e.g. clearcoat car paint, thin-surface subsurface scattering, refractive water, anisotropic velvet, and procedural bark).

In a **Traditional Megakernel** approach, ray traversal, intersection, material classification, shading evaluation, and ray bouncing are packed into a single monolithic compute shader. This design suffers from the **convoy effect**:
- **Register Spilling & Reduced Occupancy**: The compiler must allocate vector registers (VGPRs) for the union of all material branches and ray operations, drastically reducing active wavefront occupancy on SIMD execution units.
- **SIMD Lane Starvation**: When threads in a 32-lane wave diverge across different material branches, non-matching lanes are masked off and idle while matching lanes execute, serializing ALU execution.
- **Ray Death Inefficiencies**: In multi-bounce path tracing, terminated rays (via Russian roulette or sky misses) must remain allocated in the wave until all other rays in the wave terminate.

GPUBench implements and contrasts modern ray scheduling paradigms to decouple these stages:
- **Traditional Megakernel**: Monolithic compute dispatch (`vkCmdDispatch`).
- **Device-Generated Commands (DGC)**: Decoupled micro-kernels where hit records are classified into uniform queues and executed via GPU-driven command generation (`VK_EXT_device_generated_commands` / `vkCmdExecuteGeneratedCommandsEXT`).
- **Active-Ray Compaction**: Atomic queue compaction repacking surviving path-tracing rays into dense wavefronts after every bounce using wave ballot stream sort.
- **Shader Execution Reordering (SER)**: In-pipeline thread reordering (`VK_EXT_ray_tracing_invocation_reorder`, supported in hardware on NVIDIA Ada Lovelace / Blackwell and Intel Arc Battlemage Xe2 / Xe3; unsupported in hardware on AMD RDNA).

> [!NOTE]
> **Vulkan DGC vs. DirectX 12 Work Lists**:
> In DirectX 12, "Work Lists" refers to an evaluation-stage model extending `ExecuteIndirect`. In Vulkan, GPU-driven work creation and indirect command execution are standardized through **Device-Generated Commands (DGC)** (`VK_EXT_device_generated_commands`) utilizing Indirect Execution Sets (`VkIndirectExecutionSetEXT`) and Indirect Commands Layouts (`VkIndirectCommandsLayoutEXT`), paired with wavefront compaction queues. GPUBench strictly targets the Vulkan DGC specification.

---

## 2. Four-Scenario Morphology

To evaluate how geometric complexity, mesh density, occlusion, and material distribution dictate scheduling efficiency, GPUBench provides four distinct benchmark scenarios:

### 2.1 Showroom Studio (`-s showroom`)
- **Geometry**: $108,936$ triangles featuring the Khronos ToyCar glTF PBR production asset mounted on an organic cloth pedestal.
- **Textures & Materials**: 11 PBR texture maps across 3 distinct material systems:
  1. *Clearcoat Car Body*: Dual GGX specular lobes, clearcoat roughness, and metallic flake reflections.
  2. *Decal / Chrome Trim*: High-gloss Dirac conductor mirrors and stencil decals.
  3. *Velvet Pedestal*: High-roughness micro-fiber grazing sheen (Charlie model).
- **Primary Use Case**: Studio automotive showcase evaluating multi-lobe clearcoat and texture filtering under direct lighting.

### 2.2 Indoor Atrium (`-s indoor`)
- **Geometry**: $426,064$ triangles featuring the clean-room Renaissance Colonnade Atrium glTF (`assets/models/sponza.glb`) with central courtyard, perimeter colonnades, and upper galleries.
- **Optical Confinement**: $0\%$ sky escape. Every secondary ray is confined within the architectural interior, striking masonry, columns, or fabrics.
- **Textures & Materials**: 25 distinct material definitions with Cook-Torrance GGX PBR, tangent-space normal mapping, roughness/metallic maps, and ornamental drapery.
- **Primary Use Case**: High geometric occlusion, complex depth complexity, and architectural global illumination.

### 2.3 Outdoor Landscape (`-s outdoor`)
- **Geometry**: $57,216$ triangles spanning an expansive $>2,000\text{m}$ mountain terrain, procedural alpine valley, lake plane, and instanced conifer pines.
- **Optical Confinement**: High sky escape ($>60\%$ of secondary rays escape into the sky dome).
- **Lighting & Atmosphere**: Directional solar illumination, hard ray-traced sun shadows, and algebraic Rayleigh-Mie aerial perspective haze.
- **Primary Use Case**: Expansive open world with long ray distances, heavy sky miss rates, and directional shadow evaluation.

### 2.4 Open-World Forest (`-s forest`)
- **Geometry**: $1,051,696$ triangles featuring a high-density, multi-tiered natural ecosystem (256×256 terrain heightmap grid [131,072 triangles], riverbed bathymetry, 530 procedural trees [350 conifer pines, 180 deciduous birches, plus 4 foreground framing trees], 600 boulders, 180 fallen nurse logs, and 6,000 understory plants [3,500 sword ferns and 2,500 foliage shrubs]).
- **Heterogeneous Nature PBR**: 8 specialized physical material shaders:
  1. *Translucent Leaves/Needles*: Dual-sided thin-surface subsurface scattering with backlit emerald transmission.
  2. *Anisotropic Bark*: High-roughness ($0.85$) anisotropic GGX with furrow shadowing.
  3. *Weathered Granite*: Multi-frequency surface noise and steep-slope exposure.
  4. *Moist Topsoil & Silt*: Dynamic water-line moisture darkening and specular gloss modulation.
  5. *Alpine Turf & Ferns*: Micro-fiber grazing sheen (Charlie model).
  6. *Refractive River Water*: Secondary Snell's Law refraction ($\eta = 1.333$) into riverbed bathymetry with Beer-Lambert volumetric absorption.
  7. *Micro-Crystalline Snow*: Micro-facet glint distribution ($N \cdot H^{120} \times 8.0$).
  8. *Aged Timber*: Cross-plank grain reflections and ambient contact shadowing.
- **Primary Use Case**: Extreme material divergence, dense ray-geometry intersections, and secondary transmission paths.

---

## 3. Microarchitectural Analysis & Compiler Occupancy

Benchmarks and shader diagnostics were evaluated on the **AMD Radeon AI PRO R9700 (GFX1201 / RDNA 4)** at native **4K UHD (3840×2160, 8,294,400 primary rays)** using the Mesa RADV Vulkan driver and ACO compiler.

### 3.1 4K UHD Performance Summary (AMD Radeon AI PRO R9700)

| Benchmark Scenario | Triangles | Megakernel | DGC | Speedup | Bit-Exact Match | PSNR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Showroom Studio** | 108,936 | 57.6 FPS (17.37 ms) | **101.3 FPS (9.87 ms)** | **1.76x** | 8,294,400 / 8,294,400 (100%) | 120.0 dB |
| **Indoor Atrium** | 426,064 | 30.5 FPS (32.77 ms) | **68.0 FPS (14.70 ms)** | **2.23x** | 8,294,400 / 8,294,400 (100%) | 120.0 dB |
| **Outdoor Landscape** | 57,216 | 185.8 FPS (5.38 ms) | **420.0 FPS (2.38 ms)** | **2.26x** | 8,294,400 / 8,294,400 (100%) | 120.0 dB |
| **Open-World Forest** | 1,051,696 | 27.0 FPS (37.00 ms) | **55.0 FPS (18.18 ms)** | **2.04x** | 8,294,400 / 8,294,400 (100%) | 120.0 dB |

Across all scenarios, Device-Generated Commands (DGC) reduces frame render times by **43% to 56%**, doubling interactive framerates while preserving identical image output.

---

### 3.2 Compiler Resource Allocation & Wavefront Occupancy (ACO GFX1201)

Querying compiler statistics directly via `--dump-isa rayscheduling` (backed by `VK_KHR_pipeline_executable_properties` / Mesa 26.2 ACO on Slang SPIR-V shaders) shows how the monolithic megakernel differs from decoupled micro-kernels:

#### Hardware Limits (AMD Radeon AI PRO R9700 / GFX1201)
- **Compute Units (CUs)**: 64 Compute Units across 32 WGPs (128 SIMD32 units, 2 SIMDs per CU)
- **Physical Register File**: 1,536 Wave32 VGPRs per SIMD (192 KB; Dynamic VGPR allocation allows 16 waves/SIMD up to 96 VGPRs, 12 at 97–128 VGPRs, 8 at 129–192 VGPRs)
- **Maximum Waves per SIMD**: 16 waves
- **Local Data Share (LDS)**: 128 KB per WGP (64 KB per CU)

#### Shader Statistics & SIMD Wave Slot Allocation (Mesa 26.2 ACO, Slang SPIR-V)

| Compute Kernel | Pipeline Stage | Code Size | Pre-Sched / Alloc VGPRs | LDS Allocation | Waves / SIMD | Architectural Advantage |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`rt_scheduling_traditional_megakernel.comp`** | **Megakernel** | **298.7 KB** | **80 -> 96** (108 SGPRs) | **2,048 B** | **16** *(dynamic)* | Divergent material lanes + large L1I footprint (51,883 insts) |
| **`rt_scheduling_device_generated_commands_classify.comp`** | Ray Hit Classification | **84.8 KB** | **63 -> 96** | **2,048 B** | **16** | Decoupled traversal + wave ballot compaction |
| **`rt_scheduling_device_generated_commands_material.comp`** | Specialized PBR Shading | **116.2 KB** | **63 -> 96** | **0 B** | **16** | **100% uniform SIMD lanes per material queue** |
| **`rt_scheduling_device_generated_commands_bounce.comp`** | Ray Generation / Bounces | **16.2 KB** | **53 -> 96** | **2,048 B** | **16** | Fits easily in L1I cache; uniform secondary traversal |
| **`rt_scheduling_device_generated_commands_shadow.comp`** | Directional Shadows | **2.0 KB** | **32 -> 96** | **2,048 B** | **16** | Tiny L1I footprint (391 insts) |
| **`rt_scheduling_device_generated_commands_light.comp`** | Multi-Light Binning | **4.6 KB** | **33 -> 96** | **0 B** | **16** | Tiny L1I footprint; light-coherent SIMD execution |

#### Architectural Takeaways:
1. **Why the Megakernel Stalls Even at Full Wave Residency**:
   - The monolithic Slang megakernel compiles to **298.7 KB (51,883 instructions)** packing traversal, bounce management, lighting, and all material BSDFs into one binary.
   - Although ACO's dynamic VGPR scheduler fits it into 96 allocated VGPRs (avoiding register spilling, 0 scratch), divergent neighboring rays serialize through distinct material branches while the ~299 KB instruction working set stresses the L1 instruction cache.
2. **Compacted Queues & DGC Eliminate Lane Serialization & I-Cache Thrashing**:
   - Separating the pipeline into DGC stages ensures every indirect shading wave executes a single material archetype (`100%` uniform lane utilization), while auxiliary traversal/shadow/lighting passes drop to **2–16 KB** code footprints with 0 spills.

---

## 4. Hardware Telemetry & Power Management (DPM)

During benchmark passes, external GPU monitoring tools (such as desktop widgets or MangoHud) frequently report fluctuating GPU utilization (~50–84%), power draw around 46W, and memory clocks sitting at 96 MHz. We attached a continuous 20 Hz (50 ms interval) telemetry monitor directly reading GPU 1 kernel sysfs counters to investigate this behavior.

### 4.1 Real-Time Hardware Telemetry Comparison

| Workload | Active Compute Duty Cycle | GPU Busy (Active Avg) | GPU Busy (Peak) | GFX Core Clock (Peak) | VRAM Clock (Peak) | Package Power (Peak) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Device Memory Bandwidth** | **80.4%** | **90.7%** | **100%** | **3468 MHz** | **1258 MHz (DPM 5)** | **263.0 W** |
| **Ray Scheduling (Showroom Megakernel)** | **10.9%** | **55.8%** | **100%** | **3364 MHz** | **1258 MHz (DPM 5)** | **290.0 W** |
| **Ray Scheduling (Showroom DGC)** | **10.6%** | **54.2%** | **100%** | **3253 MHz** | **1124 MHz (DPM 4)** | **281.0 W** |
| **Ray Scheduling (All 4 Scenes Complete)** | **42.6%** | **85.5%** | **100%** | **3399 MHz** | **1258 MHz (DPM 5)** | **379.0 W** |

### 4.2 Root Cause Analysis

#### 1. Duty Cycle & Benchmark Duration (`target_duration_ms = 250.0`)
- Each benchmark configuration executes for **only 250 ms** of active compute, surrounded by warmup runs, CPU pipeline synchronization (`context->waitIdle()`), result collection, and tone-mapping/file export.
- External monitoring tools sample at **500 ms to 1000 ms intervals**.
- When a 1-second sampling window captures 250 ms of active 100% compute followed by 750 ms of host synchronization, the time-averaged rolling metric reads **~50–80% GPU utilization**, despite the GPU being fully active during the compute window.
- In our high-frequency 20 Hz telemetry across the full 4-scene run:
  - Active GPU compute average was **85.5%** (peaking at **100%**).
  - Peak GFX core clock hit **3,399 MHz** (exceeding the 2350 MHz base clock).
  - Peak board power hit **379.0 W**.

#### 2. L3 Infinity Cache Residency vs External GDDR6 Memory Clock
- The AMD Radeon AI PRO R9700 has **64 MB of on-chip L3 Infinity Cache (MALL)** and **8 MB of L2 cache**.
- In ray scheduling, the scene BVH trees, ray hit buffers, and classification queues fit largely inside this 64 MB Infinity Cache.
- AMD's Dynamic Power Management (DPM) firmware scales the GDDR6 memory clock (MCLK) based on memory controller (UMC) bandwidth demand. Because compute shaders hit primarily in L2/L3 cache during 250 ms bursts, external DRAM bandwidth demand is near zero. DPM intelligently keeps MCLK in low-power state 0 (96 MHz) to avoid wasting board power on idle GDDR6 PHYs.
- **Proof via Memory Bandwidth Benchmark**: When running `Device Memory Bandwidth`, which streams tens of gigabytes to DRAM, MCLK immediately ramps to its maximum DPM state (**1,258 MHz**), memory bandwidth reaches **637.79 GB/s** (98.9% of the 645 GB/s theoretical hardware maximum), and power scales to **263.0 W**.

---

## 5. Analytical Parity & Ground-Truth Verification

### 5.1 Analytical LDR & HDR Visual Parity (3840×2160)
To guarantee mathematical correctness between the monolithic megakernel and decoupled Device-Generated Commands (DGC), rendered outputs are verified pixel-by-pixel across both **8-bit LDR framebuffers** and **linear FP32 HDR buffers** (`--verify-parity` checks PSNR, RMSE, max absolute delta, and discrepant pixel counts). Reference 8-bit LDR captures at 4K UHD showed:

```
================================================================================
       RAY SCHEDULING VISUAL & ANALYTICAL PARITY: 4-SCENARIO SUITE
================================================================================
  Resolution          : 3840 x 2160 (8,294,400 primary rays)
  Showroom Studio     : 8,294,400 / 8,294,400 match (100.00%) | 120.00 dB PSNR | 0 Diff
  Indoor Atrium       : 8,294,400 / 8,294,400 match (100.00%) | 120.00 dB PSNR | 0 Diff
  Outdoor Landscape   : 8,294,400 / 8,294,400 match (100.00%) | 120.00 dB PSNR | 0 Diff
  Open-World Forest   : 8,294,400 / 8,294,400 match (100.00%) | 120.00 dB PSNR | 0 Diff
================================================================================
```

All four scenarios match within numerical precision tolerances across LDR and HDR outputs.

### 5.2 Blender Cycles Ground Truth (HIP RT)
The Open-World Forest geometry and nature PBR shaders were ported to **Blender 5.2 (Cycles Path Tracer)** for offline reference validation on GPU 1:
- Running on the exact same hardware (**AMD Radeon AI PRO R9700**) with **hardware HIP RT**, Blender Cycles required **8.43 seconds** (64 spp with OIDN) to render the offline reference.
- GPUBench's decoupled Device-Generated Commands (DGC) pipeline renders the frame in real time at **55.0 FPS (18.18 ms)**, matching perspective, thin-surface foliage subsurface transmission, and riverbed refraction.

---

## 6. Profiling & Benchmarking Workflow

### 6.1 Running Ray Scheduling Benchmarks

```bash
# Run all 4 scenes sequentially (Showroom, Indoor, Outdoor, Forest)
gpubench -d 1 -b rayscheduling -s all

# Run an individual scene
gpubench -d 1 -b rayscheduling -s showroom
gpubench -d 1 -b rayscheduling -s indoor
gpubench -d 1 -b rayscheduling -s outdoor
gpubench -d 1 -b rayscheduling -s forest

# Dump 4K UHD PNG/PPM frames, difference heatmaps, and 4-scenario comparative grid
gpubench -d 0 -b rayscheduling -s all --dump-renders

# Run a specific benchmark configuration (e.g. Config 0: Primary + Shading Megakernel, Config 1: Primary + Shading DGC, Config 17: Primary Rays RTP, Config 21: Full Pipeline 4K UHD Megakernel, Config 22: Full Pipeline 4K UHD DGC)
gpubench -d 0 -b rayscheduling -s forest -c 0
gpubench -d 0 -b rayscheduling -s forest -c 1
gpubench -d 0 -b rayscheduling -s forest -c 17
gpubench -d 0 -b rayscheduling -s forest -c 21
gpubench -d 0 -b rayscheduling -s forest -c 22

# Run in profiling snapshot mode (single submit for clean RGP trace capture)
gpubench -d 0 -b rayscheduling -s forest -c 22 --profile-snapshot
```

### 6.2 Inspecting Shader Occupancy & Compiler Statistics

To inspect ACO compiler machine code statistics, VGPR allocation, and SIMD wave occupancy:
```bash
RADV_DEBUG=shaderstats gpubench -d 1 -b rayscheduling -s showroom -c 22 --profile-snapshot --no-dump
```

### 6.3 Querying Real-Time Hardware Telemetry

Query live clocks, power, and memory utilization on GPU 1:
```bash
# Query via amd-smi (rocm-smi is deprecated)
amd-smi metric -g 1

# Direct kernel sysfs counters (zero CPU overhead)
cat /sys/bus/pci/devices/0000:4d:00.0/gpu_busy_percent
cat /sys/bus/pci/devices/0000:4d:00.0/hwmon/hwmon8/freq1_input   # GFX Clock (Hz)
cat /sys/bus/pci/devices/0000:4d:00.0/hwmon/hwmon8/freq2_input   # VRAM MCLK (Hz)
cat /sys/bus/pci/devices/0000:4d:00.0/hwmon/hwmon8/power1_average # Package Power (uW)
```
