# GPUBench

GPUBench is a high-performance cross-platform GPU benchmarking tool designed to measure raw compute capabilities, memory bandwidth, and modern hardware ray tracing pipeline architectures across graphics hardware. It supports multiple backends and a wide range of data types, from double-precision floating point (FP64) down to 4-bit integers (INT4), alongside modern hardware ray scheduling architectures.

![GitHub Version](https://img.shields.io/github/v/release/Soddentrough/GPUBench)
![License](https://img.shields.io/github/license/Soddentrough/GPUBench)

## Features

- **Multi-Backend Support**: Benchmarks using Vulkan, OpenCL, and ROCm/HIP.
- **Hardware Ray Tracing Suite**:
  - **Ray Scheduling Architectures**: Megakernel vs. Decoupled Wavefronts via Device-Generated Commands (DGC) and Shader Execution Reordering (SER, where supported by hardware).
  - **Real-World Material Divergence**: Realistic heterogeneous material distributions testing VGPR allocation pressure and SIMD wave divergence.
  - **Spatial Ray Divergence**: Parametric cone divergence measuring BVH traversal cache hit rates.
  - **Multi-Layer Alpha Testing**: AnyHit alpha evaluation through 16 stacked cutout planes.
  - **Acceleration Structure Throughput**: BLAS/TLAS build and dynamic vertex refit rates.
- **Comprehensive Compute Data Types**: 
  - Actively Supported Hardware Types: FP64, FP32, FP16, FP8 (Vulkan Cooperative Matrix / ROCm), INT8
  - Capability-Probed / Future Types: BF16 (probed; toolchain arithmetic limitation), FP6 (probed; NVIDIA SPV_NV_float6 only), FP4 (probed; software emulation avoided), INT4 (probed; lacks standardized SPIR-V types)
- **Memory & Cache Hierarchy**: Measure Device VRAM Bandwidth, Host/PCIe Bandwidth, L0 Cache Latency, and full multi-level cache latency curves (16 KB to 256 MB spanning L0 TCP, GL1, GL2, L3 MALL, and GDDR6 DRAM). Cache bandwidth rows are disabled to prevent compiler dead-code elimination.
- **Dynamic Loading**: Backends are loaded at runtime, making them optional and reducing installation dependencies.
- **Cross-Platform**: Built for Linux and Windows.

## Supported Backends

| Backend | Platform | Primary Use Case | Minimum Version |
| :--- | :--- | :--- | :--- |
| **Vulkan** | Linux, Windows | Standard cross-vendor compute & ray tracing | 1.4+ |
| **OpenCL** | Linux, Windows | Fallback cross-vendor compute | 1.2+ |
| **ROCm/HIP** | Linux | Native AMD performance | 6.4+ |

---

## Interfaces & Default Output

GPUBench provides both a standalone graphical workstation profiler with live hardware telemetry and a rich command-line interface with hierarchical Unicode reporting.

### Graphical User Interface (GUI)

The Workstation Profiler dashboard (`gpubench-gui`) provides real-time GPU telemetry (utilization, package power, core and memory clocks, temperatures, and VRAM usage), dynamic physical hardware device detection, compute API diagnostics with requirement tooltips, responsive multi-scale DPI layout (1.0x–2.5x), benchmark suite configuration with dedicated score columns, and in-application comparative ray tracing inspection:

![GPUBench Workstation Profiler GUI](docs/images/gpubench_gui.png)
*Fig: GPUBench Workstation Profiler GUI showing real-time hardware telemetry HUD, backend selection, and benchmark suite configuration.*

### Command-Line Interface (CLI)

Running `gpubench` in the terminal runs the preparation phase, compiles compute and ray tracing pipelines with an interactive ticker, and prints a hierarchical card report with throughput, memory latencies, and architectural speedup takeaways:

```text
$ gpubench -d 0

╭─ GPUBench v1.0.0 ─────────────────────────────────────────────────────────╮
│ Target Device : [GPU 0] AMD Radeon 8060S Graphics (RADV STRIX_HALO)        │
│ Backend / API : Vulkan | VRAM: 81 GB Unified Memory                        │
│ Resolution    : 3840x2160 (4K UHD)                                         │
╰────────────────────────────────────────────────────────────────────────────╯

  [1/2] Preparation Phase (compiling kernels, uploading data, building BVHs)...
  Progress: [━━━━━━━━━━━━━━━━━━━━━━━━━━━━] 100% Compiling pipelines
  ✔ Preparation complete.

  [2/2] Running Benchmarks...
  ✔ Benchmark suite completed.


  ╭─ GPUBench Hierarchical Benchmark Report ─────────────────────────────────────╮
  │ Target Device : AMD Radeon 8060S Graphics (RADV STRIX_HALO) (ID: 0)          │
  │ Canvas / Res  : 3840x2160 (4K UHD)                                           │
  ╰──────────────────────────────────────────────────────────────────────────────╯

  [Compute]
  ╭─ FP64 ───────────────────────────────────────────────────────────────────────╮
  │ Workload                                       │ Backend  │             Throughput │ Details / Speedup                       │
  ├────────────────────────────────────────────────┼──────────┼────────────────────────┼─────────────────────────────────────────┤
  │ FP64                                           │ Vulkan   │            0.43 TFLOPS │                                         │
  ╰────────────────────────────────────────────────┴──────────┴────────────────────────┴─────────────────────────────────────────╯
  ╭─ FP32 ───────────────────────────────────────────────────────────────────────╮
  │ Workload                                       │ Backend  │             Throughput │ Details / Speedup                       │
  ├────────────────────────────────────────────────┼──────────┼────────────────────────┼─────────────────────────────────────────┤
  │ FP32                                           │ Vulkan   │           18.38 TFLOPS │                                         │
  ╰────────────────────────────────────────────────┴──────────┴────────────────────────┴─────────────────────────────────────────╯
  ╭─ FP16 ───────────────────────────────────────────────────────────────────────╮
  │ Workload                                       │ Backend  │             Throughput │ Details / Speedup                       │
  ├────────────────────────────────────────────────┼──────────┼────────────────────────┼─────────────────────────────────────────┤
  │ Vector                                         │ Vulkan   │           24.37 TFLOPS │ [Baseline]                              │
  │ Matrix                                         │ Vulkan   │           54.66 TFLOPS │ └──> 2.24x (+124.3%)                    │
  ╰────────────────────────────────────────────────┴──────────┴────────────────────────┴─────────────────────────────────────────╯
  ╭─ INT8 ───────────────────────────────────────────────────────────────────────╮
  │ Workload                                       │ Backend  │             Throughput │ Details / Speedup                       │
  ├────────────────────────────────────────────────┼──────────┼────────────────────────┼─────────────────────────────────────────┤
  │ Vector                                         │ Vulkan   │             12.70 TOPS │ [Baseline]                              │
  │ Matrix                                         │ Vulkan   │             55.10 TOPS │ └──> 4.34x (+333.9%)                    │
  ╰────────────────────────────────────────────────┴──────────┴────────────────────────┴─────────────────────────────────────────╯

  [Memory]
  ╭─ Bandwidth ──────────────────────────────────────────────────────────────────╮
  │ Workload                                       │ Backend  │             Throughput │ Details / Speedup                       │
  ├────────────────────────────────────────────────┼──────────┼────────────────────────┼─────────────────────────────────────────┤
  │ Read 128 threads/group                         │ Vulkan   │            202.30 GB/s │                                         │
  │ Write 128 threads/group                        │ Vulkan   │            124.91 GB/s │                                         │
  │ R/W 128 threads/group                          │ Vulkan   │            129.96 GB/s │                                         │
  │ Read 256 threads/group                         │ Vulkan   │            170.62 GB/s │                                         │
  │ Write 256 threads/group                        │ Vulkan   │            123.14 GB/s │                                         │
  │ R/W 256 threads/group                          │ Vulkan   │            129.40 GB/s │                                         │
  │ Read 1024 threads/group                        │ Vulkan   │            198.08 GB/s │                                         │
  │ Write 1024 threads/group                       │ Vulkan   │            170.15 GB/s │                                         │
  │ R/W 1024 threads/group                         │ Vulkan   │            148.24 GB/s │                                         │
  ╰────────────────────────────────────────────────┴──────────┴────────────────────────┴─────────────────────────────────────────╯

  [Ray Tracing]
  ╭─ Intersection Tests ─────────────────────────────────────────────────────────╮
  │ Workload                                       │ Backend  │             Throughput │ Details / Speedup                       │
  ├────────────────────────────────────────────────┼──────────┼────────────────────────┼─────────────────────────────────────────┤
  │ Ray-Triangle                                   │ Vulkan   │       3,396.69 MRays/s │                                         │
  │ Ray-Box                                        │ Vulkan   │       7,375.20 MRays/s │                                         │
  ╰────────────────────────────────────────────────┴──────────┴────────────────────────┴─────────────────────────────────────────╯

  ╭─ Executive Performance Summary & Architectural Takeaways ────────────────────╮
  │ • Wavefront Scheduling Speedup : 3.08x in PBR Ray Tracing [Indoor Atrium] (56.8 vs 18.5 MRays/s) │
  │ • Peak Measured Ray Rate        : 7,375.2 MRays/s (Ray-Box)                  │
  ╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## Hardware Ray Tracing & Scheduling Architectures

Modern ray tracing performance in production games and visual effects engines is rarely bound by simple triangle intersection; it is bound by **divergence**—both spatial ray direction divergence and material shading divergence.

GPUBench evaluates how different GPU hardware architectures handle these workloads across distinct scheduling architectures:

1. **Traditional Megakernel**: Traces rays and evaluates all hit shading in a single compute pass. Suffering from the "convoy effect," a single complex material forces all lanes to allocate worst-case VGPRs and serializes execution over divergent SIMD branches.
2. **Traditional + SER (Shader Execution Reordering)**: Evaluates in-pipeline thread regrouping (`VK_EXT_ray_tracing_invocation_reorder`) on supported architectures (e.g. NVIDIA Ada Lovelace / Blackwell) to dynamically regroup divergent lanes by spatial direction and material hit ID before executing hit shaders. *(Note: Unsupported on AMD RDNA hardware, which relies on software stream compaction).*
3. **Device-Generated Commands (DGC / Wavefront Compaction)**: Compacts divergent hits into categorized material queues via ballot/atomic compaction and dispatches uniform waves using GPU-driven command generation (`VK_EXT_device_generated_commands`), providing optimal scaling across AMD RDNA and multi-vendor GPUs.

#### Four-Scenario Benchmarking Morphology
All scenario throughput measurements below were evaluated at **4K UHD (3840x2160, ~8.29M primary rays)** on the **AMD Radeon AI PRO R9700** (RDNA 4 / gfx1201):
- **Showroom Studio (`-s showroom`)**: $108,936$ triangles featuring the Khronos ToyCar glTF asset with clearcoat, decals, and velvet pedestal. Device-Generated Commands (DGC) achieve **101.3 FPS** vs. Megakernel **57.6 FPS** (**1.76x speedup**).
- **Complex Indoor Atrium (`-s indoor`)**: $262,267$ triangles featuring Crytek Sponza glTF with 25 PBR materials and 0% sky escape. Device-Generated Commands (DGC) achieve **68.0 FPS** vs. Megakernel **30.5 FPS** (**2.23x speedup**).
- **Open-World Outdoor Landscape (`-s outdoor`)**: $57,216$ triangles spanning $>2000\text{m}$ alpine terrain, lake, conifer foliage, and Rayleigh-Mie atmospheric scattering. Device-Generated Commands (DGC) achieve **420.0 FPS** vs. Megakernel **185.8 FPS** (**2.26x speedup**).
- **Open-World Forest (`-s forest`)**: $1,007,280$ triangles featuring high-density 256×256 terrain, river bathymetry, 530 trees (350 pines, 180 birches), and 8 nature PBR shaders. Device-Generated Commands (DGC) achieve **55.0 FPS** vs. Megakernel **27.0 FPS** (**2.04x speedup**).
- **100% Bit-Exact Analytical Parity**: Verified bit-exact 120.00 dB PSNR, 0.000000 MAE, and 0 discrepant pixels across all 8,294,400 pixels at 4K UHD across all four scenarios.

---

### Realistic Material Divergence

Production scenes rarely contain uniform shaders; they feature a **heterogeneous distribution of materials** with differing computational complexity and register footprints.

![Realistic Material Range Showroom](docs/images/realistic_scene_material_range.png)
*Fig 1: Representative still-life showroom scene featuring a heterogeneous distribution of production material archetypes.*

![Showroom Geometric Wireframe](docs/images/geometry_showroom_wireframe.png)
*Fig 2: Wireframe view showing the underlying geometry, mesh density, and topological curvature of the test scene.*

#### Reference Material Archetypes

![5-Material Shader Lineup](docs/images/material_lineup.png)
*Fig 3: Lineup of production material candidates on test pedestals.*

| Archetype | Reference Shading Model | Computational / SIMD Bottleneck |
| :--- | :--- | :--- |
| **Clearcoat Car Paint** | Dual-specular GGX lobes (clearcoat + metallic substrate), Beer-Lambert absorption, high-frequency Voronoi micro-flake glints. | Multi-lobe evaluation, procedural hash functions, secondary normal perturbations. |
| **Dispersive Crystal / Glass** | Snell's law refraction with total internal reflection (TIR) branching, Cauchy spectral dispersion, 450 nm thin-film wave interference. | Directional ray branching (reflection vs. transmission), trigonometric Airy interference series. |
| **Organic Jade / Wax** | Multi-channel subsurface diffusion profile ($R, G, B$ differing mean free paths), dual-lobe surface gloss. | Multi-channel exponential attenuation, non-local volumetric scattering. |
| **Anisotropic Velvet / Fabric** | Dual-axis anisotropic roughness ($a_x \neq a_y$) with tangent frame rotation, Charlie micro-fiber inverted grazing sheen ($D_{\text{charlie}}$). | Tangent-space matrix transforms, transcendental power functions ($x^{1/2\alpha}$). |
| **Weathered Industrial Rust** | 6-octave Fractal Brownian Motion (FBM) noise loops, continuous dynamic phase transition from conductor steel to porous dielectric rust. | Heavy arithmetic loop execution, divergent multi-octave iteration depth. |
| **Matte Ceramic & Concrete** | Standard Lambertian/Oren-Nayar diffuse PBR. | Minimal ALU baseline, high wave occupancy. |

#### Analytic Atmospheric Skybox Model

When secondary rays escape geometric boundaries into the surrounding environment, GPUBench evaluates an algebraic Rayleigh-Mie atmospheric scattering model with Henyey-Greenstein solar aureole forward scattering:

![Analytic Atmospheric Skybox Panorama](docs/images/skybox_analytic_preview.png)
*Fig 4: 360° equirectangular preview of the mathematical atmospheric sky model evaluated when rays miss geometry.*

Stressing arithmetic ALUs on miss without querying large VRAM texture maps ensures that BVH traversal and material divergence remain the dominant bottlenecks without cache pollution from texture filtering units.

---

### Geometry & BVH Traversal Benchmarks

![16-Layer Alpha-Testing Stack](docs/images/geometry_alpha_layers.png)
*Fig 5: 16 stacked alpha-tested cutout planes used in the `RayAnyHit` benchmark to measure BVH AnyHit invocation overhead.*

* **Multi-Layer Alpha Testing (`RayAnyHit`)**: Measures hardware performance when traversing through transparent foliage and cutout surfaces. Tests BVH traversal with stochastic opacity cutouts across 16 stacked geometric planes.
* **Spatial Cone Ray Divergence (`RayDivergence`)**: Sweeps ray cone distribution angles from $\theta = 0^\circ$ (fully coherent primary rays) to $\theta = 90^\circ$ (fully diffuse hemispherical rays) to benchmark L1/L2 cache hit rates in GPU BVH traversal units.

---

## Quick Start

### Prerequisites

Ensure you have the appropriate drivers and SDKs installed for the backends you wish to use. See [VERSION_REQUIREMENTS.md](VERSION_REQUIREMENTS.md) for details.

### Installation

Download the latest release package (`.rpm`, `.deb`, `.tar.gz`) from the [GitHub Releases](https://github.com/Soddentrough/GPUBench/releases) page or build from source following the [INSTALL.md](INSTALL.md) guide.

### Basic Usage

```bash
# List all available benchmarks
gpubench --list-benchmarks

# List compute backend availability and diagnostic status
gpubench --list-backends

# Run all benchmarks on default device
gpubench

# Run Dual-Issue & Concurrency benchmark suite across backends
gpubench -d 0 -b dualissue
gpubench -d 0 -b dualissue -k vulkan,rocm

# Run Ray Scheduling on a specific GPU device (e.g. Device 1)
gpubench -d 1 -b RayScheduling

# Select benchmark scene morphology: showroom, indoor, outdoor, forest, or all
gpubench -d 0 -b rayscheduling -s forest
gpubench -d 0 -b rayscheduling -s all

# Set render resolution preset (720p, 1080p, 1440p, 4k) or custom WxH (default: auto; 4K UHD on >=16GB VRAM)
gpubench -d 0 -b rayscheduling -s forest -r 4k

# Dump 4K UHD PPM/PNG render buffers, diff heatmaps, and 4-scenario comparative grid
gpubench -d 0 -b rayscheduling -s all --dump-renders

# Run specific config (e.g. Config 17: Primary Rays (Compute Megakernel), Config 18: Primary Rays (Wavefront - DGC), Config 29: Primary Rays (RTP)) in profiling snapshot mode
gpubench -d 0 -b rayscheduling -s forest -c 18 --profile-snapshot

# Export machine-readable results to JSON
gpubench -d 0 -b rayscheduling -s all -o benchmark_results.json
```

### Profiling & Telemetry Suite

GPUBench includes Python tools for automated thread tracing with Mesa RADV / Radeon GPU Profiler (RGP) and AMD ROCm SMI telemetry:

```bash
# Capture RGP traces, amd-smi power/clock telemetry, and RGA ISA compilation
python3 scripts/capture_gpu_profiles.py
```

## Documentation

### Core Guides
- [Installation Guide](INSTALL.md) - Cross-platform build and installation instructions.
- [Windows Environment & Build Guide](WINDOWS.md) - MinGW toolchain, packaging, and Windows guidelines.
- [Version Requirements](VERSION_REQUIREMENTS.md) - Minimum software and compute hardware requirements.
- [Release Process](RELEASING.md) - Tagging, CI packaging, and automated release deployment.

### Architectural & Technical Whitepapers
- [Ray Scheduling Architectures](docs/RAY_SCHEDULING_ARCHITECTURE.md) - Decoupled scheduling, microarchitectural ISA analysis, and RGP timeline profiling.
- [BVH Traversal Architectural Audit](docs/BVH_TRAVERSAL_ARCHITECTURAL_AUDIT.md) - Hardware BVH traversal ceilings, box and triangle peak rates on RDNA 4.
- [RDNA 4 Ray Tracing Architecture](docs/RDNA4_RAY_TRACING_ARCHITECTURE.md) - RAv3 BVH8 traversal, LDS hardware stack management, and DGC wavefront compaction.
- [RDNA 3 Ray Tracing Architecture](docs/RDNA3_RAY_TRACING_ARCHITECTURE.md) - Chiplet topology, memory fabric, and zero-LDS pure Wave32 compaction.
- [Hardware Profiling & Telemetry Guide](docs/PROFILING_GUIDE.md) - RGA disassembly, ACO compiler stats, packet dumping, and SMI telemetry.
- [Compute Performance Analysis](docs/COMPUTE_PERFORMANCE_ANALYSIS.md) - Compute pipelines, packed math, and compiler ceilings.
- [Dual-Issue & Datapath Concurrency](docs/DUAL_ISSUE_ANALYSIS.md) - RDNA 3 VOPD, RDNA 4 dual-issue SIMD32, and NVIDIA concurrent FP32+INT32 datapath analysis.
- [OpenCL Backend](docs/OPENCL_BACKEND.md) - Architecture, feature matrix, and disk binary caching.
- [Desktop Integration](docs/DESKTOP_INTEGRATION.md) - FreeDesktop XDG, Windows High-DPI manifest, and macOS app bundle integration.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
