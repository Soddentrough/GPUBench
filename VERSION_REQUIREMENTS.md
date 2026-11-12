# Version Requirements

This document specifies the minimum version requirements for all GPU compute backends and vendored middleware used in GPUBench.

## Summary

- **Vulkan**: **1.4+** uncompromising baseline (`VK_API_VERSION_1_4` / SPIR-V 1.4+ — legacy Vulkan 1.2/1.3 runtime fallbacks removed)
- **ROCm/HIP**: **6.4+** (Linux & Windows dynamic `amdhip64_7.dll` / `amdhip64_6.dll` / `libamdhip64.so` + `hipGraphLaunch` stream capture)
- **OpenCL**: **3.0** target headers (`CL_TARGET_OPENCL_VERSION=300`; compatible with OpenCL 1.2–3.0 ICD runtimes exposing required feature queries)
- **Compiler Standard**: C++23 (`-std=c++23` / `/std:c++latest`) and C11

## Backend Details

### Vulkan Requirements

**Minimum Version**: **1.4** (`VK_API_VERSION_1_4`)

GPUBench requires a native Vulkan 1.4 loader and physical device driver (`apiVersion >= VK_API_VERSION_1_4`). Devices reporting Vulkan 1.3 or earlier are cleanly skipped during physical device enumeration without silent downgrade.

All GLSL (`#version 460`) and SPIR-V (`--target-env vulkan1.4`) shaders in `shaders/` target Vulkan 1.4 core capabilities and optional hardware-probed extensions:
- **Memory Management**: Managed via **VulkanMemoryAllocator (VMA 3.5.0)** with `VK_EXT_memory_budget`, `VK_KHR_buffer_device_address` (Vulkan 1.2/1.4 core), and ReBAR (`HOST_VISIBLE | DEVICE_LOCAL`) awareness, plus persistent disk-backed `VkPipelineCache` keyed by `(vendorID, deviceID, pipelineCacheUUID)`.
- **Matrix, Sparsity & Cooperative Vector**: `VK_KHR_cooperative_matrix` (dynamic `(M, N, K)` tile selection via specialization constants), `VK_KHR_shader_bfloat16`, `VK_EXT_shader_float8`, `VK_EXT_shader_ocp_microscaling_types` (`Float6E3M2EXT` / `Float4E2M1EXT`), and `VK_NV_cooperative_vector` (`coopvecNV` inference matmul+bias+ReLU).
- **Hardware Ray Tracing & Execution Reordering**: `VK_KHR_acceleration_structure`, `VK_KHR_ray_query`, `VK_KHR_ray_tracing_pipeline`, `VK_EXT_ray_tracing_invocation_reorder` (SER), and `VK_EXT_opacity_micromap` (2-state / 4-state hardware Opacity Micromaps).
- **GPU-Driven Scheduling**: `VK_EXT_device_generated_commands` (DGC wavefront compaction) and `VK_AMDX_shader_enqueue` (`SPV_AMDX_shader_enqueue` GPU Work Graphs producer->consumer multi-node expansion).

**Affected Files**:
- `shaders/*.comp`, `shaders/*.rgen`, `shaders/*.rmiss`, `shaders/*.rchit`, `shaders/*.rahit`, `shaders/workgraphs_amdx.spvasm`

### ROCm/HIP Requirements

**Minimum Version**: **6.4+** (ROCm 6.4 – 10.x)

HIP kernels in `hip_kernels/` support:
- **RDNA 3 / 3.5 / 4 (`gfx1100`, `gfx1151`, `gfx1200`, `gfx1201`)**: Wave32 WMMA (`__builtin_amdgcn_wmma_*`) for FP16, BF16, FP8, INT8, and sub-byte INT4 (`iu4` on `gfx12+`).
- **CDNA 2 / 3 / 4 (`gfx90a`, `gfx942`, `gfx950`)**: Wave64 MFMA (`__builtin_amdgcn_mfma_*`) for FP16, BF16, FP8 (`gfx942+`), INT8, and OCP MXFP6/MXFP4 (`__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` on `gfx950`).
- **Low-Overhead Batched Timing**: Timed iteration batches are captured into reusable `hipGraphExec_t` graphs and launched via `hipGraphLaunch`.
- **Cross-Platform Runtime Loading**: Dynamically probes `libamdhip64.so` on Linux and `amdhip64_7.dll` / `amdhip64_6.dll` / `amdhip64.dll` on Windows.

**Affected Files**:
- `hip_kernels/*.hip`

### OpenCL Requirements

**Target API Version**: **OpenCL 3.0** (`CL_TARGET_OPENCL_VERSION=300`)

OpenCL kernels in `kernels/*.cl` use `fma`, vector arithmetic, and hardware capability probes (`cl_khr_fp16`, `cl_khr_fp64`, `cl_khr_subgroups`) with disk-cached compiled binaries.

**Affected Files**:
- `kernels/*.cl`

## Vendored Third-Party Middleware Versions

| Component | Version | Purpose |
| :--- | :--- | :--- |
| **VulkanMemoryAllocator (VMA)** | `3.5.0` | Unified Vulkan 1.4 buffer, image, and acceleration structure memory allocation |
| **Dear ImGui** | `1.93.0` | Workstation GUI with `ImGuiBackendFlags_RendererHasTextures` dynamic font baking |
| **ImPlot** | `v1.1` | Real-time hardware telemetry charts & Empirical Roofline log-log plotting |
| **CLI11** | `2.7.2` | Unified CLI argument parsing across `gpubench` and `gpubench-gui` |
| **yyjson** | `0.13.0` | RFC 8259 typed JSON serialization & parsing (unified v2.0 schema) |
| **SDL3** | `3.x` | Cross-platform windowing, Vulkan surface creation, and HiDPI display scaling |

## Build & Runtime Requirements

- **Vulkan SDK**: 1.4+ (`glslc`, `spirv-as`, `spirv-val`, `Vulkan-Headers` 1.4+)
- **CMake**: 3.20+ and Ninja / Make / MSBuild
- **C++23 Compiler**: GCC 13+, Clang 17+, or MSVC 19.38+ (Visual Studio 2022 17.8+)
- **Optional SDKs**: ROCm/HIP SDK 6.4+, OpenCL 3.0 ICD Loader & Headers, SDL3 (required for `gpubench-gui`)
