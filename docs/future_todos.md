# GPUBench Future Work & Roadmap

## Graphics & Ray Tracing
- **Pixel Fill Rate (ROP Throughput)**: Completed in v1.2.0 (`PixelFillRateBench.cpp`). Measures offscreen Vulkan rasterization fill rates (RGBA8, RGBA16F HDR, Alpha Blending) using a dedicated graphics pipeline.
- **Vulkan Work Graphs (`VK_AMDX_shader_enqueue`)**: Evaluate autonomous GPU node enqueue and dynamic shader dispatch graphs for ray scheduling and multi-pass traversal.

## Compute Enhancements
- **Cooperative Matrix Sub-Byte Formats**: Integrate native FP8 (`VK_EXT_shader_float8` + `VK_KHR_cooperative_matrix`) and INT4 matrix multiplication when driver toolchains provide GLSL compiler support.
- **Dynamic Workgroup Dispatch Scaling**: Dynamically scale dispatch workgroup dimensions based on hardware Compute Unit / Streaming Multiprocessor counts across low-power and high-end discrete GPUs.

## Ecosystem & Infrastructure
- **Community Leaderboard**: Opt-in JSON result submission to community database for cross-architecture and driver comparison.
- **Windows Authenticode Signing**: Implement digital signature integration for Windows release binaries.
