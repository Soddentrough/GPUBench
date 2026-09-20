# GPUBench Future Work & TODOs

## Graphics Benchmarks
- **Pixel Fill Rate (ROP Throughput)**: 
  - *Goal*: Measure the theoretical peak pixel fill rate (e.g., 380.2 GP/s on RDNA4).
  - *Implementation Needs*: Requires adding a full Vulkan graphics pipeline (Vertex + Fragment shaders) to `VulkanContext`. Compute shaders using `imageStore` bypass the Render Output Units (ROPs) and Delta Color Compression (DCC), so they cannot reach true fill rate spec limits.
  - *Design*: Render a massive offscreen framebuffer (e.g., 8192x8192) using a fullscreen quad.

## Compute Enhancements
- (Add future compute enhancements here)
