# GPUBench Architecture Recommendations

Based on a codebase review, here are the leading architectural recommendations for future iterations of GPUBench, designed to increase robustness, maintainability, and alignment with modern GPU programming standards.

## 1. Resource Management & RAII
Currently, Vulkan buffers (`VulkanBuffer`) and kernels (`VulkanKernel`) are managed via raw pointers tracking resource handles. 
It is strongly recommended to transition to full RAII (Resource Acquisition Is Initialization) semantics, potentially leveraging C++ smart pointers with custom deleters, or migrating to `Vulkan-Hpp` to prevent resource leaks and avoid undefined behavior during teardown.

## 2. Vulkan Memory Allocator (VMA)
`VulkanContext::createBuffer` calls `vkAllocateMemory` for every buffer. The Vulkan specification strictly limits the total number of simultaneous memory allocations (often capped around 4096). While GPUBench might not exceed this cap, integrating AMD's Vulkan Memory Allocator (VMA) is a cornerstone industry best-practice. It significantly reduces memory fragmentation and often enhances performance by pooling allocations.

## 3. Dynamic Dispatch Sizing (Magic Numbers)
Hardware execution workloads (e.g., `Fp32Bench::Run`) natively dispatch fixed workgroup sizes (e.g., `8192` workgroups `x 64` local size).
While this will saturate mid-to-high-end tier cards successfully, a best-practice strategy involves actively querying the device for its core count (Streaming Multiprocessors / Compute Units). Dispatch dimensions can then be scaled dynamically in the runtime (e.g., `numWorkgroups = CUs * wave_multiplier`) to adapt more efficiently to extremely low-end IGPs or next-generation Mega-GPUs.

## 4. Code Organization in `BenchmarkRunner`
The `BenchmarkRunner::run()` method spans hundreds of lines of complex logic intertwining terminal output, timing loops, iteration tracking, device enumeration, and TDR timeout logic.
Refactoring the core execution constraint (`while (total_time < 5000)`) into a reusable template or callable parameter, and moving the CLI progress prints to a distinct logging/UI abstraction, will dramatically increase modularity.

## 5. Teardown Resilience
If a Benchmark's `Setup()` rapidly fails (e.g., attempting to load a missing `.comp` file), the framework safely intercepts this but directly calls `Teardown()`. `Teardown()` implementations generally assume internal members (`kernel`, `buffer`) are initialized; however, if these aren't initialized cleanly (e.g., zero-initialized in headers), attempting to forcefully free them prompts segregation faults. Explicit initialization and null validity checks throughout teardown steps are recommended.
