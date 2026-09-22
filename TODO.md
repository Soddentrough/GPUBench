# GPUBench Roadmap & TODOs

## Completed in v1.2.0
- [x] **FP32 Dual-Issue SIMD32 RDNA4 Dual-Issue Saturation**: Expanded `hip_kernels/fp32.hip` to 32 `float4` accumulators in a ring chain, achieving 44+ TFLOPS on Navi 48.
- [x] **FP16 & BF16 Packed Math Saturation**: Upgraded `shaders/fp16.comp` and `shaders/bf16.comp` to 32 `f16vec4` accumulators (256 FLOPs/iter) and corrected FLOP accounting in `Fp16Bench.cpp` / `Bf16Bench.cpp`.
- [x] **Memory Bandwidth Write-Mode Guard**: Eliminated redundant loads from `InputBuffer` in `shaders/membw_*.comp` during write-only sweeps to dedicate 100% of memory bus bandwidth to streaming stores.
- [x] **Ray Tracing Payload Register Pressure Integrity**: Initialized and accumulated all payload fields in `shaders/raypayload_*.rgen` before/after `traceRayEXT` to accurately test register pressure and spilling.
- [x] **Asynchronous Vulkan In-Flight Command Ring**: Decoupled per-dispatch synchronous `vkWaitForFences` stalls in `VulkanContext.cpp` with a 16-frame in-flight command buffer ring and 3-second TDR hang watchdog.
- [x] **Pixel Fill Rate (ROP Throughput) Benchmark**: Built offscreen Vulkan rasterization pipeline measuring RGBA8, RGBA16F HDR, and Alpha Blending fill rates in GPixels/s.
- [x] **Real-time Hardware Telemetry HUD**: Added sysfs hardware telemetry monitoring (temperatures, power draw, core/memory clocks, VRAM usage) in `gpubench-gui`.
- [x] **Benchmark Naming & Progress Bar Fixes**: Renamed `"Performance"` to `"Device Memory Bandwidth"`, fixed `Fp6Bench` naming, and fixed ROCm compilation progress bar rendering.
- [x] **4K UHD Default Render Dimensions**: Initialized default render target dimensions to 4K UHD (`3840 x 2160`, ~8.29M primary rays) for GPUs with $\ge 16$ GB VRAM (with automatic graceful tier down to 1440p/1080p for lower VRAM cards), plus CLI `-r`/`--resolution` and GUI resolution preset selection.
- [x] **Dynamic Multi-Scale GUI Layout & Results Visibility**: Engineered responsive ImGui table column sizing with dedicated score columns, preventing result clipping under UI scaling (1.0x to 2.5x).
- [x] **Dynamic Physical Hardware Device Detection**: Enumerated physical GPU devices dynamically, filtering out phantom/unconnected devices on single-GPU workstations.
- [x] **Compute API Diagnostics & Tooltips**: Added runtime capability probing with diagnostic status notes and hover tooltips for unsupported compute backends in the GUI and CLI (`--list-backends`).
- [x] **Console-Free WIN32 GUI Launch**: Configured Windows GUI build for the WIN32 subsystem (`-mwindows` / `WIN32_EXECUTABLE`), eliminating the persistent terminal console window when launching the GUI.

---

## Future Enhancements & TODOs

### Community Leaderboard & Cloud Verification (Target: v1.3.0)
- **Goal**: Enable opt-in submission of benchmark results to a community leaderboard for comparing GPU and system performance across operating systems, driver versions, and microarchitectures.
- **Payload Schema Specification**:
  ```json
  {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "GPUBenchSubmission",
    "type": "object",
    "required": ["version", "timestamp", "system_info", "gpu_info", "benchmark_results", "signature"],
    "properties": {
      "version": { "type": "string", "example": "1.2.0" },
      "timestamp": { "type": "string", "format": "date-time" },
      "system_info": {
        "type": "object",
        "properties": {
          "os": { "type": "string", "example": "Fedora 44" },
          "kernel": { "type": "string" },
          "cpu_model": { "type": "string", "example": "AMD Threadripper 3750X" },
          "ram_gb": { "type": "number", "example": 64.0 }
        }
      },
      "gpu_info": {
        "type": "object",
        "properties": {
          "device_name": { "type": "string", "example": "AMD Radeon RX 9070 XT" },
          "driver_version": { "type": "string" },
          "rocm_version": { "type": "string" },
          "vulkan_api_version": { "type": "string" },
          "vram_size_bytes": { "type": "integer" }
        }
      },
      "benchmark_results": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "component": { "type": "string" },
            "benchmark": { "type": "string" },
            "subcategory": { "type": "string" },
            "config": { "type": "string" },
            "metric": { "type": "string" },
            "score": { "type": "number" },
            "time_ms": { "type": "number" },
            "is_emulated": { "type": "boolean" }
          }
        }
      },
      "signature": { "type": "string", "description": "HMAC-SHA256 checksum / anti-cheat token" }
    }
  }
  ```
- **Backend Architecture**:
  - Cloudflare Worker or Actix-web server running Postgres/ClickHouse database.
  - Rate limiting, anti-cheat validation, and duplicate submission filtering.
  - Web UI for interactive filtering by GPU model, driver version, backend API, and date.

### Windows Packaging & Distribution Security
- **Windows Code Signing & Authenticode Certification**:
  - Integrate Authenticode digital signing for Windows binaries (`gpubench.exe`, `gpubench-gui.exe`) and installer (`GPUBench-*-win64.exe`) in `.github/workflows/release.yml`.
  - Configure SignPath.io (free for open-source GitHub projects) or a trusted code signing certificate with `signtool.exe` to establish reputation and prevent browser / SmartScreen warnings.
  - Establish persistent developer reputation across version releases to prevent SmartScreen "unrecognized app" / "uncommonly downloaded" download blocks.
- **Windows Defender False-Positive Triage & Submission**:
  - Maintain a proactive release workflow to submit newly generated release artifacts to Microsoft Security Intelligence (WDSI) upon publishing.
  - Explore Inno Setup or WiX Toolset (.msi) generator alternatives in CPack to reduce heuristic AV flags associated with NSIS self-extracting archive stubs.

### Architectural Backlog (Best Practices)
- **Resource Management & RAII**: Transition Vulkan buffer (`VulkanBuffer`) and kernel (`VulkanKernel`) handles to full RAII semantics using C++ smart pointers with custom deleters or `Vulkan-Hpp` to guarantee leak-free teardowns.
- **Vulkan Memory Allocator (VMA)**: Integrate AMD's VMA library into `VulkanContext` to pool device memory allocations, eliminate the 4,096 allocation cap limitation, and reduce VRAM fragmentation.
- **Dynamic Workgroup Dispatch Scaling**: Scale dispatch workgroup counts dynamically at runtime based on enumerated Compute Unit / Streaming Multiprocessor counts (`numWorkgroups = CUs * wave_multiplier`), adapting to low-power IGPs and high-CU discrete GPUs alike.
- **Modular Execution Engine**: Decouple `BenchmarkRunner::run()` timing loops from terminal/GUI reporting into reusable template abstractions and dedicated logging interfaces.
- **Teardown Null-Safety**: Implement explicit initialization and null checks across all benchmark `Teardown()` routines to ensure resilience if `Setup()` aborts early.

