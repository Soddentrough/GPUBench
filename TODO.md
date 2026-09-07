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
