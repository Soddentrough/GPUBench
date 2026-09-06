# GPUBench Windows Environment & Build Guide

This document provides developer and agent guidelines for configuring, building, testing, and running GPUBench on Windows systems.

---

## 1. Hardware & System Target

- **Operating System**: Windows 11 (64-bit).
- **Target GPU**: **GPU 0** (`-d 0`) — `AMD Radeon RX 7900 XTX` (Navi 31, RDNA 3, GFX1100).
  > **Note**: Unlike the Linux environment described in `AGENTS.md` (which targets GPU 1 on a multi-GPU Threadripper node), on this Windows system GPU 0 is the primary discrete Radeon GPU.
- **Driver**: AMD Proprietary Driver 26.8.1 (LLPC / Vulkan 1.4.349).
- **API Backends**: Vulkan 1.4 (Primary). OpenCL and ROCm/HIP are not active on Windows.

---

## 2. Toolchain & Environment Paths

GPUBench on Windows is built using the **MSYS2 MinGW64** toolchain with CMake and Ninja. Use these exact paths:

| Tool | Path | Notes |
| :--- | :--- | :--- |
| **MSYS2 MinGW64 Bin** | `C:\msys64\mingw64\bin` | Contains `gcc.exe`, `g++.exe`, `ninja.exe`, `python.exe` |
| **MSYS2 Usr Bin** | `C:\msys64\usr\bin` | Contains `pacman.exe` |
| **CMake** | `C:\Program Files\CMake\bin\cmake.exe` | Minimum CMake version 3.22+ |
| **Vulkan SDK** | `C:\VulkanSDK\1.4.357.0` (or `C:\VulkanSDK\*`) | Required headers, validation layers, and `vulkan-1.lib` |
| **Python & Pillow** | `C:\msys64\mingw64\bin\python.exe` | Package: `mingw-w64-x86_64-python-pillow` |

### Python & Pillow Prerequisite
Image conversion and telemetry annotation scripts (`scripts/annotate_render.py` and `scripts/make_triptych.py`) require the Python `Pillow` library. If missing in MSYS2, install it via:
```cmd
C:\msys64\usr\bin\pacman.exe -S --noconfirm mingw-w64-x86_64-python-pillow
```

---

## 3. Quick Start via `build_windows.bat`

The root [`build_windows.bat`](file:///c:/Users/naoki/Development/GPUBench/build_windows.bat) script automates environment detection, CMake configuration, building, and copying required runtime DLLs.

### Commands

| Command | Purpose |
| :--- | :--- |
| `.\build_windows.bat` | Build GPUBench Release target and bundle MinGW DLLs |
| `.\build_windows.bat test` | Build and list detected Vulkan devices (`gpubench.exe -l`) |
| `.\build_windows.bat run [args...]` | Build and run `gpubench.exe` with forwarded arguments |
| `.\build_windows.bat clean` | Wipe `build-release/` directory and perform a fresh build |
| `.\build_windows.bat package` | Build and create a standalone CPack release `.zip` archive |

### Non-Interactive / Script / CI Execution
Pass `--no-pause` (or set `CI=1` in the environment) to prevent interactive keyboard pauses on errors:
```cmd
build_windows.bat --no-pause test
build_windows.bat --no-pause run -d 0 -b RayScheduling -s showroom --no-dump
```

---

## 4. Manual PowerShell Build & Run

If working directly in PowerShell, prepend the toolchain paths to `$env:PATH`:

```powershell
# 1. Setup toolchain environment
$env:PATH = "C:\msys64\mingw64\bin;C:\msys64\usr\bin;C:\Program Files\CMake\bin;" + $env:PATH
if (-not $env:VULKAN_SDK) { $env:VULKAN_SDK = (Get-ChildItem C:\VulkanSDK | Sort-Object Name -Descending | Select-Object -First 1).FullName }
$env:PATH = "$env:VULKAN_SDK\Bin;" + $env:PATH

# 2. Configure with Ninja
cmake -B build-release -S . -DCMAKE_BUILD_TYPE=Release -G "Ninja"

# 3. Build
cmake --build build-release

# 4. Run GPUBench
.\build-release\gpubench.exe -d 0 -b RayScheduling
```

---

## 5. Runtime DLL Requirements for `gpubench.exe`

Because the binary is built with GCC under MinGW-w64, running `gpubench.exe` standalone requires three runtime DLLs:
1. `libstdc++-6.dll`
2. `libgcc_s_seh-1.dll`
3. `libwinpthread-1.dll`

- `CMakeLists.txt` and `build_windows.bat` automatically copy these DLLs into `build-release/` post-build.
- If `gpubench.exe` exits immediately with code 1 and no output in a clean shell, verify that these three DLLs exist in `build-release/` alongside `gpubench.exe`.

---

## 6. Vulkan Device-Generated Commands (DGC) on AMD Windows Driver

When implementing or debugging Vulkan DGC (`VK_EXT_device_generated_commands`) on Windows AMD drivers (LLPC):

1. **Standard DGC (Indirect Dispatches + Push Constants)**:
   - Supported on compute stages (`VK_SHADER_STAGE_COMPUTE_BIT`).
   - Powers native DGC work list ray scheduling, achieving significant speedups over megakernels.
2. **DGC Execution Set Pipeline Binding (`VK_INDIRECT_COMMANDS_TOKEN_TYPE_EXECUTION_SET_EXT`)**:
   - `VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT::supportedIndirectCommandsShaderStagesPipelineBinding` is `0` (None) on AMD Windows driver 26.8.1 for compute shaders.
   - Dynamic pipeline switching within compute DGC sequences is not supported on this driver.
   - The engine queries `isDGCExecutionSetSupported()` and automatically uses standard indirect dispatches as fallback for specialized material micro-kernels.
3. **Buffer Usage Flags**:
   - Any buffer used as an indirect argument or sequence buffer in DGC must include `VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT` and `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT`.
4. **Vulkan Surface Extension**:
   - `VK_KHR_surface` must be enabled during `vkCreateInstance` whenever `VK_KHR_swapchain` is requested on device creation.

---

## 7. Render Artifacts & Verification

All image verification dumps and visual parity comparisons are exported to the `renders/` directory:
- Side-by-side annotated comparisons: `render_<scene>_comparison.png`, `render_<scene>_pathtracing_<spp>spp_comparison.png`
- Technique & Pipeline Breakdown grids: `render_technique_grid.png`, `render_comparison_grid.png`, `render_pathtracing_grid.png`
- Profiling JSON telemetry: `render_<scene>_profile.json`, `render_<scene>_pt1_profile.json`, `render_<scene>_pt16_profile.json`
