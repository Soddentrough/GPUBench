# OpenCL Compute Backend Support

GPUBench includes a first-class, cross-vendor **OpenCL Compute Backend** (`--backend opencl`), providing GPU compute and memory bandwidth benchmarking across AMD, Intel, NVIDIA, and mobile/embedded GPUs without requiring Vulkan 1.4 or ROCm/HIP drivers.

---

## 1. Architecture

The multi-backend architecture consists of:

1. **`IComputeContext` Interface**: Backend-agnostic abstract interface for memory buffers, kernel compilation, argument binding, execution dispatches, and device queries.
2. **`OpenCLContext` Implementation**: OpenCL driver interface utilizing dynamic loading (`libOpenCL.so.1` on Linux, `OpenCL.dll` on Windows) to avoid hard link-time package dependencies.
3. **`ComputeBackendFactory`**: Factory managing compile-time detection, runtime availability probes, and fallback backend selection (`Vulkan -> OpenCL -> ROCm`).
4. **`ShaderCache` Disk Binary Caching**: Automatic caching of compiled OpenCL program binaries (`clGetProgramInfo(CL_PROGRAM_BINARIES)`) for instant subsequent execution without re-compilation overhead.
5. **Cross-Vendor Portable OpenCL C Kernels**: High-throughput SIMD vector implementations written in standard OpenCL C 1.2+ for maximum cross-platform compatibility.

---

## 2. Benchmark Feature Matrix & Fallback Policy

GPUBench classifies tests cleanly into fully supported native workloads and transparent API limitations:

| Benchmark | OpenCL Status | Capability / Extension | Notes |
| :--- | :---: | :--- | :--- |
| **FP64 Compute** | ✅ **Supported** | `cl_khr_fp64` / `cl_amd_fp64` | Native double-precision floating-point throughput. |
| **FP32 Compute** | ✅ **Supported** | Standard OpenCL C 1.2+ `fma()` | Standard single-precision floating-point throughput. |
| **FP16 Compute** | ✅ **Supported** | `cl_khr_fp16` (`half` / `half2`) | Vector compute throughput. (Matrix mode requires cooperative matrix). |
| **BF16 Compute** | ⚠️ *API Limitation* | None (`cl_khr_bfloat16` not standard) | Reported as `UNSUPPORTED (API Limitation)` without error/crash. |
| **FP8 Compute** | ⚠️ *API Limitation* | None | Reported as `UNSUPPORTED (API Limitation)`. |
| **FP4 / INT4** | ⚠️ *Hardware Limitation* | None | Reported as `UNSUPPORTED (Hardware Limitation)`. |
| **INT8 Compute** | ✅ **Supported** | Standard vector math (`char4`) | Vector throughput across all OpenCL vendors. |
| **Device Memory Bandwidth** | ✅ **Supported** | Standard OpenCL Buffers | 128, 256, and 1024 threads/group sweep modes (Read, Write, R/W). |
| **Cache Latency** | ✅ **Supported** | Pointer-chasing buffer traversal | Single-workitem pointer chasing across L0/L1/L2/L3 cache levels. |
| **Pixel Fill Rate (ROPs)** | ⚠️ *API Limitation* | Graphics pipeline required | Reported as `UNSUPPORTED (API Limitation)` (OpenCL lacks rasterizer/ROPs). |
| **Ray Tracing (All 9 Suites)** | ⚠️ *API Limitation* | `VK_KHR_ray_tracing_pipeline` required | Reported as `UNSUPPORTED (API Limitation)` (OpenCL lacks BVH traversal ISA). |

---

## 3. Building with OpenCL Support

### Prerequisites

Install OpenCL ICD loader development packages:

* **Fedora / RHEL:**
  ```bash
  sudo dnf install ocl-icd-devel opencl-headers
  ```
* **Ubuntu / Debian:**
  ```bash
  sudo apt-get install opencl-headers ocl-icd-opencl-dev
  ```
* **Arch Linux:**
  ```bash
  sudo pacman -S opencl-headers ocl-icd
  ```

### Build Steps

```bash
mkdir -p build && cd build
cmake ..
make -j16
```

---

## 4. Command-Line Usage

```bash
# List all available backends and their runtime availability
./gpubench --list-backends

# List all OpenCL devices detected on the system
./gpubench --list-devices --backend opencl

# Run all benchmarks on OpenCL backend (automatic fallback for unsupported suites)
./gpubench --backend opencl

# Run specific compute and memory benchmarks on OpenCL device 0
./gpubench -k opencl -d 0 -b FP32,FP16,FP64,INT8,Memory

# Export machine-readable results to JSON
./gpubench -k opencl --output json --output-file results_opencl.json
```
