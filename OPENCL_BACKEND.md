# OpenCL Backend Support

GPUBench now supports both Vulkan and OpenCL compute backends, providing flexibility for devices that may not support Vulkan.

## Architecture

The dual-backend architecture consists of:

1. **IComputeContext Interface** - Abstract interface for compute backends
2. **VulkanContext** - Vulkan backend implementation
3. **OpenCLContext** - OpenCL backend implementation
4. **ComputeBackendFactory** - Factory for creating and managing backends
5. **Backend Detection** - Automatic runtime detection with fallback support

## Building with OpenCL Support

### Prerequisites

You need OpenCL development files installed:

**Fedora/RHEL:**
```bash
sudo dnf install ocl-icd-devel opencl-headers
```

**Ubuntu/Debian:**
```bash
sudo apt-get install opencl-headers ocl-icd-opencl-dev
```

**Arch Linux:**
```bash
sudo pacman -S opencl-headers ocl-icd
```

### Build Process

```bash
mkdir build
cd build
cmake ..
make
```

CMake will automatically detect available backends:
- If both Vulkan and OpenCL are found, both will be compiled in
- If only Vulkan is found, only Vulkan will be available
- If only OpenCL is found, only OpenCL will be available
- If neither is found, the build will fail

## Usage

### Command Line Options

```bash
# Use automatic backend detection (prefers Vulkan, falls back to OpenCL)
./gpubench

# Explicitly use Vulkan
./gpubench --backend vulkan

# Explicitly use OpenCL
./gpubench --backend opencl

# List available devices
./gpubench --list-devices --backend vulkan
./gpubench --list-devices --backend opencl
```

### Backend Selection

The `-k` or `--backend` flag supports three values:
- `auto` (default): Automatically detect and use the best available backend
- `vulkan`: Force use of Vulkan backend
- `opencl`: Force use of OpenCL backend

## Current Implementation Status

### ✅ Completed
- [x] Compute backend abstraction layer (IComputeContext)
- [x] VulkanContext implementation with IComputeContext interface
- [x] OpenCLContext implementation
- [x] Backend factory with automatic detection and fallback
- [x] Command-line backend selection
- [x] CMake build system updates for dual-backend support
- [x] OpenCL kernel conversions (all 6 kernels: FP64, FP32, FP16, FP8, INT8, INT4)

### 🚧 Work in Progress
- [ ] OpenCL-specific benchmark implementations

Currently, benchmarks only work with the Vulkan backend. When OpenCL is selected, the program will display:
```
Note: Benchmarks are currently only implemented for Vulkan backend.
OpenCL benchmark implementations are planned for future releases.
```

### Future Work

To complete OpenCL benchmark support, the following needs to be done:

1. **Create OpenCL Benchmark Implementations**
   - Implement OpenCL versions of all 6 benchmarks
   - Handle OpenCL buffer creation and kernel execution
   - Implement timing using OpenCL events

2. **Update IBenchmark Interface**
   - Make benchmark interface backend-agnostic
   - Support both Vulkan and OpenCL execution paths

3. **BenchmarkRunner Updates**
   - Add OpenCL execution path in BenchmarkRunner
   - Implement OpenCL timing and result collection

## Kernel Conversions

All GLSL compute shaders have been converted to OpenCL C kernels:

| Benchmark | GLSL Shader | OpenCL Kernel |
|-----------|-------------|---------------|
| FP64 | `shaders/fp64.comp` | `kernels/fp64.cl` |
| FP32 | `shaders/fp32.comp` | `kernels/fp32.cl` |
| FP16 | `shaders/fp16.comp` | `kernels/fp16.cl` |
| FP8 | `shaders/fp8.comp` | `kernels/fp8.cl` |
| INT8 | `shaders/int8.comp` | `kernels/int8.cl` |
| INT4 | `shaders/int4.comp` | `kernels/int4.cl` |

OpenCL kernels use similar algorithmic approaches to the GLSL shaders but adapted for OpenCL syntax and vector types.

## Technical Details

### Backend Abstraction

The `IComputeContext` interface provides:
```cpp
virtual ComputeBackend getBackend() const = 0;
virtual const std::vector<DeviceInfo>& getDevices() const = 0;
virtual void pickDevice(uint32_t index) = 0;
virtual DeviceInfo getCurrentDeviceInfo() const = 0;

// Backend-specific accessors
virtual VkPhysicalDevice getVulkanPhysicalDevice() const;
virtual VkDevice getVulkanDevice() const;
virtual void* getVulkanContext() const;
virtual cl_platform_id getOpenCLPlatform() const;
virtual cl_device_id getOpenCLDevice() const;
virtual cl_context getOpenCLContext() const;
```

### CMake Conditional Compilation

The build system uses preprocessor definitions:
- `HAVE_VULKAN` - Defined when Vulkan is available
- `HAVE_OPENCL` - Defined when OpenCL is available

Code can conditionally compile for available backends:
```cpp
#ifdef HAVE_VULKAN
    // Vulkan-specific code
#endif

#ifdef HAVE_OPENCL
    // OpenCL-specific code
#endif
```

## Troubleshooting

### OpenCL Not Found During Build

If CMake reports "OpenCL not found" but you have OpenCL installed:

1. Check that you have both headers and libraries:
   ```bash
   ls /usr/include/CL/cl.h
   ls /usr/lib64/libOpenCL.so  # or /usr/lib/x86_64-linux-gnu/
   ```

2. You may need to manually specify OpenCL paths:
   ```bash
   cmake -DOpenCL_INCLUDE_DIR=/usr/include \
         -DOpenCL_LIBRARY=/usr/lib64/libOpenCL.so ..
   ```

### Runtime: "OpenCL backend not available"

This means the binary was not compiled with OpenCL support. Rebuild with OpenCL development files installed.

## Contributing

To add full OpenCL benchmark support:

1. Create OpenCL benchmark classes (similar to existing Vulkan benchmarks)
2. Update BenchmarkRunner to support OpenCL execution
3. Test with various OpenCL devices (AMD, NVIDIA, Intel)
4. Ensure timing accuracy with OpenCL events
5. Submit a pull request with the changes

## License

Same as GPUBench main project.
