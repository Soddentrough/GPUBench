# Version Requirements

This document specifies the minimum version requirements for all GPU compute backends used in GPUBench.

## Summary

- **ROCm/HIP**: 6.4+
- **OpenCL**: 1.2+
- **Vulkan**: 1.4+

## Details

### ROCm/HIP Requirements

**Minimum Version**: 6.4+

All HIP kernels in the `hip_kernels/` directory require ROCm 6.4 or later. This version provides:
- Improved compiler optimizations
- Enhanced support for modern AMD GPU architectures
- Better stability and performance

**Affected Files**:
- `hip_kernels/*.hip` (all HIP kernel files)

### OpenCL Requirements

**Minimum Version**: 1.2+

All OpenCL kernels in the `kernels/` directory require OpenCL 1.2 or later. This version provides:
- Support for `fma` (fused multiply-add) operations
- Vector operations for FP16, FP32, and FP64
- Required extensions for half-precision (`cl_khr_fp16`) and double-precision (`cl_khr_fp64`) floating point

**Affected Files**:
- `kernels/*.cl` (all OpenCL kernel files)

### Vulkan Requirements

**Minimum Version**: 1.4+

All Vulkan shaders in the `shaders/` directory target GLSL version 460, which corresponds to Vulkan 1.4. This provides:
- Support for extended arithmetic types (8-bit, 16-bit storage)
- Enhanced shader capabilities
- Better optimization opportunities

**Affected Files**:
- `shaders/*.comp` (all Vulkan compute shaders)

## Version Verification

Each kernel/shader file includes a comment at the top indicating its version requirement:

- **HIP kernels**: `// Requires ROCm 6.4+`
- **OpenCL kernels**: `// Requires OpenCL 1.2+`  
- **Vulkan shaders**: `#version 460` with `// Requires Vulkan 1.4+`

## Compatibility Notes

### ROCm/HIP
- Compatible with AMD Radeon RX 6000 series and newer (RDNA 2+)
- Compatible with AMD Instinct MI series GPUs
- Requires appropriate ROCm drivers and runtime installed

### OpenCL
- Most modern GPUs support OpenCL 1.2 or later
- Verify driver support for required extensions (`cl_khr_fp16`, `cl_khr_fp64`)

### Vulkan
- Vulkan 1.4 is widely supported on modern GPUs (2020+)
- Check for extension support: `VK_EXT_shader_16bit_storage`, `VK_EXT_shader_8bit_storage`, `VK_EXT_shader_float64`

## Build Requirements

When building GPUBench, ensure your development environment meets these minimum versions:
- ROCm SDK 6.4 or later (for HIP backend)
- OpenCL SDK 1.2 or later (for OpenCL backend)
- Vulkan SDK 1.4 or later (for Vulkan backend)

## Runtime Requirements

At runtime, the system must have:
- Compatible GPU hardware
- Appropriate drivers supporting the minimum versions
- Required runtime libraries installed
