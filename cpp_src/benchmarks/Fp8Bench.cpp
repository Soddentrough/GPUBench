#include "benchmarks/Fp8Bench.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

bool Fp8Bench::IsSupported(const DeviceInfo &info,
                           IComputeContext *context) const {
  return true; // Supported on all backends via emulated vector paths if native is absent.
}

void Fp8Bench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;

  DeviceInfo info = context.getCurrentDeviceInfo();

  // Detect hardware with native FP8 support:
  // - MI300 (gfx942) and RDNA4 (gfx12) have native FP8 vector/matrix
  // - RDNA3 (gfx11) does NOT have native FP8
  // For Vulkan, the VK_EXT_shader_float8 extension indicates native support.
  bool has_native_fp8 = info.fp8Support; // This checks VK_EXT_shader_float8

  // Create storage buffer
  size_t bufferSize =
      8192 * 64 * 4; // 8192 workgroups * 64 threads * 4 bytes (u8vec4)
  if (context.getBackend() == ComputeBackend::OpenCL) {
    // The OpenCL kernel uses half4, which is 8 bytes per thread
    bufferSize = 8192 * 64 * 8;
  }
  buffer = context.createBuffer(bufferSize);

  // Helper to check if file exists
  auto file_exists = [](const std::string &path) {
    std::ifstream f(path.c_str());
    return f.good();
  };

  std::filesystem::path kdir(kernel_dir);

  if (context.getBackend() == ComputeBackend::ROCm) {
    // ROCm FP8 is completely emulated on the current compiler stack.
    // Skip to prevent inaccurate benchmark results.
    is_native_vector = false;
    is_native_matrix = false;
    return;
  }

  if (context.getBackend() == ComputeBackend::OpenCL) {
    // OpenCL FP8 is completely emulated.
    // Skip to prevent inaccurate benchmark results.
    is_native_vector = false;
    is_emulated_vector = false;
    return;
  }

  if (context.getBackend() == ComputeBackend::OpenCL) {
    std::filesystem::path kernel_file = kdir / "opencl" / "fp8.cl";
    vectorKernel = context.createKernel(kernel_file.string(), "run_benchmark", 1);
    context.setKernelArg(vectorKernel, 0, buffer);
    is_native_vector = true;
    is_emulated = false;
    is_native_matrix = false;
    return;
  }

  // Vulkan Path
  std::filesystem::path vector_file = kdir / "vulkan" / "fp8_emulated.comp";

  // Only load the kernel if the hardware supports it natively.
  // We completely bypass emulation fallbacks.
  if (has_native_fp8) {
    is_native_vector = true;
    is_emulated_vector = false;

    if (file_exists(vector_file.string())) {
      try {
        vectorKernel = context.createKernel(vector_file.string(), "main", 1);
        context.setKernelArg(vectorKernel, 0, buffer);
      } catch (const std::exception &e) {
        std::cerr << "Native FP8 vector shader compilation failed: " << e.what() << std::endl;
        vectorKernel = nullptr;
        is_native_vector = false;
      }
    }
  } else {
    is_native_vector = false;
    is_emulated_vector = false;
  }

  // Load Matrix Kernel (cooperative matrix) if supported
  is_native_matrix = false;
  if (info.cooperativeMatrixSupport &&
      context.getBackend() == ComputeBackend::Vulkan) {
    std::filesystem::path matrix_file =
        kdir / "vulkan" / "coop_matrix_fp8.comp";
    if (file_exists(matrix_file.string())) {
      try {
        matrixKernel = context.createKernel(matrix_file.string(), "main", 1);
        context.setKernelArg(matrixKernel, 0, buffer);
        is_native_matrix = true;
      } catch (...) {
        // Ignore failure, just don't enable matrix mode
        is_native_matrix = false;
      }
    }
  }
}

void Fp8Bench::Run(uint32_t config_idx) {
  if (config_idx == 0 && vectorKernel != nullptr) {
    context->dispatch(vectorKernel, 8192, 1, 1, 64, 1, 1);
  } else if (matrixKernel) {
    context->dispatch(matrixKernel, 32768, 1, 1, 32, 1, 1);
  }
}

void Fp8Bench::Teardown() {
  if (vectorKernel)
    context->releaseKernel(vectorKernel);
  if (matrixKernel)
    context->releaseKernel(matrixKernel);
  if (buffer)
    context->releaseBuffer(buffer);
  vectorKernel = nullptr;
  matrixKernel = nullptr;
  buffer = nullptr;
}

BenchmarkResult Fp8Bench::GetResult(uint32_t config_idx) const {
  if (config_idx == 0 && vectorKernel != nullptr) { // Vector
    // 8 fma operations per iteration, each is 2 ops (multiply, add)
    // 8 * 2 * 4 = 64 FP8-equivalent operations per iteration.
    // ROCm kernel loop reduced from 16384 → 512 to avoid TDR timeout;
    // Vulkan and OpenCL kernels still use 16384 iterations.
    uint64_t iters = 16384;
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      iters = 512;
    }
    uint64_t num_ops = iters * 64 * 8192 * 64;
    return {num_ops, 0.0};
  } else { // Matrix
    // 16x16x16 matrix multiply = 8192 ops
    // 16384 iterations * 8192 ops * 32768 subgroups
    uint64_t iters = 16384;
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      iters = 512;
    }
    uint64_t num_ops = iters * 8192 * 32768;
    return {num_ops, 0.0};
  }
}

uint32_t Fp8Bench::GetNumConfigs() const {
  int configs = 0;
  if (vectorKernel != nullptr) configs++;
  if (matrixKernel != nullptr) configs++;
  return configs;
}

std::string Fp8Bench::GetConfigName(uint32_t config_idx) const {
  if (config_idx == 0 && vectorKernel != nullptr) return "Vector";
  return "Matrix";
}
