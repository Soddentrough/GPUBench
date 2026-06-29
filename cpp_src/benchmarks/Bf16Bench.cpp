#include "benchmarks/Bf16Bench.h"
#include <filesystem>
#include <stdexcept>

bool Bf16Bench::IsSupported(const DeviceInfo &info,
                            IComputeContext *context) const {
  return info.bf16Support;
}

void Bf16Bench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;

  // Create storage buffer
  size_t bufferSize =
      8192 * 64 * 4 * 4; // 8MB buffer to prevent out of bounds
  buffer = context.createBuffer(bufferSize);

  // Load Vector Kernel
  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path vector_file;
  std::filesystem::path matrix_file;

  if (context.getBackend() == ComputeBackend::ROCm) {
    vector_file = kdir / "rocm" / "bf16.hip";
    matrix_file = kdir / "rocm" / "bf16_matrix.hip";
  } else if (context.getBackend() == ComputeBackend::OpenCL) {
    vector_file = kdir / "opencl" / "bf16.cl";
  } else {
    vector_file = kdir / "vulkan" / "bf16.comp";
    matrix_file = kdir / "vulkan" / "coop_matrix_bf16.comp";
  }

  std::string vec_func_name = (context.getBackend() == ComputeBackend::ROCm) ? "run_benchmark" : "main";
  try {
    vectorKernel = context.createKernel(vector_file.string(), vec_func_name, 1);
    context.setKernelArg(vectorKernel, 0, buffer);
  } catch (...) {
    vectorKernel = nullptr;
  }

  // Optionally load Matrix Kernel if supported
  bool try_load_matrix = false;
  if (context.getBackend() == ComputeBackend::Vulkan && context.getCurrentDeviceInfo().cooperativeMatrixSupport) {
      try_load_matrix = true;
  } else if (context.getBackend() == ComputeBackend::ROCm) {
      // Matrix WMMA is disabled on ROCm due to 7.1.1 backend crash for RDNA4.
      // If we are on RDNA3 (gfx1100), we could enable it, but for now we skip.
      if (context.getCurrentDeviceInfo().name.find("gfx11") != std::string::npos) {
          try_load_matrix = true;
      }
  }

  if (try_load_matrix) {
    try {
      std::string func_name = (context.getBackend() == ComputeBackend::ROCm) ? "run_benchmark" : "main";
      matrixKernel = context.createKernel(matrix_file.string(), func_name, 1);
      if (matrixKernel) {
          context.setKernelArg(matrixKernel, 0, buffer);
      }
    } catch (...) {
      matrixKernel = nullptr;
    }
  }
}

void Bf16Bench::Run(uint32_t config_idx) {
  if (config_idx == 0 && vectorKernel != nullptr) {
    context->dispatch(vectorKernel, 8192, 1, 1, 64, 1, 1);
  } else if (matrixKernel) {
    context->dispatch(matrixKernel, 32768, 1, 1, 32, 1, 1);
  }
}

void Bf16Bench::Teardown() {
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

BenchmarkResult Bf16Bench::GetResult(uint32_t config_idx) const {
  if (config_idx == 0 && vectorKernel != nullptr) { // Vector
    uint64_t iters = 16384;
    uint64_t ops_per_iter = 128; // Vulkan/OpenCL use f16vec2 (128 ops)
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      ops_per_iter = 64; // ROCm uses scalar hip_bfloat16 (64 ops)
    }
    uint64_t num_ops = iters * ops_per_iter * 8192 * 64;
    return {num_ops, 0.0};
  } else { // Matrix
    // 16x16x16 matrix multiply = 8192 ops
    // ROCm bf16_matrix.hip uses 32768 iterations.
    // Vulkan coop_matrix_bf16.comp uses 4096 iterations.
    uint64_t iters = 4096;
    if (context && context->getBackend() == ComputeBackend::ROCm) {
        iters = 32768;
    }
    uint64_t num_ops = iters * 8192 * 32768;
    return {num_ops, 0.0};
  }
}

uint32_t Bf16Bench::GetNumConfigs() const {
  int configs = 0;
  if (vectorKernel != nullptr) configs++;
  if (matrixKernel != nullptr) configs++;
  return configs;
}

std::string Bf16Bench::GetConfigName(uint32_t config_idx) const {
  if (config_idx == 0 && vectorKernel != nullptr) return "Vector";
  return "Matrix";
}
