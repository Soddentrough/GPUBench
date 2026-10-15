#include "benchmarks/Bf16Bench.h"
#include <filesystem>
#include <stdexcept>
#include <vector>

bool Bf16Bench::IsSupported(const DeviceInfo &info,
                            IComputeContext *context) const {
  this->lastCheckedContext = context;
  if (context && context->getBackend() == ComputeBackend::OpenCL) {
    // Standard OpenCL does not support native BFloat16 floating-point arithmetic.
    return false;
  }
  if (context && context->getBackend() == ComputeBackend::ROCm) {
    return info.bf16Support && info.cooperativeMatrixSupport;
  }
  return info.bf16Support;
}

void Bf16Bench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;

  // Create storage buffer and initialize with non-zero BF16 data (0.03125 = 0x3D00)
  size_t bufferSize =
      8192 * 64 * 4 * 4; // 8MB buffer to prevent out of bounds
  buffer = context.createBuffer(bufferSize);
  std::vector<uint32_t> initData(bufferSize / sizeof(uint32_t), 0x3D003D00u);
  context.writeBuffer(buffer, 0, bufferSize, initData.data());

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

  std::string vec_func_name =
      (context.getBackend() == ComputeBackend::Vulkan) ? "main" : "run_benchmark";
  try {
    vectorKernel = context.createKernel(vector_file.string(), vec_func_name, 1);
    if (vectorKernel) {
      context.setKernelArg(vectorKernel, 0, buffer);
    }
  } catch (...) {
    vectorKernel = nullptr;
  }

  // Optionally load Matrix Kernel if supported
  bool try_load_matrix = false;
  if (context.getBackend() == ComputeBackend::Vulkan && context.getCurrentDeviceInfo().cooperativeMatrixSupport) {
      try_load_matrix = true;
  } else if (context.getBackend() == ComputeBackend::ROCm) {
      if (context.getCurrentDeviceInfo().cooperativeMatrixSupport) {
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
    uint64_t iters = 32768;
    uint64_t ops_per_iter = 256; // 32 bf16vec4 dot-accumulates (64 v_dot2 * 4 ops = 256 BF16 ops)
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      iters = 16384;
      ops_per_iter = 128;
    }
    uint64_t num_ops = iters * ops_per_iter * 8192 * 64;
    return {num_ops, 0.0};
  } else { // Matrix
    // 16x16x16 matrix multiply = 8192 ops
    // Both Vulkan coop_matrix_bf16.comp and ROCm bf16_matrix.hip run
    // 4096 loop iterations x 8 independent accumulators = 32768 muladds per subgroup.
    uint64_t iters = 4096;
    uint64_t muladds = iters * 8;
    uint64_t num_ops = muladds * 8192 * 32768;
    return {num_ops, 0.0};
  }
}


