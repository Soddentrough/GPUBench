#include "benchmarks/Fp16Bench.h"
#include <filesystem>
#include <stdexcept>

bool Fp16Bench::IsSupported(const DeviceInfo &info,
                            IComputeContext *context) const {
  return info.fp16Support;
}

void Fp16Bench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;

  // Create storage buffer
  size_t bufferSize =
      8192 * 64 * 4; // 8192 workgroups * 64 threads * 4 bytes (f16vec2)
  buffer = context.createBuffer(bufferSize);

  // Initialize buffer
  std::vector<uint32_t> initData(bufferSize / sizeof(uint32_t), 0);
  context.writeBuffer(buffer, 0, bufferSize, initData.data());

  // Load Vector Kernel
  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path vector_file;
  std::filesystem::path matrix_file;

  if (context.getBackend() == ComputeBackend::ROCm) {
    vector_file = kdir / "rocm" / "fp16.hip";
    matrix_file = kdir / "rocm" / "fp16_matrix.hip";
  } else {
    vector_file = kdir / "vulkan" / "fp16.comp";
    matrix_file = kdir / "vulkan" / "coop_matrix_fp16.comp";
  }

  vectorKernel = context.createKernel(vector_file.string(), "main", 1);
  if (!vectorKernel && context.getBackend() == ComputeBackend::ROCm) {
    // HIP uses run_benchmark
    vectorKernel = context.createKernel(vector_file.string(), "run_benchmark", 1);
  }
  context.setKernelArg(vectorKernel, 0, buffer);

  // Optionally load Matrix Kernel if supported
  bool try_load_matrix = false;
  if (context.getBackend() == ComputeBackend::Vulkan && context.getCurrentDeviceInfo().cooperativeMatrixSupport) {
      try_load_matrix = true;
  } else if (context.getBackend() == ComputeBackend::ROCm) {
      // Allow ROCm to attempt to load its matrix kernel on RDNA3+
      try_load_matrix = true;
  }

  if (try_load_matrix) {
    try {
      matrixKernel = context.createKernel(matrix_file.string(), context.getBackend() == ComputeBackend::ROCm ? "run_benchmark" : "main", 1);
      if (matrixKernel) {
          context.setKernelArg(matrixKernel, 0, buffer);
      }
    } catch (...) {
      matrixKernel = nullptr;
    }
  }
}

void Fp16Bench::Run(uint32_t config_idx) {
  if (config_idx == 0) {
    context->dispatch(vectorKernel, 8192, 1, 1, 64, 1, 1);
  } else if (matrixKernel) {
    // 65536 WGs of 32 threads each — double dispatch to saturate tensor units
    context->dispatch(matrixKernel, 65536, 1, 1, 32, 1, 1);
  }
}

void Fp16Bench::Teardown() {
  if (vectorKernel) {
    context->releaseKernel(vectorKernel);
    vectorKernel = nullptr;
  }
  if (matrixKernel) {
    context->releaseKernel(matrixKernel);
    matrixKernel = nullptr;
  }
  if (buffer) {
    context->releaseBuffer(buffer);
    buffer = nullptr;
  }
}

BenchmarkResult Fp16Bench::GetResult(uint32_t config_idx) const {
  if (config_idx == 0) {
    // 32 f16vec2 FMAs per iteration = 32 * 4 = 128 FP16 ops per iteration.
    // Each f16vec2 FMA = 2 elements × (mul+add) = 4 FP16 ops.
    // Vulkan: 65536 iters. OpenCL: 16384 iters. ROCm: 2048 iters.
    uint64_t iters = 65536; // Vulkan default
    uint64_t ops_per_iter = 128; // 32 FMAs × 4 ops each
    if (context) {
      if (context->getBackend() == ComputeBackend::ROCm) {
        iters = 2048;
      } else if (context->getBackend() == ComputeBackend::OpenCL) {
        iters = 16384;
      }
    }
    // 8192 workgroups × 64 threads
    uint64_t num_ops = iters * ops_per_iter * 8192 * 64;
    return {num_ops, 0.0};
  } else {
    // coopmat 16x16x16: 16*16*16*2 = 8192 FP16 ops per coopMatMulAdd.
    // Each subgroup (32 threads) computes one tile — not multiplied by thread count.
    // Shader loops 32768 iters. Dispatch: 65536 WGs.
    uint64_t num_ops = (uint64_t)65536 * 32768 * 8192;
    return {num_ops, 0.0};
  }
}

uint32_t Fp16Bench::GetNumConfigs() const {
  return (matrixKernel != nullptr) ? 2 : 1;
}

std::string Fp16Bench::GetConfigName(uint32_t config_idx) const {
  return config_idx == 0 ? "Vector" : "Matrix";
}
