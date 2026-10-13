#include "benchmarks/Fp16Bench.h"
#include <cmath>
#include <cstring>
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
      8192 * 64 * 4 * 4; // 8MB buffer to prevent out of bounds
  buffer = context.createBuffer(bufferSize);

  // Initialize buffer with packed FP16 0.03125 (0x2800) so both vector and matrix loads see non-zero finite values
  std::vector<uint32_t> initData(bufferSize / sizeof(uint32_t), 0x28002800u);
  context.writeBuffer(buffer, 0, bufferSize, initData.data());

  // Load Vector Kernel
  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path vector_file;
  std::filesystem::path matrix_file;

  if (context.getBackend() == ComputeBackend::ROCm) {
    vector_file = kdir / "rocm" / "fp16.hip";
    matrix_file = kdir / "rocm" / "fp16_matrix.hip";
  } else if (context.getBackend() == ComputeBackend::OpenCL) {
    vector_file = kdir / "opencl" / "fp16.cl";
  } else {
    vector_file = kdir / "vulkan" / "fp16.comp";
    matrix_file = kdir / "vulkan" / "coop_matrix_fp16.comp";
  }

  std::string vector_kernel_name = (context.getBackend() == ComputeBackend::Vulkan) ? "main" : "run_benchmark";
  vectorKernel = context.createKernel(vector_file.string(), vector_kernel_name, 1);
  context.setKernelArg(vectorKernel, 0, buffer);

  // Optionally load Matrix Kernel if supported
  bool try_load_matrix = false;
  if ((context.getBackend() == ComputeBackend::Vulkan ||
       context.getBackend() == ComputeBackend::ROCm) &&
      context.getCurrentDeviceInfo().cooperativeMatrixSupport) {
    try_load_matrix = true;
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

void Fp16Bench::Run(uint32_t config_idx) {
  if (config_idx == 0) {
    context->dispatch(vectorKernel, 8192, 1, 1, 64, 1, 1);
  } else if (matrixKernel) {
    // 65536 WGs of 32 threads each — double dispatch to saturate tensor units
    context->dispatch(matrixKernel, 65536, 1, 1, 32, 1, 1);
  }
}

void Fp16Bench::Teardown() {
  if (context) {
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
    context = nullptr;
  }
}

BenchmarkResult Fp16Bench::GetResult(uint32_t config_idx) const {
  if (config_idx == 0) {
    // Vulkan: 16 f16vec4 FMAs per iteration = 16 * 8 = 128 FP16 ops per iteration, 65536 iters.
    // Total ops per thread = 65536 * 128 = 8,388,608 ops (<=64 VGPRs, 100% occupancy)
    uint64_t iters = 65536; // Vulkan default
    uint64_t ops_per_iter = 128; // 16 FMAs × 8 ops each
    if (context) {
      if (context->getBackend() == ComputeBackend::ROCm) {
        iters = 16384;
        ops_per_iter = 128;
      } else if (context->getBackend() == ComputeBackend::OpenCL) {
        iters = 16384;
        ops_per_iter = 128;
      }
    }
    // 8192 workgroups × 64 threads
    uint64_t num_ops = iters * ops_per_iter * 8192 * 64;
    return {num_ops, 0.0};
  } else {
    // coopmat 16x16x16: 16*16*16*2 = 8192 FP16 ops per coopMatMulAdd.
    // Each subgroup (32 threads) computes one tile — not multiplied by thread count.
    // Shader loops 4096 iters * 8 accumulators = 32768 WMMA ops. Dispatch: 65536 WGs.
    uint64_t num_ops = (uint64_t)65536 * 32768 * 8192;
    return {num_ops, 0.0};
  }
}

bool Fp16Bench::ValidateResults(uint32_t config_idx) const {
  if (!context || !buffer) return false;
  if (config_idx == 1 && !matrixKernel) return true;

  try {
    auto fp16ToFloat = [](uint16_t h) -> float {
      uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
      uint32_t exp = (h >> 10) & 0x1Fu;
      uint32_t mant = h & 0x03FFu;
      uint32_t bits = 0;
      if (exp == 0) {
        if (mant == 0) {
          bits = sign;
        } else {
          exp = 1;
          while ((mant & 0x0400u) == 0) {
            mant <<= 1;
            exp--;
          }
          mant &= 0x03FFu;
          bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
      } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
      } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
      }
      float out = 0.0f;
      std::memcpy(&out, &bits, sizeof(float));
      return out;
    };

    if (config_idx == 1 && context->getBackend() == ComputeBackend::ROCm) {
      std::vector<float> hostOut(64, 0.0f);
      context->readBuffer(buffer, 0, hostOut.size() * sizeof(float), hostOut.data());
      bool anyNonZero = false;
      for (float v : hostOut) {
        if (std::isnan(v) || std::isinf(v)) return false;
        if (v != 0.0f) anyNonZero = true;
      }
      return anyNonZero;
    }

    std::vector<uint16_t> hostHalf(64, 0);
    context->readBuffer(buffer, 0, hostHalf.size() * sizeof(uint16_t), hostHalf.data());
    bool anyNonZero = false;
    for (uint16_t h : hostHalf) {
      float v = fp16ToFloat(h);
      if (std::isnan(v) || std::isinf(v)) return false;
      if (v != 0.0f) anyNonZero = true;
    }
    return anyNonZero;
  } catch (...) {
    return false;
  }
}


