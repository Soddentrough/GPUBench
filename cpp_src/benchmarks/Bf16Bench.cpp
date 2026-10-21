#include "benchmarks/Bf16Bench.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

bool Bf16Bench::IsSupported(const DeviceInfo &info,
                            IComputeContext *context) const {
  this->lastCheckedContext = context;
  if (context && context->getBackend() == ComputeBackend::OpenCL) {
    // Standard OpenCL does not support native BFloat16 floating-point arithmetic.
    return false;
  }
  return info.bf16Support;
}

void Bf16Bench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;

  // Create storage buffer and initialize with non-zero BF16 data (0.03125 = 0x3D00)
  size_t bufferSize =
      8192 * 64 * 4 * 4; // 8MB buffer to prevent out of bounds
  buffer = context.createBuffer(bufferSize);
  matrixBuffer = context.createBuffer(bufferSize);
  std::vector<uint32_t> initData(bufferSize / sizeof(uint32_t), 0x3D003D00u);
  context.writeBuffer(buffer, 0, bufferSize, initData.data());
  context.writeBuffer(matrixBuffer, 0, bufferSize, initData.data());

  // Load Vector Kernel
  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path vector_file;
  std::filesystem::path matrix_file;

  if (context.getBackend() == ComputeBackend::ROCm) {
    vector_file = kdir / "rocm" / "bf16.hip";
    matrix_file = kdir / "rocm" / "bf16_matrix.hip";
  } else if (context.getBackend() == ComputeBackend::Vulkan) {
    vector_file = kdir / "vulkan" / "bf16.comp";
    matrix_file = kdir / "vulkan" / "coop_matrix_bf16.comp";
  } else {
    return;
  }

  DeviceInfo devInfo = context.getCurrentDeviceInfo();

  std::string vec_func_name =
      (context.getBackend() == ComputeBackend::Vulkan) ? "main" : "run_benchmark";
  bool try_load_vector = (context.getBackend() != ComputeBackend::Vulkan) ||
                         devInfo.shaderBfloat16DotProductSupport;
  if (try_load_vector) {
    try {
      vectorKernel = context.createKernel(vector_file.string(), vec_func_name, 1);
      if (vectorKernel) {
        context.setKernelArg(vectorKernel, 0, buffer);
      }
    } catch (...) {
      vectorKernel = nullptr;
    }
  }

  // Optionally load Matrix Kernel if supported
  bool try_load_matrix = false;
  if (context.getBackend() == ComputeBackend::Vulkan &&
      devInfo.cooperativeMatrixSupport && devInfo.shaderBfloat16CoopMatrixSupport) {
    try_load_matrix = true;
  } else if (context.getBackend() == ComputeBackend::ROCm) {
    if (devInfo.cooperativeMatrixSupport) {
      try_load_matrix = true;
    }
  }

  if (try_load_matrix) {
    try {
      std::string func_name = (context.getBackend() == ComputeBackend::ROCm) ? "run_benchmark" : "main";
      matrixKernel = context.createKernel(matrix_file.string(), func_name, 1);
      if (matrixKernel) {
        context.setKernelArg(matrixKernel, 0, matrixBuffer);
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
  if (matrixBuffer)
    context->releaseBuffer(matrixBuffer);
  vectorKernel = nullptr;
  matrixKernel = nullptr;
  buffer = nullptr;
  matrixBuffer = nullptr;
}

BenchmarkResult Bf16Bench::GetResult(uint32_t config_idx) const {
  if (config_idx == 0 && vectorKernel != nullptr) { // Vector
    uint64_t iters = 32768;
    uint64_t ops_per_iter = 256; // 32 bf16vec4 dot-accumulates (64 v_dot2 * 4 ops = 256 BF16 ops)
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

bool Bf16Bench::ValidateResults(uint32_t config_idx) const {
  if (!context || !buffer) return false;
  if (config_idx == 1 && !matrixKernel) return true;
  if (config_idx == 0 && !vectorKernel) return false;

  try {
    auto bf16ToFloat = [](uint16_t bf) -> float {
      uint32_t u = static_cast<uint32_t>(bf) << 16;
      float f = 0.0f;
      std::memcpy(&f, &u, sizeof(float));
      return f;
    };

    ComputeBuffer targetBuf = (config_idx == 1 && matrixBuffer) ? matrixBuffer : buffer;
    constexpr uint16_t kUnwrittenSeed16 = 0x3D00u;
    constexpr uint32_t kUnwrittenSeed32 = 0x3D003D00u;

    // ROCm Vector (config_idx == 0) writes 32-bit floats
    if (context->getBackend() == ComputeBackend::ROCm && config_idx == 0) {
      std::vector<float> hostOut(64, 0.0f);
      context->readBuffer(targetBuf, 0, hostOut.size() * sizeof(float), hostOut.data());
      for (float v : hostOut) {
        if (std::isnan(v) || std::isinf(v) || v == 0.0f) return false;

        // Verify that buffer was genuinely overwritten (not unwritten default)
        uint32_t rawBits = 0;
        std::memcpy(&rawBits, &v, sizeof(uint32_t));
        if (rawBits == kUnwrittenSeed32) return false;

        // hip_kernels/bf16.hip accumulates negative dot2 products (~ -36400.0f)
        if (std::abs(v) < 10.0f) return false;
      }
      return true;
    }

    // Vulkan (Config 0 & 1) and ROCm Matrix (Config 1): read back 256 16-bit bfloat16 words
    std::vector<uint16_t> hostBf16(256, 0);
    context->readBuffer(targetBuf, 0, hostBf16.size() * sizeof(uint16_t), hostBf16.data());

    for (size_t i = 0; i < hostBf16.size(); ++i) {
      uint16_t h = hostBf16[i];

      // 1. Verify buffer word changed from unwritten default seed (0x3D00)
      if (h == kUnwrittenSeed16) {
        return false;
      }

      float v = bf16ToFloat(h);

      // 2. Numerical sanity check
      if (std::isnan(v) || std::isinf(v) || v == 0.0f) {
        return false;
      }

      // 3. Expected transformed boundaries per workload configuration
      if (config_idx == 0) {
        if (v <= 0.0f) return false;
      } else {
        // Native BF16 coop_matrix produces v ~ 512.0f
        if (v < 10.0f || v > 2000.0f) return false;
      }
    }
    return true;
  } catch (...) {
    return false;
  }
}



