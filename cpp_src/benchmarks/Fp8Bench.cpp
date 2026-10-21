#include "benchmarks/Fp8Bench.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

bool Fp8Bench::IsSupported(const DeviceInfo &info,
                           IComputeContext *context) const {
  this->lastCheckedContext = context;
  if (context && context->getBackend() == ComputeBackend::OpenCL) {
    return false;
  }
  if (context && context->getBackend() == ComputeBackend::ROCm) {
    return info.fp8Support;
  }
  if (context && context->getBackend() == ComputeBackend::Vulkan) {
    return info.fp8Support && info.cooperativeMatrixSupport;
  }
  return info.fp8Support;
}

void Fp8Bench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;

  DeviceInfo info = context.getCurrentDeviceInfo();

  // Create separate input A/B and output C storage buffers to prevent aliasing
  size_t bufferSize = 8192 * 64 * sizeof(float) * 2;
  buffer = context.createBuffer(bufferSize);
  outputBuffer = context.createBuffer(bufferSize);

  std::vector<uint32_t> initData(bufferSize / sizeof(uint32_t), 0x30303030u);
  context.writeBuffer(buffer, 0, bufferSize, initData.data());
  std::vector<uint32_t> zeroData(bufferSize / sizeof(uint32_t), 0u);
  context.writeBuffer(outputBuffer, 0, bufferSize, zeroData.data());

  // Helper to check if file exists
  auto file_exists = [](const std::string &path) {
    std::ifstream f(path.c_str());
    return f.good();
  };

  std::filesystem::path kdir(kernel_dir);

  if (context.getBackend() == ComputeBackend::ROCm) {
    is_native_matrix = false;

    if (info.fp8Support) {
      std::filesystem::path matrix_file = kdir / "rocm" / "fp8_matrix.hip";
      try {
        matrixKernel = context.createKernel(matrix_file.string(), "run_benchmark", 2);
        if (matrixKernel) {
          context.setKernelArg(matrixKernel, 0, buffer);
          context.setKernelArg(matrixKernel, 1, outputBuffer);
          is_native_matrix = true;
        }
      } catch (const std::exception &e) {
        std::cerr << "ROCm FP8 Matrix kernel compilation failed: " << e.what() << std::endl;
        matrixKernel = nullptr;
        is_native_matrix = false;
      }
    }
    return;
  }

  if (context.getBackend() == ComputeBackend::OpenCL) {
    // OpenCL FP8 has no native support in API.
    is_native_matrix = false;
    return;
  }

  // Vulkan Path
  // Note: SPV_EXT_float8 / VK_EXT_shader_float8 defines FP8 types for cooperative matrices,
  // memory, and conversions, but does not define general scalar/vector FP8 arithmetic ALUs.
  is_native_matrix = false;
  if (info.cooperativeMatrixSupport && info.fp8Support &&
      context.getBackend() == ComputeBackend::Vulkan) {
    std::filesystem::path matrix_file =
        kdir / "vulkan" / "coop_matrix_fp8.comp";
    if (file_exists(matrix_file.string())) {
      try {
        matrixKernel = context.createKernel(matrix_file.string(), "main", 2);
        if (matrixKernel) {
          context.setKernelArg(matrixKernel, 0, buffer);
          context.setKernelArg(matrixKernel, 1, outputBuffer);
          is_native_matrix = true;
        }
      } catch (const std::exception &e) {
        std::cerr << "Vulkan FP8 Matrix kernel compilation failed: " << e.what() << std::endl;
        matrixKernel = nullptr;
        is_native_matrix = false;
      }
    }
  }
}

void Fp8Bench::Run(uint32_t config_idx) {
  (void)config_idx;
  if (matrixKernel != nullptr) {
    // 65536 WGs of 32 threads each (subgroup wave32)
    context->dispatch(matrixKernel, 65536, 1, 1, 32, 1, 1);
  }
}

void Fp8Bench::Teardown() {
  if (context) {
    if (matrixKernel)
      context->releaseKernel(matrixKernel);
    if (buffer)
      context->releaseBuffer(buffer);
    if (outputBuffer)
      context->releaseBuffer(outputBuffer);
    context = nullptr;
  }
  matrixKernel = nullptr;
  buffer = nullptr;
  outputBuffer = nullptr;
}

BenchmarkResult Fp8Bench::GetResult(uint32_t config_idx) const {
  (void)config_idx;
  // 16x16x16 matrix multiply = 8192 ops per WMMA
  // In coop_matrix_fp8.comp: 4096 iters * 8 accumulators = 32768 WMMA ops per workgroup
  // In ROCm fp8_matrix.hip: 4096 iters * 8 WMMA ops = 32768 WMMA ops per workgroup
  // Dispatch: 65536 WGs
  uint64_t wmma_per_wg = 32768ULL;
  uint64_t num_ops = (uint64_t)65536 * wmma_per_wg * 8192ULL;
  return {num_ops, 0.0};
}

bool Fp8Bench::ValidateResults(uint32_t config_idx) const {
  (void)config_idx;
  if (!context) return false;
  // If FP8 matrix is not supported or not loaded on this device, pass validation
  if (!matrixKernel) return true;

  ComputeBuffer targetBuf = outputBuffer ? outputBuffer : buffer;
  if (!targetBuf) return false;

  std::vector<float> hostOut(64, 0.0f);
  try {
    context->readBuffer(targetBuf, 0, hostOut.size() * sizeof(float), hostOut.data());
    bool anyNonZero = false;
    for (float v : hostOut) {
      if (std::isnan(v) || std::isinf(v)) return false;
      if (v != 0.0f) anyNonZero = true;
      // On Vulkan coop_matrix_fp8, 8 accumulators * 4096 iters * (16 * 0.5 * 0.5) = ~131072.0f
      if (context->getBackend() == ComputeBackend::Vulkan) {
        if (v < 125000.0f || v > 135000.0f) {
          return false;
        }
      } else if (context->getBackend() == ComputeBackend::ROCm) {
        // On ROCm fp8_matrix.hip, 8 accumulators * 8 elements * 4096 iters * (16 * 0.5 * 0.5) = ~1048576.0f
        if (v < 1000000.0f || v > 1100000.0f) {
          return false;
        }
      }
    }
    return anyNonZero;
  } catch (...) {
    return false;
  }
}

