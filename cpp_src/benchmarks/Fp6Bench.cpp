#include "benchmarks/Fp6Bench.h"
#include "core/VulkanContext.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

bool Fp6Bench::IsSupported(const DeviceInfo &info,
                           IComputeContext *context) const {
  if (context && context->getBackend() == ComputeBackend::OpenCL) {
    return false;
  }
  if (context && context->getBackend() == ComputeBackend::ROCm) {
    return info.fp6Support;
  }
  if (context && context->getBackend() == ComputeBackend::Vulkan) {
    return info.fp6Support && info.cooperativeMatrixSupport;
  }
  return info.fp6Support;
}

void Fp6Bench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;
  this->matrixKernel = nullptr;

  DeviceInfo info = context.getCurrentDeviceInfo();

  // Allocate 8 MB input A/B buffer and 8 MB output C buffer
  const size_t bufferSize = 8192ULL * 64ULL * sizeof(uint32_t) * 4ULL;
  buffer = context.createBuffer(bufferSize);
  outputBuffer = context.createBuffer(bufferSize);

  // Fill input buffer with packed 6-bit E3M2 +1.0 values (0b001100 = 0x0C):
  // 16 x 6-bit values = 96 bits = repeating 3-word pattern {0x0C30C30Cu, 0xC30C30C3u, 0x30C30C30u}
  std::vector<uint32_t> initData(bufferSize / sizeof(uint32_t));
  const uint32_t kPackedFp6One[3] = {0x0C30C30Cu, 0xC30C30C3u, 0x30C30C30u};
  for (size_t i = 0; i < initData.size(); ++i) {
    initData[i] = kPackedFp6One[i % 3u];
  }
  context.writeBuffer(buffer, 0, bufferSize, initData.data());

  std::vector<uint32_t> zeroData(bufferSize / sizeof(uint32_t), 0u);
  context.writeBuffer(outputBuffer, 0, bufferSize, zeroData.data());

  auto file_exists = [](const std::string &path) {
    std::ifstream f(path.c_str());
    return f.good();
  };

  std::filesystem::path kdir(kernel_dir);

  if (context.getBackend() == ComputeBackend::ROCm) {
    matrixBlockX = 64u;
    if (info.fp6Support) {
      std::filesystem::path matrix_file = kdir / "rocm" / "fp6_matrix.hip";
      try {
        matrixKernel = context.createKernel(matrix_file.string(), "run_benchmark", 2);
        if (matrixKernel) {
          context.setKernelArg(matrixKernel, 0, buffer);
          context.setKernelArg(matrixKernel, 1, outputBuffer);
        }
      } catch (const std::exception &e) {
        std::cerr << "ROCm FP6 Matrix kernel compilation failed: " << e.what() << std::endl;
        matrixKernel = nullptr;
      }
    }
    return;
  }

  if (context.getBackend() == ComputeBackend::Vulkan &&
      info.fp6Support && info.cooperativeMatrixSupport) {
    auto *vc = dynamic_cast<VulkanContext *>(&context);
    auto tile = vc ? vc->selectCoopMatrixTile(
                         VulkanContext::kVkComponentTypeFloat6E3M2EXT,
                         VK_COMPONENT_TYPE_FLOAT32_KHR)
                   : VulkanContext::CoopMatrixTile{};
    matrixBlockX = tile.supported ? tile.subgroupSize : 32u;
    std::filesystem::path matrix_file = kdir / "vulkan" / "coop_matrix_fp6.comp";
    if (file_exists(matrix_file.string()) ||
        file_exists(matrix_file.string() + ".spv")) {
      try {
        matrixKernel = context.createKernelWithSpecConstants(
            matrix_file.string(), "main", 2, {16u, 16u, 16u, matrixBlockX});
        if (matrixKernel) {
          context.setKernelArg(matrixKernel, 0, buffer);
          context.setKernelArg(matrixKernel, 1, outputBuffer);
        }
      } catch (const std::exception &e) {
        std::cerr << "Vulkan FP6 Matrix kernel compilation failed: " << e.what() << std::endl;
        matrixKernel = nullptr;
      }
    }
  }
}

void Fp6Bench::Run(uint32_t config_idx) {
  (void)config_idx;
  if (!context || !matrixKernel) {
    return;
  }
  // 32768 x 2 = 65536 workgroups (respects Vulkan baseline maxComputeWorkGroupCount[0] <= 65535)
  context->dispatch(matrixKernel, 32768, 2, 1, matrixBlockX, 1, 1);
}

BenchmarkResult Fp6Bench::GetResult(uint32_t config_idx) const {
  (void)config_idx;
  // 65536 workgroups * 32768 16x16x16 matrix ops * 8192 ops
  const uint64_t num_ops = 65536ULL * 32768ULL * 8192ULL;
  return {num_ops, 0.0};
}

bool Fp6Bench::ValidateResults(uint32_t config_idx) const {
  if (config_idx >= GetNumConfigs()) return false;
  if (!context || !matrixKernel || !outputBuffer) return false;

  std::vector<float> hostOut(64, 0.0f);
  try {
    context->readBuffer(outputBuffer, 0, hostOut.size() * sizeof(float), hostOut.data());
    for (float v : hostOut) {
      if (std::isnan(v) || std::isinf(v) || v <= 0.0f) {
        return false;
      }
      if (context->getBackend() == ComputeBackend::Vulkan) {
        // 8 accumulators * 4096 iters * (16 * 1.0 * 1.0) = ~524,288.0f
        if (v < 500000.0f || v > 550000.0f) return false;
      } else if (context->getBackend() == ComputeBackend::ROCm) {
        // 8 accumulators * 4 elements * 512 iters * (128 * 1.0 * 1.0) = ~2,097,152.0f
        if (v < 2000000.0f || v > 2200000.0f) return false;
      }
    }
    return true;
  } catch (...) {
    return false;
  }
}

void Fp6Bench::Teardown() {
  if (context) {
    if (matrixKernel) context->releaseKernel(matrixKernel);
    if (buffer) context->releaseBuffer(buffer);
    if (outputBuffer) context->releaseBuffer(outputBuffer);
    context = nullptr;
  }
  matrixKernel = nullptr;
  buffer = nullptr;
  outputBuffer = nullptr;
}
