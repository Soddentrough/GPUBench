#include "benchmarks/Fp8Bench.h"
#ifdef HAVE_VULKAN
#include "core/VulkanContext.h"
#endif
#include <algorithm>
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
  tileM = 16;
  tileN = 16;
  tileK = 16;
  matrixBlockX = 32;

  DeviceInfo info = context.getCurrentDeviceInfo();
  if (context.getBackend() == ComputeBackend::ROCm) {
    matrixBlockX = std::max(32u, info.subgroupSize);
  }

  // Create separate input A/B and output C storage buffers to prevent aliasing
  size_t bufferSize = 8192 * 64 * sizeof(float) * 2;
  buffer = context.createBuffer(bufferSize);
  outputBuffer = context.createBuffer(bufferSize);

  std::vector<uint32_t> initData(bufferSize / sizeof(uint32_t), 0x30303030u);
  context.writeBuffer(buffer, 0, bufferSize, initData.data());
  std::vector<uint32_t> zeroData(bufferSize / sizeof(uint32_t), 0u);
  context.writeBuffer(outputBuffer, 0, bufferSize, zeroData.data());

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
    is_native_matrix = false;
    return;
  }

  is_native_matrix = false;
  if (info.cooperativeMatrixSupport && info.fp8Support &&
      context.getBackend() == ComputeBackend::Vulkan) {
    std::filesystem::path matrix_file =
        kdir / "vulkan" / "coop_matrix_fp8.comp";
    if (file_exists(matrix_file.string()) ||
        file_exists(matrix_file.string() + ".spv")) {
      try {
#ifdef HAVE_VULKAN
        auto *vc = static_cast<VulkanContext *>(&context);
        (void)vc;
#ifdef VK_EXT_shader_float8
        auto tile = vc->selectCoopMatrixTile(VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT,
                                             VK_COMPONENT_TYPE_FLOAT32_KHR);
        if (tile.supported) {
          tileM = tile.M;
          tileN = tile.N;
          tileK = tile.K;
          matrixBlockX = tile.subgroupSize;
        }
#endif
#endif
        matrixKernel = context.createKernelWithSpecConstants(
            matrix_file.string(), "main", 2,
            {tileM, tileN, tileK, matrixBlockX});
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
  if (config_idx == 0 && matrixKernel != nullptr) {
    context->dispatch(matrixKernel, 32768, 2, 1, matrixBlockX, 1, 1);
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
  uint64_t wmma_per_wg = 32768ULL;
  uint64_t tile_ops = 2ULL * tileM * tileN * tileK;
  uint64_t num_ops = 65536ULL * wmma_per_wg * tile_ops;
  return {num_ops, 0.0};
}

bool Fp8Bench::ValidateResults(uint32_t config_idx) const {
  if (!context || config_idx != 0) return false;
  if (!matrixKernel) return false;

  ComputeBuffer targetBuf = outputBuffer ? outputBuffer : buffer;
  if (!targetBuf) return false;

  const size_t checkCount =
      std::max<size_t>(16u, std::min<size_t>(64u, static_cast<size_t>(tileM) * tileN));
  std::vector<float> hostOut(checkCount, 0.0f);
  try {
    context->readBuffer(targetBuf, 0, hostOut.size() * sizeof(float), hostOut.data());
    bool anyNonZero = false;
    for (float v : hostOut) {
      if (std::isnan(v) || std::isinf(v)) return false;
      if (v != 0.0f) anyNonZero = true;
      if (context->getBackend() == ComputeBackend::Vulkan) {
        if (v < 125000.0f || v > 135000.0f) {
          return false;
        }
      } else if (context->getBackend() == ComputeBackend::ROCm) {
        // RDNA 4 OCP FP8 sums to ~1,048,576; CDNA 3 FNUZ / CDNA 4 OCP MFMA sums to ~131,072..524,288
        if (v < 120000.0f || v > 1100000.0f) {
          return false;
        }
      }
    }
    return anyNonZero;
  } catch (...) {
    return false;
  }
}

