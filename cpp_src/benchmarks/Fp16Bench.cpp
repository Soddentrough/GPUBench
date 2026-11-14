#include "benchmarks/Fp16Bench.h"
#ifdef HAVE_VULKAN
#include "core/VulkanContext.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

bool Fp16Bench::IsSupported(const DeviceInfo &info,
                            IComputeContext *context) const {
  (void)context;
  return info.fp16Support;
}

void Fp16Bench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;
  tileM = 16;
  tileN = 16;
  tileK = 16;
  matrixBlockX = 32;

  // Create storage buffers (separate vector and matrix buffers prevent config aliasing)
  size_t bufferSize =
      8192 * 64 * 4 * 4; // 8MB buffer to prevent out of bounds
  buffer = context.createBuffer(bufferSize);
  matrixInputBuffer = context.createBuffer(bufferSize);
  matrixBuffer = context.createBuffer(bufferSize);

  // Initialize buffers with packed FP16 0.03125 (0x2800) so both vector and matrix loads see non-zero finite values
  std::vector<uint32_t> initData(bufferSize / sizeof(uint32_t), 0x28002800u);
  context.writeBuffer(buffer, 0, bufferSize, initData.data());
  context.writeBuffer(matrixInputBuffer, 0, bufferSize, initData.data());
  context.writeBuffer(matrixBuffer, 0, bufferSize, initData.data());

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

  DeviceInfo devInfo = context.getCurrentDeviceInfo();
  if (context.getBackend() == ComputeBackend::ROCm) {
    matrixBlockX = std::max(32u, devInfo.subgroupSize);
  }

  // Optionally load Matrix Kernel if supported
  bool try_load_matrix = false;
  if ((context.getBackend() == ComputeBackend::Vulkan ||
       context.getBackend() == ComputeBackend::ROCm) &&
      devInfo.cooperativeMatrixSupport) {
    try_load_matrix = true;
  }

  if (try_load_matrix) {
    try {
      if (context.getBackend() == ComputeBackend::Vulkan) {
#ifdef HAVE_VULKAN
        auto *vc = static_cast<VulkanContext *>(&context);
        auto tile = vc->selectCoopMatrixTile(VK_COMPONENT_TYPE_FLOAT16_KHR,
                                             VK_COMPONENT_TYPE_FLOAT32_KHR);
        if (tile.supported) {
          tileM = tile.M;
          tileN = tile.N;
          tileK = tile.K;
          matrixBlockX = tile.subgroupSize;
        }
#endif
        matrixKernel = context.createKernelWithSpecConstants(
            matrix_file.string(), "main", 2,
            {tileM, tileN, tileK, matrixBlockX});
        if (matrixKernel) {
          context.setKernelArg(matrixKernel, 0, matrixInputBuffer);
          context.setKernelArg(matrixKernel, 1, matrixBuffer);
        }
      } else {
        matrixKernel =
            context.createKernel(matrix_file.string(), "run_benchmark", 1);
        if (matrixKernel) {
          context.setKernelArg(matrixKernel, 0, matrixBuffer);
        }
      }
    } catch (...) {
      matrixKernel = nullptr;
    }
  }
}

void Fp16Bench::Run(uint32_t config_idx) {
  if (config_idx == 0) {
    if (vectorKernel) {
      context->dispatch(vectorKernel, ScaledGroups(8192), 1, 1, 64, 1, 1);
    }
  } else if (config_idx == 1 && matrixKernel) {
    // 65536 WGs (32768 x 2 2D grid <= 65535 baseline limit) of matrixBlockX threads each
    context->dispatch(matrixKernel, ScaledGroups(32768), 2, 1, matrixBlockX, 1, 1);
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
    if (matrixInputBuffer) {
      context->releaseBuffer(matrixInputBuffer);
      matrixInputBuffer = nullptr;
    }
    if (matrixBuffer) {
      context->releaseBuffer(matrixBuffer);
      matrixBuffer = nullptr;
    }
    context = nullptr;
  }
}

BenchmarkResult Fp16Bench::GetResult(uint32_t config_idx) const {
  if (config_idx == 0) {
    uint64_t iters = 65536;
    uint64_t ops_per_iter = 128;
    if (context) {
      if (context->getBackend() == ComputeBackend::ROCm ||
          context->getBackend() == ComputeBackend::OpenCL) {
        iters = 16384;
        ops_per_iter = 128;
      }
    }
    uint64_t num_ops = iters * ops_per_iter * 8192 * 64;
    return {ScaledOps(num_ops, 8192), 0.0};
  } else {
    uint64_t tile_ops = 2ULL * tileM * tileN * tileK;
    uint64_t num_ops = 65536ULL * 32768ULL * tile_ops;
    return {ScaledOps(num_ops, 32768), 0.0};
  }
}

bool Fp16Bench::ValidateResults(uint32_t config_idx) const {
  if (!context || !buffer || config_idx >= 2) return false;
  if (config_idx == 1 && !matrixKernel) return false;
  if (config_idx == 0 && !vectorKernel) return false;

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

    ComputeBuffer targetBuf = (config_idx == 1 && matrixBuffer) ? matrixBuffer : buffer;
    constexpr uint16_t kUnwrittenSeed16 = 0x2800u;
    constexpr uint32_t kUnwrittenSeed32 = 0x28002800u;

    if (config_idx == 1) {
      const size_t checkCount =
          std::max<size_t>(16u, std::min<size_t>(64u, static_cast<size_t>(tileM) * tileN));
      std::vector<float> hostOut(checkCount, 0.0f);
      context->readBuffer(targetBuf, 0, hostOut.size() * sizeof(float), hostOut.data());
      for (float v : hostOut) {
        if (std::isnan(v) || std::isinf(v) || v == 0.0f) return false;
        uint32_t rawBits = 0;
        std::memcpy(&rawBits, &v, sizeof(uint32_t));
        if (rawBits == kUnwrittenSeed32) return false;
        if (context->getBackend() == ComputeBackend::ROCm) {
          // fp16_matrix.hip sums 8 elements of (8 accumulators * 4096 * 0.015625) = ~4096.0f
          if (v < 4000.0f || v > 4200.0f) return false;
        } else {
          // coop_matrix_fp16.comp stores float32 accumulator (~5.12f)
          if (v < 3.0f || v > 10.0f) return false;
        }
      }
      return true;
    }

    std::vector<uint16_t> hostHalf(64, 0);
    context->readBuffer(targetBuf, 0, hostHalf.size() * sizeof(uint16_t), hostHalf.data());
    for (uint16_t h : hostHalf) {
      if (h == kUnwrittenSeed16) return false;
      float v = fp16ToFloat(h);
      if (std::isnan(v) || std::isinf(v) || v == 0.0f) return false;
      // Vector converges to ~4.0f (Vulkan) or ~8.0f (ROCm/OpenCL)
      if (v < 3.0f || v > 10.0f) return false;
    }
    return true;
  } catch (...) {
    return false;
  }
}


