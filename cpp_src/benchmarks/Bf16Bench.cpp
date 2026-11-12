#include "benchmarks/Bf16Bench.h"
#ifdef HAVE_VULKAN
#include "core/VulkanContext.h"
#endif
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
  vectorEmulated = false;
  tileM = 16;
  tileN = 16;
  tileK = 16;
  matrixBlockX = 32;

  // Create storage buffer and initialize with non-zero BF16 data (0.03125 = 0x3D00)
  size_t bufferSize =
      8192 * 64 * 4 * 4; // 8MB buffer to prevent out of bounds
  buffer = context.createBuffer(bufferSize);
  matrixInputBuffer = context.createBuffer(bufferSize);
  matrixBuffer = context.createBuffer(bufferSize);
  std::vector<uint32_t> initData(bufferSize / sizeof(uint32_t), 0x3D003D00u);
  context.writeBuffer(buffer, 0, bufferSize, initData.data());
  context.writeBuffer(matrixInputBuffer, 0, bufferSize, initData.data());
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
  if (context.getBackend() == ComputeBackend::ROCm) {
    matrixBlockX = std::max(32u, devInfo.subgroupSize);
  }

  std::string vec_func_name =
      (context.getBackend() == ComputeBackend::Vulkan) ? "main" : "run_benchmark";
  bool try_load_vector = devInfo.shaderBfloat16DotProductSupport;
  if (try_load_vector) {
    try {
      vectorKernel = context.createKernel(vector_file.string(), vec_func_name, 1);
      if (vectorKernel) {
        context.setKernelArg(vectorKernel, 0, buffer);
#ifdef HAVE_VULKAN
        if (context.getBackend() == ComputeBackend::Vulkan) {
          auto *vc = static_cast<VulkanContext *>(&context);
          vectorEmulated = vc->isKernelBf16Emulated(vectorKernel);
        }
#endif
      }
    } catch (const std::exception &e) {
      std::cerr << "Bf16Bench Setup vectorKernel exception: " << e.what() << std::endl;
      vectorKernel = nullptr;
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
      if (context.getBackend() == ComputeBackend::Vulkan) {
#ifdef HAVE_VULKAN
        auto *vc = static_cast<VulkanContext *>(&context);
        (void)vc;
#ifdef VK_KHR_shader_bfloat16
        auto tile = vc->selectCoopMatrixTile(VK_COMPONENT_TYPE_BFLOAT16_KHR,
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

void Bf16Bench::Run(uint32_t config_idx) {
  if (config_idx == 0) {
    if (vectorKernel != nullptr) {
      context->dispatch(vectorKernel, 8192, 1, 1, 64, 1, 1);
    }
  } else if (config_idx == 1) {
    if (matrixKernel != nullptr) {
      context->dispatch(matrixKernel, 32768, 1, 1, matrixBlockX, 1, 1);
    }
  }
}

void Bf16Bench::Teardown() {
  if (context) {
    if (vectorKernel)
      context->releaseKernel(vectorKernel);
    if (matrixKernel)
      context->releaseKernel(matrixKernel);
    if (buffer)
      context->releaseBuffer(buffer);
    if (matrixInputBuffer)
      context->releaseBuffer(matrixInputBuffer);
    if (matrixBuffer)
      context->releaseBuffer(matrixBuffer);
    context = nullptr;
  }
  vectorKernel = nullptr;
  matrixKernel = nullptr;
  buffer = nullptr;
  matrixInputBuffer = nullptr;
  matrixBuffer = nullptr;
  vectorEmulated = false;
}

BenchmarkResult Bf16Bench::GetResult(uint32_t config_idx) const {
  if (config_idx == 0 && vectorKernel != nullptr) { // Vector
    uint64_t iters = 32768;
    uint64_t ops_per_iter = 256; // 64 bf16vec2 dot-accumulates * 4 ops = 256 BF16 ops
    uint64_t num_ops = iters * ops_per_iter * 8192 * 64;
    return {num_ops, 0.0};
  } else { // Matrix
    uint64_t iters = 4096;
    uint64_t muladds = iters * 8;
    uint64_t tile_ops = 2ULL * tileM * tileN * tileK;
    uint64_t num_ops = muladds * tile_ops * 32768ULL;
    return {num_ops, 0.0};
  }
}

bool Bf16Bench::ValidateResults(uint32_t config_idx) const {
  if (!context || !buffer) {
    std::cerr << "Bf16Bench validation failed: context or buffer is null" << std::endl;
    return false;
  }
  if (config_idx == 1 && !matrixKernel) return false;
  if (config_idx == 0 && !vectorKernel) {
    std::cerr << "Bf16Bench validation failed: vectorKernel is null" << std::endl;
    return false;
  }
  if (config_idx >= 2) return false;

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

        // hip_kernels/bf16.hip accumulates positive dot2 products (> 10.0f)
        if (v < 10.0f) return false;
      }
      return true;
    }

    // Vulkan Matrix (config_idx == 1) writes 32-bit float accumulators directly
    if (context->getBackend() == ComputeBackend::Vulkan && config_idx == 1) {
      const size_t checkCount =
          std::max<size_t>(16u, std::min<size_t>(128u, static_cast<size_t>(tileM) * tileN));
      std::vector<float> hostOut(checkCount, 0.0f);
      context->readBuffer(targetBuf, 0, hostOut.size() * sizeof(float), hostOut.data());
      for (size_t i = 0; i < hostOut.size(); ++i) {
        float v = hostOut[i];
        uint32_t rawBits = 0;
        std::memcpy(&rawBits, &v, sizeof(uint32_t));
        if (rawBits == kUnwrittenSeed32) {
          std::cerr << "Bf16Bench validation failed (config 1): hostOut[" << i
                    << "] is unwritten seed 0x3D003D00" << std::endl;
          return false;
        }
        if (std::isnan(v) || std::isinf(v) || v == 0.0f) {
          std::cerr << "Bf16Bench validation failed (config 1): hostOut[" << i
                    << "] = " << v << " (raw=0x" << std::hex << rawBits << std::dec << ")" << std::endl;
          return false;
        }
        if (v < 10.0f || v > 2000.0f) {
          std::cerr << "Bf16Bench validation failed (config 1): hostOut[" << i
                    << "] = " << v << " out of [10, 2000]" << std::endl;
          return false;
        }
      }
      return true;
    }

    // Vulkan Vector (Config 0) and ROCm Matrix (Config 1): read back 256 16-bit bfloat16 words
    std::vector<uint16_t> hostBf16(256, 0);
    context->readBuffer(targetBuf, 0, hostBf16.size() * sizeof(uint16_t), hostBf16.data());

    for (size_t i = 0; i < hostBf16.size(); ++i) {
      uint16_t h = hostBf16[i];

      // 1. Verify buffer word changed from unwritten default seed (0x3D00)
      if (h == kUnwrittenSeed16) {
        std::cerr << "Bf16Bench validation failed (config " << config_idx
                  << "): hostBf16[" << i << "] is unwritten seed 0x3D00" << std::endl;
        return false;
      }

      float v = bf16ToFloat(h);

      // 2. Numerical sanity check
      if (std::isnan(v) || std::isinf(v) || v == 0.0f) {
        std::cerr << "Bf16Bench validation failed (config " << config_idx
                  << "): hostBf16[" << i << "] = " << v << " (raw=0x" << std::hex << h << std::dec << ")" << std::endl;
        return false;
      }

      // 3. Expected transformed boundaries per workload configuration
      if (config_idx == 0) {
        if (v <= 0.0f) {
          std::cerr << "Bf16Bench validation failed (config 0): hostBf16[" << i << "] = " << v << " <= 0" << std::endl;
          return false;
        }
      } else {
        // Native BF16 coop_matrix produces v ~ 512.0f
        if (v < 10.0f || v > 2000.0f) {
          std::cerr << "Bf16Bench validation failed (config 1): hostBf16[" << i << "] = " << v << " out of [10, 2000]" << std::endl;
          return false;
        }
      }
    }
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Bf16Bench validation exception (config " << config_idx << "): " << e.what() << std::endl;
    return false;
  } catch (...) {
    std::cerr << "Bf16Bench validation unknown exception (config " << config_idx << ")" << std::endl;
    return false;
  }
}



