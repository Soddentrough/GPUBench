#include "benchmarks/Int8Bench.h"
#ifdef HAVE_VULKAN
#include "core/VulkanContext.h"
#include <vulkan/vulkan.h>
#endif
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <vector>

bool Int8Bench::IsSupported(const DeviceInfo &info,
                            IComputeContext *context) const {
  (void)context;
  return info.int8Support;
}

void Int8Bench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;
  tileM = 16;
  tileN = 16;
  tileK = 16;
  matrixBlockX = 32;

  // Create storage buffers (separate matrixInputBuffer and outputBuffer prevent int8 BufferA and vector buffer from aliasing)
  size_t bufferSize =
      8192 * 64 * 4 * 2; // 4 MB buffers for safe arbitrary tile strides
  buffer = context.createBuffer(bufferSize);
  matrixInputBuffer = context.createBuffer(bufferSize);
  outputBuffer = context.createBuffer(bufferSize);
  std::vector<int8_t> vecInitData(bufferSize, 2);
  std::vector<int8_t> matInitData(bufferSize, 1);
  std::vector<int8_t> zeroData(bufferSize, 0);
  context.writeBuffer(buffer, 0, bufferSize, vecInitData.data());
  context.writeBuffer(matrixInputBuffer, 0, bufferSize, matInitData.data());
  context.writeBuffer(outputBuffer, 0, bufferSize, zeroData.data());

  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path vector_file_path;

  if (context.getBackend() == ComputeBackend::ROCm) {
    vector_file_path = kdir / "rocm" / "int8.hip";
  } else if (context.getBackend() == ComputeBackend::Vulkan) {
    vector_file_path = kdir / "vulkan" / "int8.comp";
  } else {
    vector_file_path = kdir / "opencl" / "int8.cl";
  }

  std::string kernel_name = (context.getBackend() == ComputeBackend::Vulkan)
                                ? "main"
                                : "run_benchmark";
  vectorKernel =
      context.createKernel(vector_file_path.string(), kernel_name, 1);
  context.setKernelArg(vectorKernel, 0, buffer);

  DeviceInfo devInfo = context.getCurrentDeviceInfo();
  if (context.getBackend() == ComputeBackend::ROCm) {
    matrixBlockX = std::max(32u, devInfo.subgroupSize);
  }

  // Optionally load Matrix Kernel
  if (devInfo.cooperativeMatrixSupport) {
    if (context.getBackend() == ComputeBackend::Vulkan) {
      std::filesystem::path matrix_file_path =
          kdir / "vulkan" / "coop_matrix_int8.comp";
      try {
#ifdef HAVE_VULKAN
        auto *vc = static_cast<VulkanContext *>(&context);
        auto tile = vc->selectCoopMatrixTile(VK_COMPONENT_TYPE_SINT8_KHR,
                                             VK_COMPONENT_TYPE_SINT32_KHR);
        if (tile.supported) {
          tileM = tile.M;
          tileN = tile.N;
          tileK = tile.K;
          matrixBlockX = tile.subgroupSize;
        }
#endif
        matrixKernel = context.createKernelWithSpecConstants(
            matrix_file_path.string(), "main", 2,
            {tileM, tileN, tileK, matrixBlockX});
        if (matrixKernel) {
          context.setKernelArg(matrixKernel, 0, matrixInputBuffer);
          context.setKernelArg(matrixKernel, 1, outputBuffer);
        }
      } catch (...) {
        matrixKernel = nullptr;
      }
    } else if (context.getBackend() == ComputeBackend::ROCm) {
      std::filesystem::path matrix_file_path =
          kdir / "rocm" / "int8_matrix.hip";
      try {
        matrixKernel = context.createKernel(matrix_file_path.string(), "run_benchmark", 2);
        if (matrixKernel) {
          context.setKernelArg(matrixKernel, 0, matrixInputBuffer);
          context.setKernelArg(matrixKernel, 1, outputBuffer);
        }
      } catch (const std::exception &) {
        matrixKernel = nullptr;
      }
    }
  }
}

void Int8Bench::Run(uint32_t config_idx) {
  if (config_idx == 0) {
    if (vectorKernel) {
      context->dispatch(vectorKernel, ScaledGroups(8192), 1, 1, 64, 1, 1);
    }
  } else if (config_idx == 1 && matrixKernel) {
    context->dispatch(matrixKernel, ScaledGroups(32768), 2, 1, matrixBlockX, 1, 1);
  }
}

void Int8Bench::Teardown() {
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
  if (outputBuffer) {
    context->releaseBuffer(outputBuffer);
    outputBuffer = nullptr;
  }
}

BenchmarkResult Int8Bench::GetResult(uint32_t config_idx) const {
  if (config_idx == 0) {
    uint64_t iters = 16384;
    uint64_t num_ops = iters * 64 * 8192 * 64;
    return {ScaledOps(num_ops, 8192), 0.0};
  } else {
    uint64_t tile_ops = 2ULL * tileM * tileN * tileK;
    uint64_t num_ops = 65536ULL * 32768ULL * tile_ops;
    return {ScaledOps(num_ops, 32768), 0.0};
  }
}

bool Int8Bench::ValidateResults(uint32_t config_idx) const {
  if (!context || !buffer || config_idx >= 2) return false;

  if (config_idx == 0) {
    if (!vectorKernel) return false;
    std::vector<int8_t> hostBytes(64, 0);
    try {
      context->readBuffer(buffer, 0, hostBytes.size() * sizeof(int8_t), hostBytes.data());
      for (size_t i = 0; i < hostBytes.size(); i += 4) {
        int8_t v = hostBytes[i];
        if (v == 0 || v == 2 || (v & 1) == 0) return false;
      }
      return true;
    } catch (...) {
      return false;
    }
  } else {
    if (!matrixKernel) return false;
    ComputeBuffer targetBuf = outputBuffer ? outputBuffer : buffer;
    if (!targetBuf) return false;

    const size_t checkCount =
        std::max<size_t>(16u, std::min<size_t>(64u, static_cast<size_t>(tileM) * tileN));
    std::vector<int32_t> hostInts(checkCount, 0);
    try {
      context->readBuffer(targetBuf, 0, hostInts.size() * sizeof(int32_t), hostInts.data());
      int32_t expectedBase = static_cast<int32_t>(32768u * tileK);
      for (size_t i = 0; i < hostInts.size(); ++i) {
        int32_t v = hostInts[i];
        if (context->getBackend() == ComputeBackend::Vulkan) {
          if (v != expectedBase + 72 && v != expectedBase) {
            return false;
          }
        } else {
          if (v < 4194304 || v > 4250000) return false;
        }
      }
      return true;
    } catch (...) {
      return false;
    }
  }
}
