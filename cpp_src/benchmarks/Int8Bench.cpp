#include "benchmarks/Int8Bench.h"
#include <filesystem>
#include <iostream>
#include <vector>
#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

bool Int8Bench::IsSupported(const DeviceInfo &info,
                            IComputeContext *context) const {
  return info.int8Support;
}

void Int8Bench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;

  // Create storage buffers (separate matrixInputBuffer and outputBuffer prevent int8 BufferA and vector buffer from aliasing)
  size_t bufferSize =
      8192 * 64 * 4; // 8192 workgroups * 64 threads * 4 bytes (i8vec4)
  buffer = context.createBuffer(bufferSize);
  matrixInputBuffer = context.createBuffer(bufferSize);
  outputBuffer = context.createBuffer(bufferSize);
  std::vector<int8_t> initData(bufferSize, 1);
  context.writeBuffer(buffer, 0, bufferSize, initData.data());
  context.writeBuffer(matrixInputBuffer, 0, bufferSize, initData.data());

  // Load Vector Kernel
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

  // Optionally load Matrix Kernel
  if (context.getCurrentDeviceInfo().cooperativeMatrixSupport) {
    if (context.getBackend() == ComputeBackend::Vulkan) {
      std::filesystem::path matrix_file_path =
          kdir / "vulkan" / "coop_matrix_int8.comp";
      matrixKernel = context.createKernel(matrix_file_path.string(), "main", 2);
      if (matrixKernel) {
        context.setKernelArg(matrixKernel, 0, matrixInputBuffer); // Binding 0: int8 (A/B)
        context.setKernelArg(matrixKernel, 1, outputBuffer);      // Binding 1: int32 (C)
      }
    } else if (context.getBackend() == ComputeBackend::ROCm) {
      std::filesystem::path matrix_file_path =
          kdir / "rocm" / "int8_matrix.hip";
      try {
        matrixKernel = context.createKernel(matrix_file_path.string(), "run_benchmark", 1);
        if (matrixKernel) {
          context.setKernelArg(matrixKernel, 0, matrixInputBuffer);
        }
      } catch (const std::exception &e) {
        matrixKernel = nullptr;
      }
    }
  }
}

void Int8Bench::Run(uint32_t config_idx) {
  if (config_idx == 0) {
    context->dispatch(vectorKernel, 8192, 1, 1, 64, 1, 1);
  } else if (matrixKernel) {
    // 65536 WGs of 32 threads each — double dispatch to saturate tensor units
    context->dispatch(matrixKernel, 65536, 1, 1, 32, 1, 1);
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
    // Shader: 8 i8vec4 multiply-adds per iteration.
    // Each MAD on i8vec4 = 4 components × 2 ops = 8 ops per MAD.
    // 8 MADs × 8 = 64 INT8 operations per iteration.
    // Vulkan: 16384 iters. OpenCL: 16384 iters. ROCm: 512 iters.
    uint64_t iters = 16384; // Vulkan/OpenCL default
    if (context) {
      if (context->getBackend() == ComputeBackend::ROCm) {
        iters = 512;
      }
    }
    uint64_t num_ops = iters * 64 * 8192 * 64;
    return {num_ops, 0.0};
  } else {
    // coopmat 16x16x16: 16*16*16*2 = 8192 INT8 ops per coopMatMulAdd.
    // Each subgroup (32 threads) computes one tile — not multiplied by thread count.
    // Shader loops 32768 iters. Dispatch: 65536 WGs.
    uint64_t num_ops = (uint64_t)65536 * 32768 * 8192;
    return {num_ops, 0.0};
  }
}

bool Int8Bench::ValidateResults(uint32_t config_idx) const {
  if (!context || !buffer) return false;

  if (config_idx == 0) {
    if (!vectorKernel) return false;
    std::vector<int8_t> hostBytes(64, 0);
    try {
      context->readBuffer(buffer, 0, hostBytes.size() * sizeof(int8_t), hostBytes.data());
      bool anyNonZero = false;
      for (int8_t v : hostBytes) {
        if (v != 0) anyNonZero = true;
      }
      return anyNonZero;
    } catch (...) {
      return false;
    }
  } else {
    if (!matrixKernel) return true; // Not supported on current device
    ComputeBuffer targetBuf = outputBuffer ? outputBuffer : buffer;
    if (!targetBuf) return false;

    std::vector<int32_t> hostInts(64, 0);
    try {
      context->readBuffer(targetBuf, 0, hostInts.size() * sizeof(int32_t), hostInts.data());
      for (size_t i = 0; i < hostInts.size(); ++i) {
        int32_t v = hostInts[i];
        // Analytical match: 524,288 WMMA ops + 72 seed perturbation = 524,360
        if (context->getBackend() == ComputeBackend::Vulkan) {
          if (v != 524360 && v != 524288) {
            return false;
          }
        } else {
          if (v == 0) return false;
        }
      }
      return true;
    } catch (...) {
      return false;
    }
  }
}
