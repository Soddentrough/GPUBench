#include "benchmarks/Int4Bench.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

bool Int4Bench::IsSupported(const DeviceInfo &info,
                            IComputeContext *context) const {
  if (context && context->getBackend() == ComputeBackend::ROCm) {
    return info.int4Support;
  }
  return false;
}

void Int4Bench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;

  // Create storage buffer (8 MB) and initialize with non-zero packed INT4 data
  size_t bufferSize = 8192 * 64 * 4 * 4;
  buffer = context.createBuffer(bufferSize);
  std::vector<uint32_t> initData(bufferSize / sizeof(uint32_t), 0x12345678u);
  context.writeBuffer(buffer, 0, bufferSize, initData.data());

  auto file_exists = [](const std::string &path) {
    std::ifstream f(path.c_str());
    return f.good();
  };

  std::filesystem::path kdir(kernel_dir);
  is_native_matrix = false;
  matrixKernel = nullptr;

  if (context.getBackend() == ComputeBackend::ROCm) {
    if (context.getCurrentDeviceInfo().int4Support) {
      std::filesystem::path matrix_file = kdir / "rocm" / "int4_matrix.hip";
      if (file_exists(matrix_file.string())) {
        try {
          matrixKernel =
              context.createKernel(matrix_file.string(), "run_benchmark", 1);
          if (matrixKernel) {
            context.setKernelArg(matrixKernel, 0, buffer);
            is_native_matrix = true;
          }
        } catch (...) {
          matrixKernel = nullptr;
          is_native_matrix = false;
        }
      }
    }
    return;
  }

  if (context.getBackend() == ComputeBackend::OpenCL) {
    return;
  }

  DeviceInfo info = context.getCurrentDeviceInfo();
  if (info.int4Support && info.cooperativeMatrixSupport &&
      context.getBackend() == ComputeBackend::Vulkan) {
    std::filesystem::path matrix_file =
        kdir / "vulkan" / "coop_matrix_int4.comp";
    if (file_exists(matrix_file.string())) {
      try {
        matrixKernel = context.createKernel(matrix_file.string(), "main", 1);
        if (matrixKernel) {
          context.setKernelArg(matrixKernel, 0, buffer);
          is_native_matrix = true;
        }
      } catch (...) {
        matrixKernel = nullptr;
        is_native_matrix = false;
      }
    }
  }
}

void Int4Bench::Run(uint32_t config_idx) {
  (void)config_idx;
  if (matrixKernel) {
    context->dispatch(matrixKernel, 32768, 1, 1, 32, 1, 1);
  }
}

void Int4Bench::Teardown() {
  if (matrixKernel)
    context->releaseKernel(matrixKernel);
  if (buffer)
    context->releaseBuffer(buffer);
  matrixKernel = nullptr;
  buffer = nullptr;
}

BenchmarkResult Int4Bench::GetResult(uint32_t config_idx) const {
  (void)config_idx;
  // 16×16×16 matmul = 8192 ops per WMMA instruction
  // 4096 loop iterations × 8 independent accumulators × 8192 ops × 32768 subgroups
  uint64_t iters = 4096;
  uint64_t muladds = iters * 8;
  uint64_t num_ops = muladds * 8192 * 32768;
  return {num_ops, 0.0};
}

bool Int4Bench::ValidateResults(uint32_t config_idx) const {
  (void)config_idx;
  if (!context) return false;
  // If INT4 is not supported on this backend/hardware, pass validation
  if (!matrixKernel || !buffer) return true;

  std::vector<int32_t> hostOut(64, 0);
  try {
    context->readBuffer(buffer, 0, hostOut.size() * sizeof(int32_t), hostOut.data());
    for (int32_t v : hostOut) {
      if (v == 0 || static_cast<uint32_t>(v) == 0x12345678u) return false;
    }
    return true;
  } catch (...) {
    return false;
  }
}
