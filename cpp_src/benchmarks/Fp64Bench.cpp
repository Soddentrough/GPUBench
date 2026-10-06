#include "benchmarks/Fp64Bench.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

bool Fp64Bench::IsSupported(const DeviceInfo &info,
                            IComputeContext *context) const {
  return info.fp64Support;
}

void Fp64Bench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;

  // Create storage buffer
  size_t bufferSize =
      8192 * 64 * sizeof(double); // 8192 workgroups * 64 threads * 8 bytes
  buffer = context.createBuffer(bufferSize);

  // Initialize buffer
  std::vector<double> initData(bufferSize / sizeof(double), 0.0);
  context.writeBuffer(buffer, 0, bufferSize, initData.data());

  // Create kernel
  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path kernel_file_path;
  std::string kernel_name;

  if (context.getBackend() == ComputeBackend::ROCm) {
    kernel_file_path = kdir / "rocm" / "fp64.hip";
    kernel_name = "run_benchmark";
  } else if (context.getBackend() == ComputeBackend::OpenCL) {
    kernel_file_path = kdir / "opencl" / "fp64.cl";
    kernel_name = "run_benchmark";
  } else { // Default to Vulkan
    kernel_file_path = kdir / "vulkan" / "fp64.comp";
    kernel_name = "main";
  }
  kernel = context.createKernel(kernel_file_path.string(), kernel_name, 1);
  context.setKernelArg(kernel, 0, buffer);
}

void Fp64Bench::Run(uint32_t config_idx) {
  context->dispatch(kernel, 8192, 1, 1, 64, 1, 1);
}

void Fp64Bench::Teardown() {
  if (kernel) {
    context->releaseKernel(kernel);
    kernel = nullptr;
  }
  if (buffer) {
    context->releaseBuffer(buffer);
    buffer = nullptr;
  }
}

BenchmarkResult Fp64Bench::GetResult(uint32_t config_idx) const {
  // All backends (Vulkan, ROCm, OpenCL): 256 iters * 8 independent FMAs * 2 ops = 4096 FP64 ops per thread
  uint64_t num_threads = 8192 * 64;
  uint64_t num_ops = 4096ULL * num_threads;
  return {num_ops, 0.0};
}
