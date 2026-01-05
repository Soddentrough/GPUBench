#include "benchmarks/Fp64Bench.h"
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
      4096 * 64 * sizeof(double); // 4096 workgroups * 64 threads * 8 bytes
  buffer = context.createBuffer(bufferSize);

  // Initialize buffer
  std::vector<double> initData(bufferSize / sizeof(double), 0.0);
  context.writeBuffer(buffer, 0, bufferSize, initData.data());

  // Create kernel
  std::string kernel_file;
  if (context.getBackend() == ComputeBackend::Vulkan) {
    kernel_file = kernel_dir + "/vulkan/fp64.comp";
  } else if (context.getBackend() == ComputeBackend::ROCm) {
    kernel_file = kernel_dir + "/rocm/fp64.hip";
  } else {
    kernel_file = kernel_dir + "/opencl/fp64.cl";
  }

  std::string kernel_name;
  if (context.getBackend() == ComputeBackend::Vulkan) {
    kernel_name = "main";
  } else if (context.getBackend() == ComputeBackend::ROCm) {
    kernel_name = "run_benchmark";
  } else {
    kernel_name = "run_benchmark";
  }
  kernel = context.createKernel(kernel_file, kernel_name, 1);
  context.setKernelArg(kernel, 0, buffer);
}

void Fp64Bench::Run(uint32_t config_idx) {
  context->dispatch(kernel, 4096, 1, 1, 64, 1, 1);
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
  // 2 operations per loop iteration (FMA)
  uint64_t num_threads = 4096 * 64;
  uint64_t num_ops = (uint64_t)65536 * 2 * num_threads;
  return {num_ops, 0.0};
}
