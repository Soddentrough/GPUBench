#include "benchmarks/Fp32Bench.h"
#include <stdexcept>

bool Fp32Bench::IsSupported(const DeviceInfo &info,
                            IComputeContext *context) const {
  // FP32 is universally supported
  return true;
}

void Fp32Bench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;

  // Create storage buffer
  numElements = 8192 * 64;
  size_t bufferSize = numElements * sizeof(float);
  buffer = context.createBuffer(bufferSize);

  // Create kernel
  std::string kernel_file;
  if (context.getBackend() == ComputeBackend::Vulkan) {
    kernel_file = kernel_dir + "/vulkan/fp32.comp";
  } else if (context.getBackend() == ComputeBackend::ROCm) {
    kernel_file = kernel_dir + "/rocm/fp32.hip";
  } else {
    kernel_file = kernel_dir + "/opencl/fp32.cl";
  }

  std::string kernel_name;
  if (context.getBackend() == ComputeBackend::Vulkan) {
    kernel_name = "main";
  } else if (context.getBackend() == ComputeBackend::ROCm) {
    kernel_name = "run_benchmark";
  } else {
    kernel_name = "run_benchmark";
  }
  kernel = context.createKernel(kernel_file, kernel_name, 3);
  context.setKernelArg(kernel, 0, buffer);
}

void Fp32Bench::Run(uint32_t config_idx) {
  // Pass multiplier as push constant / arg 1
  float multiplier = 1.0001f;
  context->setKernelArg(kernel, 1, sizeof(float), &multiplier);

  // Pass numElements as arg 2
  context->setKernelArg(kernel, 2, sizeof(uint32_t), &numElements);

  // Increase to 8192 workgroups for better GPU saturation
  context->dispatch(kernel, 8192, 1, 1, 64, 1, 1);
}

void Fp32Bench::Teardown() {
  if (kernel) {
    context->releaseKernel(kernel);
    kernel = nullptr;
  }
  if (buffer) {
    context->releaseBuffer(buffer);
    buffer = nullptr;
  }
}

BenchmarkResult Fp32Bench::GetResult(uint32_t config_idx) const {
  // 32 vec4 FMAs per iteration = 32 * 4 * 2 = 256 FP32 operations per iteration
  // 16384 iterations * 256 ops * 8192 workgroups * 64 threads
  uint64_t num_ops = (uint64_t)16384 * 256 * 8192 * 64;
  return {num_ops, 0.0};
}
