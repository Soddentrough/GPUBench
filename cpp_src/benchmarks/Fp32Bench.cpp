#include "benchmarks/Fp32Bench.h"
#include <cmath>
#include <filesystem>
#include <iostream>
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

  // Initialize buffer with test seed
  std::vector<float> initData(numElements, 1.0f);
  context.writeBuffer(buffer, 0, bufferSize, initData.data());

  // Create kernel
  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path kernel_file;

  if (context.getBackend() == ComputeBackend::ROCm) {
    kernel_file = kdir / "rocm" / "fp32.hip";
  } else if (context.getBackend() == ComputeBackend::OpenCL) {
    kernel_file = kdir / "opencl" / "fp32.cl";
  } else { // Vulkan
    kernel_file = kdir / "vulkan" / "fp32.comp";
  }

  std::string kernel_name;
  if (context.getBackend() == ComputeBackend::Vulkan) {
    kernel_name = "main";
  } else if (context.getBackend() == ComputeBackend::ROCm ||
             context.getBackend() == ComputeBackend::OpenCL) {
    kernel_name = "run_benchmark";
  }
  kernel = context.createKernel(kernel_file.string(), kernel_name, 1);
  context.setKernelArg(kernel, 0, buffer);
}

void Fp32Bench::Run(uint32_t config_idx) {
  // Pass multiplier as push constant / arg 1
  float multiplier = 0.999f;
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
  // All backends (Vulkan, OpenCL, ROCm):
  // 32 vec4/float4 FMAs × 4 components × 2 ops = 256 FP32 ops per iteration.
  // 16384 iterations × 256 ops = 4,194,304 ops per thread.
  // 8192 workgroups × 64 threads = 524,288 threads.
  // Total workload = 2,199,023,255,552 FP32 operations.
  uint64_t iters = 16384;
  uint64_t ops_per_iter = 256;
  uint64_t num_ops = iters * ops_per_iter * 8192ULL * 64ULL;
  return {num_ops, 0.0};
}

bool Fp32Bench::ValidateResults(uint32_t config_idx) const {
  if (!context || !buffer)
    return false;
  float val = 0.0f;
  try {
    context->readBuffer(buffer, 0, sizeof(float), &val);
    return !std::isnan(val) && !std::isinf(val) && val != 0.0f;
  } catch (...) {
    return false;
  }
}

