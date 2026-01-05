
#include "benchmarks/Fp4Bench.h"
#include <stdexcept>

bool Fp4Bench::IsSupported(const DeviceInfo &info,
                           IComputeContext *context) const {
  return info.fp4Support;
}

void Fp4Bench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;

  DeviceInfo info = context.getCurrentDeviceInfo();
  is_emulated = (info.name.find("gfx942") == std::string::npos &&
                 info.name.find("gfx11") == std::string::npos);

  // Create storage buffer
  size_t bufferSize =
      8192 * 64 * 4; // 8192 workgroups * 64 threads * 4 bytes (u8vec4)
  if (context.getBackend() == ComputeBackend::OpenCL) {
    // The OpenCL kernel uses half4, which is 8 bytes per thread
    bufferSize = 8192 * 64 * 8;
  }
  buffer = context.createBuffer(bufferSize);

  // Create kernel
  std::string kernel_file;
  std::string kernel_name = is_emulated ? "fp4_emulated" : "fp4_native";
  if (context.getBackend() == ComputeBackend::Vulkan) {
    kernel_file = kernel_dir + "/vulkan/" + kernel_name + ".comp";
  } else if (context.getBackend() == ComputeBackend::ROCm) {
    kernel_file = kernel_dir + "/rocm/" + kernel_name + ".hip";
  } else {
    kernel_file = kernel_dir + "/opencl/fp4.cl";
  }

  std::string func_name;
  if (context.getBackend() == ComputeBackend::Vulkan) {
    func_name = "main";
  } else if (context.getBackend() == ComputeBackend::ROCm) {
    func_name = "run_benchmark";
  } else {
    func_name = "run_benchmark";
  }
  kernel = context.createKernel(kernel_file, func_name, 1);
  context.setKernelArg(kernel, 0, buffer);
}

void Fp4Bench::Run(uint32_t config_idx) {
  context->dispatch(kernel, 8192, 1, 1, 64, 1, 1);
}

void Fp4Bench::Teardown() {
  if (kernel) {
    context->releaseKernel(kernel);
  }
  if (buffer) {
    context->releaseBuffer(buffer);
  }
}

BenchmarkResult Fp4Bench::GetResult(uint32_t config_idx) const {
  // 8 fma operations per iteration, each is 2 ops (multiply, add)
  // 8 * 2 * 4 = 64 FP4-equivalent operations per iteration
  // 16384 iterations * 64 ops * 8192 workgroups * 64 threads
  uint64_t num_ops = (uint64_t)16384 * 64 * 8192 * 64;
  return {num_ops, 0.0};
}
