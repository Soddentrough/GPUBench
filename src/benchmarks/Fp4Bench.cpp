#include "benchmarks/Fp4Bench.h"
#include <filesystem>
#include <fstream>
#include <iostream>
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
  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path kernel_file_path;
  std::string kernel_base_name = is_emulated ? "fp4_emulated" : "fp4_native";

  if (context.getBackend() == ComputeBackend::Vulkan) {
    kernel_file_path = kdir / "vulkan" / (kernel_base_name + ".comp");
  } else if (context.getBackend() == ComputeBackend::ROCm) {
    kernel_file_path = kdir / "rocm" / (kernel_base_name + ".hip");
  } else {
    kernel_file_path = kdir / "opencl" / "fp4.cl";
  }

  auto file_exists = [](const std::string &path) {
    std::ifstream f(path.c_str());
    return f.good();
  };

  if (!file_exists(kernel_file_path.string())) {
    // Fallback mode: try emulated if native doesn't exist, or skip completely
    if (!is_emulated) {
      kernel_base_name = "fp4_emulated";
      if (context.getBackend() == ComputeBackend::Vulkan) {
        kernel_file_path = kdir / "vulkan" / (kernel_base_name + ".comp");
      } else if (context.getBackend() == ComputeBackend::ROCm) {
        kernel_file_path = kdir / "rocm" / (kernel_base_name + ".hip");
      }
      is_emulated = true;
    }
    
    // If still doesn't exist, skip
    if (!file_exists(kernel_file_path.string())) {
        std::cerr << "FP4 kernel not found. Skipping FP4 benchmark." << std::endl;
        kernel = nullptr;
        return;
    }
  }

  std::string func_name;
  if (context.getBackend() == ComputeBackend::Vulkan) {
    func_name = "main";
  } else {
    func_name = "run_benchmark";
  }
  kernel = context.createKernel(kernel_file_path.string(), func_name, 1);
  context.setKernelArg(kernel, 0, buffer);
}

void Fp4Bench::Run(uint32_t config_idx) {
  if (kernel) {
    context->dispatch(kernel, 8192, 1, 1, 64, 1, 1);
  }
}

void Fp4Bench::Teardown() {
  if (kernel) {
    context->releaseKernel(kernel);
    kernel = nullptr;
  }
  if (buffer) {
    context->releaseBuffer(buffer);
    buffer = nullptr;
  }
}

BenchmarkResult Fp4Bench::GetResult(uint32_t config_idx) const {
  if (!kernel) return {0, 0.0};
  
  // Shader (fp4_emulated.comp): 16 f16vec4 FMAs per iteration.
  // Each FMA on f16vec4 = 4 components × 2 ops (mul+add) = 8 ops per FMA.
  // 16 FMAs × 8 = 128 FP4-equivalent operations per iteration.
  // 16384 iterations × 128 ops × 8192 workgroups × 64 threads
  uint64_t num_ops = (uint64_t)16384 * 128 * 8192 * 64;
  return {num_ops, 0.0};
}

uint32_t Fp4Bench::GetNumConfigs() const {
  return kernel ? 1 : 0;
}

std::string Fp4Bench::GetConfigName(uint32_t config_idx) const {
  return is_emulated ? "Vector (Emulated)" : "Vector";
}
