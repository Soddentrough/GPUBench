#include "benchmarks/Int4Bench.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

bool Int4Bench::IsSupported(const DeviceInfo &info,
                            IComputeContext *context) const {
  return info.int4Support;
}

void Int4Bench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;

  // Create storage buffer
  size_t bufferSize =
      8192 * 64 * 4; // 8192 workgroups * 64 threads * 4 bytes (i8vec4)
  buffer = context.createBuffer(bufferSize);

  auto file_exists = [](const std::string &path) {
    std::ifstream f(path.c_str());
    return f.good();
  };

  std::filesystem::path kdir(kernel_dir);

  if (context.getBackend() == ComputeBackend::ROCm) {
    // HIP INT4 is currently completely emulated. Skip to prevent inaccurate results.
    is_native_vector = false;
    is_native_matrix = false;
    return;
  }

  if (context.getBackend() == ComputeBackend::OpenCL) {
    // OpenCL INT4 is completely emulated. Skip to prevent inaccurate results.
    is_native_vector = false;
    is_native_matrix = false;
    return;
  }

  if (context.getBackend() == ComputeBackend::OpenCL) {
    std::filesystem::path kernel_file = kdir / "opencl" / "int4.cl";
    vectorKernel = context.createKernel(kernel_file.string(), "run_benchmark", 1);
    context.setKernelArg(vectorKernel, 0, buffer);
    is_native_vector = true;
    is_emulated_vector = false;
    is_native_matrix = false;
    return;
  }

  // Vulkan Path
  // INT4 vector shader uses i8vec4 with masking — this is emulated regardless
  // of hardware since there is no native INT4 vector ISA in Vulkan/SPIR-V.
  // Skip to prevent inaccurate results.
  is_native_vector = false;
  is_emulated_vector = false;
  vectorKernel = nullptr;

  DeviceInfo info = context.getCurrentDeviceInfo();

  // Cooperative Matrix path: on RDNA4, the matrix cores handle INT4 natively
  // via the cooperative matrix interface with int8_t types (HW packs/unpacks).
  bool is_rdna4 =
      info.name.find("gfx12") != std::string::npos ||
      info.name.find("GFX12") != std::string::npos ||
      info.name.find("rx 9070") != std::string::npos ||
      info.name.find("R9700") != std::string::npos ||
      info.name.find("Radeon AI") != std::string::npos;
      
  is_native_matrix = false;
  if (info.cooperativeMatrixSupport &&
      context.getBackend() == ComputeBackend::Vulkan && is_rdna4) {
    std::filesystem::path matrix_file =
        kdir / "vulkan" / "coop_matrix_int4.comp";
    if (file_exists(matrix_file.string())) {
      try {
        matrixKernel = context.createKernel(matrix_file.string(), "main", 1);
        context.setKernelArg(matrixKernel, 0, buffer);
        is_native_matrix = true;
      } catch (...) {
        is_native_matrix = false;
      }
    }
  }
}

void Int4Bench::Run(uint32_t config_idx) {
  if (config_idx == 0 && vectorKernel != nullptr) {
    context->dispatch(vectorKernel, 8192, 1, 1, 64, 1, 1);
  } else if (matrixKernel) {
    context->dispatch(matrixKernel, 32768, 1, 1, 32, 1, 1);
  }
}

void Int4Bench::Teardown() {
  if (vectorKernel)
    context->releaseKernel(vectorKernel);
  if (matrixKernel)
    context->releaseKernel(matrixKernel);
  if (buffer)
    context->releaseBuffer(buffer);
  vectorKernel = nullptr;
  matrixKernel = nullptr;
  buffer = nullptr;
}

BenchmarkResult Int4Bench::GetResult(uint32_t config_idx) const {
  if (config_idx == 0 && vectorKernel != nullptr) { // Vector
    uint64_t iters = 32768;
    uint64_t ops_per_iter = 32;
    if (context) {
      if (context->getBackend() == ComputeBackend::OpenCL) {
        iters = 16384;
        ops_per_iter = 96;
      } else if (context->getBackend() == ComputeBackend::ROCm) {
        iters = 16384;
        ops_per_iter = 96;
      }
    }
    uint64_t num_ops = (uint64_t)iters * ops_per_iter * 8192 * 64;
    return {num_ops, 0.0};
  } else { // Matrix
    // 16×16×16 matmul = 8192 ops per iteration
    // 16384 iterations × 8192 ops × 32768 subgroups
    uint64_t num_ops = (uint64_t)16384 * 8192 * 32768;
    return {num_ops, 0.0};
  }
}

uint32_t Int4Bench::GetNumConfigs() const {
  int configs = 0;
  if (vectorKernel != nullptr) configs++;
  if (matrixKernel != nullptr) configs++;
  return configs;
}

std::string Int4Bench::GetConfigName(uint32_t config_idx) const {
  if (config_idx == 0 && vectorKernel != nullptr) return "Vector";
  return "Matrix";
}
