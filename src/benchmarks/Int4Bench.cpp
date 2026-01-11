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

  // Helper to check if file exists
  auto file_exists = [](const std::string &path) {
    std::ifstream f(path.c_str());
    return f.good();
  };

  std::filesystem::path kdir(kernel_dir);

  if (context.getBackend() == ComputeBackend::ROCm) {
    // HIP Path
    std::filesystem::path kernel_file = kdir / "rocm" / "int4.hip";
    if (file_exists(kernel_file.string())) {
      vectorKernel =
          context.createKernel(kernel_file.string(), "run_benchmark", 1);
      context.setKernelArg(vectorKernel, 0, buffer);
      is_native_vector = true;
      is_emulated = false;
    } else {
      std::cerr << "Native INT4 HIP kernel missing: " << kernel_file.string()
                << std::endl;
      is_native_vector = false; // Not supported
                                // We can return here or let it be null.
    }
    is_native_matrix = false;
    return;
  }

  // Vulkan Path
  // Use emulated vector kernel for stability on Windows
  std::filesystem::path vector_file = kdir / "vulkan" / "int4.comp";
  is_native_vector = false;
  is_emulated = true;

  try {
    // Initialize buffer to zero for stability
    std::vector<int8_t> zeros(bufferSize, 0);
    context.writeBuffer(buffer, 0, bufferSize, zeros.data());

    vectorKernel = context.createKernel(vector_file.string(), "main", 1);
    context.setKernelArg(vectorKernel, 0, buffer);

    // Pass element count (number of i8vec4) as push constant at offset 0
    uint32_t elementCount = (uint32_t)(8192 * 64);
    context.setKernelArg(vectorKernel, 1, sizeof(uint32_t), &elementCount);
  } catch (const std::exception &e) {
    std::cerr << "INT4 vector kernel failed to load: " << e.what() << std::endl;
    throw;
  }

  bool is_rdna4 =
      context.getCurrentDeviceInfo().name.find("gfx12") != std::string::npos;
  if (context.getCurrentDeviceInfo().cooperativeMatrixSupport &&
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
  if (config_idx == 0) {
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
  if (config_idx == 0) { // Vector
    // 4 i8vec4 operations per iteration, each with multiply-add + AND = 3 ops
    // per component 4 * 4 * 3 = 48 INT4-equivalent operations per iteration
    uint64_t num_ops = (uint64_t)32768 * 48 * 8192 * 64;
    return {num_ops, 0.0};
  } else { // Matrix
    // 16x16x16 matmul = 8192 ops per iteration
    // 16384 iterations * 8192 ops * 32768 subgroups
    uint64_t num_ops = (uint64_t)16384 * 8192 * 32768;
    return {num_ops, 0.0};
  }
}

uint32_t Int4Bench::GetNumConfigs() const {
  return (matrixKernel != nullptr) ? 2 : 1;
}

std::string Int4Bench::GetConfigName(uint32_t config_idx) const {
  return (config_idx == 0) ? "Vector" : "Matrix";
}
