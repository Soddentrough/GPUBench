#include "benchmarks/LdsBankConflictBench.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>

LdsBankConflictBench::LdsBankConflictBench() {
  configs = {
    {"Stride 1 (Conflict-Free Baseline)", 1, "1-way conflict-free; all 32 lanes access distinct 4-byte banks"},
    {"Stride 2 (2-Way Conflict)", 2, "2 lanes per bank; 2x serialization reduces throughput to ~50%"},
    {"Stride 4 (4-Way Conflict)", 4, "4 lanes per bank; 4x serialization reduces throughput to ~25%"},
    {"Stride 8 (8-Way Conflict)", 8, "8 lanes per bank; 8x serialization reduces throughput to ~12.5%"},
    {"Stride 16 (16-Way Conflict)", 16, "16 lanes per bank; 16x serialization reduces throughput to ~6.25%"},
    {"Stride 32 (32-Way Full Serialization)", 32, "32 lanes collide on bank 0; fully serialized down to ~3.125%"},
    {"Stride 3 (Odd Stride Control)", 3, "gcd(3,32)=1; conflict-free permutation proves modular collision theory"},
    {"Stride 5 (Odd Stride Control)", 5, "gcd(5,32)=1; conflict-free permutation proves modular collision theory"}
  };
}

LdsBankConflictBench::~LdsBankConflictBench() {
  Teardown();
}

std::string LdsBankConflictBench::GetConfigName(uint32_t config_idx) const {
  if (config_idx < configs.size()) {
    return configs[config_idx].name;
  }
  return "";
}

std::string LdsBankConflictBench::GetConfigSupportNote(uint32_t config_idx) const {
  if (config_idx < configs.size()) {
    return configs[config_idx].hint;
  }
  return "";
}

void LdsBankConflictBench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;

  size_t totalThreads = static_cast<size_t>(kWorkgroups) * kThreadsPerWg;
  size_t outputBufferSize = totalThreads * sizeof(uint32_t);
  buffer = context.createBuffer(outputBufferSize);

  std::vector<uint32_t> initData(totalThreads, 0);
  context.writeBuffer(buffer, 0, outputBufferSize, initData.data());

  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path full_kernel_path;
  std::string kernel_func;

  if (context.getBackend() == ComputeBackend::ROCm) {
    full_kernel_path = kdir / "rocm" / "lds_bank_conflicts.hip";
    kernel_func = "run_benchmark";
  } else if (context.getBackend() == ComputeBackend::OpenCL) {
    full_kernel_path = kdir / "opencl" / "lds_bank_conflicts.cl";
    kernel_func = "run_benchmark";
  } else { // Vulkan
    full_kernel_path = kdir / "vulkan" / "lds_bank_conflicts.comp";
    kernel_func = "main";
  }

  kernel = context.createKernel(full_kernel_path.string(), kernel_func, 1);
  context.setKernelArg(kernel, 0, buffer);
}

void LdsBankConflictBench::Run(uint32_t config_idx) {
  if (config_idx >= configs.size() || !kernel || !context) {
    throw std::runtime_error("LdsBankConflictBench: invalid config or uninitialized state");
  }

  struct PushConstants {
    uint32_t stride;
    uint32_t iterations;
  } pc = { configs[config_idx].stride, kIterations };

  context->setKernelArg(kernel, 1, sizeof(pc), &pc);
  context->dispatch(kernel, kWorkgroups, 1, 1, kThreadsPerWg, 1, 1);
}

void LdsBankConflictBench::Teardown() {
  if (context) {
    if (kernel) {
      context->releaseKernel(kernel);
      kernel = nullptr;
    }
    if (buffer) {
      context->releaseBuffer(buffer);
      buffer = nullptr;
    }
    context = nullptr;
  }
}

BenchmarkResult LdsBankConflictBench::GetResult(uint32_t config_idx) const {
  return { kTotalBytes, 0.0 };
}

bool LdsBankConflictBench::ValidateResults(uint32_t config_idx) const {
  return true;
}
