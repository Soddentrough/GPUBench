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

const char *LdsBankConflictBench::GetDescription(uint32_t config_idx) const {
  if (config_idx < configs.size()) {
    return configs[config_idx].hint.c_str();
  }
  return GetDescription();
}

BenchmarkExplanation LdsBankConflictBench::GetExplanation(uint32_t config_idx) const {
  if (config_idx >= configs.size()) {
    return {
      GetDescription(),
      "Measures on-chip workgroup shared memory (LDS) bank conflict penalties.",
      "32-lane shared-memory load sweep across 32 hardware banks."
    };
  }
  const uint32_t stride = configs[config_idx].stride;
  switch (stride) {
  case 1u:
    return {
      "Peak conflict-free on-chip shared memory (LDS) bandwidth when all 32 lanes hit 32 separate 4-byte banks in parallel.",
      "Baseline speed for tile-based deferred lighting, screen-space blur/SSR filters, workgroup reductions, and matrix shared-memory tiling in games.",
      "2,048 workgroups x 32 threads each execute 50,000 iterations of 8 LDS dword loads at stride 1 (128 B bank-preserving offsets)."
    };
  case 2u:
  case 4u:
  case 8u:
  case 16u:
    return {
      "Shared memory (LDS) bandwidth degradation under a " + std::to_string(stride) + "-way bank conflict, where " + std::to_string(stride) + " threads collide on the same memory bank and must serialize.",
      "Demonstrates how power-of-two array strides in shared memory (e.g., 16x16 image tiles or FFT/radix-sort tables without padding) quietly cut shader throughput by " + std::to_string(stride) + "x.",
      "Same LDS load loop with lane address stride " + std::to_string(stride) + " dwords, forcing " + std::to_string(stride) + "-way bank serialization across 32 hardware banks."
    };
  case 32u:
    return {
      "Worst-case 32-way LDS bank serialization where every thread in the wave hits bank 0 simultaneously (~3.1% of peak bandwidth).",
      "Exposes catastrophic shared-memory broadcast/collision stalls when shader indexing collapses onto a single bank.",
      "Same LDS load loop with stride 32 dwords, mapping all 32 lanes onto bank 0."
    };
  default:
    return {
      "Conflict-free odd-stride control (stride " + std::to_string(stride) + ", coprime to 32 banks) verifying full LDS throughput recovery.",
      "Proves to rendering engineers how adding +1 dword padding (odd stride) to shared-memory tile rows eliminates bank conflicts in image filters and reductions.",
      "Same LDS load loop with odd stride " + std::to_string(stride) + "; gcd(" + std::to_string(stride) + ", 32) = 1 permutes lanes across all 32 banks without collisions."
    };
  }
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

  uint32_t stride = configs[config_idx].stride;
  uint32_t iterations = kIterations;

  context->setKernelArg(kernel, 1, sizeof(uint32_t), &stride);
  context->setKernelArg(kernel, 2, sizeof(uint32_t), &iterations);
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
  if (config_idx >= configs.size() || !context || !buffer) {
    return false;
  }

  const size_t totalThreads = static_cast<size_t>(kWorkgroups) * kThreadsPerWg;
  const size_t outputBufferSize = totalThreads * sizeof(uint32_t);
  constexpr uint32_t expectedVal = kIterations * kLoadsPerIter * kLdsInitVal;

  std::vector<uint32_t> results(totalThreads, 0);
  try {
    context->readBuffer(buffer, 0, outputBufferSize, results.data());
    for (size_t i = 0; i < totalThreads; ++i) {
      if (results[i] != expectedVal) {
        std::cerr << "LdsBankConflictBench validation failed (config "
                  << config_idx << "): results[" << i << "] = "
                  << results[i] << ", expected " << expectedVal << "\n";
        return false;
      }
    }
    return true;
  } catch (const std::exception &e) {
    std::cerr << "LdsBankConflictBench validation exception (config "
              << config_idx << "): " << e.what() << "\n";
    return false;
  } catch (...) {
    return false;
  }
}
