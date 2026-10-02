#include "benchmarks/DualIssueBench.h"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

DualIssueBench::DualIssueBench() {
  configs = {
    {
      "Standard FP32",
      "dual_issue_ilp4.comp",
      "run_dual_issue_ilp4",
      16384ULL * 32ULL, // 524,288 ops/thread
      "TFLOPS",
      "Standard single-issue FP32 baseline (1 instruction per cycle)"
    },
    {
      "Dual-Issue FP32 (Partial Co-Issue)",
      "dual_issue_ilp8.comp",
      "run_dual_issue_ilp8",
      16384ULL * 64ULL, // 1,048,576 ops/thread
      "TFLOPS",
      "Partial dual-issue FP32 (transition threshold to 2 instructions per cycle)"
    },
    {
      "Dual-Issue FP32 (FP32+FP32)",
      "dual_issue_ilp16.comp",
      "run_dual_issue_ilp16",
      16384ULL * 128ULL, // 2,097,152 ops/thread
      "TFLOPS",
      "Full dual-issue FP32 (2 FP32 instructions per cycle peak)"
    },
    {
      "Standard INT32",
      "dual_issue_int32_4.comp",
      "run_dual_issue_int32_4",
      16384ULL * 32ULL, // 524,288 ops/thread
      "TOPS",
      "Standard single-issue INT32 baseline (1 instruction per cycle)"
    },
    {
      "Dual-Issue INT32 (Partial Co-Issue)",
      "dual_issue_int32_8.comp",
      "run_dual_issue_int32_8",
      16384ULL * 64ULL, // 1,048,576 ops/thread
      "TOPS",
      "Partial dual-issue INT32 (transition threshold to 2 instructions per cycle)"
    },
    {
      "Dual-Issue INT32 (INT32+INT32)",
      "dual_issue_int32.comp",
      "run_dual_issue_int32",
      16384ULL * 128ULL, // 2,097,152 ops/thread
      "TOPS",
      "Full dual-issue INT32 (2 INT32 instructions per cycle peak)"
    },
    {
      "Dual-Issue Mixed (FP32+INT32)",
      "dual_issue_mixed.comp",
      "run_dual_issue_mixed",
      16384ULL * 128ULL, // 2,097,152 ops/thread (64 FP32 + 64 INT32)
      "TOPS",
      "Dual-issue concurrent execution (1 FP32 + 1 INT32 instruction per cycle)"
    }
  };
}

bool DualIssueBench::IsSupported(const DeviceInfo &info,
                                 IComputeContext *context) const {
  return true;
}

const char *DualIssueBench::GetMetric(uint32_t config_idx) const {
  if (config_idx < configs.size()) {
    return configs[config_idx].metric.c_str();
  }
  return "TFLOPS";
}

std::string DualIssueBench::GetConfigName(uint32_t config_idx) const {
  if (config_idx < configs.size()) {
    return configs[config_idx].name;
  }
  return "";
}

std::string DualIssueBench::GetConfigSupportNote(uint32_t config_idx) const {
  if (config_idx < configs.size()) {
    return configs[config_idx].hint;
  }
  return "";
}

void DualIssueBench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;

  numElements = 8192 * 64;
  size_t bufferSize = numElements * sizeof(float);
  buffer = context.createBuffer(bufferSize);

  std::vector<float> initData(numElements, 1.0f);
  context.writeBuffer(buffer, 0, bufferSize, initData.data());

  std::filesystem::path kdir(kernel_dir);
  kernels.resize(configs.size(), nullptr);

  for (size_t i = 0; i < configs.size(); ++i) {
    std::filesystem::path kernel_file;
    std::string kernel_func;

    if (context.getBackend() == ComputeBackend::ROCm) {
      kernel_file = kdir / "rocm" / "dual_issue.hip";
      kernel_func = configs[i].kernelName;
    } else if (context.getBackend() == ComputeBackend::OpenCL) {
      kernel_file = kdir / "opencl" / "dual_issue.cl";
      kernel_func = configs[i].kernelName;
    } else { // Vulkan
      kernel_file = kdir / "vulkan" / configs[i].vulkanShader;
      kernel_func = "main";
    }

    kernels[i] = context.createKernel(kernel_file.string(), kernel_func, 1);
    context.setKernelArg(kernels[i], 0, buffer);
  }
}

void DualIssueBench::Run(uint32_t config_idx) {
  if (config_idx >= kernels.size() || !kernels[config_idx]) {
    throw std::runtime_error("DualIssueBench: kernel not loaded for config " +
                             std::to_string(config_idx));
  }

  float multiplier = 1.0f;
  context->setKernelArg(kernels[config_idx], 1, sizeof(float), &multiplier);
  context->setKernelArg(kernels[config_idx], 2, sizeof(uint32_t), &numElements);

  context->dispatch(kernels[config_idx], 8192, 1, 1, 64, 1, 1);
}

void DualIssueBench::Teardown() {
  for (auto &k : kernels) {
    if (k) {
      context->releaseKernel(k);
      k = nullptr;
    }
  }
  kernels.clear();
  if (buffer) {
    context->releaseBuffer(buffer);
    buffer = nullptr;
  }
}

BenchmarkResult DualIssueBench::GetResult(uint32_t config_idx) const {
  if (config_idx >= configs.size()) return {0, 0.0};
  uint64_t total_ops = configs[config_idx].opsPerThread * 8192ULL * 64ULL;
  return {total_ops, 0.0};
}

bool DualIssueBench::ValidateResults(uint32_t config_idx) const {
  if (!context || !buffer) return false;
  float val = 0.0f;
  try {
    context->readBuffer(buffer, 0, sizeof(float), &val);
    return !std::isnan(val) && !std::isinf(val) && val != 0.0f;
  } catch (...) {
    return false;
  }
}
