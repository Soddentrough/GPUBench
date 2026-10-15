#include "benchmarks/DualIssueBench.h"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

DualIssueBench::DualIssueBench() {
  configs = {
    {
      "FP32 Baseline (Ping-Pong)",
      "dual_issue_ilp4.comp",
      "run_dual_issue_ilp4",
      16384ULL * 32ULL, // 524,288 ops/thread
      "TFLOPS",
      "Single-issue FP32 baseline (1 FMA/cycle); sequential ping-pong dependency prevents superscalar latency hiding and dual-issuing"
    },
    {
      "FP32 Moderate ILP (8 Chains)",
      "dual_issue_ilp8.comp",
      "run_dual_issue_ilp8",
      16384ULL * 64ULL, // 1,048,576 ops/thread
      "TFLOPS",
      "Moderate ILP (8 independent chains); measures latency hiding across arithmetic stages before peak saturation"
    },
    {
      "FP32 Peak ILP / Co-Issue (16 Chains)",
      "dual_issue_ilp16.comp",
      "run_dual_issue_ilp16",
      16384ULL * 128ULL, // 2,097,152 ops/thread
      "TFLOPS",
      "Peak ILP saturation (16 independent chains); saturates pipeline depth and evaluates dual-issue capacity on architectures with dual VALUs (e.g. RDNA 4 VOPD)"
    },
    {
      "INT32 Baseline (Ping-Pong)",
      "dual_issue_int32_4.comp",
      "run_dual_issue_int32_4",
      16384ULL * 32ULL, // 524,288 ops/thread
      "TOPS",
      "Single-issue integer baseline (1 ALU/cycle); sequential dependency prevents pipelined execution"
    },
    {
      "INT32 Moderate ILP (8 Chains)",
      "dual_issue_int32_8.comp",
      "run_dual_issue_int32_8",
      16384ULL * 64ULL, // 1,048,576 ops/thread
      "TOPS",
      "Moderate integer ILP (8 independent chains); evaluates integer ALU latency hiding under typical instruction-level parallelism"
    },
    {
      "INT32 Peak ILP (16 Chains)",
      "dual_issue_int32.comp",
      "run_dual_issue_int32",
      16384ULL * 128ULL, // 2,097,152 ops/thread
      "TOPS",
      "Peak integer ILP (16 independent chains); reveals whether integer datapath supports dual ALUs or single ALU/cycle"
    },
    {
      "Concurrent Mixed (FP32+INT32)",
      "dual_issue_mixed.comp",
      "run_dual_issue_mixed",
      16384ULL * 128ULL, // 2,097,152 ops/thread (64 FP32 + 64 INT32)
      "TOPS",
      "Concurrent FP32 + INT32; evaluates simultaneous execution across decoupled float and integer ALU datapaths"
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
  buffers.resize(configs.size(), nullptr);
  std::vector<float> sentinelData(numElements, 0.0f); // 0.0f sentinel prevents self-validating defaults
  for (size_t i = 0; i < configs.size(); ++i) {
    buffers[i] = context.createBuffer(bufferSize);
    context.writeBuffer(buffers[i], 0, bufferSize, sentinelData.data());
  }
  buffer = buffers.empty() ? nullptr : buffers[0]; // backward-compatibility for direct test harness references

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
    context.setKernelArg(kernels[i], 0, buffers[i]);
  }
}

void DualIssueBench::Run(uint32_t config_idx) {
  if (config_idx >= kernels.size() || !kernels[config_idx] ||
      config_idx >= buffers.size() || !buffers[config_idx]) {
    throw std::runtime_error("DualIssueBench: kernel/buffer not loaded for config " +
                             std::to_string(config_idx));
  }

  context->setKernelArg(kernels[config_idx], 0, buffers[config_idx]);
  float multiplier = 0.999f;
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
  for (auto &b : buffers) {
    if (b) {
      context->releaseBuffer(b);
      b = nullptr;
    }
  }
  buffers.clear();
  buffer = nullptr;
}

BenchmarkResult DualIssueBench::GetResult(uint32_t config_idx) const {
  if (config_idx >= configs.size()) return {0, 0.0};
  uint64_t total_ops = configs[config_idx].opsPerThread * 8192ULL * 64ULL;
  return {total_ops, 0.0};
}

bool DualIssueBench::ValidateResults(uint32_t config_idx) const {
  if (!context || config_idx >= configs.size() || config_idx >= buffers.size()) return false;
  ComputeBuffer targetBuf = buffers[config_idx];
  if (!targetBuf) return false;

  std::vector<float> hostOut(64, 0.0f);
  try {
    context->readBuffer(targetBuf, 0, hostOut.size() * sizeof(float), hostOut.data());

    for (float v : hostOut) {
      if (std::isnan(v) || std::isinf(v) || v <= 0.0f) {
        return false;
      }
    }

    float v0 = hostOut[0];
    switch (config_idx) {
      case 0: // FP32 Baseline (Ping-Pong) -> analytical recurrence envelope [30.1f, 60.6f] across all passes
        return v0 >= 28.0f && v0 <= 65.0f;
      case 1: // FP32 Moderate ILP (8 Chains) -> analytical recurrence closed form (8*base + 2.8)*(1+m)^65536 in [20.78f, 36.19f]
        return v0 >= 20.0f && v0 <= 38.0f;
      case 2: // FP32 Peak ILP / Co-Issue (16 Chains) -> analytical recurrence closed form (16*base + 6.0)*(1+m)^65536 in [42.34f, 73.14f]
        return v0 >= 40.0f && v0 <= 75.0f;
      case 3: // INT32 Baseline (Ping-Pong) -> 24-bit integer bitwise OR 1u in [1, 16777215], rejecting 0.0f sentinel
        return v0 >= 1.0f && v0 <= 16777215.0f;
      case 4: // INT32 Moderate ILP (8 Chains) -> 24-bit integer bitwise OR 1u in [1, 16777215], rejecting 0.0f sentinel
        return v0 >= 1.0f && v0 <= 16777215.0f;
      case 5: // INT32 Peak ILP (16 Chains) -> 24-bit integer bitwise OR 1u in [1, 16777215], rejecting 0.0f sentinel
        return v0 >= 1.0f && v0 <= 16777215.0f;
      case 6: // Concurrent Mixed (FP32+INT32) -> fres in [20.78, 36.19] + int in [1, 16777215] >= 21.78f
        return v0 >= 21.0f && v0 <= 16777255.0f;
      default:
        return false;
    }
  } catch (...) {
    return false;
  }
}
