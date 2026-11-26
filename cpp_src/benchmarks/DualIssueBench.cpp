#include "benchmarks/DualIssueBench.h"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

DualIssueBench::DualIssueBench() {
  configs = {
    {
      "FP32 Serialized (1 Chain / Low ILP)",
      "dual_issue_ilp4.comp",
      "run_dual_issue_ilp4",
      16384ULL * 32ULL, // 524,288 ops/thread
      "TFLOPS",
      "Serialized FP32 dependency chain (1 FMA/cycle); prevents superscalar latency hiding and dual-issue co-execution"
    },
    {
      "FP32 Moderate ILP (8 Chains)",
      "dual_issue_ilp8.comp",
      "run_dual_issue_ilp8",
      16384ULL * 64ULL, // 1,048,576 ops/thread
      "TFLOPS",
      "Moderate ILP (8 independent FP32 chains); measures ALU latency hiding across arithmetic pipeline stages"
    },
    {
      "FP32 Peak ILP / Dual-Issue (16 Chains)",
      "dual_issue_ilp16.comp",
      "run_dual_issue_ilp16",
      16384ULL * 128ULL, // 2,097,152 ops/thread
      "TFLOPS",
      "Peak ILP saturation (16 independent FP32 chains); saturates pipeline depth and unlocks dual-issue VALUs (e.g. RDNA 3/4 VOPD)"
    },
    {
      "INT32 Serialized (1 Chain / Low ILP)",
      "dual_issue_int32_4.comp",
      "run_dual_issue_int32_4",
      16384ULL * 32ULL, // 524,288 ops/thread
      "TOPS",
      "Serialized INT32 dependency chain (1 ALU/cycle); exposes integer pipeline latency without parallel instruction chains"
    },
    {
      "INT32 Moderate ILP (8 Chains)",
      "dual_issue_int32_8.comp",
      "run_dual_issue_int32_8",
      16384ULL * 64ULL, // 1,048,576 ops/thread
      "TOPS",
      "Moderate integer ILP (8 independent INT32 chains); evaluates integer ALU latency hiding"
    },
    {
      "INT32 Peak ILP / Dual-Issue (16 Chains)",
      "dual_issue_int32.comp",
      "run_dual_issue_int32",
      16384ULL * 128ULL, // 2,097,152 ops/thread
      "TOPS",
      "Peak integer ILP (16 independent INT32 chains); reveals whether integer datapaths support dual-issue or single ALU/cycle"
    },
    {
      "Concurrent Mixed (FP32 + INT32)",
      "dual_issue_mixed.comp",
      "run_dual_issue_mixed",
      16384ULL * 192ULL, // 3,145,728 ops/thread (16384 outer iters * 4x unroll * [16 FP32 FMAs = 32 FLOPs + 16 INT32 ALU insts = 16 IOPs])
      "TOPS",
      "Concurrent FP32 + INT32 (1:1 instruction mix); evaluates simultaneous execution across float and integer ALU datapaths"
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

const char *DualIssueBench::GetDescription(uint32_t config_idx) const {
  if (config_idx < configs.size()) {
    return configs[config_idx].hint.c_str();
  }
  return GetDescription();
}

BenchmarkExplanation DualIssueBench::GetExplanation(uint32_t config_idx) const {
  switch (config_idx) {
  case 0u:
    return {
      "Serialized FP32 shader math throughput when every instruction depends on the previous result (low instruction-level parallelism).",
      "Reveals how much shader performance is lost when game shaders have tight dependency chains (ray marching, iterative loops) that starve ALU latency hiding and block dual-issue execution.",
      "Compute shader executing tightly dependent serialized FP32 FMA chains (1 independent chain per lane)."
    };
  case 1u:
    return {
      "Mid-level FP32 instruction-level parallelism (8 independent chains) showing how well the GPU hides arithmetic pipeline latency.",
      "Reflects typical well-scheduled PBR pixel and compute shaders—shows engineers how much throughput unlocks before hitting full dual-issue saturation.",
      "Compute shader interleaving 8 independent FP32 FMA accumulator chains across 16,384 loop iterations."
    };
  case 2u:
    return {
      "Peak FP32 ALU saturation and dual-issue co-execution (e.g., AMD RDNA 3/4 VOPD) when 16 independent chains are available.",
      "Shows the upper ceiling compilers can reach in unrolled lighting, post-FX, and compute kernels on dual-VALU architectures—explaining why two GPUs with similar clock/CU counts can differ in shader speed.",
      "Compute shader interleaving 16 independent FP32 FMA chains so the shader compiler can pair dual-issue instructions."
    };
  case 3u:
    return {
      "Serialized 32-bit integer (INT32) ALU throughput under a single dependent chain.",
      "Baseline for address calculation, bit-packing, hash/RNG, and BVH index math when integer operations cannot be overlapped.",
      "Compute shader executing serialized dependent INT32 ALU chains (1 chain per lane)."
    };
  case 4u:
    return {
      "Moderate INT32 instruction-level parallelism (8 independent chains) measuring integer pipeline latency hiding.",
      "Shows how well integer address math, texture coordinate hashing, and compression shaders hide ALU latency.",
      "Compute shader interleaving 8 independent INT32 ALU chains across 16,384 loop iterations."
    };
  case 5u:
    return {
      "Peak INT32 ALU throughput and whether integer units support dual-issue or only single-ALU-per-cycle execution.",
      "Important for shaders heavy on bit-manipulation, procedural noise, software rasterization, and indexing—often a hidden bottleneck vs. FP32.",
      "Compute shader interleaving 16 independent INT32 ALU chains to saturate integer execution pipes."
    };
  case 6u:
    return {
      "Simultaneous FP32 float + INT32 integer co-execution efficiency under a 1:1 mixed instruction stream.",
      "Real game shaders constantly mix FP32 shading math with INT32 buffer/texture address math—shows whether INT32 steals cycles from FP32 or runs in parallel.",
      "Compute shader executing 16 FP32 FMAs paired 1:1 with 16 INT32 ALU ops per inner unroll step."
    };
  default:
    return {
      GetDescription(config_idx),
      "Measures shader instruction-level parallelism (ILP) and dual-issue ALU utilization.",
      "Compute shader ILP sweep across FP32 and INT32 accumulator chains."
    };
  }
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

  float multiplier = 0.999f;
  if (config_idx <= 2 || config_idx == 6) {
    // Pre-scale on host across all backends so multiplier stays directly in a
    // push-constant / kernarg SGPR without VALU v_mul_f32 lowering (which blocks VOPD).
    multiplier = 0.999f * 0.00001f;
  }
  context->setKernelArg(kernels[config_idx], 1, sizeof(float), &multiplier);
  context->setKernelArg(kernels[config_idx], 2, sizeof(uint32_t), &numElements);

  context->dispatch(kernels[config_idx], ScaledGroups(8192), 1, 1, 64, 1, 1);
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
  return {ScaledOps(total_ops, 8192), 0.0};
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
      case 6: // Concurrent Mixed (FP32+INT32) -> fres in [42.34, 73.14] + int in [1, 16777215] >= 43.34f
        return v0 >= 42.0f && v0 <= 16777300.0f;
      default:
        return false;
    }
  } catch (...) {
    return false;
  }
}
