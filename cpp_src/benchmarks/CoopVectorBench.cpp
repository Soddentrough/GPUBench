#include "CoopVectorBench.h"
#include "core/VulkanContext.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

std::string CoopVectorBench::GetConfigName(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "FP16 Per-Thread MLP (Shader Baseline)";
  case 1:
    return "FP16 Per-Thread MLP (VK_NV_cooperative_vector)";
  case 2:
    return "INT8 Per-Thread MLP (Shader Baseline)";
  case 3:
    return "INT8 Per-Thread MLP (VK_NV_cooperative_vector)";
  default:
    return "";
  }
}

const char *CoopVectorBench::GetDescription(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "Evaluates a 256-layer 16x16 FP16 per-invocation neural shading MLP using standard shader packed f16vec2 arithmetic.";
  case 1:
    return "Evaluates a 256-layer 16x16 FP16 per-invocation neural shading MLP using VK_NV_cooperative_vector (coopVecMatMulAddNV).";
  case 2:
    return "Evaluates a 256-layer 16x16 INT8 quantized per-invocation neural material MLP using standard shader integer arithmetic.";
  case 3:
    return "Evaluates a 256-layer 16x16 INT8 quantized per-invocation neural material MLP using VK_NV_cooperative_vector.";
  default:
    return "Per-invocation neural shading MLP inference benchmark comparing shader baselines against VK_NV_cooperative_vector.";
  }
}

BenchmarkExplanation CoopVectorBench::GetExplanation(uint32_t config_idx) const {
  switch (config_idx) {
  case 0u:
    return {
      "Baseline per-pixel/per-ray neural network (MLP) speed using standard FP16 packed shader ALU math without Cooperative Vector hardware.",
      "Shows the cost of evaluating neural materials (BRDFs), neural radiance caches, or neural texture decompression in normal shader cores.",
      "Vulkan compute shader evaluating a 256-layer 16x16 FP16 MLP per invocation using packed f16vec2 arithmetic."
    };
  case 1u:
    return {
      "Hardware-accelerated per-thread neural shading throughput via VK_NV_cooperative_vector in FP16—no workgroup-wide matrix sync needed.",
      "Key building block for next-gen neural shaders in games (neural materials, compressed neural textures, and real-time neural radiance caching inside ray/fragment shaders).",
      "Vulkan compute shader evaluating a 256-layer 16x16 FP16 MLP per invocation using coopVecMatMulAddNV."
    };
  case 2u:
    return {
      "Baseline per-thread quantized INT8 neural MLP speed using standard integer shader ALU instructions.",
      "Reference point to measure how much faster dedicated Cooperative Vector hardware runs quantized in-shader neural networks.",
      "Vulkan compute shader evaluating a 256-layer 16x16 INT8 MLP per invocation using standard integer arithmetic."
    };
  case 3u:
    return {
      "Hardware-accelerated INT8 quantized per-thread neural shading throughput via VK_NV_cooperative_vector.",
      "Enables ultra-compact INT8 neural materials and radiance caches to run inline inside ray hit and pixel shaders at minimal frame-time cost.",
      "Vulkan compute shader evaluating a 256-layer 16x16 INT8 MLP per invocation using INT8 coopVecMatMulAddNV."
    };
  default:
    return {
      GetDescription(config_idx),
      "Measures in-shader per-invocation neural network inference for neural rendering.",
      "Vulkan compute shader executing 256-layer 16x16 MLP inference."
    };
  }
}

bool CoopVectorBench::IsSupported(const DeviceInfo &info,
                                  IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) return false;
  return info.fp16Support || info.int8Support || info.cooperativeVectorSupport;
}

bool CoopVectorBench::IsConfigSupported(uint32_t config_idx,
                                        const DeviceInfo &info,
                                        IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) return false;
  switch (config_idx) {
  case 0:
    return info.fp16Support;
  case 1:
    return info.cooperativeVectorSupport;
  case 2:
    return info.int8Support;
  case 3:
    return info.cooperativeVectorSupport;
  default:
    return false;
  }
}

IBenchmark::SupportLimitation
CoopVectorBench::GetSupportLimitation(const DeviceInfo &info,
                                      IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return SupportLimitation::kApi;
  }
  if (!info.fp16Support && !info.int8Support && !info.cooperativeVectorSupport) {
    return SupportLimitation::kHardware;
  }
  return SupportLimitation::kNone;
}

IBenchmark::SupportLimitation
CoopVectorBench::GetConfigSupportLimitation(uint32_t config_idx,
                                            const DeviceInfo &info,
                                            IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return SupportLimitation::kApi;
  }
  if ((config_idx == 1u || config_idx == 3u) && !info.cooperativeVectorSupport) {
    return SupportLimitation::kHardware;
  }
  if (config_idx == 0u && !info.fp16Support) return SupportLimitation::kHardware;
  if (config_idx == 2u && !info.int8Support) return SupportLimitation::kHardware;
  return SupportLimitation::kNone;
}

std::string CoopVectorBench::GetSupportNote(const DeviceInfo &info,
                                            IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return "Requires Vulkan 1.4";
  }
  if (!info.fp16Support && !info.int8Support && !info.cooperativeVectorSupport) {
    return "Requires shaderFloat16, shaderInt8, or VK_NV_cooperative_vector support";
  }
  return "";
}

std::string CoopVectorBench::GetConfigSupportNote(uint32_t config_idx,
                                                  const DeviceInfo &info,
                                                  IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return (config_idx == 1u || config_idx == 3u)
               ? "Requires Vulkan 1.4 with VK_NV_cooperative_vector"
               : "Requires Vulkan 1.4 compute pipeline";
  }
  if (config_idx == 0u && !info.fp16Support) {
    return "shaderFloat16 not supported by GPU / Vulkan driver";
  }
  if (config_idx == 2u && !info.int8Support) {
    return "shaderInt8 not supported by GPU / Vulkan driver";
  }
  if ((config_idx == 1u || config_idx == 3u) && !info.cooperativeVectorSupport) {
    return "VK_NV_cooperative_vector not supported by GPU / Vulkan driver";
  }
  return "";
}

void CoopVectorBench::Setup(IComputeContext &context,
                            const std::string &kernel_dir) {
  m_context = &context;
  const DeviceInfo info = context.getCurrentDeviceInfo();
  std::filesystem::path vkDir = std::filesystem::path(kernel_dir) / "vulkan";

  // FP16 16x16 weights = 256 fp16 values (0x2C00 = 0.0625f -> row sum = 1.0f)
  std::vector<uint16_t> fp16W(256, 0x2C00u);
  std::vector<uint16_t> fp16B(16, 0x2000u); // 0.0078125f
  std::vector<uint16_t> fp16Z(4096 * 16, 0u);
  m_fp16Weights = context.createBuffer(fp16W.size() * sizeof(uint16_t), fp16W.data());
  m_fp16Biases = context.createBuffer(fp16B.size() * sizeof(uint16_t), fp16B.data());
  m_fp16BaselineOut = context.createBuffer(fp16Z.size() * sizeof(uint16_t), fp16Z.data());
  m_fp16Out = context.createBuffer(fp16Z.size() * sizeof(uint16_t), fp16Z.data());

  std::vector<int8_t> int8W(256, 1);
  std::vector<int32_t> int8B(16, 4);
  std::vector<int32_t> int8Z(4096 * 16, 0);
  m_int8Weights = context.createBuffer(int8W.size() * sizeof(int8_t), int8W.data());
  m_int8Biases = context.createBuffer(int8B.size() * sizeof(int32_t), int8B.data());
  m_int8BaselineOut = context.createBuffer(int8Z.size() * sizeof(int32_t), int8Z.data());
  m_int8Out = context.createBuffer(int8Z.size() * sizeof(int32_t), int8Z.data());

  uint32_t nThreads = kNumThreads;
  uint32_t nIters = kIterations;

  auto bindFp16Kernel = [&](ComputeKernel k, ComputeBuffer outBuf) {
    context.setKernelArg(k, 0, m_fp16Weights);
    context.setKernelArg(k, 1, m_fp16Biases);
    context.setKernelArg(k, 2, outBuf);
    context.setKernelArg(k, 3, sizeof(uint32_t), &nThreads);
    context.setKernelArg(k, 4, sizeof(uint32_t), &nIters);
  };
  auto bindInt8Kernel = [&](ComputeKernel k, ComputeBuffer outBuf) {
    context.setKernelArg(k, 0, m_int8Weights);
    context.setKernelArg(k, 1, m_int8Biases);
    context.setKernelArg(k, 2, outBuf);
    context.setKernelArg(k, 3, sizeof(uint32_t), &nThreads);
    context.setKernelArg(k, 4, sizeof(uint32_t), &nIters);
  };

  if (info.fp16Support) {
    m_fp16BaselineKernel = context.createKernel(
        (vkDir / "coop_vector_fp16_baseline.comp").string(), "main", 3);
    bindFp16Kernel(m_fp16BaselineKernel, m_fp16BaselineOut);
  }
  if (info.int8Support) {
    m_int8BaselineKernel = context.createKernel(
        (vkDir / "coop_vector_int8_baseline.comp").string(), "main", 3);
    bindInt8Kernel(m_int8BaselineKernel, m_int8BaselineOut);
  }
  if (info.cooperativeVectorSupport) {
    m_fp16Kernel = context.createKernel(
        (vkDir / "coop_vector_fp16.comp").string(), "main", 3);
    bindFp16Kernel(m_fp16Kernel, m_fp16Out);
    m_int8Kernel = context.createKernel(
        (vkDir / "coop_vector_int8.comp").string(), "main", 3);
    bindInt8Kernel(m_int8Kernel, m_int8Out);
  }
}

void CoopVectorBench::Run(uint32_t config_idx) {
  if (!m_context) return;
  ComputeKernel k = nullptr;
  switch (config_idx) {
  case 0: k = m_fp16BaselineKernel; break;
  case 1: k = m_fp16Kernel; break;
  case 2: k = m_int8BaselineKernel; break;
  case 3: k = m_int8Kernel; break;
  default: return;
  }
  if (!k) return;
  m_context->dispatch(k, kNumThreads / 128u, 1, 1, 128, 1, 1);
}

BenchmarkResult CoopVectorBench::GetResult(uint32_t /*config_idx*/) const {
  // 16x16 MatMulAdd = 2 * 16 * 16 = 512 MAC operations per layer per invocation
  uint64_t ops = static_cast<uint64_t>(kNumThreads) *
                 static_cast<uint64_t>(kIterations) * 512ULL;
  return {ops, 0.0};
}

bool CoopVectorBench::ValidateResults(uint32_t config_idx) const {
  if (!m_context) return false;
  if (config_idx == 0u || config_idx == 1u) {
    ComputeKernel k = (config_idx == 0u) ? m_fp16BaselineKernel : m_fp16Kernel;
    ComputeBuffer outBuf = (config_idx == 0u) ? m_fp16BaselineOut : m_fp16Out;
    if (!k || !outBuf) return false;
    std::vector<uint16_t> out(256, 0u);
    m_context->readBuffer(outBuf, 0, out.size() * sizeof(uint16_t),
                          out.data());
    bool anyPositive = false;
    for (uint16_t h : out) {
      uint16_t expBits = (h >> 10) & 0x1Fu;
      if (expBits == 0x1Fu) return false; // NaN or Inf
      if ((h & 0x7FFFu) != 0u) anyPositive = true;
    }
    return anyPositive;
  }
  if (config_idx == 2u || config_idx == 3u) {
    ComputeKernel k = (config_idx == 2u) ? m_int8BaselineKernel : m_int8Kernel;
    ComputeBuffer outBuf = (config_idx == 2u) ? m_int8BaselineOut : m_int8Out;
    if (!k || !outBuf) return false;
    std::vector<int32_t> out(256, 0);
    m_context->readBuffer(outBuf, 0, out.size() * sizeof(int32_t),
                          out.data());
    bool anyNonZero = false;
    for (int32_t v : out) {
      if (v > 0) anyNonZero = true;
    }
    return anyNonZero;
  }
  return false;
}

void CoopVectorBench::Teardown() {
  if (m_context) {
    if (m_fp16BaselineKernel) m_context->releaseKernel(m_fp16BaselineKernel);
    if (m_fp16Kernel) m_context->releaseKernel(m_fp16Kernel);
    if (m_int8BaselineKernel) m_context->releaseKernel(m_int8BaselineKernel);
    if (m_int8Kernel) m_context->releaseKernel(m_int8Kernel);
    if (m_fp16Weights) m_context->releaseBuffer(m_fp16Weights);
    if (m_fp16Biases) m_context->releaseBuffer(m_fp16Biases);
    if (m_fp16BaselineOut) m_context->releaseBuffer(m_fp16BaselineOut);
    if (m_fp16Out) m_context->releaseBuffer(m_fp16Out);
    if (m_int8Weights) m_context->releaseBuffer(m_int8Weights);
    if (m_int8Biases) m_context->releaseBuffer(m_int8Biases);
    if (m_int8BaselineOut) m_context->releaseBuffer(m_int8BaselineOut);
    if (m_int8Out) m_context->releaseBuffer(m_int8Out);
  }
  m_fp16BaselineKernel = nullptr;
  m_fp16Kernel = nullptr;
  m_int8BaselineKernel = nullptr;
  m_int8Kernel = nullptr;
  m_fp16Weights = nullptr;
  m_fp16Biases = nullptr;
  m_fp16BaselineOut = nullptr;
  m_fp16Out = nullptr;
  m_int8Weights = nullptr;
  m_int8Biases = nullptr;
  m_int8BaselineOut = nullptr;
  m_int8Out = nullptr;
  m_context = nullptr;
}
