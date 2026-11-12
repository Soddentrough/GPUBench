#include "CoopVectorBench.h"
#include "core/VulkanContext.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

std::string CoopVectorBench::GetConfigName(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "FP16 Per-Thread MLP (16x16)";
  case 1:
    return "INT8 Per-Thread MLP (16x16)";
  default:
    return "";
  }
}

const char *CoopVectorBench::GetDescription(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "Evaluates a 256-layer 16x16 FP16 per-invocation neural shading MLP using VK_NV_cooperative_vector (coopVecMatMulAddNV).";
  case 1:
    return "Evaluates a 256-layer 16x16 INT8 quantized per-invocation neural material MLP using VK_NV_cooperative_vector.";
  default:
    return "Per-invocation neural shading MLP inference benchmark via VK_NV_cooperative_vector.";
  }
}

bool CoopVectorBench::IsSupported(const DeviceInfo &info,
                                  IComputeContext *context) const {
  if (!info.cooperativeVectorSupport) return false;
  if (context && context->getBackend() != ComputeBackend::Vulkan) return false;
  return true;
}

bool CoopVectorBench::IsConfigSupported(uint32_t config_idx,
                                        const DeviceInfo &info,
                                        IComputeContext *context) const {
  return (config_idx < 2u) && IsSupported(info, context);
}

IBenchmark::SupportLimitation
CoopVectorBench::GetSupportLimitation(const DeviceInfo &info,
                                      IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return SupportLimitation::kApi;
  }
  if (!info.cooperativeVectorSupport) {
    return SupportLimitation::kHardware;
  }
  return SupportLimitation::kNone;
}

std::string CoopVectorBench::GetSupportNote(const DeviceInfo &info,
                                            IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return "Requires Vulkan 1.4 with VK_NV_cooperative_vector";
  }
  if (!info.cooperativeVectorSupport) {
    return "VK_NV_cooperative_vector not supported by GPU / Vulkan driver";
  }
  return "";
}

void CoopVectorBench::Setup(IComputeContext &context,
                            const std::string &kernel_dir) {
  m_context = &context;
  std::filesystem::path vkDir = std::filesystem::path(kernel_dir) / "vulkan";

  // FP16 16x16 weights = 256 fp16 values (0x2C00 = 0.0625f -> row sum = 1.0f)
  std::vector<uint16_t> fp16W(256, 0x2C00u);
  std::vector<uint16_t> fp16B(16, 0x2000u); // 0.0078125f
  std::vector<uint16_t> fp16Z(4096 * 16, 0u);
  m_fp16Weights = context.createBuffer(fp16W.size() * sizeof(uint16_t), fp16W.data());
  m_fp16Biases = context.createBuffer(fp16B.size() * sizeof(uint16_t), fp16B.data());
  m_fp16Out = context.createBuffer(fp16Z.size() * sizeof(uint16_t), fp16Z.data());

  std::vector<int8_t> int8W(256, 1);
  std::vector<int32_t> int8B(16, 4);
  std::vector<int32_t> int8Z(4096 * 16, 0);
  m_int8Weights = context.createBuffer(int8W.size() * sizeof(int8_t), int8W.data());
  m_int8Biases = context.createBuffer(int8B.size() * sizeof(int32_t), int8B.data());
  m_int8Out = context.createBuffer(int8Z.size() * sizeof(int32_t), int8Z.data());

  m_fp16Kernel = context.createKernel(
      (vkDir / "coop_vector_fp16.comp").string(), "main", 3);
  m_int8Kernel = context.createKernel(
      (vkDir / "coop_vector_int8.comp").string(), "main", 3);

  uint32_t nThreads = kNumThreads;
  uint32_t nIters = kIterations;

  context.setKernelArg(m_fp16Kernel, 0, m_fp16Weights);
  context.setKernelArg(m_fp16Kernel, 1, m_fp16Biases);
  context.setKernelArg(m_fp16Kernel, 2, m_fp16Out);
  context.setKernelArg(m_fp16Kernel, 3, sizeof(uint32_t), &nThreads);
  context.setKernelArg(m_fp16Kernel, 4, sizeof(uint32_t), &nIters);

  context.setKernelArg(m_int8Kernel, 0, m_int8Weights);
  context.setKernelArg(m_int8Kernel, 1, m_int8Biases);
  context.setKernelArg(m_int8Kernel, 2, m_int8Out);
  context.setKernelArg(m_int8Kernel, 3, sizeof(uint32_t), &nThreads);
  context.setKernelArg(m_int8Kernel, 4, sizeof(uint32_t), &nIters);
}

void CoopVectorBench::Run(uint32_t config_idx) {
  if (!m_context) return;
  ComputeKernel k = (config_idx == 0) ? m_fp16Kernel : m_int8Kernel;
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
  if (config_idx == 0) {
    if (!m_fp16Out) return false;
    std::vector<uint16_t> out(256, 0u);
    m_context->readBuffer(m_fp16Out, 0, out.size() * sizeof(uint16_t),
                          out.data());
    bool anyPositive = false;
    for (uint16_t h : out) {
      uint16_t expBits = (h >> 10) & 0x1Fu;
      if (expBits == 0x1Fu) return false; // NaN or Inf
      if ((h & 0x7FFFu) != 0u) anyPositive = true;
    }
    return anyPositive;
  }
  if (config_idx == 1) {
    if (!m_int8Out) return false;
    std::vector<int32_t> out(256, 0);
    m_context->readBuffer(m_int8Out, 0, out.size() * sizeof(int32_t),
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
    if (m_fp16Kernel) m_context->releaseKernel(m_fp16Kernel);
    if (m_int8Kernel) m_context->releaseKernel(m_int8Kernel);
    if (m_fp16Weights) m_context->releaseBuffer(m_fp16Weights);
    if (m_fp16Biases) m_context->releaseBuffer(m_fp16Biases);
    if (m_fp16Out) m_context->releaseBuffer(m_fp16Out);
    if (m_int8Weights) m_context->releaseBuffer(m_int8Weights);
    if (m_int8Biases) m_context->releaseBuffer(m_int8Biases);
    if (m_int8Out) m_context->releaseBuffer(m_int8Out);
  }
  m_fp16Kernel = nullptr;
  m_int8Kernel = nullptr;
  m_fp16Weights = nullptr;
  m_fp16Biases = nullptr;
  m_fp16Out = nullptr;
  m_int8Weights = nullptr;
  m_int8Biases = nullptr;
  m_int8Out = nullptr;
  m_context = nullptr;
}
