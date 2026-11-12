#include "SparseMatrixBench.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace {
bool hardwarePotentiallyHas2to4Sparsity(const DeviceInfo &info) {
  if (info.structuredSparsitySupport) return true;
  // RDNA 4, CDNA 3+, or Ada/Blackwell/Ampere expose FP8 matrix or >= SM 8.0
  if (info.vendorID == 0x10DE && info.cooperativeMatrixSupport) return true;
  if (info.vendorID == 0x1002 && info.fp8Support) return true;
  return false;
}
} // namespace

std::string SparseMatrixBench::GetConfigName(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "FP16 2:4 Sparse";
  case 1:
    return "FP8 2:4 Sparse";
  case 2:
    return "INT8 2:4 Sparse";
  default:
    return "";
  }
}

const char *SparseMatrixBench::GetDescription(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "Executes hardware 2:4 structured sparse FP16 matrix-multiply-accumulate instructions (v_swmmac / v_smfmac).";
  case 1:
    return "Executes hardware 2:4 structured sparse FP8 (E4M3) matrix-multiply-accumulate instructions at 2x dense K-dimension throughput.";
  case 2:
    return "Executes hardware 2:4 structured sparse INT8 matrix-multiply-accumulate instructions with INT32 accumulation.";
  default:
    return "Hardware 2:4 structured sparse matrix throughput benchmark.";
  }
}

bool SparseMatrixBench::IsSupported(const DeviceInfo &info,
                                    IComputeContext *context) const {
  if (!info.structuredSparsitySupport) return false;
  if (context && context->getBackend() != ComputeBackend::ROCm) return false;
  return true;
}

bool SparseMatrixBench::IsConfigSupported(uint32_t config_idx,
                                          const DeviceInfo &info,
                                          IComputeContext *context) const {
  return (config_idx < 3u) && IsSupported(info, context);
}

IBenchmark::SupportLimitation
SparseMatrixBench::GetSupportLimitation(const DeviceInfo &info,
                                        IComputeContext *context) const {
  if (!hardwarePotentiallyHas2to4Sparsity(info)) {
    return SupportLimitation::kHardware;
  }
  if (context && context->getBackend() != ComputeBackend::ROCm) {
    return SupportLimitation::kApi;
  }
  return SupportLimitation::kHardware;
}

std::string SparseMatrixBench::GetSupportNote(const DeviceInfo &info,
                                              IComputeContext *context) const {
  if (!hardwarePotentiallyHas2to4Sparsity(info)) {
    return "GPU lacks hardware 2:4 structured sparse matrix units (requires RDNA 4, CDNA 3+, or Sparse Tensor Cores)";
  }
  if (context && context->getBackend() == ComputeBackend::Vulkan) {
    return "Vulkan lacks a cross-vendor 2:4 structured sparse cooperative matrix extension (supported via ROCm v_swmmac / v_smfmac)";
  }
  if (context && context->getBackend() == ComputeBackend::OpenCL) {
    return "OpenCL C lacks 2:4 structured sparse matrix intrinsics";
  }
  return "Requires ROCm backend on RDNA 4 (gfx12) or CDNA 3+ (gfx942/gfx950)";
}

void SparseMatrixBench::Setup(IComputeContext &context,
                              const std::string &kernel_dir) {
  m_context = &context;
  DeviceInfo info = context.getCurrentDeviceInfo();
  m_blockSize = (info.subgroupSize == 64) ? 64u : 32u;
  uint32_t cus = (info.computeUnits > 0) ? info.computeUnits : 64u;
  m_gridX = cus * 128u;

  constexpr size_t kElements = 1048576;
  std::vector<uint32_t> initWords(kElements);
  for (size_t i = 0; i < kElements; ++i) {
    // Non-zero packed FP16 (0x3400 = 0.25) / FP8 / INT8 values
    initWords[i] = 0x34003400u;
  }
  m_inBuf = context.createBuffer(kElements * sizeof(uint32_t), initWords.data());
  std::vector<uint32_t> zeroWords(524288, 0u);
  m_outBuf = context.createBuffer(zeroWords.size() * sizeof(uint32_t),
                                  zeroWords.data());

  std::filesystem::path hipDir = std::filesystem::path(kernel_dir) / "hip";
  m_fp16Kernel = context.createKernel(
      (hipDir / "sparse_fp16_matrix.co").string(), "run_benchmark", 2);
  m_fp8Kernel = context.createKernel(
      (hipDir / "sparse_fp8_matrix.co").string(), "run_benchmark", 2);
  m_int8Kernel = context.createKernel(
      (hipDir / "sparse_int8_matrix.co").string(), "run_benchmark", 2);

  for (ComputeKernel k : {m_fp16Kernel, m_fp8Kernel, m_int8Kernel}) {
    if (k) {
      context.setKernelArg(k, 0, m_inBuf);
      context.setKernelArg(k, 1, m_outBuf);
    }
  }
}

void SparseMatrixBench::Run(uint32_t config_idx) {
  if (!m_context) return;
  ComputeKernel k = nullptr;
  if (config_idx == 0) k = m_fp16Kernel;
  else if (config_idx == 1) k = m_fp8Kernel;
  else if (config_idx == 2) k = m_int8Kernel;
  if (!k) return;
  m_context->dispatch(k, m_gridX, 1, 1, m_blockSize, 1, 1);
}

BenchmarkResult SparseMatrixBench::GetResult(uint32_t /*config_idx*/) const {
  // Each workgroup (1 wave) executes 536,870,912 effective 2:4 sparse ops
  uint64_t opsPerWave = 536870912ULL;
  return {static_cast<uint64_t>(m_gridX) * opsPerWave, 0.0};
}

bool SparseMatrixBench::ValidateResults(uint32_t config_idx) const {
  if (!m_context || !m_outBuf) return false;
  if (config_idx == 2) {
    std::vector<int32_t> vals(4096, 0);
    m_context->readBuffer(m_outBuf, 0, vals.size() * sizeof(int32_t),
                          vals.data());
    bool anyNonZero = false;
    for (int32_t v : vals) {
      if (v != 0) anyNonZero = true;
    }
    return anyNonZero;
  }
  std::vector<float> vals(4096, 0.0f);
  m_context->readBuffer(m_outBuf, 0, vals.size() * sizeof(float), vals.data());
  bool anyNonZero = false;
  for (float v : vals) {
    if (!std::isfinite(v)) return false;
    if (std::fabs(v) > 1e-5f) anyNonZero = true;
  }
  return anyNonZero;
}

void SparseMatrixBench::Teardown() {
  if (m_context) {
    if (m_fp16Kernel) m_context->releaseKernel(m_fp16Kernel);
    if (m_fp8Kernel) m_context->releaseKernel(m_fp8Kernel);
    if (m_int8Kernel) m_context->releaseKernel(m_int8Kernel);
    if (m_inBuf) m_context->releaseBuffer(m_inBuf);
    if (m_outBuf) m_context->releaseBuffer(m_outBuf);
  }
  m_fp16Kernel = nullptr;
  m_fp8Kernel = nullptr;
  m_int8Kernel = nullptr;
  m_inBuf = nullptr;
  m_outBuf = nullptr;
  m_context = nullptr;
}
