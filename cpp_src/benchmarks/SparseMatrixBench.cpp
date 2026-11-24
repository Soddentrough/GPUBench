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

IBenchmark::SupportDiscrepancy SparseMatrixBench::GetConfigSupportDiscrepancy(
    uint32_t /*config_idx*/, const DeviceInfo &info,
    IComputeContext *context) const {
  SupportDiscrepancy d{};
  if (hardwarePotentiallyHas2to4Sparsity(info) && !info.structuredSparsitySupport) {
    d.differs = true;
    d.hardwareStatus = "Supported in hardware (GPU architecture includes 2:4 sparse matrix units)";
    if (context && context->getBackend() == ComputeBackend::Vulkan) {
      d.driverStatus = "Unsupported (Vulkan driver/API lacks 2:4 structured sparse matrix extension)";
      d.summary = "Hardware has 2:4 sparse units, but Vulkan API/driver does not expose them";
    } else if (context && context->getBackend() == ComputeBackend::OpenCL) {
      d.driverStatus = "Unsupported (OpenCL C lacks 2:4 sparse matrix intrinsics)";
      d.summary = "Hardware has 2:4 sparse units, but OpenCL API does not expose them";
    } else {
      d.driverStatus = "Unsupported by active runtime/driver";
      d.summary = "Hardware has 2:4 sparse units, but active driver does not expose them";
    }
  }
  return d;
}

void SparseMatrixBench::Setup(IComputeContext &context,
                              const std::string &kernel_dir) {
  m_context = &context;
  DeviceInfo info = context.getCurrentDeviceInfo();
  m_blockSize = (info.subgroupSize == 64) ? 64u : 32u;
  uint32_t cus = (info.computeUnits > 0) ? info.computeUnits : 64u;
  m_gridX = cus * 128u;

  constexpr size_t kElements = 1048576;
  // Non-zero packed payloads per precision so 2:4 sparse indices (0, 2) never multiply by zero:
  //   - FP16: 0x3400 = 0.25f per half
  //   - FP8 : 0x34   = 0.75f (OCP E4M3) / 0.375f (E4M3 FNUZ) in every byte
  //   - INT8: 0x02   = 2 in every byte (keeps 8-lane sum well inside signed int32)
  std::vector<uint32_t> fp16Words(kElements, 0x34003400u);
  std::vector<uint32_t> fp8Words(kElements, 0x34343434u);
  std::vector<uint32_t> int8Words(kElements, 0x02020202u);
  m_fp16InBuf =
      context.createBuffer(kElements * sizeof(uint32_t), fp16Words.data());
  m_fp8InBuf =
      context.createBuffer(kElements * sizeof(uint32_t), fp8Words.data());
  m_int8InBuf =
      context.createBuffer(kElements * sizeof(uint32_t), int8Words.data());
  std::vector<uint32_t> zeroWords(524288, 0u);
  m_outBuf = context.createBuffer(zeroWords.size() * sizeof(uint32_t),
                                  zeroWords.data());

  std::filesystem::path rocmDir = std::filesystem::path(kernel_dir) / "rocm";
  m_fp16Kernel = context.createKernel(
      (rocmDir / "sparse_fp16_matrix.hip").string(), "run_benchmark", 2);
  m_fp8Kernel = context.createKernel(
      (rocmDir / "sparse_fp8_matrix.hip").string(), "run_benchmark", 2);
  m_int8Kernel = context.createKernel(
      (rocmDir / "sparse_int8_matrix.hip").string(), "run_benchmark", 2);

  if (m_fp16Kernel) {
    context.setKernelArg(m_fp16Kernel, 0, m_fp16InBuf);
    context.setKernelArg(m_fp16Kernel, 1, m_outBuf);
  }
  if (m_fp8Kernel) {
    context.setKernelArg(m_fp8Kernel, 0, m_fp8InBuf);
    context.setKernelArg(m_fp8Kernel, 1, m_outBuf);
  }
  if (m_int8Kernel) {
    context.setKernelArg(m_int8Kernel, 0, m_int8InBuf);
    context.setKernelArg(m_int8Kernel, 1, m_outBuf);
  }
}

void SparseMatrixBench::PrepareConfig(uint32_t /*config_idx*/) {
  if (!m_context || !m_outBuf) return;
  std::vector<uint32_t> zeroWords(524288, 0u);
  m_context->writeBuffer(m_outBuf, 0, zeroWords.size() * sizeof(uint32_t),
                         zeroWords.data());
  m_context->waitIdle();
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
    // Seed-only sum is < 10,000; 4096 non-zero sparse MACs produce >= 1,000,000.
    bool hasValidAccum = false;
    for (int32_t v : vals) {
      if (v < 0) return false;
      if (v >= 100000) hasValidAccum = true;
    }
    return hasValidAccum;
  }
  std::vector<float> vals(4096, 0.0f);
  m_context->readBuffer(m_outBuf, 0, vals.size() * sizeof(float), vals.data());
  // Seed-only sum is < 50.0f; non-zero 2:4 sparse MAC accumulation exceeds 1,000.0f.
  bool hasValidAccum = false;
  for (float v : vals) {
    if (!std::isfinite(v) || v < 0.0f) return false;
    if (v > 1000.0f) hasValidAccum = true;
  }
  return hasValidAccum;
}

void SparseMatrixBench::Teardown() {
  if (m_context) {
    if (m_fp16Kernel) m_context->releaseKernel(m_fp16Kernel);
    if (m_fp8Kernel) m_context->releaseKernel(m_fp8Kernel);
    if (m_int8Kernel) m_context->releaseKernel(m_int8Kernel);
    if (m_fp16InBuf) m_context->releaseBuffer(m_fp16InBuf);
    if (m_fp8InBuf) m_context->releaseBuffer(m_fp8InBuf);
    if (m_int8InBuf) m_context->releaseBuffer(m_int8InBuf);
    if (m_outBuf) m_context->releaseBuffer(m_outBuf);
  }
  m_fp16Kernel = nullptr;
  m_fp8Kernel = nullptr;
  m_int8Kernel = nullptr;
  m_fp16InBuf = nullptr;
  m_fp8InBuf = nullptr;
  m_int8InBuf = nullptr;
  m_outBuf = nullptr;
  m_context = nullptr;
}
