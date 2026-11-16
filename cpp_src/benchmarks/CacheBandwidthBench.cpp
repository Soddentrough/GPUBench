#include "benchmarks/CacheBandwidthBench.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

CacheBandwidthBench::CacheBandwidthBench() = default;

CacheBandwidthBench::~CacheBandwidthBench() {
  Teardown();
}

std::string CacheBandwidthBench::GetConfigName(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "L0 Cache Bandwidth (16 KB)";
  case 1:
    return "L1 Cache Bandwidth (128 KB)";
  case 2:
    return "L2 Cache Bandwidth";
  case 3:
    return "L3 / Infinity Cache Bandwidth";
  default:
    return "";
  }
}

const char *CacheBandwidthBench::GetDescription(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "Multi-CU streaming bandwidth resident in per-CU/WGP L0 vector cache (16 KB working set).";
  case 1:
    return "Multi-CU streaming bandwidth resident in shared L1 / GL1 cache (128 KB working set).";
  case 2:
    return "Full-GPU streaming bandwidth scaled inside hardware L2 cache capacity.";
  case 3:
    return "Full-GPU streaming bandwidth scaled inside hardware L3 / Infinity Cache / MALL capacity.";
  default:
    return "Multi-CU streaming GPU cache bandwidth across L0, L1, L2, and L3/MALL cache tiers.";
  }
}

bool CacheBandwidthBench::IsConfigSupported(uint32_t config_idx,
                                            const DeviceInfo &info,
                                            IComputeContext *context) const {
  (void)context;
  if (config_idx == 0u || config_idx == 1u) {
    return true;
  }
  if (config_idx == 2u) {
    return info.l2CacheSize > 0;
  }
  if (config_idx == 3u) {
    return info.l3CacheSize > 0;
  }
  return false;
}

IBenchmark::SupportLimitation CacheBandwidthBench::GetConfigSupportLimitation(
    uint32_t config_idx, const DeviceInfo &info,
    IComputeContext *context) const {
  (void)context;
  if (config_idx == 2u && info.l2CacheSize == 0) {
    return SupportLimitation::kHardware;
  }
  if (config_idx == 3u && info.l3CacheSize == 0) {
    return SupportLimitation::kHardware;
  }
  if (config_idx >= 4u) {
    return SupportLimitation::kHardware;
  }
  return SupportLimitation::kNone;
}

std::string CacheBandwidthBench::GetConfigSupportNote(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "Measures nearest vector cache bandwidth: AMD RDNA WGP Vector L0 (TCP, 16-32 KB); NVIDIA L1; Intel L1";
  case 2:
    return "Device does not have an L2 cache reported";
  case 3:
    return "Device does not have an L3 / Infinity Cache (l3CacheSize = 0)";
  default:
    return "";
  }
}

std::string CacheBandwidthBench::GetConfigSupportNote(
    uint32_t config_idx, const DeviceInfo &info,
    IComputeContext *context) const {
  (void)context;
  if (config_idx == 0u) {
    return "Measures nearest vector cache bandwidth: AMD RDNA WGP Vector L0 (TCP, 16-32 KB); NVIDIA L1; Intel L1";
  }
  if (config_idx == 2u && info.l2CacheSize == 0) {
    return "Device does not have an L2 cache reported";
  }
  if (config_idx == 3u && info.l3CacheSize == 0) {
    return "Device does not have an L3 / Infinity Cache (l3CacheSize = 0)";
  }
  return "";
}

uint64_t CacheBandwidthBench::computeWorkingSetBytes(
    uint32_t config_idx, const DeviceInfo &info) const {
  uint64_t sz = 16ULL * 1024ULL;
  if (config_idx == 0u) {
    sz = 16ULL * 1024ULL;
  } else if (config_idx == 1u) {
    sz = 128ULL * 1024ULL;
  } else if (config_idx == 2u) {
    sz = (info.l2CacheSize > 0)
             ? std::clamp<uint64_t>(info.l2CacheSize / 2ULL,
                                    512ULL * 1024ULL,
                                    16ULL * 1024ULL * 1024ULL)
             : (4ULL * 1024ULL * 1024ULL);
  } else {
    sz = (info.l3CacheSize > 0)
             ? std::clamp<uint64_t>(info.l3CacheSize / 2ULL,
                                    8ULL * 1024ULL * 1024ULL,
                                    128ULL * 1024ULL * 1024ULL)
             : (64ULL * 1024ULL * 1024ULL);
  }
  return std::max<uint64_t>(16384ULL, (sz + 16383ULL) & ~16383ULL);
}

void CacheBandwidthBench::Setup(IComputeContext &context,
                                const std::string &kernel_dir) {
  this->context = &context;
  activeConfigIdx = -1;

  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path full_kernel_path;
  std::string kernel_name;
  if (context.getBackend() == ComputeBackend::ROCm) {
    full_kernel_path = kdir / "rocm" / "cache_bw_robust.hip";
    kernel_name = "run_benchmark";
  } else if (context.getBackend() == ComputeBackend::OpenCL) {
    full_kernel_path = kdir / "opencl" / "cache_bw_robust.cl";
    kernel_name = "run_benchmark";
  } else {
    full_kernel_path = kdir / "vulkan" / "cache_bw_robust.comp";
    kernel_name = "main";
  }

  constexpr size_t kOutFloats = 65536;
  std::vector<float> zeroOut(kOutFloats, 0.0f);
  outBuffer = context.createBuffer(kOutFloats * sizeof(float));
  context.writeBuffer(outBuffer, 0, kOutFloats * sizeof(float), zeroOut.data());

  const uint32_t numBuffers =
      (context.getBackend() == ComputeBackend::Vulkan) ? 2u : 3u;
  kernel = context.createKernel(full_kernel_path.string(), kernel_name, numBuffers);
  context.setKernelArg(kernel, 1, outBuffer);
}

void CacheBandwidthBench::PrepareConfig(uint32_t config_idx) {
  if (!context || !kernel || config_idx >= GetNumConfigs()) {
    return;
  }
  if (activeConfigIdx == static_cast<int32_t>(config_idx) && buffer != nullptr) {
    return;
  }
  ReleaseConfigResources(config_idx);

  const DeviceInfo info = context->getCurrentDeviceInfo();
  const uint64_t wsBytes = computeWorkingSetBytes(config_idx, info);
  const size_t numFloats = static_cast<size_t>(wsBytes / sizeof(float));

  std::vector<float> bwInit(numFloats);
  for (size_t i = 0; i < numFloats; ++i) {
    bwInit[i] = 1.0f + static_cast<float>(i & 15u) * 0.0625f;
  }

  buffer = context->createBuffer(wsBytes);
  context->writeBuffer(buffer, 0, wsBytes, bwInit.data());

  constexpr size_t kOutFloats = 65536;
  std::vector<float> zeroOut(kOutFloats, 0.0f);
  context->writeBuffer(outBuffer, 0, kOutFloats * sizeof(float), zeroOut.data());

  struct PushConstants {
    uint32_t stride;
    uint32_t numVec4;
    uint32_t iterations;
    uint32_t padding;
  } pc = {1024u, static_cast<uint32_t>(wsBytes / 16ULL), 2000u, 0u};

  context->setKernelArg(kernel, 0, buffer);
  context->setKernelArg(kernel, 1, outBuffer);
  if (context->getBackend() == ComputeBackend::Vulkan) {
    context->setKernelArg(kernel, 2, sizeof(pc), &pc);
  } else {
    pcBuffer = context->createBuffer(sizeof(PushConstants));
    context->writeBuffer(pcBuffer, 0, sizeof(PushConstants), &pc);
    context->setKernelArg(kernel, 2, pcBuffer);
  }
  activeConfigIdx = static_cast<int32_t>(config_idx);
}

void CacheBandwidthBench::ReleaseConfigResources(uint32_t /*config_idx*/) {
  if (!context) return;
  context->waitIdle();
  if (buffer) {
    context->releaseBuffer(buffer);
    buffer = nullptr;
  }
  if (pcBuffer) {
    context->releaseBuffer(pcBuffer);
    pcBuffer = nullptr;
  }
  activeConfigIdx = -1;
}

void CacheBandwidthBench::Run(uint32_t config_idx) {
  if (!context || !kernel || config_idx >= GetNumConfigs()) {
    throw std::runtime_error("CacheBandwidthBench: invalid config or uninitialized state");
  }
  if (activeConfigIdx != static_cast<int32_t>(config_idx) || !buffer) {
    PrepareConfig(config_idx);
  }
  context->dispatch(kernel, kNumWorkgroups, 1, 1, 128, 1, 1);
}

void CacheBandwidthBench::Teardown() {
  ReleaseConfigResources(0);
  if (context) {
    if (kernel) {
      context->releaseKernel(kernel);
      kernel = nullptr;
    }
    if (outBuffer) {
      context->releaseBuffer(outBuffer);
      outBuffer = nullptr;
    }
    context = nullptr;
  }
  activeConfigIdx = -1;
}

BenchmarkResult CacheBandwidthBench::GetResult(uint32_t /*config_idx*/) const {
  const uint64_t numThreads = static_cast<uint64_t>(kNumWorkgroups) * 128ULL;
  // cache_bw_robust loops 2,000 times * 8 float4 loads * 16 bytes
  const uint64_t ops = numThreads * 16000ULL * 16ULL;
  return {ops, 0.0};
}

bool CacheBandwidthBench::ValidateResults(uint32_t config_idx) const {
  if (config_idx >= GetNumConfigs() || !context || !kernel || !buffer || !outBuffer) {
    return false;
  }
  if (activeConfigIdx != static_cast<int32_t>(config_idx)) {
    return false;
  }
  try {
    std::vector<float> samples(64, 0.0f);
    context->readBuffer(outBuffer, 0, samples.size() * sizeof(float), samples.data());
    for (float v : samples) {
      if (!std::isfinite(v) || v < 60000.0f || v > 130000.0f) {
        std::cerr << "CacheBandwidthBench validation failed (config "
                  << config_idx << "): sample=" << v << "\n";
        return false;
      }
    }
    return true;
  } catch (const std::exception &e) {
    std::cerr << "CacheBandwidthBench validation error: " << e.what() << "\n";
    return false;
  }
}
