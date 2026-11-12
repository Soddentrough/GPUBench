#include "benchmarks/CacheLatencyCurveBench.h"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

CacheLatencyCurveBench::CacheLatencyCurveBench() {
  configs = {
    {"16 KB", 16ULL * 1024, "L0 TCP Cache", 0},
    {"32 KB", 32ULL * 1024, "L0 TCP Boundary", 0},
    {"64 KB", 64ULL * 1024, "GL1 Cache Transition", 0},
    {"128 KB", 128ULL * 1024, "GL1 Cache", 0},
    {"256 KB", 256ULL * 1024, "GL1 Boundary", 0},
    {"512 KB", 512ULL * 1024, "GL2 Cache", 0},
    {"1 MB", 1ULL * 1024 * 1024, "GL2 Cache", 0},
    {"2 MB", 2ULL * 1024 * 1024, "GL2 Cache", 0},
    {"4 MB", 4ULL * 1024 * 1024, "GL2 Boundary", 0},
    {"8 MB", 8ULL * 1024 * 1024, "L3 MALL Cache", 0},
    {"16 MB", 16ULL * 1024 * 1024, "L3 MALL Cache", 0},
    {"32 MB", 32ULL * 1024 * 1024, "L3 MALL Cache", 0},
    {"64 MB", 64ULL * 1024 * 1024, "L3 MALL Cache", 0},
    {"128 MB", 128ULL * 1024 * 1024, "L3 MALL Boundary", 0},
    {"256 MB", 256ULL * 1024 * 1024, "GDDR6 VRAM DRAM", 0}
  };
}

CacheLatencyCurveBench::~CacheLatencyCurveBench() {
  Teardown();
}

std::string CacheLatencyCurveBench::GetConfigName(uint32_t config_idx) const {
  if (config_idx < configs.size()) {
    return configs[config_idx].name;
  }
  return "";
}

std::string CacheLatencyCurveBench::ComputeAdaptiveHint(
    uint64_t workingSetBytes, const DeviceInfo &info) const {
  uint64_t l0Bytes = info.l1CacheSize > 0 ? info.l1CacheSize : (32ULL * 1024);
  uint64_t l1Bytes = 256ULL * 1024;
  uint64_t l2Bytes =
      info.l2CacheSize > 0 ? info.l2CacheSize : (4ULL * 1024 * 1024);
  uint64_t l3Bytes = info.l3CacheSize;

  if (workingSetBytes < l0Bytes) {
    return "L0 Vector Cache";
  }
  if (workingSetBytes == l0Bytes) {
    return "L0 Vector Cache Boundary";
  }
  if (workingSetBytes < l1Bytes) {
    return "L1 / GL1 Cache";
  }
  if (workingSetBytes == l1Bytes) {
    return "L1 / GL1 Cache Boundary";
  }
  if (workingSetBytes < l2Bytes) {
    return "L2 Cache";
  }
  if (workingSetBytes == l2Bytes) {
    return "L2 Cache Boundary";
  }
  if (l3Bytes > 0) {
    if (workingSetBytes < l3Bytes) {
      return "L3 / MALL Infinity Cache";
    }
    if (workingSetBytes == l3Bytes) {
      return "L3 / MALL Cache Boundary";
    }
  }
  return info.isApu ? "Unified System DRAM" : "VRAM DRAM";
}

std::string CacheLatencyCurveBench::GetConfigSupportNote(uint32_t config_idx) const {
  if (config_idx < configs.size()) {
    if (context) {
      return ComputeAdaptiveHint(configs[config_idx].workingSetBytes,
                                 context->getCurrentDeviceInfo());
    }
    return configs[config_idx].cacheLevelHint;
  }
  return "";
}

std::string CacheLatencyCurveBench::GetConfigSupportNote(
    uint32_t config_idx, const DeviceInfo &info,
    IComputeContext *context) const {
  (void)context;
  if (config_idx < configs.size()) {
    return ComputeAdaptiveHint(configs[config_idx].workingSetBytes, info);
  }
  return "";
}

void CacheLatencyCurveBench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;
  activeConfigIdx = -1;
  totalBufferBytes = 0;

  DeviceInfo info = context.getCurrentDeviceInfo();
  for (auto &cfg : configs) {
    cfg.cacheLevelHint = ComputeAdaptiveHint(cfg.workingSetBytes, info);
    cfg.startWordIndex = 0;
    cfg.expectedTerminalIndex = 0;
  }

  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path full_kernel_path;
  std::string kernel_func;

  if (context.getBackend() == ComputeBackend::ROCm) {
    full_kernel_path = kdir / "rocm" / "cache_latency_curve.hip";
    kernel_func = "run_benchmark";
  } else if (context.getBackend() == ComputeBackend::OpenCL) {
    full_kernel_path = kdir / "opencl" / "cache_latency_curve.cl";
    kernel_func = "run_benchmark";
  } else { // Vulkan
    full_kernel_path = kdir / "vulkan" / "cache_latency_curve.comp";
    kernel_func = "main";
  }

  kernel = context.createKernel(full_kernel_path.string(), kernel_func, 1);
}

void CacheLatencyCurveBench::PrepareConfig(uint32_t config_idx) {
  if (!context || !kernel || config_idx >= configs.size()) {
    return;
  }
  if (activeConfigIdx == static_cast<int32_t>(config_idx) && buffer != nullptr) {
    return;
  }
  if (buffer != nullptr) {
    context->waitIdle();
    context->releaseBuffer(buffer);
    buffer = nullptr;
  }

  uint64_t stepBytes = configs[config_idx].workingSetBytes;
  size_t stepWords = static_cast<size_t>(stepBytes / sizeof(uint32_t));
  std::vector<uint32_t> hostBuffer(stepWords, 0);

  uint64_t numLines = stepBytes / 128ULL;
  if (numLines == 0) {
    numLines = 1;
  }

  std::vector<uint32_t> perm(static_cast<size_t>(numLines));
  std::iota(perm.begin(), perm.end(), 0u);
  std::mt19937 g(1337u + config_idx);
  std::shuffle(perm.begin(), perm.end(), g);

  for (size_t i = 0; i < static_cast<size_t>(numLines); ++i) {
    uint32_t curLine = perm[i];
    uint32_t nextLine = perm[(i + 1) % static_cast<size_t>(numLines)];
    size_t curWord = static_cast<size_t>(curLine) * 32u;
    uint32_t nextWord = nextLine * 32u;
    hostBuffer[curWord] = nextWord;
  }

  configs[config_idx].startWordIndex = perm[0] * 32u;
  size_t targetStep = kIterations % static_cast<size_t>(numLines);
  configs[config_idx].expectedTerminalIndex = perm[targetStep] * 32u;

  buffer = context->createBuffer(stepBytes);
  context->writeBuffer(buffer, 0, stepBytes, hostBuffer.data());
  context->setKernelArg(kernel, 0, buffer);
  totalBufferBytes = stepBytes;
  activeConfigIdx = static_cast<int32_t>(config_idx);
}

void CacheLatencyCurveBench::ReleaseConfigResources(uint32_t /*config_idx*/) {
  if (context && buffer) {
    context->waitIdle();
    context->releaseBuffer(buffer);
    buffer = nullptr;
  }
  totalBufferBytes = 0;
  activeConfigIdx = -1;
}

void CacheLatencyCurveBench::Run(uint32_t config_idx) {
  if (config_idx >= configs.size() || !kernel || !context) {
    throw std::runtime_error("CacheLatencyCurveBench: invalid config or uninitialized state");
  }
  if (activeConfigIdx != static_cast<int32_t>(config_idx) || !buffer) {
    PrepareConfig(config_idx);
  }

  uint32_t iterations = kIterations;
  uint32_t startIndex = configs[config_idx].startWordIndex;

  context->setKernelArg(kernel, 1, sizeof(uint32_t), &iterations);
  context->setKernelArg(kernel, 2, sizeof(uint32_t), &startIndex);
  context->dispatch(kernel, 1, 1, 1, 1, 1, 1);
}

void CacheLatencyCurveBench::Teardown() {
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
  totalBufferBytes = 0;
  activeConfigIdx = -1;
}

BenchmarkResult CacheLatencyCurveBench::GetResult(uint32_t config_idx) const {
  (void)config_idx;
  return { kIterations, 0.0 };
}

bool CacheLatencyCurveBench::ValidateResults(uint32_t config_idx) const {
  if (config_idx >= configs.size() || !context || !buffer) {
    return false;
  }

  uint32_t expected = configs[config_idx].expectedTerminalIndex;
  uint32_t readback = 0;
  uint64_t byteOffset = static_cast<uint64_t>(configs[config_idx].startWordIndex + 1) * sizeof(uint32_t);

  try {
    context->readBuffer(buffer, byteOffset, sizeof(uint32_t), &readback);
    if (readback != expected) {
      std::cerr << "CacheLatencyCurveBench validation failed (config "
                << config_idx << "): readback " << readback
                << " != expected " << expected << "\n";
      return false;
    }
    return true;
  } catch (const std::exception &e) {
    std::cerr << "CacheLatencyCurveBench validation error (config "
              << config_idx << "): " << e.what() << "\n";
    return false;
  }
}

double CacheLatencyCurveBench::FilterDuration(uint32_t config_idx, uint64_t total_invocations, double duration_ms) {
  (void)config_idx;
  (void)total_invocations;
  return duration_ms;
}
