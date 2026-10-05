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

std::string CacheLatencyCurveBench::GetConfigSupportNote(uint32_t config_idx) const {
  if (config_idx < configs.size()) {
    return configs[config_idx].cacheLevelHint;
  }
  return "";
}

void CacheLatencyCurveBench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;

  // Calculate total buffer allocation
  totalBufferBytes = 0;
  for (const auto &cfg : configs) {
    totalBufferBytes += cfg.workingSetBytes;
  }

  size_t totalWords = totalBufferBytes / sizeof(uint32_t);
  std::vector<uint32_t> hostBuffer(totalWords, 0);

  uint64_t currentByteOffset = 0;
  for (size_t c = 0; c < configs.size(); ++c) {
    uint32_t baseWordOffset = static_cast<uint32_t>(currentByteOffset / sizeof(uint32_t));
    uint64_t numLines = configs[c].workingSetBytes / 128; // 128 bytes per cache line
    if (numLines == 0) numLines = 1;

    std::vector<uint32_t> perm(numLines);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 g(1337 + static_cast<uint32_t>(c));
    std::shuffle(perm.begin(), perm.end(), g);

    for (size_t i = 0; i < numLines; ++i) {
      uint32_t curLine = perm[i];
      uint32_t nextLine = perm[(i + 1) % numLines];
      // Store next line's word index in the first word of curLine
      size_t curWord = baseWordOffset + static_cast<size_t>(curLine) * 32;
      uint32_t nextWord = baseWordOffset + nextLine * 32;
      hostBuffer[curWord] = nextWord;
    }

    configs[c].startWordIndex = baseWordOffset + perm[0] * 32;
    currentByteOffset += configs[c].workingSetBytes;
  }

  buffer = context.createBuffer(totalBufferBytes);
  context.writeBuffer(buffer, 0, totalBufferBytes, hostBuffer.data());

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
  context.setKernelArg(kernel, 0, buffer);
}

void CacheLatencyCurveBench::Run(uint32_t config_idx) {
  if (config_idx >= configs.size() || !kernel || !context) {
    throw std::runtime_error("CacheLatencyCurveBench: invalid config or uninitialized state");
  }

  struct PushConstants {
    uint32_t iterations;
    uint32_t startIndex;
  } pc = { kIterations, configs[config_idx].startWordIndex };

  context->setKernelArg(kernel, 1, sizeof(pc), &pc);
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
}

BenchmarkResult CacheLatencyCurveBench::GetResult(uint32_t config_idx) const {
  return { kIterations, 0.0 };
}

bool CacheLatencyCurveBench::ValidateResults(uint32_t config_idx) const {
  return true;
}
