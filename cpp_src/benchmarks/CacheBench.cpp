#include "benchmarks/CacheBench.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h>
#define ALIGNED_ALLOC(alignment, size) _aligned_malloc(size, alignment)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#define ALIGNED_ALLOC(alignment, size) aligned_alloc(alignment, size)
#define ALIGNED_FREE(ptr) free(ptr)
#endif

CacheBench::CacheBench(std::string name, std::string metric,
                       uint64_t bufferSize, std::string kernelFile,
                       std::vector<uint32_t> initData,
                       std::vector<std::string> aliases, int targetCacheLevel)
    : name(name), aliases(aliases), metric(metric), bufferSize(bufferSize),
      kernelFile(kernelFile), initData(initData),
      targetCacheLevel(targetCacheLevel) {}

CacheBench::~CacheBench() {
  Teardown();
}

bool CacheBench::IsSupported(const DeviceInfo &info,
                             IComputeContext *context) const {
  (void)context;
  if (targetCacheLevel == 3 && info.l3CacheSize == 0) {
    return false;
  }
  if (targetCacheLevel == 2 && info.l2CacheSize == 0) {
    return false;
  }
  return true;
}

void CacheBench::Setup(IComputeContext &context,
                       const std::string &kernel_dir) {
  this->context = &context;

  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path full_kernel_path;
  if (context.getBackend() == ComputeBackend::ROCm) {
    full_kernel_path = kdir / "rocm" / (kernelFile + ".hip");
  } else if (context.getBackend() == ComputeBackend::OpenCL) {
    full_kernel_path = kdir / "opencl" / (kernelFile + ".cl");
  } else {
    full_kernel_path = kdir / "vulkan" / (kernelFile + ".comp");
  }

  const std::string kernel_name =
      (context.getBackend() == ComputeBackend::Vulkan) ? "main"
                                                       : "run_benchmark";

  if (metric == "GB/s") {
    const DeviceInfo info = context.getCurrentDeviceInfo();
    if (targetCacheLevel == 2 && info.l2CacheSize > 0) {
      bufferSize = std::clamp<uint64_t>(info.l2CacheSize / 2,
                                        512ULL * 1024ULL,
                                        16ULL * 1024ULL * 1024ULL);
    } else if (targetCacheLevel == 3 && info.l3CacheSize > 0) {
      bufferSize = std::clamp<uint64_t>(info.l3CacheSize / 2,
                                        8ULL * 1024ULL * 1024ULL,
                                        128ULL * 1024ULL * 1024ULL);
    }
    bufferSize = std::max<uint64_t>(16384ULL, (bufferSize + 16383ULL) & ~16383ULL);
    numWorkgroups = 2048u; // 2048 WGs x 128 threads = 262,144 threads (strictly <= 65535 WGs)

    const size_t numFloats = static_cast<size_t>(bufferSize / sizeof(float));
    std::vector<float> bwInit(numFloats);
    for (size_t i = 0; i < numFloats; ++i) {
      bwInit[i] = 1.0f + static_cast<float>(i & 15u) * 0.0625f;
    }

    buffer = context.createBuffer(bufferSize);
    context.writeBuffer(buffer, 0, bufferSize, bwInit.data());

    constexpr size_t kOutFloats = 65536;
    std::vector<float> zeroOut(kOutFloats, 0.0f);
    outBuffer = context.createBuffer(kOutFloats * sizeof(float));
    context.writeBuffer(outBuffer, 0, kOutFloats * sizeof(float), zeroOut.data());

    const uint32_t numBuffers =
        (context.getBackend() == ComputeBackend::Vulkan) ? 2u : 3u;
    kernel = context.createKernel(full_kernel_path.string(), kernel_name, numBuffers);

    struct PushConstants {
      uint32_t stride;
      uint32_t numVec4;
      uint32_t iterations;
      uint32_t padding;
    } pc = {1024u, static_cast<uint32_t>(bufferSize / 16ULL), 2000u, 0u};

    context.setKernelArg(kernel, 0, buffer);
    context.setKernelArg(kernel, 1, outBuffer);
    if (context.getBackend() == ComputeBackend::Vulkan) {
      context.setKernelArg(kernel, 2, sizeof(pc), &pc);
    } else {
      pcBuffer = context.createBuffer(sizeof(PushConstants));
      context.writeBuffer(pcBuffer, 0, sizeof(PushConstants), &pc);
      context.setKernelArg(kernel, 2, pcBuffer);
    }
    return;
  }

  // Latency path (metric == "ns"): inter-cache-line pointer chasing across 128-byte cache lines
  numWorkgroups = 1u;
  if (bufferSize > 0) {
    constexpr size_t kLineBytes = 128;
    size_t numLines = bufferSize / kLineBytes;
    if (numLines == 0) numLines = 1;
    size_t totalWords = bufferSize / sizeof(uint32_t);
    initData.assign(totalWords, 0);

    std::vector<uint32_t> perm(numLines);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 g(1337);
    std::shuffle(perm.begin(), perm.end(), g);

    for (size_t i = 0; i < numLines; ++i) {
      uint32_t curLine = perm[i];
      uint32_t nextLine = perm[(i + 1) % numLines];
      initData[curLine * (kLineBytes / sizeof(uint32_t))] =
          static_cast<uint32_t>(nextLine * (kLineBytes / sizeof(uint32_t)));
    }

    buffer = context.createBuffer(bufferSize);
    context.writeBuffer(buffer, 0, initData.size() * sizeof(uint32_t), initData.data());
  }

  const uint32_t numBuffers =
      (context.getBackend() == ComputeBackend::Vulkan) ? 1u : 2u;
  kernel = context.createKernel(full_kernel_path.string(), kernel_name, numBuffers);
  if (buffer) {
    context.setKernelArg(kernel, 0, buffer);

    struct PushConstants {
      uint32_t stride;
      uint32_t mask;
      uint32_t iterations;
      uint32_t padding;
    } pc = {1u, static_cast<uint32_t>(bufferSize / 4ULL - 1ULL), 1000000u, 0u};

    if (context.getBackend() == ComputeBackend::Vulkan) {
      context.setKernelArg(kernel, 1, sizeof(pc), &pc);
    } else {
      pcBuffer = context.createBuffer(sizeof(PushConstants));
      context.writeBuffer(pcBuffer, 0, sizeof(PushConstants), &pc);
      context.setKernelArg(kernel, 1, pcBuffer);
    }
  }
}

void CacheBench::Run(uint32_t config_idx) {
  (void)config_idx;
  if (!context || !kernel) {
    throw std::runtime_error("CacheBench is not set up");
  }
  if (metric == "GB/s") {
    context->dispatch(kernel, std::min<uint32_t>(numWorkgroups, 65535u), 1, 1, 128, 1, 1);
  } else {
    context->dispatch(kernel, 1, 1, 1, 1, 1, 1);
  }
}

void CacheBench::Teardown() {
  if (context) {
    if (kernel) {
      context->releaseKernel(kernel);
      kernel = nullptr;
    }
    if (buffer) {
      context->releaseBuffer(buffer);
      buffer = nullptr;
    }
    if (outBuffer) {
      context->releaseBuffer(outBuffer);
      outBuffer = nullptr;
    }
    if (pcBuffer) {
      context->releaseBuffer(pcBuffer);
      pcBuffer = nullptr;
    }
    context = nullptr;
  }
  if (hostMem) {
    ALIGNED_FREE(hostMem);
    hostMem = nullptr;
  }
}

const char *CacheBench::GetName() const { return name.c_str(); }

std::vector<std::string> CacheBench::GetAliases() const { return aliases; }

const char *CacheBench::GetMetric() const { return metric.c_str(); }

BenchmarkResult CacheBench::GetResult(uint32_t config_idx) const {
  (void)config_idx;
  uint64_t operations = 0;
  uint64_t num_threads_bw =
      static_cast<uint64_t>(std::min<uint32_t>(numWorkgroups, 65535u)) * 128ULL;

  if (metric == "GB/s") {
    // cache_bw_robust loops 2,000 times * 8 float4 loads * 16 bytes
    operations = num_threads_bw * 16000ULL * 16ULL;
  } else {
    // 1,000,000 pointer chasing steps
    operations = 1000000ULL;
  }
  return {operations, 0.0};
}

const char *CacheBench::GetComponent(uint32_t config_idx) const {
  (void)config_idx;
  return "Memory";
}

const char *CacheBench::GetSubCategory(uint32_t config_idx) const {
  (void)config_idx;
  return (metric == "GB/s") ? "Cache Bandwidth" : "Latency";
}

int CacheBench::GetSortWeight() const { return (metric == "GB/s") ? 150 : 200; }

bool CacheBench::ValidateResults(uint32_t config_idx) const {
  if (config_idx >= GetNumConfigs()) {
    return false;
  }
  if (!context || !kernel || !buffer) {
    return false;
  }

  if (metric == "GB/s") {
    if (!outBuffer) {
      return false;
    }
    try {
      std::vector<float> samples(64, 0.0f);
      context->readBuffer(outBuffer, 0, samples.size() * sizeof(float),
                          samples.data());
      for (float v : samples) {
        if (!std::isfinite(v) || v < 60000.0f || v > 130000.0f) {
          std::cerr << "CacheBench bandwidth validation failed: sample=" << v
                    << "\n";
          return false;
        }
      }
      return true;
    } catch (const std::exception &e) {
      std::cerr << "CacheBench bandwidth validation error: " << e.what()
                << "\n";
      return false;
    }
  }

  if (metric == "ns") {
    constexpr size_t kLineBytes = 128;
    size_t numLines = bufferSize / kLineBytes;
    if (numLines == 0) numLines = 1;

    uint32_t expectedIndex = 0;
    if (!initData.empty()) {
      size_t targetStep = 1000000 % numLines;
      for (size_t i = 0; i < targetStep; ++i) {
        expectedIndex = initData[expectedIndex];
      }
    }

    uint32_t readbackIndex = 0;
    try {
      context->readBuffer(buffer, sizeof(uint32_t), sizeof(uint32_t),
                          &readbackIndex);
      if (readbackIndex != expectedIndex) {
        std::cerr << "CacheBench validation failed: readback " << readbackIndex
                  << " != expected " << expectedIndex << "\n";
        return false;
      }
      return true;
    } catch (const std::exception &e) {
      std::cerr << "CacheBench validation error: " << e.what() << "\n";
      return false;
    }
  }

  return false;
}

