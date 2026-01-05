#include "benchmarks/SysMemLatencyBench.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#define ALIGNED_ALLOC(alignment, size) _aligned_malloc(size, alignment)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#define ALIGNED_ALLOC(alignment, size) aligned_alloc(alignment, size)
#define ALIGNED_FREE(ptr) free(ptr)
#endif

SysMemLatencyBench::SysMemLatencyBench() {}

SysMemLatencyBench::~SysMemLatencyBench() { Teardown(); }

const char *SysMemLatencyBench::GetName() const {
  return "System Memory Latency";
}

const char *SysMemLatencyBench::GetMetric() const { return "ns"; }

bool SysMemLatencyBench::IsSupported(const DeviceInfo &info,
                                     IComputeContext *context) const {
  return true;
}

void SysMemLatencyBench::Setup(IComputeContext &context,
                               const std::string &kernel_dir) {
  // 512MB buffer to ensure we bypass CPU caches (including large L3)
  bufferSize = 512ULL * 1024ULL * 1024ULL;

  buffer = ALIGNED_ALLOC(64, bufferSize);
  if (!buffer) {
    throw std::runtime_error(
        "Failed to allocate system memory buffer for latency test");
  }

  // Initialize with shuffled indices for pointer chasing
  uint32_t numElements = (uint32_t)(bufferSize / sizeof(uint32_t));
  uint32_t *pBuffer = reinterpret_cast<uint32_t *>(buffer);

  std::vector<uint32_t> indices(numElements);
  std::iota(indices.begin(), indices.end(), 0);

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices.begin(), indices.end(), g);

  // Create chasing chain: pBuffer[indices[i]] = indices[i+1]
  for (uint32_t i = 0; i < numElements - 1; ++i) {
    pBuffer[indices[i]] = indices[i + 1];
  }
  pBuffer[indices[numElements - 1]] = indices[0]; // Close the loop
}

void SysMemLatencyBench::Run(uint32_t config_idx) {
  uint32_t *pBuffer = reinterpret_cast<uint32_t *>(buffer);
  uint32_t index = 0;

  // Warm up
  for (int i = 0; i < 1000; i++) {
    index = pBuffer[index];
  }

  const uint64_t iterations = 1000000; // 1M jumps

  auto start = std::chrono::high_resolution_clock::now();

  // Pointer chasing loop
  for (uint64_t i = 0; i < iterations; ++i) {
    index = pBuffer[index];
  }

  auto end = std::chrono::high_resolution_clock::now();

  // Use index to prevent optimization
  volatile uint32_t sink = index;
  (void)sink;

  double elapsedNs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  lastRunTimeMs = elapsedNs / 1000000.0;
  lastRunOps = iterations;
}

void SysMemLatencyBench::Teardown() {
  if (buffer) {
    ALIGNED_FREE(buffer);
    buffer = nullptr;
  }
}

BenchmarkResult SysMemLatencyBench::GetResult(uint32_t config_idx) const {
  return {lastRunOps, lastRunTimeMs};
}

uint32_t SysMemLatencyBench::GetNumConfigs() const { return 1; }

std::string SysMemLatencyBench::GetConfigName(uint32_t config_idx) const {
  return "Default";
}
