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
  // 256MB buffer to ensure we bypass CPU caches (including 32MB - 128MB L3)
  bufferSize = 256ULL * 1024ULL * 1024ULL;

  buffer = ALIGNED_ALLOC(64, bufferSize);
  if (!buffer) {
    throw std::runtime_error(
        "Failed to allocate system memory buffer for latency test");
  }

  // Pointer chasing across 64-byte cache lines.
  // Striding by 64 bytes (16 uint32_t elements) guarantees that every jump
  // targets a distinct cache line, defeating hardware prefetchers and ensuring
  // true DRAM access latency while reducing index generation time from 8s to <100ms.
  constexpr uint32_t kStrideElements = 16; // 16 * sizeof(uint32_t) = 64 bytes
  uint32_t numLines = static_cast<uint32_t>(bufferSize / (kStrideElements * sizeof(uint32_t)));
  uint32_t *pBuffer = reinterpret_cast<uint32_t *>(buffer);

  std::vector<uint32_t> lineIndices(numLines);
  std::iota(lineIndices.begin(), lineIndices.end(), 0);

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(lineIndices.begin(), lineIndices.end(), g);

  // Create chasing chain on cache-line boundaries:
  for (uint32_t i = 0; i < numLines - 1; ++i) {
    pBuffer[lineIndices[i] * kStrideElements] = lineIndices[i + 1] * kStrideElements;
  }
  pBuffer[lineIndices[numLines - 1] * kStrideElements] = lineIndices[0] * kStrideElements; // Close the cycle
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
