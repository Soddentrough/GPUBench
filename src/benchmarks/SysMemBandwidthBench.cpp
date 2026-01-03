#include "benchmarks/SysMemBandwidthBench.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

// Check for AVX2 support
// This is a GCC/Clang builtin. For MSVC one would use __cpuid.
static bool hasAVX2() { return __builtin_cpu_supports("avx2"); }

SysMemBandwidthBench::SysMemBandwidthBench() {
  configs.push_back({"Read Bandwidth", SysMemTestMode::Read, 0});
  configs.push_back({"Write Bandwidth", SysMemTestMode::Write, 0});
  configs.push_back({"Copy Bandwidth", SysMemTestMode::ReadWrite, 0});

  // Single Threaded (Scaling / Channel Bandwidth approximation)
  configs.push_back({"Read Bandwidth (1 Thread)", SysMemTestMode::Read, 1});
  configs.push_back({"Write Bandwidth (1 Thread)", SysMemTestMode::Write, 1});
  configs.push_back(
      {"Copy Bandwidth (1 Thread)", SysMemTestMode::ReadWrite, 1});
}

SysMemBandwidthBench::~SysMemBandwidthBench() { Teardown(); }

const char *SysMemBandwidthBench::GetName() const {
  return "System Memory Bandwidth";
}

const char *SysMemBandwidthBench::GetMetric() const { return "GB/s"; }

bool SysMemBandwidthBench::IsSupported(const DeviceInfo &info,
                                       IComputeContext *context) const {
  return true;
}

void SysMemBandwidthBench::Setup(IComputeContext &context,
                                 const std::string &kernel_dir) {
  // 4GB buffer
  bufferSize = 4ULL * 1024ULL * 1024ULL * 1024ULL;

  // Use aligned_alloc for AVX
  buffer = aligned_alloc(64, bufferSize);
  destBuffer = aligned_alloc(64, bufferSize);

  if (!buffer || !destBuffer) {
    throw std::runtime_error("Failed to allocate system memory buffers");
  }

  // Initialize memory to avoid page faults during timed run (Linux lazy
  // allocation)
  std::memset(buffer, 1, bufferSize);
  std::memset(destBuffer, 0, bufferSize); // Touch pages
}

// AVX2 kernels
__attribute__((target("avx2"))) void run_read_avx2(const void *src,
                                                   size_t size) {
  const __m256i *pSrc = reinterpret_cast<const __m256i *>(src);
  size_t count = size / sizeof(__m256i);

  // Unroll 4x
  __m256i accum = _mm256_setzero_si256();
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m256i v0 = _mm256_load_si256(pSrc + i);
    __m256i v1 = _mm256_load_si256(pSrc + i + 1);
    __m256i v2 = _mm256_load_si256(pSrc + i + 2);
    __m256i v3 = _mm256_load_si256(pSrc + i + 3);

    accum = _mm256_xor_si256(accum, v0);
    accum = _mm256_xor_si256(accum, v1);
    accum = _mm256_xor_si256(accum, v2);
    accum = _mm256_xor_si256(accum, v3);
  }

  volatile __m256i sink = accum;
  (void)sink;
}

__attribute__((target("avx2"))) void run_write_avx2(void *dst, size_t size) {
  __m256i *pDst = reinterpret_cast<__m256i *>(dst);
  size_t count = size / sizeof(__m256i);
  __m256i val = _mm256_set1_epi32(0xAAAAAAAA);

  // Stream stores (bypass cache) are best for pure memory bandwidth writing.
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    _mm256_stream_si256(pDst + i, val);
    _mm256_stream_si256(pDst + i + 1, val);
    _mm256_stream_si256(pDst + i + 2, val);
    _mm256_stream_si256(pDst + i + 3, val);
  }
  _mm_sfence();
}

__attribute__((target("avx2"))) void run_copy_avx2(const void *src, void *dst,
                                                   size_t size) {
  const __m256i *pSrc = reinterpret_cast<const __m256i *>(src);
  __m256i *pDst = reinterpret_cast<__m256i *>(dst);
  size_t count = size / sizeof(__m256i);

  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m256i v0 = _mm256_load_si256(pSrc + i);
    __m256i v1 = _mm256_load_si256(pSrc + i + 1);
    __m256i v2 = _mm256_load_si256(pSrc + i + 2);
    __m256i v3 = _mm256_load_si256(pSrc + i + 3);

    _mm256_stream_si256(pDst + i, v0);
    _mm256_stream_si256(pDst + i + 1, v1);
    _mm256_stream_si256(pDst + i + 2, v2);
    _mm256_stream_si256(pDst + i + 3, v3);
  }
  _mm_sfence();
}

// Fallbacks
void run_read_fallback(const void *src, size_t size) {
  const uint64_t *pSrc = reinterpret_cast<const uint64_t *>(src);
  size_t count = size / sizeof(uint64_t);
  volatile uint64_t sink = 0;
  for (size_t i = 0; i < count; ++i) {
    sink ^= pSrc[i];
  }
}

void run_write_fallback(void *dst, size_t size) {
  uint64_t *pDst = reinterpret_cast<uint64_t *>(dst);
  size_t count = size / sizeof(uint64_t);
  // Standard stores will go through cache hierarchy
  for (size_t i = 0; i < count; ++i) {
    pDst[i] = 0xAAAAAAAAULL;
  }
}

void run_copy_fallback(const void *src, void *dst, size_t size) {
  std::memcpy(dst, src, size);
}

void SysMemBandwidthBench::Run(uint32_t config_idx) {
  if (config_idx >= configs.size())
    return;

  const auto &config = configs[config_idx];
  bool useAVX2 = hasAVX2();

  // Determine thread count
  unsigned int threadCount = config.numThreads;
  if (threadCount == 0) {
    threadCount = std::thread::hardware_concurrency();
    if (threadCount == 0)
      threadCount = 4; // Fallback
  }

  // Split buffer among threads
  size_t chunkSize = bufferSize / threadCount;
  // Align chunk size to 256 bytes (safe for AVX unroll)
  chunkSize = (chunkSize / 256) * 256;

  std::vector<std::thread> threads;
  std::atomic<int> barrier_counter(0);

  auto thread_func = [&](int tid) {
    size_t offset = tid * chunkSize;
    char *tSrc = (char *)buffer + offset;
    char *tDst = (char *)destBuffer + offset;

    // Simple barrier
    barrier_counter++;
    while (barrier_counter < (int)threadCount) {
      std::this_thread::yield();
    }

    if (useAVX2) {
      if (config.mode == SysMemTestMode::Read) {
        run_read_avx2(tSrc, chunkSize);
      } else if (config.mode == SysMemTestMode::Write) {
        run_write_avx2(tSrc, chunkSize);
      } else {
        run_copy_avx2(tSrc, tDst, chunkSize);
      }
    } else {
      if (config.mode == SysMemTestMode::Read) {
        run_read_fallback(tSrc, chunkSize);
      } else if (config.mode == SysMemTestMode::Write) {
        run_write_fallback(tSrc, chunkSize);
      } else {
        run_copy_fallback(tSrc, tDst, chunkSize);
      }
    }
  };

  auto start = std::chrono::high_resolution_clock::now();

  for (unsigned int i = 0; i < threadCount; ++i) {
    threads.emplace_back(thread_func, i);
  }

  for (auto &t : threads) {
    t.join();
  }

  auto end = std::chrono::high_resolution_clock::now();

  double elapsedMs =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1000.0;

  lastRunTimeMs = elapsedMs;
  // Calculate bytes transferred
  uint64_t totalBytes = chunkSize * threadCount;
  if (config.mode == SysMemTestMode::ReadWrite) {
    totalBytes *= 2;
  }
  lastRunBytes = totalBytes;
}

void SysMemBandwidthBench::Teardown() {
  if (buffer) {
    std::free(buffer);
    buffer = nullptr;
  }
  if (destBuffer) {
    std::free(destBuffer);
    destBuffer = nullptr;
  }
}

BenchmarkResult SysMemBandwidthBench::GetResult(uint32_t config_idx) const {
  return {lastRunBytes, lastRunTimeMs};
}

uint32_t SysMemBandwidthBench::GetNumConfigs() const { return configs.size(); }

std::string SysMemBandwidthBench::GetConfigName(uint32_t config_idx) const {
  if (config_idx >= configs.size())
    return "Invalid";
  return configs[config_idx].name;
}
