#include "benchmarks/SysMemBandwidthBench.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#if !defined(_MSC_VER) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#define ENABLE_AVX2 1
#else
#define ENABLE_AVX2 0
#endif

#if ENABLE_AVX2
#include <immintrin.h>  // AVX2 intrinsics (GCC/Clang only)
#endif
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#define ALIGNED_ALLOC(alignment, size) _aligned_malloc(size, alignment)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#define ALIGNED_ALLOC(alignment, size) aligned_alloc(alignment, size)
#define ALIGNED_FREE(ptr) free(ptr)
#endif

// Check for AVX2 support.
// __builtin_cpu_supports is a GCC/Clang builtin. MSVC does not support it;
// on MSVC we always disable AVX2 and use the scalar fallback.
#if ENABLE_AVX2
static bool hasAVX2() { return __builtin_cpu_supports("avx2"); }
#else
static bool hasAVX2() { return false; }
#endif

SysMemBandwidthBench::SysMemBandwidthBench() {
  configs.push_back({"Read", SysMemTestMode::Read, 0});
  configs.push_back({"Write", SysMemTestMode::Write, 0});
  configs.push_back({"Copy", SysMemTestMode::ReadWrite, 0});

  // Single Threaded (Scaling / Channel Bandwidth approximation)
  configs.push_back({"Read (1 Thread)", SysMemTestMode::Read, 1});
  configs.push_back({"Write (1 Thread)", SysMemTestMode::Write, 1});
  configs.push_back({"Copy (1 Thread)", SysMemTestMode::ReadWrite, 1});
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
  buffer = ALIGNED_ALLOC(64, bufferSize);
  destBuffer = ALIGNED_ALLOC(64, bufferSize);

  if (!buffer || !destBuffer) {
    throw std::runtime_error("Failed to allocate system memory buffers");
  }

  // Determine thread count and initialize persistent worker thread pool
  threadCount = std::thread::hardware_concurrency();
  if (threadCount == 0) threadCount = 4;
  if (threadCount > 64) threadCount = 64; // Cap to 64 to avoid excessive memory bus contention

  // Perform parallel first-touch page initialization across worker threads so pages
  // are distributed across NUMA/CCD nodes rather than faulted onto thread 0's node.
  {
    std::vector<std::thread> initThreads;
    initThreads.reserve(threadCount);
    size_t chunkSize = (bufferSize / threadCount / 256) * 256;
    for (unsigned int i = 0; i < threadCount; ++i) {
      initThreads.emplace_back([this, i, chunkSize]() {
        size_t offset = i * chunkSize;
        size_t len = (i + 1 == threadCount) ? (bufferSize - offset) : chunkSize;
        std::memset(static_cast<char *>(buffer) + offset, 1, len);
        std::memset(static_cast<char *>(destBuffer) + offset, 0, len);
      });
    }
    for (auto &t : initThreads) {
      t.join();
    }
  }

  stopPool = false;
  activeConfigIdx = 0xFFFFFFFFu;
  sliceIndex = 0;
  workGeneration = 0;
  completedWorkers = 0;
  workers.clear();
  workers.reserve(threadCount);
  for (unsigned int i = 0; i < threadCount; ++i) {
    workers.emplace_back(&SysMemBandwidthBench::workerLoop, this, i);
  }
}

#if ENABLE_AVX2
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

    accum = _mm256_add_epi64(accum, v0);
    accum = _mm256_add_epi64(accum, v1);
    accum = _mm256_add_epi64(accum, v2);
    accum = _mm256_add_epi64(accum, v3);
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
#endif

// Fallbacks
void run_read_fallback(const void *src, size_t size) {
  const uint64_t *pSrc = reinterpret_cast<const uint64_t *>(src);
  size_t count = size / sizeof(uint64_t);
  volatile uint64_t sink = 0;
  for (size_t i = 0; i < count; ++i) {
    sink ^= pSrc[i];
  }
  (void)sink;
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

void SysMemBandwidthBench::workerLoop(unsigned int tid) {
  bool useAVX2 = hasAVX2();
  uint32_t localGen = 0;
  constexpr size_t kQuantumBytes = 512ULL * 1024ULL * 1024ULL; // Fixed 512 MB per Run()

  while (true) {
    uint32_t cfgIdx = 0;
    uint32_t curSlice = 0;
    {
      std::unique_lock<std::mutex> lock(poolMutex);
      cvStart.wait(lock, [&] { return stopPool.load() || workGeneration != localGen; });
      if (stopPool.load()) {
        return;
      }
      localGen = workGeneration;
      cfgIdx = activeConfigIdx;
      curSlice = sliceIndex;
    }

    if (cfgIdx < configs.size()) {
      const auto &config = configs[cfgIdx];
      uint32_t activeThreads = (config.numThreads > 0) ? config.numThreads : threadCount;
      if (tid < activeThreads) {
        size_t numSlices = std::max<size_t>(1, bufferSize / kQuantumBytes);
        size_t sliceBase = (curSlice % numSlices) * kQuantumBytes;

        size_t startInSlice =
            ((static_cast<uint64_t>(tid) * kQuantumBytes) / activeThreads) & ~size_t(255);
        size_t endInSlice = (tid + 1 == activeThreads)
            ? kQuantumBytes
            : (((static_cast<uint64_t>(tid + 1) * kQuantumBytes) / activeThreads) & ~size_t(255));
        size_t chunkSize = endInSlice - startInSlice;

        size_t offset = sliceBase + startInSlice;
        char *tSrc = static_cast<char *>(buffer) + offset;
        char *tDst = static_cast<char *>(destBuffer) + offset;

#if ENABLE_AVX2
        if (useAVX2) {
          if (config.mode == SysMemTestMode::Read) {
            run_read_avx2(tSrc, chunkSize);
            reinterpret_cast<uint64_t *>(destBuffer)[tid] = 0xDEADBEEFCAFEBABEULL;
          } else if (config.mode == SysMemTestMode::Write) {
            run_write_avx2(tDst, chunkSize);
          } else {
            run_copy_avx2(tSrc, tDst, chunkSize);
          }
        } else
#endif
        {
          if (config.mode == SysMemTestMode::Read) {
            run_read_fallback(tSrc, chunkSize);
            reinterpret_cast<uint64_t *>(destBuffer)[tid] = 0xDEADBEEFCAFEBABEULL;
          } else if (config.mode == SysMemTestMode::Write) {
            run_write_fallback(tDst, chunkSize);
          } else {
            run_copy_fallback(tSrc, tDst, chunkSize);
          }
        }
      }
    }

    if (++completedWorkers == threadCount) {
      cvDone.notify_one();
    }
  }
}

void SysMemBandwidthBench::Run(uint32_t config_idx) {
  if (config_idx >= configs.size() || workers.empty() || threadCount == 0)
    return;

  const auto &config = configs[config_idx];
  constexpr uint64_t kQuantumBytes = 512ULL * 1024ULL * 1024ULL; // Fixed 512 MB per Run()

  auto start = std::chrono::high_resolution_clock::now();

  {
    std::unique_lock<std::mutex> lock(poolMutex);
    completedWorkers = 0;
    if (activeConfigIdx != config_idx) {
      activeConfigIdx = config_idx;
      sliceIndex = 0;
    } else {
      sliceIndex++;
    }
    workGeneration++;
    cvStart.notify_all();
    cvDone.wait(lock, [&] { return completedWorkers.load() == threadCount; });
  }

  auto end = std::chrono::high_resolution_clock::now();

  double elapsedMs =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1000.0;

  lastRunTimeMs = elapsedMs;
  uint64_t totalBytes = kQuantumBytes;
  if (config.mode == SysMemTestMode::ReadWrite) {
    totalBytes *= 2;
  }
  lastRunBytes = totalBytes;
}

void SysMemBandwidthBench::Teardown() {
  {
    std::unique_lock<std::mutex> lock(poolMutex);
    stopPool = true;
    cvStart.notify_all();
  }
  for (auto &w : workers) {
    if (w.joinable()) {
      w.join();
    }
  }
  workers.clear();

  if (buffer) {
    ALIGNED_FREE(buffer);
    buffer = nullptr;
  }
  if (destBuffer) {
    ALIGNED_FREE(destBuffer);
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

bool SysMemBandwidthBench::IsEmulated(uint32_t config_idx) const {
  (void)config_idx;
  return !hasAVX2();
}

std::string SysMemBandwidthBench::GetConfigCaveat(uint32_t config_idx) const {
  (void)config_idx;
  if (!hasAVX2()) {
    return "Compiled without AVX2 streaming non-temporal stores; memory writes pass through CPU cache hierarchy";
  }
  return "";
}

bool SysMemBandwidthBench::ValidateResults(uint32_t config_idx) const {
  if (config_idx >= configs.size() || !buffer || !destBuffer || bufferSize == 0) {
    return false;
  }
  if (lastRunBytes == 0 || lastRunTimeMs <= 0.0) {
    return false;
  }

  const auto &config = configs[config_idx];
  const uint32_t *pDst32 = reinterpret_cast<const uint32_t *>(destBuffer);
  const size_t checkElements = std::min<size_t>(bufferSize / sizeof(uint32_t), 65536);

  if (config.mode == SysMemTestMode::Write) {
    // Write mode fills with 0xAAAAAAAA
    constexpr uint32_t expected = 0xAAAAAAAAu;
    for (size_t i = 0; i < checkElements; ++i) {
      if (pDst32[i] != expected) {
        std::cerr << "SysMemBandwidthBench Write validation failed at index " << i
                  << ": got 0x" << std::hex << pDst32[i] << ", expected 0x" << expected << std::dec << "\n";
        return false;
      }
    }
  } else if (config.mode == SysMemTestMode::ReadWrite) {
    // Copy mode copies buffer (initialized with byte 0x01 -> uint32 0x01010101)
    constexpr uint32_t expected = 0x01010101u;
    for (size_t i = 0; i < checkElements; ++i) {
      if (pDst32[i] != expected) {
        std::cerr << "SysMemBandwidthBench Copy validation failed at index " << i
                  << ": got 0x" << std::hex << pDst32[i] << ", expected 0x" << expected << std::dec << "\n";
        return false;
      }
    }
  } else {
    // Read mode verifies worker completion indicator
    uint32_t activeThreads = (config.numThreads > 0) ? config.numThreads : threadCount;
    const uint64_t *pDst64 = reinterpret_cast<const uint64_t *>(destBuffer);
    for (uint32_t t = 0; t < activeThreads; ++t) {
      if (pDst64[t] != 0xDEADBEEFCAFEBABEULL) {
        std::cerr << "SysMemBandwidthBench Read validation failed: worker " << t << " did not record valid completion\n";
        return false;
      }
    }
  }

  return true;
}
