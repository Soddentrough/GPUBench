#include "benchmarks/SysMemBandwidthBench.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define GPUBENCH_X86_SIMD 1
#ifdef _MSC_VER
#include <intrin.h>
#include <immintrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>
#endif
#else
#define GPUBENCH_X86_SIMD 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define GPUBENCH_ARM64_NEON 1
#include <arm_neon.h>
#else
#define GPUBENCH_ARM64_NEON 0
#endif

#ifdef _WIN32
#include <malloc.h>
#define ALIGNED_ALLOC(alignment, size) _aligned_malloc(size, alignment)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#define ALIGNED_ALLOC(alignment, size) aligned_alloc(alignment, size)
#define ALIGNED_FREE(ptr) free(ptr)
#endif

namespace {

enum class HostSimdTier { Scalar, ArmNeon, Avx2, Avx512 };

#if GPUBENCH_X86_SIMD
static void queryCpuid(int leaf, int subleaf, int regs[4]) {
#ifdef _MSC_VER
  __cpuidex(regs, leaf, subleaf);
#else
  __cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
#endif
}

static uint64_t queryXcr0() {
#ifdef _MSC_VER
  return _xgetbv(0);
#else
  uint32_t eax = 0, edx = 0;
  __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
  return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
}
#endif

static HostSimdTier detectHostSimdTier() {
  static const HostSimdTier cachedTier = []() -> HostSimdTier {
#if GPUBENCH_X86_SIMD
    int regs[4] = {0, 0, 0, 0};
    queryCpuid(0, 0, regs);
    const int maxLeaf = regs[0];
    if (maxLeaf < 7) {
      return HostSimdTier::Scalar;
    }
    queryCpuid(1, 0, regs);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) {
      return HostSimdTier::Scalar;
    }
    const uint64_t xcr0 = queryXcr0();
    const bool ymmEnabled = (xcr0 & 0x6ULL) == 0x6ULL;
    const bool zmmEnabled = (xcr0 & 0xE6ULL) == 0xE6ULL;

    queryCpuid(7, 0, regs);
    const bool avx2 = (regs[1] & (1 << 5)) != 0;
    const bool avx512f = (regs[1] & (1 << 16)) != 0;

    if (zmmEnabled && avx512f) {
      return HostSimdTier::Avx512;
    }
    if (ymmEnabled && avx2) {
      return HostSimdTier::Avx2;
    }
    return HostSimdTier::Scalar;
#elif GPUBENCH_ARM64_NEON
    return HostSimdTier::ArmNeon;
#else
    return HostSimdTier::Scalar;
#endif
  }();
  return cachedTier;
}

static bool hasStreamingSimd() {
  return detectHostSimdTier() != HostSimdTier::Scalar;
}

#if GPUBENCH_X86_SIMD
#if !defined(_MSC_VER) || defined(__clang__)
#define GPUBENCH_TARGET_AVX2 __attribute__((target("avx2")))
#define GPUBENCH_TARGET_AVX512 __attribute__((target("avx512f")))
#else
#define GPUBENCH_TARGET_AVX2
#define GPUBENCH_TARGET_AVX512
#endif

GPUBENCH_TARGET_AVX512 uint64_t run_read_avx512(const void *src, size_t size) {
  const __m512i *pSrc = reinterpret_cast<const __m512i *>(src);
  size_t count = size / sizeof(__m512i);

  __m512i accum0 = _mm512_setzero_si512();
  __m512i accum1 = _mm512_setzero_si512();
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m512i v0 = _mm512_load_si512(pSrc + i);
    __m512i v1 = _mm512_load_si512(pSrc + i + 1);
    __m512i v2 = _mm512_load_si512(pSrc + i + 2);
    __m512i v3 = _mm512_load_si512(pSrc + i + 3);

    accum0 = _mm512_add_epi64(accum0, _mm512_add_epi64(v0, v1));
    accum1 = _mm512_add_epi64(accum1, _mm512_add_epi64(v2, v3));
  }
  for (; i < count; ++i) {
    accum0 = _mm512_add_epi64(accum0, _mm512_load_si512(pSrc + i));
  }
  __m512i total = _mm512_add_epi64(accum0, accum1);
  alignas(64) uint64_t lanes[8];
  _mm512_store_si512(reinterpret_cast<__m512i *>(lanes), total);
  uint64_t x = 0;
  for (uint64_t v : lanes) {
    x ^= v;
  }
  return (x ^ 0xDEADBEEFCAFEBABEULL) | 1ULL;
}

GPUBENCH_TARGET_AVX512 void run_write_avx512(void *dst, size_t size) {
  __m512i *pDst = reinterpret_cast<__m512i *>(dst);
  size_t count = size / sizeof(__m512i);
  __m512i val = _mm512_set1_epi32(static_cast<int>(0xAAAAAAAAu));

  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    _mm512_stream_si512(pDst + i, val);
    _mm512_stream_si512(pDst + i + 1, val);
    _mm512_stream_si512(pDst + i + 2, val);
    _mm512_stream_si512(pDst + i + 3, val);
  }
  for (; i < count; ++i) {
    _mm512_stream_si512(pDst + i, val);
  }
  _mm_sfence();
}

GPUBENCH_TARGET_AVX512 void run_copy_avx512(const void *src, void *dst, size_t size) {
  const __m512i *pSrc = reinterpret_cast<const __m512i *>(src);
  __m512i *pDst = reinterpret_cast<__m512i *>(dst);
  size_t count = size / sizeof(__m512i);

  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    __m512i v0 = _mm512_load_si512(pSrc + i);
    __m512i v1 = _mm512_load_si512(pSrc + i + 1);
    __m512i v2 = _mm512_load_si512(pSrc + i + 2);
    __m512i v3 = _mm512_load_si512(pSrc + i + 3);

    _mm512_stream_si512(pDst + i, v0);
    _mm512_stream_si512(pDst + i + 1, v1);
    _mm512_stream_si512(pDst + i + 2, v2);
    _mm512_stream_si512(pDst + i + 3, v3);
  }
  for (; i < count; ++i) {
    _mm512_stream_si512(pDst + i, _mm512_load_si512(pSrc + i));
  }
  _mm_sfence();
}

GPUBENCH_TARGET_AVX2 uint64_t run_read_avx2(const void *src, size_t size) {
  const __m256i *pSrc = reinterpret_cast<const __m256i *>(src);
  size_t count = size / sizeof(__m256i);

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
  for (; i < count; ++i) {
    accum = _mm256_add_epi64(accum, _mm256_load_si256(pSrc + i));
  }

  alignas(32) uint64_t lanes[4];
  _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), accum);
  return ((lanes[0] ^ lanes[1] ^ lanes[2] ^ lanes[3]) ^ 0xDEADBEEFCAFEBABEULL) | 1ULL;
}

GPUBENCH_TARGET_AVX2 void run_write_avx2(void *dst, size_t size) {
  __m256i *pDst = reinterpret_cast<__m256i *>(dst);
  size_t count = size / sizeof(__m256i);
  __m256i val = _mm256_set1_epi32(static_cast<int>(0xAAAAAAAAu));

  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    _mm256_stream_si256(pDst + i, val);
    _mm256_stream_si256(pDst + i + 1, val);
    _mm256_stream_si256(pDst + i + 2, val);
    _mm256_stream_si256(pDst + i + 3, val);
  }
  for (; i < count; ++i) {
    _mm256_stream_si256(pDst + i, val);
  }
  _mm_sfence();
}

GPUBENCH_TARGET_AVX2 void run_copy_avx2(const void *src, void *dst, size_t size) {
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
  for (; i < count; ++i) {
    _mm256_stream_si256(pDst + i, _mm256_load_si256(pSrc + i));
  }
  _mm_sfence();
}
#endif

#if GPUBENCH_ARM64_NEON
uint64_t run_read_neon(const void *src, size_t size) {
  const uint64_t *pSrc = reinterpret_cast<const uint64_t *>(src);
  size_t vecCount = size / 16;
  uint64x2_t acc0 = vdupq_n_u64(0);
  uint64x2_t acc1 = vdupq_n_u64(0);

  size_t i = 0;
  for (; i + 4 <= vecCount; i += 4) {
    uint64x2_t v0 = vld1q_u64(pSrc + i * 2);
    uint64x2_t v1 = vld1q_u64(pSrc + (i + 1) * 2);
    uint64x2_t v2 = vld1q_u64(pSrc + (i + 2) * 2);
    uint64x2_t v3 = vld1q_u64(pSrc + (i + 3) * 2);
    acc0 = vaddq_u64(acc0, vaddq_u64(v0, v1));
    acc1 = vaddq_u64(acc1, vaddq_u64(v2, v3));
  }
  for (; i < vecCount; ++i) {
    acc0 = vaddq_u64(acc0, vld1q_u64(pSrc + i * 2));
  }
  uint64x2_t sum = vaddq_u64(acc0, acc1);
  uint64_t x = vgetq_lane_u64(sum, 0) ^ vgetq_lane_u64(sum, 1);
  return (x ^ 0xDEADBEEFCAFEBABEULL) | 1ULL;
}

void run_write_neon(void *dst, size_t size) {
  uint8_t *pDst = reinterpret_cast<uint8_t *>(dst);
  uint32x4_t val = vdupq_n_u32(0xAAAAAAAAu);
  size_t offset = 0;
#if !defined(_MSC_VER)
  for (; offset + 64 <= size; offset += 64) {
    uint8_t *ptr = pDst + offset;
    __asm__ volatile(
        "stnp %q0, %q0, [%1, #0]\n\t"
        "stnp %q0, %q0, [%1, #32]"
        :
        : "w"(val), "r"(ptr)
        : "memory");
  }
  __asm__ volatile("dmb ishst" ::: "memory");
#endif
  for (; offset + 16 <= size; offset += 16) {
    vst1q_u32(reinterpret_cast<uint32_t *>(pDst + offset), val);
  }
}

void run_copy_neon(const void *src, void *dst, size_t size) {
  const uint8_t *pSrc = reinterpret_cast<const uint8_t *>(src);
  uint8_t *pDst = reinterpret_cast<uint8_t *>(dst);
  size_t offset = 0;
#if !defined(_MSC_VER)
  for (; offset + 64 <= size; offset += 64) {
    const uint32_t *s = reinterpret_cast<const uint32_t *>(pSrc + offset);
    uint8_t *d = pDst + offset;
    uint32x4_t v0 = vld1q_u32(s);
    uint32x4_t v1 = vld1q_u32(s + 4);
    uint32x4_t v2 = vld1q_u32(s + 8);
    uint32x4_t v3 = vld1q_u32(s + 12);
    __asm__ volatile(
        "stnp %q0, %q1, [%4, #0]\n\t"
        "stnp %q2, %q3, [%4, #32]"
        :
        : "w"(v0), "w"(v1), "w"(v2), "w"(v3), "r"(d)
        : "memory");
  }
  __asm__ volatile("dmb ishst" ::: "memory");
#endif
  for (; offset + 16 <= size; offset += 16) {
    uint32x4_t v = vld1q_u32(reinterpret_cast<const uint32_t *>(pSrc + offset));
    vst1q_u32(reinterpret_cast<uint32_t *>(pDst + offset), v);
  }
}
#endif

uint64_t run_read_fallback(const void *src, size_t size) {
  const uint64_t *pSrc = reinterpret_cast<const uint64_t *>(src);
  size_t count = size / sizeof(uint64_t);
  uint64_t sink = 0;
  for (size_t i = 0; i < count; ++i) {
    sink ^= pSrc[i];
  }
  return (sink ^ 0xDEADBEEFCAFEBABEULL) | 1ULL;
}

void run_write_fallback(void *dst, size_t size) {
  uint64_t *pDst = reinterpret_cast<uint64_t *>(dst);
  size_t count = size / sizeof(uint64_t);
  for (size_t i = 0; i < count; ++i) {
    pDst[i] = 0xAAAAAAAAAAAAAAAAULL;
  }
}

void run_copy_fallback(const void *src, void *dst, size_t size) {
  std::memcpy(dst, src, size);
}

} // namespace

SysMemBandwidthBench::SysMemBandwidthBench() {
  configs.push_back({"Read", SysMemTestMode::Read, 0});
  configs.push_back({"Write", SysMemTestMode::Write, 0});
  configs.push_back({"Copy", SysMemTestMode::ReadWrite, 0});

  configs.push_back({"Read (1 Thread)", SysMemTestMode::Read, 1});
  configs.push_back({"Write (1 Thread)", SysMemTestMode::Write, 1});
  configs.push_back({"Copy (1 Thread)", SysMemTestMode::ReadWrite, 1});
}

SysMemBandwidthBench::~SysMemBandwidthBench() { Teardown(); }

const char *SysMemBandwidthBench::GetName() const {
  return "System Memory Bandwidth";
}

const char *SysMemBandwidthBench::GetMetric() const { return "GB/s"; }

const char *SysMemBandwidthBench::GetDescription(uint32_t config_idx) const {
  if (config_idx >= 3) {
    return "Single-core host DRAM bandwidth using non-temporal SIMD streaming (AVX-512 / AVX2 / NEON).";
  }
  return "Multi-threaded NUMA/CCD first-touch host DRAM bandwidth using non-temporal SIMD streaming (AVX-512 / AVX2 / NEON).";
}

bool SysMemBandwidthBench::IsSupported(const DeviceInfo & /*info*/,
                                       IComputeContext * /*context*/) const {
  return true;
}

void SysMemBandwidthBench::Setup(IComputeContext & /*context*/,
                                 const std::string & /*kernel_dir*/) {
  // 4 GB working set
  bufferSize = 4ULL * 1024ULL * 1024ULL * 1024ULL;

  // 64-byte alignment satisfies both AVX-512 (64B) and AVX2 (32B) streaming stores
  buffer = ALIGNED_ALLOC(64, bufferSize);
  destBuffer = ALIGNED_ALLOC(64, bufferSize);

  if (!buffer || !destBuffer) {
    throw std::runtime_error("Failed to allocate system memory buffers");
  }

  threadCount = std::thread::hardware_concurrency();
  if (threadCount == 0) threadCount = 4;
  if (threadCount > 64) threadCount = 64;

  // Parallel first-touch page initialization across NUMA/CCD domains
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
  readSinks.assign(threadCount, 0ULL);
  workers.clear();
  workers.reserve(threadCount);
  for (unsigned int i = 0; i < threadCount; ++i) {
    workers.emplace_back(&SysMemBandwidthBench::workerLoop, this, i);
  }
}

void SysMemBandwidthBench::workerLoop(unsigned int tid) {
  const HostSimdTier simdTier = detectHostSimdTier();
  uint32_t localGen = 0;
  constexpr size_t kQuantumBytes = 512ULL * 1024ULL * 1024ULL;

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

#if GPUBENCH_X86_SIMD
        if (simdTier == HostSimdTier::Avx512) {
          if (config.mode == SysMemTestMode::Read) {
            readSinks[tid] = run_read_avx512(tSrc, chunkSize);
            reinterpret_cast<uint64_t *>(destBuffer)[tid] = 0xDEADBEEFCAFEBABEULL;
          } else if (config.mode == SysMemTestMode::Write) {
            run_write_avx512(tDst, chunkSize);
          } else {
            run_copy_avx512(tSrc, tDst, chunkSize);
          }
        } else if (simdTier == HostSimdTier::Avx2) {
          if (config.mode == SysMemTestMode::Read) {
            readSinks[tid] = run_read_avx2(tSrc, chunkSize);
            reinterpret_cast<uint64_t *>(destBuffer)[tid] = 0xDEADBEEFCAFEBABEULL;
          } else if (config.mode == SysMemTestMode::Write) {
            run_write_avx2(tDst, chunkSize);
          } else {
            run_copy_avx2(tSrc, tDst, chunkSize);
          }
        } else
#endif
#if GPUBENCH_ARM64_NEON
        if (simdTier == HostSimdTier::ArmNeon) {
          if (config.mode == SysMemTestMode::Read) {
            readSinks[tid] = run_read_neon(tSrc, chunkSize);
            reinterpret_cast<uint64_t *>(destBuffer)[tid] = 0xDEADBEEFCAFEBABEULL;
          } else if (config.mode == SysMemTestMode::Write) {
            run_write_neon(tDst, chunkSize);
          } else {
            run_copy_neon(tSrc, tDst, chunkSize);
          }
        } else
#endif
        {
          if (config.mode == SysMemTestMode::Read) {
            readSinks[tid] = run_read_fallback(tSrc, chunkSize);
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
  constexpr uint64_t kQuantumBytes = 512ULL * 1024ULL * 1024ULL;

  auto start = std::chrono::high_resolution_clock::now();

  {
    std::unique_lock<std::mutex> lock(poolMutex);
    completedWorkers = 0;
    if (activeConfigIdx != config_idx) {
      activeConfigIdx = config_idx;
      sliceIndex = 0;
      if (config.mode == SysMemTestMode::Read) {
        std::fill(readSinks.begin(), readSinks.end(), 0ULL);
      }
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

BenchmarkResult SysMemBandwidthBench::GetResult(uint32_t /*config_idx*/) const {
  return {lastRunBytes, lastRunTimeMs};
}

uint32_t SysMemBandwidthBench::GetNumConfigs() const {
  return static_cast<uint32_t>(configs.size());
}

std::string SysMemBandwidthBench::GetConfigName(uint32_t config_idx) const {
  if (config_idx >= configs.size())
    return "Invalid";
  return configs[config_idx].name;
}

bool SysMemBandwidthBench::IsEmulated(uint32_t config_idx) const {
  (void)config_idx;
  return !hasStreamingSimd();
}

std::string SysMemBandwidthBench::GetConfigCaveat(uint32_t config_idx) const {
  (void)config_idx;
  switch (detectHostSimdTier()) {
  case HostSimdTier::Avx512:
    return "AVX-512 512-bit non-temporal streaming";
  case HostSimdTier::Avx2:
    return "";
  case HostSimdTier::ArmNeon:
    return "ARM64 NEON 128-bit streaming";
  case HostSimdTier::Scalar:
    return "Executing without SIMD non-temporal stores; memory writes pass through CPU cache hierarchy";
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
  constexpr size_t kQuantumBytes = 512ULL * 1024ULL * 1024ULL;
  const size_t numSlices = std::max<size_t>(1, bufferSize / kQuantumBytes);
  const size_t lastSliceOffset32 =
      (sliceIndex % numSlices) * (kQuantumBytes / sizeof(uint32_t));
  const uint32_t *pDst32 =
      reinterpret_cast<const uint32_t *>(destBuffer) + lastSliceOffset32;
  const size_t checkElements =
      std::min<size_t>(kQuantumBytes / sizeof(uint32_t), 65536);

  if (config.mode == SysMemTestMode::Write) {
    constexpr uint32_t expected = 0xAAAAAAAAu;
    for (size_t i = 0; i < checkElements; ++i) {
      if (pDst32[i] != expected) {
        std::cerr << "SysMemBandwidthBench Write validation failed at index " << i
                  << ": got 0x" << std::hex << pDst32[i] << ", expected 0x" << expected << std::dec << "\n";
        return false;
      }
    }
  } else if (config.mode == SysMemTestMode::ReadWrite) {
    constexpr uint32_t expected = 0x01010101u;
    for (size_t i = 0; i < checkElements; ++i) {
      if (pDst32[i] != expected) {
        std::cerr << "SysMemBandwidthBench Copy validation failed at index " << i
                  << ": got 0x" << std::hex << pDst32[i] << ", expected 0x" << expected << std::dec << "\n";
        return false;
      }
    }
  } else {
    const uint64_t *pDst64 = reinterpret_cast<const uint64_t *>(destBuffer);
    uint32_t activeThreads = (config.numThreads > 0) ? config.numThreads : threadCount;
    if (readSinks.size() < activeThreads) {
      return false;
    }
    for (uint32_t t = 0; t < activeThreads; ++t) {
      if (pDst64[t] != 0xDEADBEEFCAFEBABEULL || readSinks[t] == 0ULL) {
        std::cerr << "SysMemBandwidthBench Read validation failed: worker " << t
                  << " did not record valid completion\n";
        return false;
      }
    }
  }

  return true;
}
