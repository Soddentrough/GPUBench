#include "test_harness.h"
#include "core/ComputeBackendFactory.h"
#include "core/VulkanContext.h"
#include "core/ResultFormatter.h"
#include "utils/KernelPath.h"

#define private public
#include "benchmarks/MemBandwidthBench.h"
#include "benchmarks/CacheBench.h"
#include "benchmarks/CacheLatencyCurveBench.h"
#include "benchmarks/SysMemBandwidthBench.h"
#include "benchmarks/SysMemLatencyBench.h"
#undef private

#include <vector>
#include <iostream>
#include <limits>
#include <cmath>
#include <cstring>

namespace {

std::unique_ptr<IComputeContext> createVulkanContext() {
  if (!ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
    return nullptr;
  }
  auto ctx = ComputeBackendFactory::create(ComputeBackend::Vulkan, false, false);
  ctx->pickDevice(0);
  return ctx;
}

} // namespace

// ============================================================================
// 1. MemBandwidthBench Adversarial Negative Validation Tests
// ============================================================================

TEST_CASE(M3_AdversarialChallenge, MemBandwidth_PositiveValidation_AllModes) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  MemBandwidthBench bench;
  bench.Setup(*ctx, kdir);

  uint32_t numConfigs = bench.GetNumConfigs();
  ASSERT_EQ(numConfigs, 9u);

  // Run and validate every configuration (Read, Write, R/W across 128, 256, 1024 threads)
  for (uint32_t cfg = 0; cfg < numConfigs; ++cfg) {
    bench.Run(cfg);
    ctx->waitIdle();
    bool valid = bench.ValidateResults(cfg);
    ASSERT_TRUE(valid);
  }

  bench.Teardown();
}

TEST_CASE(M3_AdversarialChallenge, MemBandwidth_NegativeValidation_ZeroedBuffer) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  MemBandwidthBench bench;
  bench.Setup(*ctx, kdir);

  // Config 0: Read mode (expects 2048.0f)
  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));

  // Deliberately wipe entire checked region of outputBuffer with zeros
  std::vector<float> zeros(16384 * 4, 0.0f);
  ctx->writeBuffer(bench.outputBuffer, 0, zeros.size() * sizeof(float), zeros.data());
  ctx->waitIdle();

  // ValidateResults MUST reject the zeroed buffer
  ASSERT_FALSE(bench.ValidateResults(0));

  // Config 1: Write mode (expects 1.0f)
  bench.Run(1);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(1));

  ctx->writeBuffer(bench.outputBuffer, 0, zeros.size() * sizeof(float), zeros.data());
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(1));

  // Config 2: ReadWrite mode (expects 2.0f)
  bench.Run(2);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(2));

  ctx->writeBuffer(bench.outputBuffer, 0, zeros.size() * sizeof(float), zeros.data());
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(2));

  bench.Teardown();
}

TEST_CASE(M3_AdversarialChallenge, MemBandwidth_NegativeValidation_CorruptedFloatValues) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  MemBandwidthBench bench;
  bench.Setup(*ctx, kdir);

  // Test single corrupted float element at element 0
  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));

  float corruptVal = 9999.0f;
  ctx->writeBuffer(bench.outputBuffer, 0, sizeof(float), &corruptVal);
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(0));

  // Test single corrupted float element at lane 100 component 2
  bench.Run(6); // Read 1024
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(6));

  size_t offsetBytes = (100 * 4 + 2) * sizeof(float);
  ctx->writeBuffer(bench.outputBuffer, offsetBytes, sizeof(float), &corruptVal);
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(6));

  bench.Teardown();
}

// ============================================================================
// 2. CacheBench (L0 Vector Cache Latency) Negative Validation Tests
// ============================================================================

TEST_CASE(M3_AdversarialChallenge, CacheBench_PositiveAndNegativeValidation_Word1) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  const size_t l0_size = 16 * 1024;
  CacheBench bench(
      "L0 Vector Cache Latency (16 KB)", "ns", l0_size, "l0_cache_latency",
      std::vector<uint32_t>{},
      std::vector<std::string>{"l0l"}, 0);

  bench.Setup(*ctx, kdir);

  // 1. Legitimate execution must validate
  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));

  // Read the genuine terminal index from word 1 (offset 4)
  uint32_t genuineTerminal = 0;
  ctx->readBuffer(bench.buffer, sizeof(uint32_t), sizeof(uint32_t), &genuineTerminal);

  // 2. Adversarial corruption: write genuineTerminal + 1 to word 1
  uint32_t badTerminal = genuineTerminal + 1;
  ctx->writeBuffer(bench.buffer, sizeof(uint32_t), sizeof(uint32_t), &badTerminal);
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(0));

  // 3. Adversarial corruption: zero out word 1
  uint32_t zeroTerminal = 0;
  ctx->writeBuffer(bench.buffer, sizeof(uint32_t), sizeof(uint32_t), &zeroTerminal);
  ctx->waitIdle();
  if (genuineTerminal != 0) {
    ASSERT_FALSE(bench.ValidateResults(0));
  }

  // 4. Adversarial corruption: write 0xDEADBEEF to word 1
  uint32_t deadBeef = 0xDEADBEEFu;
  ctx->writeBuffer(bench.buffer, sizeof(uint32_t), sizeof(uint32_t), &deadBeef);
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(0));

  bench.Teardown();
}

// ============================================================================
// 3. CacheLatencyCurveBench Negative Validation Tests
// ============================================================================

TEST_CASE(M3_AdversarialChallenge, CacheLatencyCurve_PositiveAndNegativeValidation_Word1) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  CacheLatencyCurveBench bench;
  bench.Setup(*ctx, kdir);

  uint32_t numConfigs = bench.GetNumConfigs();
  ASSERT_EQ(numConfigs, 15u);

  // Test across L0, L1, L2, L3, and DRAM boundary configs:
  // Config 0 (16 KB), Config 4 (256 KB), Config 7 (2 MB), Config 11 (32 MB), Config 14 (256 MB)
  uint32_t testConfigs[] = {0, 4, 7, 11, 14};

  for (uint32_t cfg : testConfigs) {
    bench.Run(cfg);
    ctx->waitIdle();
    ASSERT_TRUE(bench.ValidateResults(cfg));

    uint64_t word1Offset = static_cast<uint64_t>(bench.configs[cfg].startWordIndex + 1) * sizeof(uint32_t);
    uint32_t expectedTerminal = bench.configs[cfg].expectedTerminalIndex;

    // Corrupt word 1 with expectedTerminal + 1
    uint32_t badVal = expectedTerminal + 1;
    ctx->writeBuffer(bench.buffer, word1Offset, sizeof(uint32_t), &badVal);
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(cfg));

    // Restore genuine value and re-verify
    ctx->writeBuffer(bench.buffer, word1Offset, sizeof(uint32_t), &expectedTerminal);
    ctx->waitIdle();
    ASSERT_TRUE(bench.ValidateResults(cfg));
  }

  bench.Teardown();
}

// ============================================================================
// 4. SysMemBandwidthBench Negative Validation Tests
// ============================================================================

TEST_CASE(M3_AdversarialChallenge, SysMemBandwidth_PositiveAndNegativeValidation) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  SysMemBandwidthBench bench;
  bench.Setup(*ctx, kdir);

  // Config 0: Write mode (expects 0xAAAAAAAA)
  bench.Run(0);
  ASSERT_TRUE(bench.ValidateResults(0));

  // Deliberately corrupt destBuffer
  uint32_t *pDst32 = reinterpret_cast<uint32_t *>(bench.destBuffer);
  pDst32[0] = 0x12345678u;
  ASSERT_FALSE(bench.ValidateResults(0));

  // Config 2: Copy mode (expects 0x01010101)
  bench.Run(2);
  ASSERT_TRUE(bench.ValidateResults(2));

  pDst32[0] = 0x00000000u;
  ASSERT_FALSE(bench.ValidateResults(2));

  // Config 1: Read mode (expects sentinel 0xDEADBEEFCAFEBABEULL per thread)
  bench.Run(1);
  ASSERT_TRUE(bench.ValidateResults(1));

  uint64_t *pDst64 = reinterpret_cast<uint64_t *>(bench.destBuffer);
  pDst64[0] = 0ULL;
  ASSERT_FALSE(bench.ValidateResults(1));

  bench.Teardown();
}

// ============================================================================
// 5. SysMemLatencyBench Negative Validation Tests
// ============================================================================

TEST_CASE(M3_AdversarialChallenge, SysMemLatency_PositiveAndNegativeValidation) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  SysMemLatencyBench bench;
  bench.Setup(*ctx, kdir);

  // 1. Positive validation
  bench.Run(0);
  ASSERT_TRUE(bench.ValidateResults(0));

  uint32_t *pBuffer = reinterpret_cast<uint32_t *>(bench.buffer);
  uint32_t originalTarget = pBuffer[0];

  // 2. Corrupt target to unaligned index (must be multiple of 16)
  pBuffer[0] = 5u; // 5 % 16 != 0
  ASSERT_FALSE(bench.ValidateResults(0));

  // 3. Corrupt target to self-loop (0 points to 0)
  pBuffer[0] = 0u;
  ASSERT_FALSE(bench.ValidateResults(0));

  // 4. Corrupt target to out-of-bounds index
  uint32_t totalElements = static_cast<uint32_t>(bench.bufferSize / sizeof(uint32_t));
  pBuffer[0] = totalElements + 16u;
  ASSERT_FALSE(bench.ValidateResults(0));

  // Restore
  pBuffer[0] = originalTarget;
  ASSERT_TRUE(bench.ValidateResults(0));

  bench.Teardown();
}

// ============================================================================
// 6. No Self-Validating Stubs Audit
// ============================================================================

TEST_CASE(M3_AdversarialChallenge, NoSelfValidatingStubs_OOBAndUninitialized) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  // 1. MemBandwidthBench
  {
    MemBandwidthBench bench;
    // Uninitialized must return false
    ASSERT_FALSE(bench.ValidateResults(0));
    bench.Setup(*ctx, kdir);
    // Out of bounds config must return false
    ASSERT_FALSE(bench.ValidateResults(999));
    ASSERT_FALSE(bench.ValidateResults(bench.GetNumConfigs()));
    bench.Teardown();
    // Post teardown must return false
    ASSERT_FALSE(bench.ValidateResults(0));
  }

  // 2. CacheLatencyCurveBench
  {
    CacheLatencyCurveBench bench;
    ASSERT_FALSE(bench.ValidateResults(0));
    bench.Setup(*ctx, kdir);
    ASSERT_FALSE(bench.ValidateResults(999));
    ASSERT_FALSE(bench.ValidateResults(bench.GetNumConfigs()));
    bench.Teardown();
    ASSERT_FALSE(bench.ValidateResults(0));
  }

  // 3. SysMemBandwidthBench
  {
    SysMemBandwidthBench bench;
    ASSERT_FALSE(bench.ValidateResults(0));
    bench.Setup(*ctx, kdir);
    ASSERT_FALSE(bench.ValidateResults(999));
    ASSERT_FALSE(bench.ValidateResults(bench.GetNumConfigs()));
    bench.Teardown();
    ASSERT_FALSE(bench.ValidateResults(0));
  }

  // 4. SysMemLatencyBench
  {
    SysMemLatencyBench bench;
    ASSERT_FALSE(bench.ValidateResults(0));
    bench.Setup(*ctx, kdir);
    ASSERT_FALSE(bench.ValidateResults(999));
    ASSERT_FALSE(bench.ValidateResults(bench.GetNumConfigs()));
    bench.Teardown();
    ASSERT_FALSE(bench.ValidateResults(0));
  }
}
