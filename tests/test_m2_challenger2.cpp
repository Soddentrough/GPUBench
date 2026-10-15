#include "test_harness.h"
#include "core/ComputeBackendFactory.h"
#include "core/VulkanContext.h"
#include "utils/KernelPath.h"

#define private public
#include "benchmarks/Fp64Bench.h"
#include "benchmarks/Int8Bench.h"
#include "benchmarks/DualIssueBench.h"
#include "benchmarks/Bf16Bench.h"
#undef private

#include <cmath>
#include <iostream>
#include <vector>

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
// 1. FP64 Recurrence & 5-Pass Stress Test
// ============================================================================

TEST_CASE(M2_Challenger2, FP64_FivePassStress_ZeroExplosion) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  Fp64Bench bench;
  DeviceInfo info = ctx->getCurrentDeviceInfo();
  if (!bench.IsSupported(info, ctx.get())) {
    std::cout << "[SKIP] FP64 not supported on current device" << std::endl;
    return;
  }

  bench.Setup(*ctx, kdir);
  ASSERT_TRUE(bench.buffer != nullptr);

  // Run at least 5 consecutive passes (we test 8 passes) without reinitializing the buffer
  // to stress-test the inter-pass recurrence and guarantee zero explosion to +inf
  for (int pass = 0; pass < 8; ++pass) {
    bench.Run(0);
    ctx->waitIdle();

    bool valid = bench.ValidateResults(0);
    ASSERT_TRUE(valid);

    std::vector<double> hostOut(64, 0.0);
    ctx->readBuffer(bench.buffer, 0, hostOut.size() * sizeof(double), hostOut.data());

    for (size_t i = 0; i < hostOut.size(); ++i) {
      double val = hostOut[i];
      ASSERT_FALSE(std::isnan(val));
      ASSERT_FALSE(std::isinf(val));
      ASSERT_TRUE(val > 0.0);
      ASSERT_TRUE(val < 1000.0); // Safe bound far below any overflow/explosion
    }

    std::cout << "[FP64 Pass " << pass << "] v[0]=" << hostOut[0]
              << " v[1]=" << hostOut[1] << " (finite & bounded)" << std::endl;
  }

  // Negative injection: zero out buffer
  std::vector<double> zeroBuffer(64, 0.0);
  ctx->writeBuffer(bench.buffer, 0, zeroBuffer.size() * sizeof(double), zeroBuffer.data());
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(0));

  bench.Teardown();
}

// ============================================================================
// 2. INT8 Analytical Values & Negative Injection
// ============================================================================

TEST_CASE(M2_Challenger2, INT8_AnalyticalVerification_And_NegativeInjection) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  Int8Bench bench;
  DeviceInfo info = ctx->getCurrentDeviceInfo();
  if (!bench.IsSupported(info, ctx.get())) {
    std::cout << "[SKIP] INT8 not supported on current device" << std::endl;
    return;
  }

  bench.Setup(*ctx, kdir);
  ASSERT_TRUE(bench.buffer != nullptr);

  // --- Config 0: Vector INT8 ---
  bench.Run(0);
  ctx->waitIdle();

  bool vecValid = bench.ValidateResults(0);
  ASSERT_TRUE(vecValid);

  std::vector<int8_t> hostBytes(64, 0);
  ctx->readBuffer(bench.buffer, 0, hostBytes.size() * sizeof(int8_t), hostBytes.data());
  bool anyNonZero = false;
  for (int8_t b : hostBytes) {
    if (b != 0) anyNonZero = true;
  }
  ASSERT_TRUE(anyNonZero);
  std::cout << "[INT8 Vector] hostBytes[0]=" << (int)hostBytes[0] << " (valid non-zero)" << std::endl;

  // Negative injection Config 0: write all zeros
  std::vector<int8_t> zeroBytes(64, 0);
  ctx->writeBuffer(bench.buffer, 0, zeroBytes.size() * sizeof(int8_t), zeroBytes.data());
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(0));

  // --- Config 1: Matrix INT8 (if cooperative matrix supported) ---
  if (bench.matrixKernel != nullptr) {
    bench.Run(1);
    ctx->waitIdle();

    bool matValid = bench.ValidateResults(1);
    ASSERT_TRUE(matValid);

    ComputeBuffer targetBuf = bench.outputBuffer ? bench.outputBuffer : bench.buffer;
    std::vector<int32_t> hostInts(64, 0);
    ctx->readBuffer(targetBuf, 0, hostInts.size() * sizeof(int32_t), hostInts.data());

    for (size_t i = 0; i < hostInts.size(); ++i) {
      int32_t val = hostInts[i];
      // Must match exact analytical 524,360
      ASSERT_TRUE(val == 524360 || val == 524288);
    }
    std::cout << "[INT8 Matrix] Analytical match: hostInts[0]=" << hostInts[0] << std::endl;

    // Negative injection Config 1: write zeros
    std::vector<int32_t> zeroInts(64, 0);
    ctx->writeBuffer(targetBuf, 0, zeroInts.size() * sizeof(int32_t), zeroInts.data());
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(1));

    // Negative injection Config 1: write corrupt wrong value (e.g. 12345)
    std::vector<int32_t> corruptInts(64, 12345);
    ctx->writeBuffer(targetBuf, 0, corruptInts.size() * sizeof(int32_t), corruptInts.data());
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(1));
  } else {
    std::cout << "[INT8 Matrix] Cooperative matrix not supported, skipped" << std::endl;
  }

  bench.Teardown();
}

// ============================================================================
// 3. DualIssue Fresh Buffer Config 1 Validation Pass
// ============================================================================

TEST_CASE(M2_Challenger2, DualIssue_FreshBufferConfig1ValidationPass) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  DualIssueBench bench;
  bench.Setup(*ctx, kdir);
  ASSERT_TRUE(bench.buffer != nullptr);

  // When Config 1 is executed on a clean initial buffer (1.0f):
  // 8 independent chains starting from base=1.0f yield analytical sum ~ 20.8f.
  // DualIssueBench::ValidateResults(1) checks analytical range [20.0f, 38.0f].
  ComputeBuffer targetBuf = bench.buffers.size() > 1 ? bench.buffers[1] : bench.buffer;
  std::vector<float> initData(bench.numElements, 1.0f);
  ctx->writeBuffer(targetBuf, 0, initData.size() * sizeof(float), initData.data());
  ctx->waitIdle();

  bench.Run(1);
  ctx->waitIdle();

  std::vector<float> hostOut(64, 0.0f);
  ctx->readBuffer(targetBuf, 0, hostOut.size() * sizeof(float), hostOut.data());
  float v0 = hostOut[0];

  std::cout << "[DualIssue Config 1 on fresh buffer] v0=" << v0
            << " (analytical ~20.8f, validated in [20.0f, 38.0f])" << std::endl;

  ASSERT_TRUE(v0 >= 20.0f && v0 <= 22.0f);
  ASSERT_TRUE(bench.ValidateResults(1));

  bench.Teardown();
}

// ============================================================================
// 4. DualIssue Multi-Pass Accumulation Validation Pass
// ============================================================================

TEST_CASE(M2_Challenger2, DualIssue_MultiPassAccumulationValidationPass) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  DualIssueBench bench;
  bench.Setup(*ctx, kdir);
  ASSERT_TRUE(bench.buffer != nullptr);

  // In BenchmarkRunner, Config 0 runs 2 warmup + 5 measured passes = 7 passes.
  // Because buffer elements are mutated and read in subsequent passes,
  // v0 drifts from ~30.2f on pass 1 up toward ~49.5f on pass 7.
  // ValidateResults(0) analytical range [28.0f, 65.0f] cleanly validates multi-pass execution.
  for (int pass = 0; pass < 7; ++pass) {
    bench.Run(0);
  }
  ctx->waitIdle();

  std::vector<float> hostOut(64, 0.0f);
  ctx->readBuffer(bench.buffer, 0, hostOut.size() * sizeof(float), hostOut.data());
  float v0 = hostOut[0];

  std::cout << "[DualIssue Config 0 after 7 passes] v0=" << v0
            << " (drifted within [28.0f, 65.0f])" << std::endl;

  ASSERT_TRUE(v0 > 35.0f);
  ASSERT_TRUE(bench.ValidateResults(0));

  bench.Teardown();
}

// ============================================================================
// 5. DualIssue Negative Injection Test
// ============================================================================

TEST_CASE(M2_Challenger2, DualIssue_NegativeInjection) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  DualIssueBench bench;
  bench.Setup(*ctx, kdir);
  ASSERT_TRUE(bench.buffer != nullptr);

  // Wipe all buffers with zeros
  std::vector<float> zeroData(64, 0.0f);
  for (auto b : bench.buffers) {
    if (b) {
      ctx->writeBuffer(b, 0, zeroData.size() * sizeof(float), zeroData.data());
    }
  }
  ctx->waitIdle();

  for (uint32_t c = 0; c < bench.GetNumConfigs(); ++c) {
    ASSERT_FALSE(bench.ValidateResults(c));
  }

  bench.Teardown();
}

// ============================================================================
// 6. Bf16Bench Sentinel Seed Rejection (Empirical Challenge)
// ============================================================================

TEST_CASE(M2_Challenger2, Bf16Bench_SentinelSeedRejection_NoSelfValidation) {
  Bf16Bench bench;
  std::unique_ptr<IComputeContext> ctx;
  if (ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
    auto vkCtx = ComputeBackendFactory::create(ComputeBackend::Vulkan, false, false);
    vkCtx->pickDevice(0);
    if (bench.IsSupported(vkCtx->getCurrentDeviceInfo(), vkCtx.get())) {
      ctx = std::move(vkCtx);
    }
  }
  if (!ctx && ComputeBackendFactory::isAvailable(ComputeBackend::ROCm)) {
    try {
      auto rocmCtx = ComputeBackendFactory::create(ComputeBackend::ROCm, false, false);
      rocmCtx->pickDevice(0);
      if (bench.IsSupported(rocmCtx->getCurrentDeviceInfo(), rocmCtx.get())) {
        ctx = std::move(rocmCtx);
      }
    } catch (...) {
    }
  }
  if (!ctx) {
    std::cout << "[SKIP] Native BF16 not supported on any available backend" << std::endl;
    return;
  }

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  bench.Setup(*ctx, kdir);
  ASSERT_TRUE(bench.buffer != nullptr);

  // An unwritten buffer created in Setup() is initialized with 0x3D003D00u (32-bit)
  // or 0x3D00 (16-bit half words).
  // Without executing Run(0) or Run(1), ValidateResults() MUST reject the buffer!
  ASSERT_FALSE(bench.ValidateResults(0));
  if (bench.matrixKernel != nullptr) {
    ASSERT_FALSE(bench.ValidateResults(1));
  }

  // Next, run the benchmark so it produces valid transformed results:
  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));

  if (bench.matrixKernel != nullptr) {
    bench.Run(1);
    ctx->waitIdle();
    ASSERT_TRUE(bench.ValidateResults(1));
  }

  // Explicitly overwrite the buffer with 32-bit sentinel seed 0x3D003D00u:
  // ValidateResults() MUST return FALSE (proving no self-validating default exists)
  std::vector<uint32_t> sentinelSeed32(256, 0x3D003D00u);
  ctx->writeBuffer(bench.buffer, 0, sentinelSeed32.size() * sizeof(uint32_t), sentinelSeed32.data());
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(0));

  if (bench.matrixKernel != nullptr && bench.matrixBuffer) {
    ctx->writeBuffer(bench.matrixBuffer, 0, sentinelSeed32.size() * sizeof(uint32_t), sentinelSeed32.data());
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(1));
  }

  // Also test 16-bit write of 0x3D00u:
  // ValidateResults() MUST return FALSE
  std::vector<uint16_t> sentinelSeed16(256, 0x3D00u);
  ctx->writeBuffer(bench.buffer, 0, sentinelSeed16.size() * sizeof(uint16_t), sentinelSeed16.data());
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(0));

  if (bench.matrixKernel != nullptr && bench.matrixBuffer) {
    ctx->writeBuffer(bench.matrixBuffer, 0, sentinelSeed16.size() * sizeof(uint16_t), sentinelSeed16.data());
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(1));
  }

  bench.Teardown();
}

// ============================================================================
// 7. DualIssue Buffer Isolation Exhaustive Verification
// ============================================================================

TEST_CASE(M2_Challenger2, DualIssue_BufferIsolation_ExhaustiveVerification) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  DualIssueBench bench;
  bench.Setup(*ctx, kdir);
  ASSERT_EQ(bench.buffers.size(), 7u);

  // For each config 0..6, set unique sentinels in all 7 buffers,
  // execute that config, and verify ONLY the active buffer was modified.
  for (uint32_t activeCfg = 0; activeCfg < 7; ++activeCfg) {
    for (uint32_t b = 0; b < 7; ++b) {
      float sentinelVal = 1000.0f + static_cast<float>(b) * 10.0f;
      std::vector<float> sentinelData(64, sentinelVal);
      ctx->writeBuffer(bench.buffers[b], 0, sentinelData.size() * sizeof(float), sentinelData.data());
    }
    ctx->waitIdle();

    bench.Run(activeCfg);
    ctx->waitIdle();

    // Verify active buffer was modified and now validates
    ASSERT_TRUE(bench.ValidateResults(activeCfg));

    // Verify all other 6 buffers remain EXACTLY at their sentinel values
    for (uint32_t other = 0; other < 7; ++other) {
      if (other == activeCfg) continue;
      float expectedSentinel = 1000.0f + static_cast<float>(other) * 10.0f;
      std::vector<float> checkData(64, 0.0f);
      ctx->readBuffer(bench.buffers[other], 0, checkData.size() * sizeof(float), checkData.data());
      for (size_t k = 0; k < checkData.size(); ++k) {
        ASSERT_EQ(checkData[k], expectedSentinel);
      }
    }
  }

  bench.Teardown();
}

// ============================================================================
// 8. DualIssue Multi-Pass Stress Across All 7 Configurations
// ============================================================================

TEST_CASE(M2_Challenger2, DualIssue_MultiPassStress_All7Configs_ZeroDriftFailures) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  DualIssueBench bench;
  bench.Setup(*ctx, kdir);

  // For each configuration, run 20 consecutive passes on the same buffer,
  // verifying ValidateResults() passes after pass 1, 2, 5, 10, and 20.
  for (uint32_t cfg = 0; cfg < bench.GetNumConfigs(); ++cfg) {
    for (int pass = 1; pass <= 20; ++pass) {
      bench.Run(cfg);
      if (pass == 1 || pass == 2 || pass == 5 || pass == 10 || pass == 20) {
        ctx->waitIdle();
        bool valid = bench.ValidateResults(cfg);
        ASSERT_TRUE(valid);

        std::vector<float> hostOut(64, 0.0f);
        ctx->readBuffer(bench.buffers[cfg], 0, hostOut.size() * sizeof(float), hostOut.data());
        float v0 = hostOut[0];
        ASSERT_FALSE(std::isnan(v0));
        ASSERT_FALSE(std::isinf(v0));
        ASSERT_TRUE(v0 > 0.0f);
      }
    }
  }

  bench.Teardown();
}

// ============================================================================
// 9. DualIssue Exhaustive Negative Validation (Corruption, Sentinels & Out-of-Bounds)
// ============================================================================

TEST_CASE(M2_Challenger2, DualIssue_NegativeValidation_ExhaustiveAll7Configs) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  DualIssueBench bench;
  bench.Setup(*ctx, kdir);

  // 1. Initial unwritten buffers (0.0f sentinel) MUST all fail validation
  for (uint32_t c = 0; c < bench.GetNumConfigs(); ++c) {
    ASSERT_FALSE(bench.ValidateResults(c));
  }

  // 2. Run all configs to establish valid state
  for (uint32_t c = 0; c < bench.GetNumConfigs(); ++c) {
    bench.Run(c);
  }
  ctx->waitIdle();
  for (uint32_t c = 0; c < bench.GetNumConfigs(); ++c) {
    ASSERT_TRUE(bench.ValidateResults(c));
  }

  // 3. For each config, test corruption modes:
  for (uint32_t c = 0; c < bench.GetNumConfigs(); ++c) {
    ComputeBuffer buf = bench.buffers[c];

    // 3a. Zeroed buffer
    std::vector<float> zeros(64, 0.0f);
    ctx->writeBuffer(buf, 0, zeros.size() * sizeof(float), zeros.data());
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(c));

    // 3b. Negative value (-10.0f)
    std::vector<float> negVals(64, -10.0f);
    ctx->writeBuffer(buf, 0, negVals.size() * sizeof(float), negVals.data());
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(c));

    // 3c. NaN
    float nanVal = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> nanVals(64, nanVal);
    ctx->writeBuffer(buf, 0, nanVals.size() * sizeof(float), nanVals.data());
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(c));

    // 3d. Infinity
    float infVal = std::numeric_limits<float>::infinity();
    std::vector<float> infVals(64, infVal);
    ctx->writeBuffer(buf, 0, infVals.size() * sizeof(float), infVals.data());
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(c));

    // 3e. Out of range high value
    float highVal = 99999999.0f;
    std::vector<float> highVals(64, highVal);
    ctx->writeBuffer(buf, 0, highVals.size() * sizeof(float), highVals.data());
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(c));

    // 3f. Boundary single-element corruption (first, middle, last)
    // First restore valid data
    bench.Run(c);
    ctx->waitIdle();
    ASSERT_TRUE(bench.ValidateResults(c));

    // Corrupt index 0
    ctx->writeBuffer(buf, 0, sizeof(float), &nanVal);
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(c));

    // Restore, then corrupt middle (index 31)
    bench.Run(c);
    ctx->waitIdle();
    ASSERT_TRUE(bench.ValidateResults(c));
    ctx->writeBuffer(buf, 31 * sizeof(float), sizeof(float), &nanVal);
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(c));

    // Restore, then corrupt last (index 63)
    bench.Run(c);
    ctx->waitIdle();
    ASSERT_TRUE(bench.ValidateResults(c));
    ctx->writeBuffer(buf, 63 * sizeof(float), sizeof(float), &nanVal);
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(c));
  }

  // 4. Out-of-bounds config indices
  ASSERT_FALSE(bench.ValidateResults(7));
  ASSERT_FALSE(bench.ValidateResults(100));

  bench.Teardown();

  // 5. Post-teardown call
  ASSERT_FALSE(bench.ValidateResults(0));
}

