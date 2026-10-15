#include "test_harness.h"
#include "core/ComputeBackendFactory.h"
#include "core/VulkanContext.h"
#include "core/ResultFormatter.h"
#include "utils/KernelPath.h"

#define private public
#include "benchmarks/LdsBankConflictBench.h"
#include "benchmarks/Bf16Bench.h"
#undef private

#include <vector>
#include <iostream>
#include <limits>
#include <cmath>

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
// 1. LdsBankConflictBench Adversarial & Negative Validation Tests
// ============================================================================

TEST_CASE(M2_AdversarialChallenge, LdsBankConflicts_PositiveValidation_AllConfigs) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  LdsBankConflictBench bench;
  bench.Setup(*ctx, kdir);

  uint32_t numConfigs = bench.GetNumConfigs();
  ASSERT_EQ(numConfigs, 8u);

  // Run and validate every configuration (Strides 1, 2, 4, 8, 16, 32, 3, 5)
  for (uint32_t cfg = 0; cfg < numConfigs; ++cfg) {
    bench.Run(cfg);
    ctx->waitIdle();
    bool valid = bench.ValidateResults(cfg);
    ASSERT_TRUE(valid);
  }

  bench.Teardown();
}

TEST_CASE(M2_AdversarialChallenge, LdsBankConflicts_NegativeValidation_ZeroedBuffer) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  LdsBankConflictBench bench;
  bench.Setup(*ctx, kdir);

  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));

  // Deliberately wipe entire output buffer with zeros
  const size_t totalThreads = 2048 * 32;
  const size_t bufferSize = totalThreads * sizeof(uint32_t);
  std::vector<uint32_t> zeros(totalThreads, 0);
  ctx->writeBuffer(bench.buffer, 0, bufferSize, zeros.data());
  ctx->waitIdle();

  // ValidateResults MUST reject the zeroed buffer
  ASSERT_FALSE(bench.ValidateResults(0));

  bench.Teardown();
}

TEST_CASE(M2_AdversarialChallenge, LdsBankConflicts_NegativeValidation_SingleElementCorruption) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  LdsBankConflictBench bench;
  bench.Setup(*ctx, kdir);

  const size_t totalThreads = 2048 * 32;
  constexpr uint32_t badVal = 12345678u;

  // Test corruption at element 0 (first element boundary)
  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));
  ctx->writeBuffer(bench.buffer, 0, sizeof(uint32_t), &badVal);
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(0));

  // Test corruption at middle element
  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));
  size_t midIdx = totalThreads / 2;
  ctx->writeBuffer(bench.buffer, midIdx * sizeof(uint32_t), sizeof(uint32_t), &badVal);
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(0));

  // Test corruption at last element (last element boundary: 65535)
  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));
  size_t lastIdx = totalThreads - 1;
  ctx->writeBuffer(bench.buffer, lastIdx * sizeof(uint32_t), sizeof(uint32_t), &badVal);
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(0));

  bench.Teardown();
}

TEST_CASE(M2_AdversarialChallenge, LdsBankConflicts_NegativeValidation_OffByOneExpectedValue) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  LdsBankConflictBench bench;
  bench.Setup(*ctx, kdir);

  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));

  constexpr uint32_t expected = 50000u * 8u * 32u; // 12,800,000
  uint32_t offByOneHigh = expected + 1;
  uint32_t offByOneLow = expected - 1;

  // Expected + 1 must fail
  ctx->writeBuffer(bench.buffer, 100 * sizeof(uint32_t), sizeof(uint32_t), &offByOneHigh);
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(0));

  // Expected - 1 must fail
  ctx->writeBuffer(bench.buffer, 100 * sizeof(uint32_t), sizeof(uint32_t), &offByOneLow);
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(0));

  bench.Teardown();
}

TEST_CASE(M2_AdversarialChallenge, LdsBankConflicts_NegativeValidation_OOBAndTeardown) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  LdsBankConflictBench bench;
  bench.Setup(*ctx, kdir);

  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));

  // Out of bounds config index
  ASSERT_FALSE(bench.ValidateResults(999));
  ASSERT_FALSE(bench.ValidateResults(8));

  bench.Teardown();

  // Post teardown validation must fail safely
  ASSERT_FALSE(bench.ValidateResults(0));
}

TEST_CASE(M2_AdversarialChallenge, LdsBankConflicts_SerializationCurve_EmpiricalVerification) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  LdsBankConflictBench bench;
  bench.Setup(*ctx, kdir);

  // Measure elapsed times for Strides 1, 4, 8, 16, 32 and Odd Strides 3, 5
  auto measureTimeMs = [&](uint32_t cfg) -> double {
    bench.Run(cfg);
    ctx->waitIdle();
    ctx->startTiming();
    bench.Run(cfg);
    return ctx->stopTiming();
  };

  // Warmup
  measureTimeMs(0);

  double timeStride1 = measureTimeMs(0);
  double timeStride4 = measureTimeMs(2);
  double timeStride8 = measureTimeMs(3);
  double timeStride16 = measureTimeMs(4);
  double timeStride32 = measureTimeMs(5);
  double timeStride3 = measureTimeMs(6);
  double timeStride5 = measureTimeMs(7);

  // 1. Serialization ordering: each higher power-of-two conflict must increase execution time
  // timeStride1 < timeStride4 < timeStride8 < timeStride16 < timeStride32
  ASSERT_LT(timeStride1, timeStride4);
  ASSERT_LT(timeStride4, timeStride8);
  ASSERT_LT(timeStride8, timeStride16);
  ASSERT_LT(timeStride16, timeStride32);

  // 2. Severe degradation at Stride 32: must take at least 5x longer than Stride 1 (theoretical 32x)
  double slowdown32 = timeStride32 / timeStride1;
  ASSERT_GT(slowdown32, 5.0);

  // 3. Odd stride controls: gcd(3, 32)==1 and gcd(5, 32)==1 must be close to Stride 1 (within 35%)
  double ratioStride3 = timeStride3 / timeStride1;
  double ratioStride5 = timeStride5 / timeStride1;
  ASSERT_LT(ratioStride3, 1.35);
  ASSERT_LT(ratioStride5, 1.35);

  bench.Teardown();
}

// ============================================================================
// 2. Bf16Bench Adversarial & Native Capability Gating Honesty Tests
// ============================================================================

namespace {

std::unique_ptr<IComputeContext> createBf16Context() {
  Bf16Bench probe;
  if (ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
    auto vkCtx = ComputeBackendFactory::create(ComputeBackend::Vulkan, false, false);
    vkCtx->pickDevice(0);
    if (probe.IsSupported(vkCtx->getCurrentDeviceInfo(), vkCtx.get())) {
      return vkCtx;
    }
  }
  if (ComputeBackendFactory::isAvailable(ComputeBackend::ROCm)) {
    try {
      auto rocmCtx = ComputeBackendFactory::create(ComputeBackend::ROCm, false, false);
      rocmCtx->pickDevice(0);
      if (probe.IsSupported(rocmCtx->getCurrentDeviceInfo(), rocmCtx.get())) {
        return rocmCtx;
      }
    } catch (...) {
    }
  }
  return nullptr;
}

} // namespace

TEST_CASE(M2_AdversarialChallenge, Bf16Bench_HonestyContract_NoSilentFp16Fallback) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  Bf16Bench bench;
  DeviceInfo info = ctx->getCurrentDeviceInfo();
  if (!info.bf16Support) {
    // On Vulkan devices lacking VK_KHR_shader_bfloat16, Bf16Bench MUST report
    // IsSupported == false rather than silently substituting IEEE FP16!
    ASSERT_FALSE(bench.IsSupported(info, ctx.get()));
    std::string note = bench.GetSupportNote(info, ctx.get());
    ASSERT_FALSE(note.empty());
    ASSERT_NE(note.find("BF16"), std::string::npos);
  } else {
    ASSERT_TRUE(bench.IsSupported(info, ctx.get()));
    ASSERT_FALSE(bench.IsEmulated(0));
    ASSERT_FALSE(bench.IsEmulated(1));
  }
}

TEST_CASE(M2_AdversarialChallenge, Bf16Bench_PositiveValidation_VectorAndMatrix) {
  auto ctx = createBf16Context();
  if (!ctx) {
    std::cout << "[SKIP] Native BF16 not supported on any available backend" << std::endl;
    return;
  }

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  Bf16Bench bench;
  bench.Setup(*ctx, kdir);

  // Config 0: Vector BF16
  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));

  // Config 1: Matrix BF16 (if cooperative matrix / WMMA supported)
  DeviceInfo info = ctx->getCurrentDeviceInfo();
  if (info.cooperativeMatrixSupport && bench.matrixKernel) {
    bench.Run(1);
    ctx->waitIdle();
    ASSERT_TRUE(bench.ValidateResults(1));
  }

  bench.Teardown();
}

TEST_CASE(M2_AdversarialChallenge, Bf16Bench_NegativeValidation_ZeroedBuffer) {
  auto ctx = createBf16Context();
  if (!ctx) {
    std::cout << "[SKIP] Native BF16 not supported on any available backend" << std::endl;
    return;
  }

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  Bf16Bench bench;
  bench.Setup(*ctx, kdir);

  // Config 0: Wipe buffer with zeros
  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));

  std::vector<uint32_t> zeros(256, 0);
  ctx->writeBuffer(bench.buffer, 0, zeros.size() * sizeof(uint32_t), zeros.data());
  ctx->waitIdle();
  // Zeroed output must be rejected by ValidateResults(0)
  ASSERT_FALSE(bench.ValidateResults(0));

  // Config 1: Wipe matrixBuffer with zeros
  DeviceInfo info = ctx->getCurrentDeviceInfo();
  if (info.cooperativeMatrixSupport && bench.matrixBuffer && bench.matrixKernel) {
    bench.Run(1);
    ctx->waitIdle();
    ASSERT_TRUE(bench.ValidateResults(1));

    ctx->writeBuffer(bench.matrixBuffer, 0, zeros.size() * sizeof(uint32_t), zeros.data());
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(1));
  }

  bench.Teardown();
}

TEST_CASE(M2_AdversarialChallenge, Bf16Bench_NegativeValidation_NaNInjection) {
  auto ctx = createBf16Context();
  if (!ctx) {
    std::cout << "[SKIP] Native BF16 not supported on any available backend" << std::endl;
    return;
  }

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  Bf16Bench bench;
  bench.Setup(*ctx, kdir);

  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));

  // 0x7FC00000 is float32 quiet-NaN, and its lower/upper 16-bit half (0x7FC0) is also BF16 quiet-NaN
  uint32_t nanBits = 0x7FC07FC0u;
  ctx->writeBuffer(bench.buffer, 0, sizeof(uint32_t), &nanBits);
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(0));

  DeviceInfo info = ctx->getCurrentDeviceInfo();
  if (info.cooperativeMatrixSupport && bench.matrixBuffer && bench.matrixKernel) {
    bench.Run(1);
    ctx->waitIdle();
    ASSERT_TRUE(bench.ValidateResults(1));

    ctx->writeBuffer(bench.matrixBuffer, 0, sizeof(uint32_t), &nanBits);
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(1));
  }

  bench.Teardown();
}

TEST_CASE(M2_AdversarialChallenge, Bf16Bench_NegativeValidation_InfInjection) {
  auto ctx = createBf16Context();
  if (!ctx) {
    std::cout << "[SKIP] Native BF16 not supported on any available backend" << std::endl;
    return;
  }

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  Bf16Bench bench;
  bench.Setup(*ctx, kdir);

  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));

  // 0x7F800000 is float32 +Inf, and 0x7F80 is BF16 +Inf
  uint32_t infBits = (ctx->getBackend() == ComputeBackend::ROCm) ? 0x7F800000u : 0x7F807F80u;
  ctx->writeBuffer(bench.buffer, 0, sizeof(uint32_t), &infBits);
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(0));

  DeviceInfo info = ctx->getCurrentDeviceInfo();
  if (info.cooperativeMatrixSupport && bench.matrixBuffer && bench.matrixKernel) {
    bench.Run(1);
    ctx->waitIdle();
    ASSERT_TRUE(bench.ValidateResults(1));

    ctx->writeBuffer(bench.matrixBuffer, 0, sizeof(uint32_t), &infBits);
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(1));
  }

  bench.Teardown();
}

TEST_CASE(M2_AdversarialChallenge, Bf16Bench_RecurrenceStability_MultiPassStress) {
  auto ctx = createBf16Context();
  if (!ctx) {
    std::cout << "[SKIP] Native BF16 not supported on any available backend" << std::endl;
    return;
  }

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  Bf16Bench bench;
  bench.Setup(*ctx, kdir);

  // Run 10 consecutive iterations of both vector and matrix workloads without reallocating buffers
  // to ensure no runaway feedback loop or cumulative overflow to Inf occurs
  for (int iter = 0; iter < 10; ++iter) {
    bench.Run(0);
    ctx->waitIdle();
    ASSERT_TRUE(bench.ValidateResults(0));

    if (bench.matrixKernel) {
      bench.Run(1);
      ctx->waitIdle();
      ASSERT_TRUE(bench.ValidateResults(1));
    }
  }

  bench.Teardown();
}
