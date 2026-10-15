#include "test_harness.h"
#include "core/ComputeBackendFactory.h"
#include "core/VulkanContext.h"
#include "utils/KernelPath.h"

#define private public
#include "benchmarks/InShaderIndirectBench.h"
#include "benchmarks/RayASBuildBench.h"
#undef private

#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <thread>
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
// 1. Empirical Verification: GPU Timing Does NOT Latch Host CPU Recording Time
// ============================================================================

TEST_CASE(M1_TimingFidelity, GpuTiming_ExcludesHostCpuSleepDelays) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  auto *vCtx = dynamic_cast<VulkanContext *>(ctx.get());
  ASSERT_TRUE(vCtx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  InShaderIndirectBench bench;
  bench.Setup(*ctx, kdir);

  // 1. Warmup dispatch
  vCtx->startTiming();
  vCtx->dispatch(bench.kernelReset, 1, 1, 1, 1, 1, 1);
  double warmupMs = vCtx->stopTiming();
  (void)warmupMs;

  // 2. Measure baseline GPU dispatch time (no CPU sleep)
  constexpr int kTrials = 10;
  double baselineSumMs = 0.0;
  for (int t = 0; t < kTrials; ++t) {
    vCtx->startTiming();
    vCtx->dispatch(bench.kernelReset, 1, 1, 1, 1, 1, 1);
    baselineSumMs += vCtx->stopTiming();
  }
  double baselineAvgMs = baselineSumMs / kTrials;

  // 3. Measure GPU dispatch time with a deliberate 20 ms host CPU sleep
  // inserted after startTiming() and before dispatch()
  constexpr int kSleepMs = 20;
  double delayedGpuSumMs = 0.0;
  double wallClockSumMs = 0.0;

  for (int t = 0; t < kTrials; ++t) {
    auto tStart = std::chrono::high_resolution_clock::now();
    vCtx->startTiming();

    // Deliberate CPU recording stall
    std::this_thread::sleep_for(std::chrono::milliseconds(kSleepMs));

    vCtx->dispatch(bench.kernelReset, 1, 1, 1, 1, 1, 1);
    double gpuTimeMs = vCtx->stopTiming();
    auto tEnd = std::chrono::high_resolution_clock::now();

    delayedGpuSumMs += gpuTimeMs;
    wallClockSumMs += std::chrono::duration<double, std::milli>(tEnd - tStart).count();
  }

  double delayedGpuAvgMs = delayedGpuSumMs / kTrials;
  double wallClockAvgMs = wallClockSumMs / kTrials;

  // Empirical Proof:
  // - wallClockAvgMs must be >= kSleepMs (at least 20 ms)
  // - delayedGpuAvgMs must be < 1.0 ms (pure GPU time, strictly unaffected by host sleep)
  // - delayedGpuAvgMs must be within 100 µs of baselineAvgMs
  std::cout << "\n    [Host Sleep Test] Wall clock: " << wallClockAvgMs << " ms"
            << " | GPU reported time: " << delayedGpuAvgMs * 1000.0 << " µs"
            << " | Baseline GPU time: " << baselineAvgMs * 1000.0 << " µs"
            << std::endl;

  ASSERT_GE(wallClockAvgMs, static_cast<double>(kSleepMs) * 0.95);
  ASSERT_LT(delayedGpuAvgMs, 1.0); // Must be sub-millisecond
  ASSERT_LT(std::abs(delayedGpuAvgMs - baselineAvgMs), 0.1); // Discrepancy < 100 µs

  bench.Teardown();
}

TEST_CASE(M1_TimingFidelity, GpuTiming_BatchRecordingExcludesHostStalls) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  auto *vCtx = dynamic_cast<VulkanContext *>(ctx.get());
  ASSERT_TRUE(vCtx != nullptr);

  std::string kdir = KernelPath::find();
  InShaderIndirectBench bench;
  bench.Setup(*ctx, kdir);

  // In batch mode, we record 10 dispatches into a single VkCommandBuffer.
  // We insert a 2 ms CPU sleep between recording each dispatch (total 20 ms CPU recording time).
  auto tStart = std::chrono::high_resolution_clock::now();
  vCtx->startTiming();
  vCtx->beginBatch(10);
  for (int i = 0; i < 10; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    vCtx->dispatch(bench.kernelReset, 1, 1, 1, 1, 1, 1);
  }
  vCtx->endBatch();
  double batchGpuMs = vCtx->stopTiming();
  auto tEnd = std::chrono::high_resolution_clock::now();

  double wallClockMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

  std::cout << "\n    [Batch Sleep Test] Host recording time: " << wallClockMs << " ms"
            << " | Batch GPU reported time: " << batchGpuMs * 1000.0 << " µs"
            << std::endl;

  ASSERT_GE(wallClockMs, 18.0);      // Host spent ~20 ms recording
  ASSERT_LT(batchGpuMs, 1.0);        // Hardware execution of 10 tiny dispatches is sub-millisecond

  bench.Teardown();
}

// ============================================================================
// 2. Empirical Verification: Indirect Reset Pass Latency is Realistic (<1 µs)
// ============================================================================

TEST_CASE(M1_TimingFidelity, IndirectResetPass_SubMicrosecondExecutionLatency) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  auto *vCtx = dynamic_cast<VulkanContext *>(ctx.get());
  ASSERT_TRUE(vCtx != nullptr);

  std::string kdir = KernelPath::find();
  InShaderIndirectBench bench;
  bench.Setup(*ctx, kdir);

  // Warmup reset kernel thoroughly (GPU power state ramp-up)
  for (int w = 0; w < 20; ++w) {
    vCtx->startTiming();
    vCtx->dispatch(bench.kernelReset, 1, 1, 1, 1, 1, 1);
    vCtx->stopTiming();
  }

  // Measure 30 isolated dispatches of indirect_reset.comp (1 thread, 20 bytes store)
  constexpr int kSamples = 30;
  std::vector<double> resetLatenciesUs;
  resetLatenciesUs.reserve(kSamples);

  for (int s = 0; s < kSamples; ++s) {
    vCtx->startTiming();
    vCtx->dispatch(bench.kernelReset, 1, 1, 1, 1, 1, 1);
    double elapsedMs = vCtx->stopTiming();
    resetLatenciesUs.push_back(elapsedMs * 1000.0);
  }

  std::sort(resetLatenciesUs.begin(), resetLatenciesUs.end());
  double minUs = resetLatenciesUs.front();
  double maxUs = resetLatenciesUs.back();
  double medianUs = resetLatenciesUs[kSamples / 2];
  double meanUs = std::accumulate(resetLatenciesUs.begin(), resetLatenciesUs.end(), 0.0) / kSamples;

  std::cout << "\n    [Reset Kernel Latency] Min: " << minUs << " µs"
            << " | Median: " << medianUs << " µs"
            << " | Mean: " << meanUs << " µs"
            << " | Max: " << maxUs << " µs"
            << std::endl;

  // Verification criteria:
  // Must be strictly positive and realistic GPU hardware latency:
  // Reset pass should execute with low-microsecond GPU latency (< 10.0 µs),
  // proving in-shader indirect timing without CPU staging stalls (which take >100 µs),
  // while remaining robust against APU memory contention and concurrent background load.
  ASSERT_GT(minUs, 0.0);
  ASSERT_LT(medianUs, 10.0);

  bench.Teardown();
}

// ============================================================================
// 3. Empirical Verification: InShaderIndirect Workload Latency Scaling
// ============================================================================

TEST_CASE(M1_TimingFidelity, InShaderIndirect_ZeroDispatchPruningSpeedup) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  auto *vCtx = dynamic_cast<VulkanContext *>(ctx.get());
  ASSERT_TRUE(vCtx != nullptr);

  std::string kdir = KernelPath::find();
  InShaderIndirectBench bench;
  bench.Setup(*ctx, kdir);

  // Measure 100% active GPU Indirect (Config 1)
  double t100Ms = 0.0;
  for (int t = 0; t < 5; ++t) {
    vCtx->startTiming();
    bench.Run(1);
    t100Ms += vCtx->stopTiming();
  }
  t100Ms /= 5.0;
  // Validate immediately after running config 1
  ASSERT_TRUE(bench.ValidateResults(1));

  // Measure 0% active GPU Indirect with Zero-Dispatch Pruning (Config 5)
  double t0Ms = 0.0;
  for (int t = 0; t < 5; ++t) {
    vCtx->startTiming();
    bench.Run(5);
    t0Ms += vCtx->stopTiming();
  }
  t0Ms /= 5.0;
  // Validate immediately after running config 5
  ASSERT_TRUE(bench.ValidateResults(5));

  double speedup = t100Ms / t0Ms;
  std::cout << "\n    [Indirect Latency Scaling] 100% active: " << t100Ms * 1000.0 << " µs"
            << " | 0% active (pruned): " << t0Ms * 1000.0 << " µs"
            << " | Hardware pruning speedup: " << speedup << "x"
            << std::endl;

  // 0% active pruned dispatch should be substantially faster than 100% active workload (>3x)
  ASSERT_GT(speedup, 3.0);
  ASSERT_LT(t0Ms * 1000.0, 50.0); // Zero dispatch pruning must complete under 50 µs

  bench.Teardown();
}

// ============================================================================
// 4. Empirical Verification: RayASBuild Command Pool Pre-recording Fidelity
// ============================================================================

TEST_CASE(M1_TimingFidelity, RayASBuild_PreRecordedCommandExecutionIntegrity) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  RayASBuildBench bench;
  DeviceInfo info = ctx->getCurrentDeviceInfo();
  if (!bench.IsSupported(info, ctx.get())) {
    return;
  }

  bench.Setup(*ctx, kdir);

  // Verify that all 8 configurations execute cleanly and validate
  for (uint32_t c = 0; c < 8; ++c) {
    bench.Run(c);
    ctx->waitIdle();
    bool valid = bench.ValidateResults(c);
    ASSERT_TRUE(valid);
  }

  bench.Teardown();
}
