#include "test_harness.h"
#include "core/ComputeBackendFactory.h"
#include "core/VulkanContext.h"
#include "core/ResultFormatter.h"
#include "core/ResultImporter.h"
#include "utils/KernelPath.h"

#define private public
#include "benchmarks/InShaderIndirectBench.h"
#include "benchmarks/RayASBuildBench.h"
#undef private

#include <vector>
#include <iostream>
#include <limits>

namespace {

struct TestGpuIndirectCommand {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t activeCount;
  uint32_t workgroupsFinished;
};

std::unique_ptr<IComputeContext> createVulkanContext() {
  if (!ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
    return nullptr;
  }
  try {
    auto ctx = ComputeBackendFactory::create(ComputeBackend::Vulkan, false, false);
    if (!ctx || ctx->getDevices().empty()) {
      return nullptr;
    }
    ctx->pickDevice(0);
    return ctx;
  } catch (...) {
    return nullptr;
  }
}

} // namespace

// ============================================================================
// 1. RayASBuildBench Negative Validation Tests
// ============================================================================

TEST_CASE(IndirectAndASBuildValidation, RayASBuild_NegativeValidation_ZeroedBlasBuffer) {
  auto ctx = createVulkanContext();
  if (!ctx) { return; }

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  RayASBuildBench bench;
  DeviceInfo info = ctx->getCurrentDeviceInfo();
  if (!bench.IsSupported(info, ctx.get())) {
    return; // Ray tracing AS not supported on this device/backend
  }

  bench.Setup(*ctx, kdir);

  // 1. Run legitimate BLAS build (config 0: 1M Tris)
  bench.Run(0);
  ctx->waitIdle();

  // Baseline validation must pass
  bool baselineValid = bench.ValidateResults(0);
  ASSERT_TRUE(baselineValid);

  // 2. Deliberately zero out the BLAS header in GPU memory
  ASSERT_TRUE(bench.blases.size() > 0);
  ASSERT_TRUE(bench.blases[0].buffer != nullptr);
  uint32_t zeroHeader[16] = {0};
  ctx->writeBuffer(bench.blases[0].buffer, 0, sizeof(zeroHeader), zeroHeader);
  ctx->waitIdle();

  // Negative validation: ValidateResults MUST return false when buffer is zeroed!
  bool zeroedValid = bench.ValidateResults(0);
  ASSERT_FALSE(zeroedValid);

  // 3. Out-of-bounds config index must return false
  bool oobValid = bench.ValidateResults(999);
  ASSERT_FALSE(oobValid);

  bench.Teardown();

  // 4. After teardown, ValidateResults must return false
  bool postTeardownValid = bench.ValidateResults(0);
  ASSERT_FALSE(postTeardownValid);
}

TEST_CASE(IndirectAndASBuildValidation, RayASBuild_NegativeValidation_ZeroedTlasBuffer) {
  auto ctx = createVulkanContext();
  if (!ctx) { return; }

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  RayASBuildBench bench;
  DeviceInfo info = ctx->getCurrentDeviceInfo();
  if (!bench.IsSupported(info, ctx.get())) {
    return;
  }

  bench.Setup(*ctx, kdir);

  // Run legitimate TLAS build (config 5: Indoor Corridor)
  bench.Run(5);
  ctx->waitIdle();

  // Baseline validation must pass
  bool baselineValid = bench.ValidateResults(5);
  ASSERT_TRUE(baselineValid);

  // Deliberately zero out the TLAS header in GPU memory
  ASSERT_TRUE(bench.tlases.size() > 0);
  ASSERT_TRUE(bench.tlases[0].buffer != nullptr);
  uint32_t zeroHeader[16] = {0};
  ctx->writeBuffer(bench.tlases[0].buffer, 0, sizeof(zeroHeader), zeroHeader);
  ctx->waitIdle();

  // Negative validation: ValidateResults MUST return false when TLAS buffer is zeroed!
  bool zeroedValid = bench.ValidateResults(5);
  ASSERT_FALSE(zeroedValid);

  bench.Teardown();
}

// ============================================================================
// 2. InShaderIndirectBench Negative Validation Tests
// ============================================================================

TEST_CASE(IndirectAndASBuildValidation, InShaderIndirect_NegativeValidation_IndirectBuffer) {
  auto ctx = createVulkanContext();
  if (!ctx) { return; }

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  InShaderIndirectBench bench;
  bench.Setup(*ctx, kdir);

  // Run config 1: 100% active GPU indirect
  bench.Run(1);
  ctx->waitIdle();

  bool baselineValid = bench.ValidateResults(1);
  ASSERT_TRUE(baselineValid);

  // Subcase A: Deliberately zero indirectBuffer
  TestGpuIndirectCommand zeroCmd{0, 0, 0, 0, 0};
  ctx->writeBuffer(bench.indirectBuffer, 0, sizeof(zeroCmd), &zeroCmd);
  ctx->waitIdle();
  bool zeroIndirectValid = bench.ValidateResults(1);
  ASSERT_FALSE(zeroIndirectValid); // Must fail because workgroupsFinished == 0

  // Subcase B: Corrupt workgroupsFinished (e.g. 12345 instead of 65536)
  TestGpuIndirectCommand corruptWgfCmd{65536, 1, 1, 4194304, 12345};
  ctx->writeBuffer(bench.indirectBuffer, 0, sizeof(corruptWgfCmd), &corruptWgfCmd);
  ctx->waitIdle();
  bool corruptWgfValid = bench.ValidateResults(1);
  ASSERT_FALSE(corruptWgfValid); // Must fail

  // Subcase C: Corrupt cmd.x (0 instead of 65536 for 100% selectivity)
  TestGpuIndirectCommand corruptXCmd{0, 1, 1, 4194304, 65536};
  ctx->writeBuffer(bench.indirectBuffer, 0, sizeof(corruptXCmd), &corruptXCmd);
  ctx->waitIdle();
  bool corruptXValid = bench.ValidateResults(1);
  ASSERT_FALSE(corruptXValid); // Must fail

  // Subcase D: Corrupt 0% active config (config 5) with non-zero activeCount
  bench.Run(5);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(5));
  TestGpuIndirectCommand corrupt0PctCmd{1, 1, 1, 100, 65536};
  ctx->writeBuffer(bench.indirectBuffer, 0, sizeof(corrupt0PctCmd), &corrupt0PctCmd);
  ctx->waitIdle();
  bool corrupt0PctValid = bench.ValidateResults(5);
  ASSERT_FALSE(corrupt0PctValid); // Must fail

  // Out of bounds config index
  ASSERT_FALSE(bench.ValidateResults(999));

  bench.Teardown();
  ASSERT_FALSE(bench.ValidateResults(1));
}

TEST_CASE(IndirectAndASBuildValidation, InShaderIndirect_NegativeValidation_OutputBufferWiped_AllActiveConfigs) {
  // Empirically verify that wiping outputBuffer with zeros causes ValidateResults()
  // to return FALSE for all active configurations:
  // - CPU direct configurations: 0 (100% active), 6 (10% active)
  // - Active GPU indirect configurations: 1 (100%), 2 (50%), 3 (10%), 4 (1%)
  auto ctx = createVulkanContext();
  if (!ctx) { return; }

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  const std::vector<uint32_t> activeConfigs = {0, 1, 2, 3, 4, 6};

  for (uint32_t cfgIdx : activeConfigs) {
    InShaderIndirectBench bench;
    bench.Setup(*ctx, kdir);

    // 1. Legitimate execution must validate cleanly
    bench.Run(cfgIdx);
    ctx->waitIdle();
    ASSERT_TRUE(bench.ValidateResults(cfgIdx));

    // 2. Deliberately wipe outputBuffer completely with zeros
    std::vector<float> zeros(1024, 0.0f);
    ctx->writeBuffer(bench.outputBuffer, 0, zeros.size() * sizeof(float), zeros.data());
    ctx->waitIdle();

    // 3. Negative validation: ValidateResults MUST reject the wiped buffer
    bool wipedValid = bench.ValidateResults(cfgIdx);
    ASSERT_FALSE(wipedValid);

    bench.Teardown();
  }
}

TEST_CASE(IndirectAndASBuildValidation, InShaderIndirect_NegativeValidation_OutputBufferCorrupted_NaN_Inf_WrongValue) {
  // Adversarially corrupt outputBuffer with NaNs, Infs, and mathematically incorrect floats
  // across all active configurations (Configs 0, 1, 2, 3, 4, 6).
  auto ctx = createVulkanContext();
  if (!ctx) { return; }

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  const std::vector<uint32_t> activeConfigs = {0, 1, 2, 3, 4, 6};

  for (uint32_t cfgIdx : activeConfigs) {
    InShaderIndirectBench bench;
    bench.Setup(*ctx, kdir);

    // Baseline execution
    bench.Run(cfgIdx);
    ctx->waitIdle();
    ASSERT_TRUE(bench.ValidateResults(cfgIdx));

    // Attack A: quiet_NaN injection
    std::vector<float> nans(1024, std::numeric_limits<float>::quiet_NaN());
    ctx->writeBuffer(bench.outputBuffer, 0, nans.size() * sizeof(float), nans.data());
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(cfgIdx));

    // Attack B: infinity injection
    std::vector<float> infs(1024, std::numeric_limits<float>::infinity());
    ctx->writeBuffer(bench.outputBuffer, 0, infs.size() * sizeof(float), infs.data());
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(cfgIdx));

    // Re-run to restore valid state
    bench.Run(cfgIdx);
    ctx->waitIdle();
    ASSERT_TRUE(bench.ValidateResults(cfgIdx));

    // Attack C: Systematic offset corruption (+1.0f delta from expected values)
    std::vector<float> corrupted(1024);
    ctx->readBuffer(bench.outputBuffer, 0, corrupted.size() * sizeof(float), corrupted.data());
    for (auto &v : corrupted) {
      v += 1.0f;
    }
    ctx->writeBuffer(bench.outputBuffer, 0, corrupted.size() * sizeof(float), corrupted.data());
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(cfgIdx));

    bench.Teardown();
  }
}

TEST_CASE(IndirectAndASBuildValidation, InShaderIndirect_NegativeValidation_SingleElementCorruption) {
  // Verify that corrupting even a single active element in outputBuffer causes ValidateResults() to fail
  auto ctx = createVulkanContext();
  if (!ctx) { return; }

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  const std::vector<uint32_t> activeConfigs = {0, 1, 2, 3, 4, 6};

  for (uint32_t cfgIdx : activeConfigs) {
    InShaderIndirectBench bench;
    bench.Setup(*ctx, kdir);

    bench.Run(cfgIdx);
    ctx->waitIdle();
    ASSERT_TRUE(bench.ValidateResults(cfgIdx));

    float badVal = 999.0f;
    if (cfgIdx == 0 || cfgIdx == 6) {
      // Find the first active gid in CPU direct mode
      uint32_t selectivity = (cfgIdx == 0) ? 100 : 10;
      uint32_t targetGid = 0;
      for (uint32_t gid = 0; gid < 1024; ++gid) {
        if (((gid * 2654435761u) % 100u) < selectivity) {
          targetGid = gid;
          break;
        }
      }
      ctx->writeBuffer(bench.outputBuffer, targetGid * sizeof(float), sizeof(float), &badVal);
    } else {
      // In indirect mode, workListBuffer[0] is always an active item
      ctx->writeBuffer(bench.outputBuffer, 0, sizeof(float), &badVal);
    }
    ctx->waitIdle();

    ASSERT_FALSE(bench.ValidateResults(cfgIdx));

    bench.Teardown();
  }
}

TEST_CASE(IndirectAndASBuildValidation, InShaderIndirect_PrunedConfig5_ZeroWorkgroupsAndUncorruptedBuffer) {
  // Config 5 (0% active): Verify zero-dispatch pruning
  // 1. Validate that indirect dispatch sizing evaluates to (0, 0, 0) and 0 workgroups ran
  // 2. Empirically verify that outputBuffer is completely uncorrupted (untouched by GPU)
  // 3. Verify negative validation when indirect command fields are maliciously mutated
  auto ctx = createVulkanContext();
  if (!ctx) { return; }

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  InShaderIndirectBench bench;
  bench.Setup(*ctx, kdir);

  // Fill outputBuffer with a distinctive sentinel / canary pattern before dispatch
  constexpr float kCanary = 42.1337f;
  std::vector<float> canaryInit(1024, kCanary);
  ctx->writeBuffer(bench.outputBuffer, 0, canaryInit.size() * sizeof(float), canaryInit.data());
  ctx->waitIdle();

  // Execute Config 5 (0% active selectivity)
  bench.Run(5);
  ctx->waitIdle();

  // 1. ValidateResults(5) must return TRUE for the legitimate 0% dispatch
  ASSERT_TRUE(bench.ValidateResults(5));

  // 2. Inspect indirectBuffer directly on GPU to verify 0 workgroups were scheduled
  TestGpuIndirectCommand cmd{};
  ctx->readBuffer(bench.indirectBuffer, 0, sizeof(cmd), &cmd);
  ASSERT_EQ(cmd.x, 0u);
  ASSERT_EQ(cmd.y, 0u);
  ASSERT_EQ(cmd.z, 0u);
  ASSERT_EQ(cmd.activeCount, 0u);
  ASSERT_EQ(cmd.workgroupsFinished, 65536u);

  // 3. Empirically verify that outputBuffer was NOT touched by the GPU (strictly preserved canary)
  std::vector<float> readback(1024, 0.0f);
  ctx->readBuffer(bench.outputBuffer, 0, readback.size() * sizeof(float), readback.data());
  for (size_t i = 0; i < readback.size(); ++i) {
    ASSERT_EQ(readback[i], kCanary);
  }

  // 4. Negative validation: Corrupting cmd.x to 1 must cause ValidateResults(5) to return FALSE
  TestGpuIndirectCommand badCmd = cmd;
  badCmd.x = 1u;
  ctx->writeBuffer(bench.indirectBuffer, 0, sizeof(badCmd), &badCmd);
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(5));

  // 5. Negative validation: Corrupting activeCount to 10 must cause ValidateResults(5) to return FALSE
  badCmd = cmd;
  badCmd.activeCount = 10u;
  ctx->writeBuffer(bench.indirectBuffer, 0, sizeof(badCmd), &badCmd);
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(5));

  // 6. Negative validation: Corrupting workgroupsFinished must fail
  badCmd = cmd;
  badCmd.workgroupsFinished = 1234u;
  ctx->writeBuffer(bench.indirectBuffer, 0, sizeof(badCmd), &badCmd);
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(5));

  // 7. Negative validation: null outputBuffer must cause ValidateResults(5) to return FALSE
  auto savedOutput = bench.outputBuffer;
  bench.outputBuffer = nullptr;
  ASSERT_FALSE(bench.ValidateResults(5));
  bench.outputBuffer = savedOutput;

  bench.Teardown();
}

TEST_CASE(IndirectAndASBuildValidation, InShaderIndirect_SamplingWindowBoundaryAndWorklistIntegrity) {
  // Boundary Condition Testing:
  // 1. Verify that corrupting the last checked element (index 1023) is caught by ValidateResults()
  // 2. Characterize the sampling window: elements at index >= 1024 are outside the sampling window
  // 3. Verify that corrupting workListBuffer (OOB item ID or non-matching hash) is caught by ValidateResults()
  auto ctx = createVulkanContext();
  if (!ctx) { return; }

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  InShaderIndirectBench bench;
  bench.Setup(*ctx, kdir);

  // Run Config 1 (100% active GPU Indirect)
  bench.Run(1);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(1));

  // Boundary Test 1: Corrupt the very last element in the 1024 sample window (index 1023)
  float badVal = 999.0f;
  ctx->writeBuffer(bench.outputBuffer, 1023 * sizeof(float), sizeof(float), &badVal);
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(1)); // MUST catch boundary corruption at index 1023

  // Re-run to restore clean state
  bench.Run(1);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(1));

  // Boundary Test 2: Corrupt element at index 1024 (just beyond the 1024 sample window)
  ctx->writeBuffer(bench.outputBuffer, 1024 * sizeof(float), sizeof(float), &badVal);
  ctx->waitIdle();
  // Validates sampling window contract: elements >= 1024 are not read back to bound validation latency
  ASSERT_TRUE(bench.ValidateResults(1));

  // Worklist Integrity Test 1: Corrupt workListBuffer[0] with out-of-bounds itemId (kTotalItems + 100)
  uint32_t oobItemId = 4194304u + 100u;
  ctx->writeBuffer(bench.workListBuffer, 0, sizeof(uint32_t), &oobItemId);
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(1));

  // Teardown
  bench.Teardown();
}



// ============================================================================
// 3. Query Pool Safety & Timestamp Overflow Stress Test
// ============================================================================

TEST_CASE(IndirectAndASBuildValidation, TimestampQueryPool_ExceedsCapacity_SafetyGuard) {
  auto ctx = createVulkanContext();
  if (!ctx) { return; }

  auto *vCtx = dynamic_cast<VulkanContext *>(ctx.get());
  ASSERT_TRUE(vCtx != nullptr);

  std::string kdir = KernelPath::find();
  InShaderIndirectBench bench;
  bench.Setup(*ctx, kdir);

  // VulkanContext::kMaxTimestamps is 4096 (2048 query pairs).
  // We test what happens if an unbatched loop triggers >2048 dispatches while timing is active.
  vCtx->startTiming();

  // Issue 2500 single dispatches (which would need 5000 queries, exceeding 4096 capacity)
  // The safety guard `if (currentQueryIndex + 2 > kMaxTimestamps) return;` must prevent
  // out-of-bounds query indices and prevent driver crashes / Vulkan validation errors.
  for (int i = 0; i < 2500; ++i) {
    vCtx->dispatch(bench.kernelReset, 1, 1, 1, 1, 1, 1);
  }

  // Stop timing and retrieve results: must NOT crash or hang
  double elapsedMs = vCtx->stopTiming();
  ASSERT_GT(elapsedMs, 0.0);

  bench.Teardown();
}

TEST_CASE(IndirectAndASBuildValidation, TimestampQueryPool_BatchingLargeIterationSafety) {
  auto ctx = createVulkanContext();
  if (!ctx) { return; }

  auto *vCtx = dynamic_cast<VulkanContext *>(ctx.get());
  ASSERT_TRUE(vCtx != nullptr);

  std::string kdir = KernelPath::find();
  InShaderIndirectBench bench;
  bench.Setup(*ctx, kdir);

  // In batching mode (beginBatch / endBatch), only ONE timestamp pair is recorded
  // for the entire batch regardless of iteration count.
  // Test a batch of 5000 dispatches:
  vCtx->startTiming();
  vCtx->beginBatch(5000);
  for (int i = 0; i < 5000; ++i) {
    vCtx->dispatch(bench.kernelReset, 1, 1, 1, 1, 1, 1);
  }
  vCtx->endBatch();

  double batchElapsedMs = vCtx->stopTiming();
  ASSERT_GT(batchElapsedMs, 0.0);

  bench.Teardown();
}

TEST_CASE(IndirectAndASBuildValidation, TimestampQueryPool_RapidRecyclingStress) {
  auto ctx = createVulkanContext();
  if (!ctx) { return; }

  auto *vCtx = dynamic_cast<VulkanContext *>(ctx.get());
  ASSERT_TRUE(vCtx != nullptr);

  std::string kdir = KernelPath::find();
  InShaderIndirectBench bench;
  bench.Setup(*ctx, kdir);

  // Rapidly cycle startTiming() / dispatch / stopTiming() 500 times
  // Verifies that vkResetQueryPool safely resets currentQueryIndex to 0 on every cycle
  for (int cycle = 0; cycle < 500; ++cycle) {
    vCtx->startTiming();
    vCtx->dispatch(bench.kernelReset, 1, 1, 1, 1, 1, 1);
    double ms = vCtx->stopTiming();
    ASSERT_GT(ms, 0.0);
  }

  bench.Teardown();
}

// ============================================================================
// 4. RFC 8259 JSON Floating-Point Sanitization Stress Test
// ============================================================================

TEST_CASE(IndirectAndASBuildValidation, ResultFormatter_RFC8259_NonFiniteFloatsSanitization) {
  std::vector<ResultData> results;
  ResultData r{};
  r.backendName = "Vulkan";
  r.deviceName = "AMD Radeon AI PRO R9700";
  r.benchmarkName = "ContentionDivZero";
  r.component = "Compute";
  r.metric = "TFLOPS";
  r.operations = 1000ULL;
  r.time_ms = 1.0;
  r.isValid = true;
  r.hasPerfQueryTelemetry = true;
  r.perfCounters.gpuActiveCycles = 0ULL;
  r.perfCounters.valuBusyPct = std::numeric_limits<float>::quiet_NaN();
  r.perfCounters.saluBusyPct = std::numeric_limits<float>::infinity();
  r.perfCounters.l0CacheHitRatio = -std::numeric_limits<float>::quiet_NaN();
  r.perfCounters.l1CacheHitRatio = -std::numeric_limits<float>::infinity();
  r.perfCounters.rawCounters["salu_cycles_per_wave"] = std::numeric_limits<double>::infinity();
  r.perfCounters.rawCounters["valu_cycles_per_wave"] = std::numeric_limits<double>::quiet_NaN();
  r.hasPowerTelemetry = true;
  r.powerWatts = std::numeric_limits<float>::quiet_NaN();
  results.push_back(r);

  std::string jsonStr = resultsToJson(results);

  ASSERT_EQ(jsonStr.find(": nan"), std::string::npos);
  ASSERT_EQ(jsonStr.find(": -nan"), std::string::npos);
  ASSERT_EQ(jsonStr.find(": inf"), std::string::npos);
  ASSERT_EQ(jsonStr.find(": -inf"), std::string::npos);

  ImportedRun run;
  std::string err;
  bool ok = ResultImporter::loadFromString(jsonStr, run, err);
  ASSERT_TRUE(ok);
  ASSERT_EQ(run.results.size(), 1u);
  ASSERT_EQ(run.results[0].perfCounters.valuBusyPct, 0.0f);
  ASSERT_EQ(run.results[0].perfCounters.saluBusyPct, 0.0f);
}
