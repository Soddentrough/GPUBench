#include "test_harness.h"
#include "core/ComputeBackendFactory.h"
#include "utils/IsaDumper.h"
#include "utils/KernelPath.h"

#include "benchmarks/AsyncOverlapBench.h"
#include "benchmarks/Bf16Bench.h"
#include "benchmarks/DualIssueBench.h"
#include "benchmarks/Fp16Bench.h"
#include "benchmarks/Fp32Bench.h"
#include "benchmarks/Fp64Bench.h"
#include "benchmarks/Fp8Bench.h"
#include "benchmarks/InShaderIndirectBench.h"
#include "benchmarks/Int4Bench.h"
#include "benchmarks/Int8Bench.h"
#include "benchmarks/InterconnectBench.h"
#include "benchmarks/LdsBankConflictBench.h"
#include "benchmarks/MemBandwidthBench.h"
#include "benchmarks/SparseMatrixBench.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

std::unique_ptr<IComputeContext> createRocmTestContext(uint32_t *outDevIdx = nullptr) {
  if (!ComputeBackendFactory::isAvailable(ComputeBackend::ROCm)) {
    return nullptr;
  }
  try {
    auto ctx = ComputeBackendFactory::create(ComputeBackend::ROCm, false, false);
    if (!ctx || !ctx->isAvailable() || ctx->getDevices().empty()) {
      return nullptr;
    }
    const uint32_t devIdx =
        ::gpubench::test::preferredTestDeviceIndex(ctx->getDevices().size());
    ctx->pickDevice(devIdx);
    if (outDevIdx) {
      *outDevIdx = devIdx;
    }
    return ctx;
  } catch (...) {
    return nullptr;
  }
}

void runAndValidateConfig(IBenchmark &bench, IComputeContext &ctx,
                          uint32_t cfgIdx) {
  bench.PrepareConfig(cfgIdx);
  bench.Run(cfgIdx);
  ctx.waitIdle();
  ASSERT_TRUE(bench.ValidateResults(cfgIdx));
  bench.ReleaseConfigResources(cfgIdx);
}

} // namespace

TEST_CASE(ROCmParityValidation, ROCm102_RuntimeAndDeviceInfoParity) {
  uint32_t devIdx = 0;
  auto ctx = createRocmTestContext(&devIdx);
  if (!ctx) {
    SKIP_TEST("ROCm backend or compatible AMD GPU unavailable");
  }

  const DeviceInfo info = ctx->getCurrentDeviceInfo();
  std::cout << "[ROCm Device " << devIdx << "] " << info.name
            << " | arch=" << info.archName
            << " | version=" << info.driverVersionStr
            << " (" << info.driverVersion << ")"
            << " | pci=" << info.pcieBusId
            << " | devID=0x" << std::hex << info.deviceID << std::dec
            << " | rebar=" << info.rebarSupported
            << " (" << info.rebarSizeBytes << " B)" << std::endl;

  // ROCm 10.2 ships HIP 7.17+ (71700000); reject older system HIP 7.2 (70200000).
  ASSERT_GE(info.driverVersion, 71700000u);
  ASSERT_FALSE(info.pcieBusId.empty());
  ASSERT_NE(info.deviceID, 0u);
  if (!info.isApu) {
    ASSERT_GT(info.dedicatedVramBytes, uint64_t{0});
    if (info.rebarSupported) {
      ASSERT_GT(info.rebarSizeBytes, uint64_t{0});
    }
  }
}

TEST_CASE(ROCmParityValidation, ROCm_CoreCompute_PositiveAndNegativeValidation) {
  auto ctx = createRocmTestContext();
  if (!ctx) {
    SKIP_TEST("ROCm backend or compatible AMD GPU unavailable");
  }
  const std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());
  const DeviceInfo info = ctx->getCurrentDeviceInfo();

  // FP64
  {
    Fp64Bench bench;
    if (bench.IsSupported(info, ctx.get())) {
      bench.Setup(*ctx, kdir);
      runAndValidateConfig(bench, *ctx, 0);
      bench.Teardown();
    }
  }

  // FP32
  {
    Fp32Bench bench;
    ASSERT_TRUE(bench.IsSupported(info, ctx.get()));
    bench.Setup(*ctx, kdir);
    runAndValidateConfig(bench, *ctx, 0);
    bench.Teardown();
  }

  // DualIssue (all 7 configs)
  {
    DualIssueBench bench;
    ASSERT_TRUE(bench.IsSupported(info, ctx.get()));
    bench.Setup(*ctx, kdir);
    for (uint32_t c = 0; c < bench.GetNumConfigs(); ++c) {
      runAndValidateConfig(bench, *ctx, c);
    }
    bench.Teardown();
  }

  // LDS Bank Conflicts (Strides 1 & 32 + LDS-aware occupancy telemetry)
  {
    LdsBankConflictBench bench;
    ASSERT_TRUE(bench.IsSupported(info, ctx.get()));
    bench.Setup(*ctx, kdir);
    runAndValidateConfig(bench, *ctx, 0);
    const KernelResourceUsage usage = ctx->getLastKernelResourceUsage();
    ASSERT_TRUE(usage.available);
    ASSERT_EQ(usage.ldsSizeBytes, 5120u);
    if (info.subgroupSize == 32u) {
      ASSERT_LE(usage.maxWavesPerSimd, 8u);
      ASSERT_GE(usage.maxWavesPerSimd, 4u);
    }
    runAndValidateConfig(bench, *ctx, 5);
    bench.Teardown();
  }

  // FP16 Vector + Matrix
  {
    Fp16Bench bench;
    if (bench.IsSupported(info, ctx.get())) {
      bench.Setup(*ctx, kdir);
      runAndValidateConfig(bench, *ctx, 0);
      if (bench.IsConfigSupported(1, info, ctx.get())) {
        runAndValidateConfig(bench, *ctx, 1);
      }
      ASSERT_EQ(bench.GetResult(0).operations, 65536ULL * 128ULL * 8192ULL * 64ULL);
      bench.Teardown();
    }
  }

  // BF16 Vector + Matrix
  {
    Bf16Bench bench;
    if (bench.IsSupported(info, ctx.get())) {
      bench.Setup(*ctx, kdir);
      if (bench.IsConfigSupported(0, info, ctx.get())) {
        runAndValidateConfig(bench, *ctx, 0);
      }
      if (bench.IsConfigSupported(1, info, ctx.get())) {
        runAndValidateConfig(bench, *ctx, 1);
      }
      bench.Teardown();
    }
  }

  // FP8 Matrix
  {
    Fp8Bench bench;
    if (bench.IsSupported(info, ctx.get())) {
      bench.Setup(*ctx, kdir);
      runAndValidateConfig(bench, *ctx, 0);
      bench.Teardown();
    }
  }

  // INT8 Vector + Matrix
  {
    Int8Bench bench;
    if (bench.IsSupported(info, ctx.get())) {
      bench.Setup(*ctx, kdir);
      runAndValidateConfig(bench, *ctx, 0);
      if (bench.IsConfigSupported(1, info, ctx.get())) {
        runAndValidateConfig(bench, *ctx, 1);
      }
      bench.Teardown();
    }
  }

  // INT4 Matrix (ROCm-only on RDNA 3/4)
  {
    Int4Bench bench;
    if (bench.IsSupported(info, ctx.get())) {
      bench.Setup(*ctx, kdir);
      runAndValidateConfig(bench, *ctx, 0);
      bench.Teardown();
    }
  }

  // 2:4 Structured Sparse Matrix + buffer-isolation & zero negative check
  {
    SparseMatrixBench bench;
    if (bench.IsSupported(info, ctx.get())) {
      bench.Setup(*ctx, kdir);
      for (uint32_t c = 0; c < bench.GetNumConfigs(); ++c) {
        bench.PrepareConfig(c);
        // Zeroed buffer before Run() must fail ValidateResults
        ASSERT_FALSE(bench.ValidateResults(c));
        bench.Run(c);
        ctx->waitIdle();
        ASSERT_TRUE(bench.ValidateResults(c));
        bench.ReleaseConfigResources(c);
      }
      bench.Teardown();
    }
  }
}

TEST_CASE(ROCmParityValidation, ROCm_MemBandwidth_CuScalingAndValidation) {
  auto ctx = createRocmTestContext();
  if (!ctx) {
    SKIP_TEST("ROCm backend or compatible AMD GPU unavailable");
  }
  const std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());
  const DeviceInfo info = ctx->getCurrentDeviceInfo();

  MemBandwidthBench bench;
  ASSERT_TRUE(bench.IsSupported(info, ctx.get()));
  bench.Setup(*ctx, kdir);

  if (info.computeUnits >= 32u) {
    ASSERT_GT(bench.configs[0].numWorkgroups * bench.configs[0].workgroupSize,
              65536u);
  }

  for (uint32_t cfg = 0; cfg <= 5u; ++cfg) {
    if (bench.IsConfigSupported(cfg, info, ctx.get())) {
      runAndValidateConfig(bench, *ctx, cfg);
    }
  }
  bench.Teardown();
}

TEST_CASE(ROCmParityValidation, ROCm_Interconnect_DMA_BiDir_P2P_AndHonesty) {
  auto ctx = createRocmTestContext();
  if (!ctx) {
    SKIP_TEST("ROCm backend or compatible AMD GPU unavailable");
  }
  const std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());
  const DeviceInfo info = ctx->getCurrentDeviceInfo();

  InterconnectBench bench;
  ASSERT_TRUE(bench.IsSupported(info, ctx.get()));
  ASSERT_TRUE(bench.IsConfigSupported(0, info, ctx.get()));
  ASSERT_TRUE(bench.IsConfigSupported(1, info, ctx.get()));
  ASSERT_TRUE(bench.IsConfigSupported(2, info, ctx.get()));
  ASSERT_FALSE(bench.IsConfigSupported(3, info, ctx.get()));
  ASSERT_EQ(static_cast<int>(bench.GetConfigSupportLimitation(3, info, ctx.get())),
            static_cast<int>(IBenchmark::SupportLimitation::kApi));

  bench.Setup(*ctx, kdir);
  for (uint32_t cfg : {0u, 1u, 2u}) {
    bench.PrepareConfig(cfg);
    bench.Run(cfg);
    ctx->waitIdle();
    ASSERT_TRUE(bench.ValidateResults(cfg));
    ASSERT_GT(bench.GetResult(cfg).elapsedTime, 0.0);
    bench.ReleaseConfigResources(cfg);
  }

  if (bench.IsConfigSupported(5, info, ctx.get())) {
    bench.PrepareConfig(5);
    bench.Run(5);
    ctx->waitIdle();
    ASSERT_TRUE(bench.ValidateResults(5));
    ASSERT_GT(bench.GetResult(5).elapsedTime, 0.0);
    bench.ReleaseConfigResources(5);
  }
  bench.Teardown();
}

TEST_CASE(ROCmParityValidation, ROCm_AsyncOverlap_ComputeCopy_AndHonesty) {
  auto ctx = createRocmTestContext();
  if (!ctx) {
    SKIP_TEST("ROCm backend or compatible AMD GPU unavailable");
  }
  const std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());
  const DeviceInfo info = ctx->getCurrentDeviceInfo();

  AsyncOverlapBench bench;
  ASSERT_TRUE(bench.IsSupported(info, ctx.get()));
  ASSERT_FALSE(bench.IsConfigSupported(0, info, ctx.get()));
  ASSERT_FALSE(bench.IsConfigSupported(1, info, ctx.get()));
  ASSERT_EQ(static_cast<int>(bench.GetConfigSupportLimitation(0, info, ctx.get())),
            static_cast<int>(IBenchmark::SupportLimitation::kApi));
  ASSERT_EQ(static_cast<int>(bench.GetConfigSupportLimitation(1, info, ctx.get())),
            static_cast<int>(IBenchmark::SupportLimitation::kApi));
  ASSERT_TRUE(bench.IsConfigSupported(2, info, ctx.get()));

  bench.Setup(*ctx, kdir);
  for (uint32_t cfg : {2u, 3u}) {
    if (!bench.IsConfigSupported(cfg, info, ctx.get())) {
      continue;
    }
    bench.PrepareConfig(cfg);
    bench.Run(cfg);
    ctx->waitIdle();
    ASSERT_TRUE(bench.ValidateResults(cfg));
    ASSERT_GT(bench.GetResult(cfg).elapsedTime, 0.0);
    bench.ReleaseConfigResources(cfg);
  }
  bench.Teardown();
}

TEST_CASE(ROCmParityValidation, ROCm_IsaDumper_HotLoopAndSymbols) {
  uint32_t devIdx = 0;
  auto ctx = createRocmTestContext(&devIdx);
  if (!ctx) {
    SKIP_TEST("ROCm backend or compatible AMD GPU unavailable");
  }
  const DeviceInfo info = ctx->getCurrentDeviceInfo();

  auto checkTarget = [&](const std::string &target, bool expectWmma) {
    IsaDumperOptions opts;
    opts.targets = {target};
    opts.backends = {"rocm"};
    opts.deviceIndex = static_cast<int>(devIdx);
    std::string asic = info.archName;
    size_t colon = asic.find(':');
    if (colon != std::string::npos) {
      asic = asic.substr(0, colon);
    }
    opts.asic = asic;

    const IsaInspectionResult res = IsaDumper::inspect(opts);
    ASSERT_TRUE(res.ok);
    ASSERT_FALSE(res.artifacts.empty());
    bool foundExec = false;
    bool foundHotLoop = false;
    bool foundExpectedMix = false;
    for (const auto &art : res.artifacts) {
      ASSERT_EQ(art.backend, std::string("rocm"));
      for (const auto &exec : art.executables) {
        foundExec = true;
        ASSERT_GT(exec.mix.total, 0);
        const uint32_t expectedSg = (asic.rfind("gfx9", 0) == 0) ? 64u : 32u;
        ASSERT_EQ(exec.subgroupSize, expectedSg);
        if (!exec.hotLoopLines.empty()) {
          foundHotLoop = true;
        }
        if (expectWmma ? (exec.mix.wmma > 0) : (exec.mix.valu > 0)) {
          foundExpectedMix = true;
        }
      }
    }
    ASSERT_TRUE(foundExec);
    ASSERT_TRUE(foundHotLoop);
    ASSERT_TRUE(foundExpectedMix);
  };

  checkTarget("fp32", false);
  if (info.bf16Support) {
    checkTarget("bf16", true);
  }

  // Verify ROCm ASIC gating: FP6/FP4 require CDNA 4 (gfx950) and must cleanly skip on non-gfx950 ASICs
  std::string asic = info.archName;
  size_t colon = asic.find(':');
  if (colon != std::string::npos) {
    asic = asic.substr(0, colon);
  }
  if (asic != "gfx950") {
    for (const char *t : {"fp6", "fp4"}) {
      IsaDumperOptions opts;
      opts.targets = {t};
      opts.backends = {"rocm"};
      opts.deviceIndex = static_cast<int>(devIdx);
      opts.asic = asic;
      const IsaInspectionResult res = IsaDumper::inspect(opts);
      ASSERT_TRUE(res.ok);
      ASSERT_FALSE(res.artifacts.empty());
      for (const auto &art : res.artifacts) {
        ASSERT_FALSE(art.skipReason.empty());
        ASSERT_TRUE(art.executables.empty());
      }
    }
  }
}

TEST_CASE(ROCmParityValidation, ROCm_Indirect_SchedulingParity) {
  auto ctx = createRocmTestContext();
  if (!ctx) {
    SKIP_TEST("ROCm backend or compatible AMD GPU unavailable");
  }
  const std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());
  const DeviceInfo info = ctx->getCurrentDeviceInfo();

  // InShaderIndirectBench: Configs 0..4 pass positive & negative check; 5..6 kApi unsupported
  {
    InShaderIndirectBench bench;
    ASSERT_TRUE(bench.IsSupported(info, ctx.get()));
    ASSERT_EQ(std::string(bench.GetComponent()), std::string("Scheduling"));
    ASSERT_EQ(bench.GetConfigName(2, info, ctx.get()),
              std::string("Persistent Work-Queue Compaction (100% Active)"));
    ASSERT_EQ(bench.GetConfigName(3, info, ctx.get()),
              std::string("Persistent Work-Queue Compaction (10% Active)"));
    ASSERT_EQ(bench.GetConfigName(4, info, ctx.get()),
              std::string("Persistent Work-Queue Early-Exit (0% Active)"));
    for (uint32_t c = 2u; c <= 4u; ++c) {
      const std::string desc = bench.GetDescription(c, info, ctx.get());
      ASSERT_TRUE(desc.find("vkCmdDispatchIndirect") == std::string::npos);
      ASSERT_TRUE(desc.find("VkDispatchIndirectCommand") == std::string::npos);
    }
    for (uint32_t c = 0; c <= 4u; ++c) {
      ASSERT_TRUE(bench.IsConfigSupported(c, info, ctx.get()));
    }
    ASSERT_FALSE(bench.IsConfigSupported(5, info, ctx.get()));
    ASSERT_FALSE(bench.IsConfigSupported(6, info, ctx.get()));
    ASSERT_FALSE(bench.IsConfigSupported(7, info, ctx.get()));
    ASSERT_EQ(static_cast<int>(bench.GetConfigSupportLimitation(5, info, ctx.get())),
              static_cast<int>(IBenchmark::SupportLimitation::kApi));
    ASSERT_EQ(static_cast<int>(bench.GetConfigSupportLimitation(6, info, ctx.get())),
              static_cast<int>(IBenchmark::SupportLimitation::kApi));
    ASSERT_EQ(static_cast<int>(bench.GetConfigSupportLimitation(7, info, ctx.get())),
              static_cast<int>(IBenchmark::SupportLimitation::kApi));

    bench.Setup(*ctx, kdir);
    for (uint32_t c = 0; c <= 4u; ++c) {
      bench.PrepareConfig(c);
      bench.Run(c);
      ctx->waitIdle();
      ASSERT_TRUE(bench.ValidateResults(c));
      bench.ReleaseConfigResources(c);
    }

    // Negative check: zeroed indirect command buffer must fail validation on Config 2
    uint32_t zeroCmd[5] = {0u, 0u, 0u, 0u, 0u};
    ctx->writeBuffer(bench.indirectBuffer, 0, sizeof(zeroCmd), zeroCmd);
    ctx->waitIdle();
    ASSERT_FALSE(bench.ValidateResults(2));

    bench.Teardown();
  }
}
