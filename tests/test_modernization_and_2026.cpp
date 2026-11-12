#include "test_harness.h"

#include "benchmarks/AsyncOverlapBench.h"
#include "benchmarks/Bf16Bench.h"
#include "benchmarks/CacheBench.h"
#include "benchmarks/CoopVectorBench.h"
#include "benchmarks/Fp16Bench.h"
#include "benchmarks/Fp32Bench.h"
#include "benchmarks/Fp64Bench.h"
#include "benchmarks/InShaderIndirectBench.h"
#include "benchmarks/InterconnectBench.h"
#include "benchmarks/MemBandwidthBench.h"
#include "benchmarks/RayOMMBench.h"
#include "benchmarks/RaySchedulingBench.h"
#include "benchmarks/SparseMatrixBench.h"
#include "benchmarks/SysMemBandwidthBench.h"
#include "benchmarks/SysMemLatencyBench.h"
#include "benchmarks/WorkGraphsBench.h"
#include "core/ComputeBackendFactory.h"
#include "core/ResultFormatter.h"
#include "core/ResultImporter.h"
#include "core/RooflineModel.h"
#include "core/RunnerAPI.h"
#include "utils/HardwareTelemetry.h"
#include "utils/KernelPath.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace {

std::unique_ptr<IComputeContext> createVulkanTestContext() {
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

void exerciseVulkanBenchmarkConfigs(IComputeContext &ctx,
                                    const std::string &kdir,
                                    IBenchmark &bench) {
  const DeviceInfo &info = ctx.getCurrentDeviceInfo();
  if (!bench.IsSupported(info, &ctx)) {
    const std::string note = bench.GetSupportNote(info, &ctx);
    ASSERT_FALSE(note.empty());
    return;
  }

  bench.Setup(ctx, kdir);
  const uint32_t numConfigs = bench.GetNumConfigs();
  for (uint32_t c = 0; c < numConfigs; ++c) {
    if (!bench.IsConfigSupported(c, info, &ctx)) {
      std::string cfgNote = bench.GetConfigSupportNote(c, info, &ctx);
      if (cfgNote.empty()) {
        cfgNote = bench.GetSupportNote(info, &ctx);
      }
      ASSERT_FALSE(cfgNote.empty());
      continue;
    }

    bench.PrepareConfig(c);
    bench.Run(c);
    ctx.waitIdle();

    const bool valid = bench.ValidateResults(c);
    if (!valid) {
      std::cerr << "[Validation Failure] " << bench.GetName() << " config " << c
                << " (" << bench.GetConfigName(c) << ")\n";
    }
    ASSERT_TRUE(valid);

    const BenchmarkResult br = bench.GetResult(c);
    ASSERT_GT(br.operations, 0ULL);
    if (bench.UsesCustomTiming(c)) {
      const double customMs = bench.GetLastRunDurationMs(c);
      ASSERT_TRUE(std::isfinite(customMs));
      ASSERT_GT(customMs, 0.0);
    }

    bench.ReleaseConfigResources(c);
  }
  bench.Teardown();
}

} // namespace

TEST_CASE(ModernizationAnd2026, UnifiedSchemaV2AndLegacySessionImport) {
  DeviceProfile dp{};
  dp.backend = "Vulkan";
  dp.deviceIndex = 0;
  dp.deviceName = "AMD Radeon RX 7900 XTX";
  dp.vendorID = 0x1002;
  dp.deviceID = 0x744C;
  dp.driverName = "AMD proprietary driver";
  dp.driverVersion = "26.10.1";
  dp.apiVersion = "1.4.357";
  dp.vramTotalMb = 24560;
  dp.computeUnits = 96;
  dp.subgroupSize = 32;
  dp.maxWorkGroupSize = 1024;
  dp.rebarSupported = true;
  dp.rayTracingSupported = true;
  dp.workGraphsSupported = true;
  dp.cooperativeMatrixSupported = true;
  dp.opacityMicromapSupported = false;
  dp.structuredSparsitySupported = false;
  dp.cooperativeVectorSupported = false;
  dp.float16Supported = true;
  dp.int8Supported = true;

  UnifiedSessionMetadata sess{};
  sess.scene = "indoor";
  sess.samplesPerPixel = 4;
  sess.dumpRenders = false;
  sess.uiScale = 1.25f;

  std::vector<ResultData> results;
  ResultData bw{};
  bw.backendName = "Vulkan";
  bw.deviceName = dp.deviceName;
  bw.deviceIndex = 0;
  bw.benchmarkName = "Device Memory Bandwidth (Read)";
  bw.component = "Memory";
  bw.subcategory = "VRAM Bandwidth";
  bw.metric = "GB/s";
  bw.operations = 1000000000ULL;
  bw.time_ms = 1.0; // 1000 GB/s
  bw.operationalIntensity = 0.0;
  bw.isValid = true;
  results.push_back(bw);

  ResultData fp32{};
  fp32.backendName = "Vulkan";
  fp32.deviceName = dp.deviceName;
  fp32.deviceIndex = 0;
  fp32.benchmarkName = "FP32";
  fp32.component = "Compute";
  fp32.subcategory = "FP32";
  fp32.metric = "TFLOPS";
  fp32.operations = 50000000000ULL;
  fp32.time_ms = 1.0; // 50 TFLOPS
  fp32.operationalIntensity = 2048.0;
  fp32.isValid = true;
  results.push_back(fp32);

  const std::string jsonV2 = resultsToJson(results, {dp}, &sess);
  ASSERT_TRUE(jsonV2.find("\"schema_version\": \"2.0\"") != std::string::npos);
  ASSERT_TRUE(jsonV2.find("\"gui_session\"") != std::string::npos);
  ASSERT_TRUE(jsonV2.find("\"roofline\"") != std::string::npos);

  ImportedRun runV2;
  std::string err;
  ASSERT_TRUE(ResultImporter::loadFromString(jsonV2, runV2, err));
  ASSERT_EQ(runV2.schemaVersion, "2.0");
  ASSERT_TRUE(runV2.hasGuiSession);
  ASSERT_EQ(runV2.sessionScene, "indoor");
  ASSERT_EQ(runV2.sessionSamplesPerPixel, 4u);
  ASSERT_FALSE(runV2.sessionDumpRenders);
  ASSERT_NEAR(runV2.sessionUiScale, 1.25f, 1e-3f);
  ASSERT_EQ(runV2.deviceProfile.computeUnits, 96u);
  ASSERT_TRUE(runV2.deviceProfile.rebarSupported);
  ASSERT_TRUE(runV2.deviceProfile.workGraphsSupported);
  ASSERT_EQ(runV2.results.size(), 2u);
  ASSERT_NEAR(runV2.results[1].operationalIntensity, 2048.0, 1e-3);

  // Verify backward-compatible import of legacy GUI v1.0 JSON format
  const std::string legacyGuiJson = R"({
    "version": "1.0",
    "backend": "Vulkan",
    "resolution": "1920x1080",
    "scene": "showroom",
    "samples_per_pixel": 2,
    "results": [
      {
        "benchmark": "FP32",
        "component": "Compute",
        "subcategory": "FP32",
        "metric": "TFLOPS",
        "operations": 48000000000,
        "time_ms": 1.0,
        "status": "SUCCESS"
      }
    ]
  })";
  ImportedRun legacyRun;
  ASSERT_TRUE(ResultImporter::loadFromString(legacyGuiJson, legacyRun, err));
  ASSERT_EQ(legacyRun.results.size(), 1u);
  ASSERT_EQ(legacyRun.results[0].benchmarkName, "FP32");
  ASSERT_NEAR(computeResultValue(legacyRun.results[0]), 48.0, 1e-3);
}

TEST_CASE(ModernizationAnd2026, EmpiricalRooflineAnalysisAndEdgeCases) {
  // 1. Empty input
  const auto emptyOut = gpubench::RooflineModel::analyze({});
  ASSERT_TRUE(emptyOut.empty());

  // 2. Unsupported, invalid, and NaN/zero entries must be safely ignored
  std::vector<ResultData> noisy;
  ResultData bad1{};
  bad1.backendName = "Vulkan";
  bad1.deviceName = "Test GPU";
  bad1.benchmarkName = "FP32";
  bad1.metric = "TFLOPS";
  bad1.operations = 100000ULL;
  bad1.time_ms = 0.0;
  bad1.isValid = true;
  noisy.push_back(bad1);

  ResultData bad2 = bad1;
  bad2.time_ms = 1.0;
  bad2.isUnsupported = true;
  noisy.push_back(bad2);

  ResultData bad3 = bad1;
  bad3.time_ms = std::numeric_limits<double>::quiet_NaN();
  noisy.push_back(bad3);

  ASSERT_TRUE(gpubench::RooflineModel::analyze(noisy).empty());

  // 3. Realistic multi-ceiling empirical Roofline
  std::vector<ResultData> valid;
  auto addRes = [&](const char *name, const char *comp, const char *subcat,
                    const char *metric, uint64_t ops, double ms, double oi) {
    ResultData r{};
    r.backendName = "Vulkan";
    r.deviceName = "Test RDNA GPU";
    r.deviceIndex = 0;
    r.benchmarkName = name;
    r.component = comp;
    r.subcategory = subcat;
    r.metric = metric;
    r.operations = ops;
    r.time_ms = ms;
    r.operationalIntensity = oi;
    r.isValid = true;
    valid.push_back(r);
  };

  addRes("Device Memory Bandwidth (Read)", "Memory", "VRAM Bandwidth", "GB/s",
         900000000ULL, 1.0, 0.25); // 900 GB/s
  addRes("L3 Cache Bandwidth (64 MB)", "Memory", "Cache Bandwidth", "GB/s",
         3000000000ULL, 1.0, 0.25); // 3000 GB/s
  addRes("FP32", "Compute", "FP32", "TFLOPS", 50000000000ULL, 1.0,
         2048.0); // 50 TFLOPS
  addRes("FP16 (Matrix)", "Compute", "FP16", "TFLOPS", 200000000000ULL, 1.0,
         4096.0); // 200 TFLOPS

  const auto rooflines = gpubench::RooflineModel::analyze(valid);
  ASSERT_EQ(rooflines.size(), 1u);
  const auto &dr = rooflines[0];
  ASSERT_NEAR(dr.primaryVramGBps(), 900.0, 1e-2);
  ASSERT_NEAR(dr.primaryFp32Gops(), 50000.0, 1e-1);
  ASSERT_NEAR(dr.highestComputeGops(), 200000.0, 1e-1);
  ASSERT_FALSE(dr.ridgePoints.empty());
  ASSERT_FALSE(dr.workloadPoints.empty());

  bool foundFp32VramRidge = false;
  for (const auto &rp : dr.ridgePoints) {
    if (rp.memoryLevel == "VRAM" && rp.computeCeiling == "FP32 Vector") {
      foundFp32VramRidge = true;
      ASSERT_NEAR(rp.ridgeOpsPerByte, 50000.0 / 900.0, 1e-2);
    }
  }
  ASSERT_TRUE(foundFp32VramRidge);

  for (const auto &wp : dr.workloadPoints) {
    ASSERT_GT(wp.attainmentPct, 0.0);
    ASSERT_LE(wp.attainmentPct, 100.0);
  }
}

TEST_CASE(ModernizationAnd2026, BenchmarkMetadataCatalogAndBarrierSemantics) {
  const auto catalog = GetBenchmarkMetadataCatalogAPI();
  ASSERT_FALSE(catalog.empty());

  std::set<std::string> benchNames;
  for (const auto &m : catalog) {
    benchNames.insert(m.benchmarkName);
    ASSERT_FALSE(m.displayName.empty());
    ASSERT_FALSE(m.component.empty());
    ASSERT_FALSE(m.metric.empty());
    ASSERT_FALSE(m.description.empty());
  }

  ASSERT_TRUE(benchNames.count("PCIe & Host-Device Interconnect") > 0);
  ASSERT_TRUE(benchNames.count("Async Compute & Copy Overlap") > 0);
  ASSERT_TRUE(benchNames.count("2:4 Structured Sparse Matrix") > 0);
  ASSERT_TRUE(benchNames.count("Cooperative Vector (Neural Shading)") > 0);
  ASSERT_TRUE(benchNames.count("GPU Work Graphs (Producer-Consumer)") > 0);
  ASSERT_TRUE(benchNames.count("Ray Tracing Opacity Micromap") > 0);

  // Barrier semantics contract: independent compute/cache throughput kernels omit
  // unnecessary cache-flushing inter-iteration barriers, while stateful kernels and
  // full-VRAM L3-evicting sweeps retain them.
  Fp32Bench fp32;
  Fp16Bench fp16;
  CacheBench l0bw("L0 Cache Bandwidth", "GB/s", 16 * 1024, "l0_cache_bw", {}, {"l0b"}, 0);
  MemBandwidthBench membw;
  Fp64Bench fp64;
  InShaderIndirectBench ind;
  WorkGraphsBench wg;
  RaySchedulingBench rs(RaySchedulingBench::SceneType::Showroom);

  ASSERT_FALSE(fp32.RequiresInterIterationBarrier(0));
  ASSERT_FALSE(fp16.RequiresInterIterationBarrier(0));
  ASSERT_FALSE(l0bw.RequiresInterIterationBarrier(0));
  ASSERT_TRUE(membw.RequiresInterIterationBarrier(0));
  ASSERT_TRUE(fp64.RequiresInterIterationBarrier(0));
  ASSERT_TRUE(ind.RequiresInterIterationBarrier(1));
  ASSERT_TRUE(wg.RequiresInterIterationBarrier(0));
  ASSERT_TRUE(rs.RequiresInterIterationBarrier(10));
}

TEST_CASE(ModernizationAnd2026, TelemetryEnergyCounterAndHostSimdValidation) {
  GpuTelemetryData s0{};
  s0.hasEnergyCounter = true;
  s0.energyMicroJoules = 1000000000ULL;

  GpuTelemetryData s1{};
  s1.hasEnergyCounter = true;
  s1.energyMicroJoules = 1250000000ULL; // +250 J over 1000 ms = 250.0 W

  const float watts = HardwareTelemetry::computeWindowPowerWatts(s0, s1, 1000.0);
  ASSERT_NEAR(watts, 250.0f, 1e-2f);

  // Case-insensitive PDH LUID matching (Windows 11 lowercase vs uppercase hex LUIDs)
  const char *pdhLower =
      "pid_4120_luid_0x00000000_0x000123ab_phys_0_eng_0_engtype_3D";
  const char *pdhUpper =
      "pid_4120_luid_0x00000000_0x000123AB_phys_0_eng_0_engtype_Compute_0";
  ASSERT_TRUE(HardwareTelemetry::containsAsciiCaseInsensitive(
      pdhLower, "luid_0x00000000_0x000123ab"));
  ASSERT_TRUE(HardwareTelemetry::containsAsciiCaseInsensitive(
      pdhLower, "luid_0x00000000_0x000123AB"));
  ASSERT_TRUE(HardwareTelemetry::containsAsciiCaseInsensitive(
      pdhUpper, "luid_0x00000000_0x000123ab"));
  ASSERT_FALSE(HardwareTelemetry::containsAsciiCaseInsensitive(
      pdhLower, "luid_0x00000000_0x000123ac"));

  // Live GPU telemetry query must never emit NaN/Inf or fabricate Junction temp
  const GpuTelemetryData live = HardwareTelemetry::queryGpu(0);
  ASSERT_TRUE(std::isfinite(live.powerWatts));
  ASSERT_TRUE(std::isfinite(live.temperatureC));
  ASSERT_TRUE(std::isfinite(live.tempJunctionC));
  ASSERT_TRUE(std::isfinite(live.tempMemoryC));
  if (!live.hasTempJct) {
    ASSERT_NEAR(live.tempJunctionC, 0.0f, 1e-5f);
  }

  // Host SIMD memory bandwidth (AVX-512 / AVX2 / NEON runtime dispatch)
  auto vkCtx = createVulkanTestContext();
  if (vkCtx) {
    SysMemBandwidthBench bwBench;
    bwBench.Setup(*vkCtx, "");
    bwBench.Run(0); // Read
    ASSERT_TRUE(bwBench.ValidateResults(0));
    bwBench.Run(1); // Non-temporal streaming Write
    ASSERT_TRUE(bwBench.ValidateResults(1));
    bwBench.Teardown();

    SysMemLatencyBench latBench;
    latBench.Setup(*vkCtx, "");
    latBench.Run(0);
    ASSERT_TRUE(latBench.ValidateResults(0));
    latBench.Teardown();
  }
}

TEST_CASE(ModernizationAnd2026, Vulkan14_2026BenchmarksEndToEndValidation) {
  auto ctx = createVulkanTestContext();
  if (!ctx) {
    std::cout << "[SKIP] Vulkan backend unavailable" << std::endl;
    return;
  }

  const std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  InterconnectBench interconnect;
  if (auto *vk = dynamic_cast<VulkanContext *>(ctx.get())) {
    const std::string caveat2 =
        interconnect.GetConfigCaveat(2, ctx->getCurrentDeviceInfo(), ctx.get());
    if (!vk->hasDedicatedTransferQueue() || !vk->hasSecondaryTransferQueue()) {
      ASSERT_FALSE(caveat2.empty());
    } else {
      ASSERT_TRUE(caveat2.empty());
    }
  }
  exerciseVulkanBenchmarkConfigs(*ctx, kdir, interconnect);

  Fp16Bench fp16;
  exerciseVulkanBenchmarkConfigs(*ctx, kdir, fp16);

  Bf16Bench bf16;
  exerciseVulkanBenchmarkConfigs(*ctx, kdir, bf16);

  AsyncOverlapBench asyncOverlap;
  exerciseVulkanBenchmarkConfigs(*ctx, kdir, asyncOverlap);

  WorkGraphsBench workGraphs;
  exerciseVulkanBenchmarkConfigs(*ctx, kdir, workGraphs);

  RayOMMBench rayOmm;
  rayOmm.SetResolution(320, 180);
  exerciseVulkanBenchmarkConfigs(*ctx, kdir, rayOmm);

  SparseMatrixBench sparseMatrix;
  exerciseVulkanBenchmarkConfigs(*ctx, kdir, sparseMatrix);

  CoopVectorBench coopVector;
  exerciseVulkanBenchmarkConfigs(*ctx, kdir, coopVector);
}
