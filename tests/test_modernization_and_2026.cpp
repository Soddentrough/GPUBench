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
#include "gui/TelemetryWorker.h"
#include "utils/HardwareTelemetry.h"
#include "utils/IsaDumper.h"
#include "utils/KernelPath.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <thread>
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

TEST_CASE(ModernizationAnd2026, IsaDumper_HonestReportingAndHotLoopAccuracy) {
  auto ctx = createVulkanTestContext();
  if (!ctx) {
    std::cout << "[SKIP] Vulkan backend unavailable" << std::endl;
    return;
  }

  const DeviceInfo info = ctx->getCurrentDeviceInfo();

  // 1. BF16 Vector ISA inspection must match Bf16Bench support (skipped if emulated/unsupported)
  {
    IsaDumperOptions opts;
    opts.targets = {"bf16"};
    opts.backends = {"vulkan"};
    opts.configs = {0};
    const IsaInspectionResult res = IsaDumper::inspect(opts);
    ASSERT_TRUE(res.ok);
    ASSERT_EQ(res.artifacts.size(), size_t{1});
    if (!info.shaderBfloat16DotProductSupport) {
      ASSERT_FALSE(res.artifacts[0].skipReason.empty());
      ASSERT_TRUE(res.artifacts[0].executables.empty());
    }
  }

  // 2. Unrolled direct_worker (indirect config 2) must not fabricate a bogus prologue hot loop
  {
    IsaDumperOptions opts;
    opts.targets = {"indirect"};
    opts.backends = {"vulkan"};
    opts.configs = {2};
    const IsaInspectionResult res = IsaDumper::inspect(opts);
    ASSERT_TRUE(res.ok);
    ASSERT_EQ(res.artifacts.size(), size_t{1});
    if (res.artifacts[0].skipReason.empty() && !res.artifacts[0].executables.empty()) {
      const auto &exec = res.artifacts[0].executables[0];
      if (!exec.fullIsaLines.empty()) {
        ASSERT_EQ(exec.hotLoopStartLine, -1);
        ASSERT_TRUE(exec.hotLoopLines.empty());
      }
    }
  }

  // 3. WorkGraphs shaders must not mistake forward branch or s_endpgm exit stub for a hot loop
  if (info.workGraphsSupported) {
    IsaDumperOptions opts;
    opts.targets = {"workgraphs"};
    opts.backends = {"vulkan"};
    const IsaInspectionResult res = IsaDumper::inspect(opts);
    ASSERT_TRUE(res.ok);
    for (const auto &art : res.artifacts) {
      for (const auto &exec : art.executables) {
        if (!exec.fullIsaLines.empty()) {
          ASSERT_EQ(exec.hotLoopStartLine, -1);
          ASSERT_TRUE(exec.hotLoopLines.empty());
        }
      }
    }
  }

  // 4. Genuine loop shader (membw config 0) must still detect a valid non-empty hot loop
  {
    IsaDumperOptions opts;
    opts.targets = {"membw"};
    opts.backends = {"vulkan"};
    opts.configs = {0};
    const IsaInspectionResult res = IsaDumper::inspect(opts);
    ASSERT_TRUE(res.ok);
    ASSERT_EQ(res.artifacts.size(), size_t{1});
    if (res.artifacts[0].skipReason.empty() && !res.artifacts[0].executables.empty()) {
      const auto &exec = res.artifacts[0].executables[0];
      if (!exec.fullIsaLines.empty()) {
        ASSERT_TRUE(exec.hotLoopStartLine > 0);
        ASSERT_FALSE(exec.hotLoopLines.empty());
        ASSERT_TRUE(exec.hotLoopEndLine <= static_cast<int>(exec.fullIsaLines.size()));
        for (const auto &line : exec.hotLoopLines) {
          ASSERT_TRUE(line.find("s_endpgm") == std::string::npos);
        }
      }
    }
  }

  // 5. RayTracingPipeline alias & honest unavailableReason reporting when text IR is absent
  if (info.rayTracingSupport) {
    IsaDumperOptions opts;
    opts.targets = {"raydiv"};
    opts.backends = {"vulkan"};
    const IsaInspectionResult res = IsaDumper::inspect(opts);
    ASSERT_TRUE(res.ok);
    ASSERT_EQ(res.artifacts.size(), size_t{1});
    ASSERT_EQ(res.artifacts[0].benchName, std::string("RayDivergence"));
    if (res.artifacts[0].skipReason.empty() && !res.artifacts[0].executables.empty()) {
      const auto &exec = res.artifacts[0].executables[0];
      ASSERT_TRUE(!exec.fullIsaLines.empty() || !exec.unavailableReason.empty());
    }
  }

  // 6. RayScheduling non-SER configs must never fall back to SER, and GUI benchmark names must resolve
  {
    ASSERT_TRUE(IsaDumper::isDeferredWarmupTarget("RayScheduling", 0));
    ASSERT_TRUE(IsaDumper::isDeferredWarmupTarget("RayScheduling", 32));
    ASSERT_FALSE(IsaDumper::isDeferredWarmupTarget("RayScheduling", 1));
    ASSERT_FALSE(IsaDumper::isDeferredWarmupTarget("RayScheduling", 33));

    auto inspectSingle = [](const std::string &target, int cfg) {
      IsaDumperOptions opts;
      opts.targets = {target};
      opts.backends = {"vulkan"};
      opts.configs = {cfg};
      return IsaDumper::inspect(opts);
    };

    const IsaInspectionResult rs33 = inspectSingle("RayScheduling", 33);
    ASSERT_TRUE(rs33.ok);
    ASSERT_EQ(rs33.artifacts.size(), size_t{1});
    ASSERT_EQ(rs33.artifacts[0].configIndex, 33);
    ASSERT_EQ(rs33.artifacts[0].artifactFile,
              std::string("rt_scheduling_device_generated_commands_light.comp.spv"));
    ASSERT_TRUE(rs33.artifacts[0].skipReason.empty());
    ASSERT_FALSE(rs33.artifacts[0].executables.empty());

    const IsaInspectionResult rs28 = inspectSingle("RayScheduling", 28);
    ASSERT_TRUE(rs28.ok);
    ASSERT_EQ(rs28.artifacts.size(), size_t{1});
    ASSERT_EQ(rs28.artifacts[0].artifactFile, std::string("rt_scheduling.rgen.spv"));
    if (info.rayTracingSupport) {
      ASSERT_TRUE(rs28.artifacts[0].skipReason.empty());
    }

    const IsaInspectionResult rs1 = inspectSingle("RayScheduling", 1);
    ASSERT_TRUE(rs1.ok);
    ASSERT_EQ(rs1.artifacts.size(), size_t{1});
    ASSERT_EQ(rs1.artifacts[0].artifactFile, std::string("rt_scheduling_ser.rgen.spv"));

    const IsaInspectionResult l0bw = inspectSingle("L0 Cache Bandwidth", 0);
    ASSERT_TRUE(l0bw.ok);
    ASSERT_EQ(l0bw.artifacts.size(), size_t{1});
    ASSERT_EQ(l0bw.artifacts[0].artifactFile, std::string("cache_bw_robust.comp.spv"));
    ASSERT_TRUE(l0bw.artifacts[0].skipReason.empty());

    const IsaInspectionResult l0lat = inspectSingle("L0 Vector Cache Latency (16 KB)", 0);
    ASSERT_TRUE(l0lat.ok);
    ASSERT_EQ(l0lat.artifacts.size(), size_t{1});
    ASSERT_EQ(l0lat.artifacts[0].artifactFile, std::string("l0_cache_latency.comp.spv"));

    const IsaInspectionResult wg0 = inspectSingle("GPU Work Graphs (Producer-Consumer)", 0);
    ASSERT_TRUE(wg0.ok);
    ASSERT_EQ(wg0.artifacts.size(), size_t{1});
    ASSERT_EQ(wg0.artifacts[0].artifactFile, std::string("workgraphs_producer.comp.spv"));

    const IsaInspectionResult async2 = inspectSingle("Async Compute & Copy Overlap", 2);
    ASSERT_TRUE(async2.ok);
    ASSERT_EQ(async2.artifacts.size(), size_t{1});
    ASSERT_EQ(async2.artifacts[0].artifactFile, std::string("async_compute_heavy.comp.spv"));

    const IsaInspectionResult membw4 = inspectSingle("Device Memory Bandwidth", 4);
    ASSERT_TRUE(membw4.ok);
    ASSERT_EQ(membw4.artifacts.size(), size_t{1});
    ASSERT_EQ(membw4.artifacts[0].artifactFile, std::string("membw_256.comp.spv"));

    const IsaInspectionResult ic4 = inspectSingle("PCIe & Host-Device Interconnect", 4);
    ASSERT_TRUE(ic4.ok);
    ASSERT_EQ(ic4.artifacts.size(), size_t{1});
    ASSERT_EQ(ic4.artifacts[0].artifactFile, std::string("interconnect_latency.comp.spv"));

    const IsaInspectionResult cv0 = inspectSingle("Cooperative Vector (Neural Shading)", 0);
    ASSERT_TRUE(cv0.ok);
    ASSERT_EQ(cv0.artifacts.size(), size_t{1});
    ASSERT_EQ(cv0.artifacts[0].artifactFile, std::string("coop_vector_fp16.comp.spv"));

    const IsaInspectionResult omm2 = inspectSingle("Ray Tracing Opacity Micromap", 2);
    ASSERT_TRUE(omm2.ok);
    ASSERT_EQ(omm2.artifacts.size(), size_t{1});
    ASSERT_EQ(omm2.artifacts[0].artifactFile, std::string("rayomm_omm.rgen.spv"));

    const IsaInspectionResult ind6 = inspectSingle("In-Shader Indirect Synthesis", 6);
    ASSERT_TRUE(ind6.ok);
    ASSERT_EQ(ind6.artifacts.size(), size_t{1});
    ASSERT_EQ(ind6.artifacts[0].artifactFile, std::string("direct_worker.comp.spv"));

    const IsaInspectionResult asb0 = inspectSingle("RayASBuild", 0);
    ASSERT_TRUE(asb0.ok);
    ASSERT_EQ(asb0.artifacts.size(), size_t{1});
    ASSERT_FALSE(asb0.artifacts[0].skipReason.empty());
  }
}

TEST_CASE(Modernization2026, IsaDiskCacheAndWarmupValidation) {
  auto ctx = createVulkanTestContext();
  if (!ctx) {
    return;
  }
  const DeviceInfo &info = ctx->getCurrentDeviceInfo();

  IsaCacheContext cacheCtx;
  cacheCtx.vendorId = info.vendorID != 0 ? info.vendorID : 0x1002u;
  cacheCtx.deviceId = info.deviceID != 0 ? info.deviceID : 0x744Cu;
  cacheCtx.driverVersion = !info.driverVersionStr.empty()
                               ? info.driverVersionStr
                               : std::to_string(info.driverVersion);
  cacheCtx.driverInfo = "GPUBench-TestDriver";

  IsaDumperOptions opts;
  opts.targets = {"fp64"};
  opts.backends = {"vulkan"};
  opts.configs = {0};
  opts.deviceIndex = 0;
  opts.fullIsa = true;

  const IsaInspectionResult inspected = IsaDumper::inspect(opts);
  ASSERT_TRUE(inspected.ok);
  ASSERT_FALSE(inspected.artifacts.empty());

  ASSERT_TRUE(IsaDumper::saveCached(opts, cacheCtx, inspected));

  IsaInspectionResult loaded;
  ASSERT_TRUE(IsaDumper::tryLoadCached(opts, cacheCtx, loaded));
  ASSERT_TRUE(loaded.ok);
  ASSERT_EQ(loaded.artifacts.size(), inspected.artifacts.size());
  ASSERT_EQ(loaded.artifacts[0].benchName, inspected.artifacts[0].benchName);
  ASSERT_EQ(loaded.artifacts[0].executables.size(), inspected.artifacts[0].executables.size());
  if (!loaded.artifacts[0].executables.empty()) {
    ASSERT_EQ(loaded.artifacts[0].executables[0].vgprCount,
              inspected.artifacts[0].executables[0].vgprCount);
    ASSERT_EQ(loaded.artifacts[0].executables[0].fullIsaLines.size(),
              inspected.artifacts[0].executables[0].fullIsaLines.size());
  }

  // Strict invalidation on driver update / identity mismatch / backend change
  {
    IsaCacheContext staleDriver = cacheCtx;
    staleDriver.driverVersion = cacheCtx.driverVersion + "-updated";
    IsaInspectionResult miss;
    ASSERT_FALSE(IsaDumper::tryLoadCached(opts, staleDriver, miss));
  }
  {
    IsaCacheContext staleInfo = cacheCtx;
    staleInfo.driverInfo = cacheCtx.driverInfo + "-new";
    IsaInspectionResult miss;
    ASSERT_FALSE(IsaDumper::tryLoadCached(opts, staleInfo, miss));
  }
  {
    IsaCacheContext otherGpu = cacheCtx;
    otherGpu.deviceId = cacheCtx.deviceId ^ 0x1u;
    IsaInspectionResult miss;
    ASSERT_FALSE(IsaDumper::tryLoadCached(opts, otherGpu, miss));
  }
  {
    IsaDumperOptions rocmOpts = opts;
    rocmOpts.backends = {"rocm"};
    IsaInspectionResult miss;
    ASSERT_FALSE(IsaDumper::tryLoadCached(rocmOpts, cacheCtx, miss));
  }

  // warmCacheBatch: multi-item shared session + cancel token test
  std::vector<IsaDumperOptions> batch(3, opts);
  batch[1].targets = {"fp32"};
  batch[2].targets = {"membw"};
  batch[2].configs = {0};

  std::atomic<bool> cancel{false};
  size_t readyCount = 0;
  IsaDumper::warmCacheBatch(batch, cacheCtx, &cancel,
                            [&](const IsaDumperOptions &, IsaInspectionResult &&res) {
                              ASSERT_TRUE(res.ok);
                              ASSERT_FALSE(res.artifacts.empty());
                              ++readyCount;
                            });
  ASSERT_EQ(readyCount, size_t{3});

  // Cancel after first item
  cancel.store(false);
  size_t canceledCount = 0;
  IsaDumper::warmCacheBatch(batch, cacheCtx, &cancel,
                            [&](const IsaDumperOptions &, IsaInspectionResult &&) {
                              ++canceledCount;
                              cancel.store(true);
                            });
  ASSERT_EQ(canceledCount, size_t{1});
}

TEST_CASE(ModernizationAnd2026, TelemetryWorkerRecordsOnlyDuringRunAndStops) {
  gpubench::gui::TelemetryWorker worker;
  worker.start();

  // Idle before any test run: must not accumulate lower-graph trace samples.
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  ASSERT_FALSE(worker.isRecording());
  ASSERT_FALSE(worker.hasCompletedRun());

  if (worker.getDeviceCount() > 0) {
    gpubench::gui::DeviceTelemetrySnapshot idleSnap{};
    ASSERT_TRUE(worker.getSnapshot(0, idleSnap));
    ASSERT_EQ(idleSnap.timeHistory.size(), size_t{0});

    // Active test run: must record samples until stopped.
    worker.startRecording();
    ASSERT_TRUE(worker.isRecording());
    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    gpubench::gui::DeviceTelemetrySnapshot recSnap{};
    ASSERT_TRUE(worker.getSnapshot(0, recSnap));
    if (recSnap.isAvailable) {
      ASSERT_GE(recSnap.timeHistory.size(), size_t{2});
    }

    // Stopped after test run: sample count must freeze and remain stopped.
    worker.stopRecording();
    ASSERT_FALSE(worker.isRecording());
    ASSERT_TRUE(worker.hasCompletedRun());

    gpubench::gui::DeviceTelemetrySnapshot stoppedSnap{};
    ASSERT_TRUE(worker.getSnapshot(0, stoppedSnap));
    const size_t frozenCount = stoppedSnap.timeHistory.size();

    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    gpubench::gui::DeviceTelemetrySnapshot afterIdleSnap{};
    ASSERT_TRUE(worker.getSnapshot(0, afterIdleSnap));
    ASSERT_EQ(afterIdleSnap.timeHistory.size(), frozenCount);

    // CLR / resetHistory must clear history and remain stopped while idle.
    worker.resetHistory();
    ASSERT_FALSE(worker.hasCompletedRun());
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    gpubench::gui::DeviceTelemetrySnapshot clearedSnap{};
    ASSERT_TRUE(worker.getSnapshot(0, clearedSnap));
    ASSERT_EQ(clearedSnap.timeHistory.size(), size_t{0});
  }

  worker.stop();
}
