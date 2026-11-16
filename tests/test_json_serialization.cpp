#include "test_harness.h"
#include "benchmarks/RaySchedulingBench.h"
#include "core/ResultFormatter.h"
#include "core/ResultImporter.h"
#include "core/RunnerAPI.h"
#include "third_party/yyjson.h"
#include <iostream>
#include <sstream>

TEST_CASE(JsonSerialization, RoundTripCompletePayload) {
  // Construct a synthetic DeviceProfile
  std::vector<DeviceProfile> profiles;
  DeviceProfile dp{};
  dp.backend = "Vulkan";
  dp.deviceIndex = 0;
  dp.deviceName = "AMD Radeon 8060S Graphics";
  dp.vendorID = 0x1002;
  dp.deviceID = 0x1586;
  dp.driverName = "radv";
  dp.driverInfo = "Mesa 26.2.3";
  dp.driverVersion = "26.2.3";
  dp.apiVersion = "1.4.341";
  dp.vramTotalMb = 81920;
  dp.subgroupSize = 32;
  dp.maxWorkGroupSize = 1024;
  dp.rayTracingSupported = true;
  dp.serSupported = false;
  dp.workGraphsSupported = false;
  dp.cooperativeMatrixSupported = false;
  dp.float16Supported = true;
  dp.int8Supported = true;
  dp.performanceQuerySupported = true;
  profiles.push_back(dp);

  // Construct synthetic benchmark results covering multiple metrics & configs
  std::vector<ResultData> results;

  ResultData r1{};
  r1.backendName = "Vulkan";
  r1.deviceName = "AMD Radeon 8060S Graphics";
  r1.benchmarkName = "FP32";
  r1.component = "Compute";
  r1.subcategory = "FP32";
  r1.metric = "TFLOPS";
  r1.operations = 200000000000ULL;
  r1.time_ms = 10.15;
  r1.min_time_ms = 9.95;
  r1.median_time_ms = 10.15;
  r1.p95_time_ms = 10.35;
  r1.sample_count = 5;
  r1.isValid = true;
  r1.hasPowerTelemetry = true;
  r1.powerWatts = 53.4f;
  r1.energyJoules = 0.542;
  r1.unitPerWatt = 369.0;
  r1.efficiencyUnit = "GFLOPS/W";
  r1.hasRegisterTelemetry = true;
  r1.vgprCount = 48;
  r1.sgprCount = 32;
  r1.hasPerfQueryTelemetry = true;
  r1.perfCounters.gpuActiveCycles = 25000000ULL;
  r1.perfCounters.waves = 16384;
  r1.perfCounters.l0CacheHitRatio = 0.345f;
  results.push_back(r1);

  ResultData r2{};
  r2.backendName = "Vulkan";
  r2.deviceName = "AMD Radeon 8060S Graphics";
  r2.benchmarkName = "RayScheduling (Wavefront DGC)";
  r2.component = "Ray Tracing";
  r2.subcategory = "Primary Rays";
  r2.metric = "MRays/s";
  r2.operations = 8294400ULL;
  r2.time_ms = 5.25;
  r2.isValid = true;
  r2.configIndex = 18;
  r2.baselineConfigIndex = 17;
  results.push_back(r2);

  // LDS Bank Conflicts (TB/s metric)
  ResultData r3{};
  r3.backendName = "Vulkan";
  r3.deviceName = "AMD Radeon 8060S Graphics";
  r3.benchmarkName = "Stride 1 (Conflict-Free Baseline)";
  r3.component = "Compute";
  r3.subcategory = "LDS & Scratchpad";
  r3.metric = "TB/s";
  r3.operations = 104857600000ULL;
  r3.time_ms = 10.0; // 10.48576 TB/s
  r3.configIndex = 0;
  r3.sortWeight = 28;
  r3.isValid = true;
  results.push_back(r3);

  // In-Shader Indirect Synthesis (MItems/s metric)
  ResultData r4{};
  r4.backendName = "Vulkan";
  r4.deviceName = "AMD Radeon 8060S Graphics";
  r4.benchmarkName = "Indirect Synthesis (10% Active)";
  r4.component = "Compute";
  r4.subcategory = "Indirect Command Synthesis";
  r4.metric = "MItems/s";
  r4.operations = 1048576ULL;
  r4.time_ms = 0.5; // 2097.152 MItems/s
  r4.configIndex = 3;
  r4.baselineConfigIndex = 2;
  r4.sortWeight = 30;
  r4.isValid = true;
  results.push_back(r4);

  // 1. Export to JSON
  std::string jsonPayload = resultsToJson(results, profiles);
  ASSERT_FALSE(jsonPayload.empty());
  ASSERT_TRUE(jsonPayload.find("\"device_profiles\"") != std::string::npos);
  ASSERT_TRUE(jsonPayload.find("\"results\"") != std::string::npos);
  ASSERT_TRUE(jsonPayload.find("AMD Radeon 8060S Graphics") != std::string::npos);
  ASSERT_TRUE(jsonPayload.find("0x1586") != std::string::npos);
  ASSERT_TRUE(jsonPayload.find("TFLOPS") != std::string::npos);
  ASSERT_TRUE(jsonPayload.find("power_telemetry") != std::string::npos);
  // Ensure TB/s and MItems/s did not serialize with value 0.000000
  ASSERT_TRUE(jsonPayload.find("10.485760") != std::string::npos);
  ASSERT_TRUE(jsonPayload.find("2097.152000") != std::string::npos);

  // 2. Parse back via ResultImporter
  ImportedRun importedRun;
  std::string errorMsg;
  bool importSuccess = ResultImporter::loadFromString(jsonPayload, importedRun, errorMsg);
  if (!importSuccess) {
    std::cerr << "Import failed with error: " << errorMsg << "\n";
  }
  ASSERT_TRUE(importSuccess);

  // 3. Verify reconstructed device profile
  ASSERT_EQ(importedRun.deviceProfile.backend, "Vulkan");
  ASSERT_EQ(importedRun.deviceProfile.deviceName, "AMD Radeon 8060S Graphics");
  ASSERT_EQ(importedRun.deviceProfile.deviceId, "0x1586");
  ASSERT_EQ(importedRun.deviceProfile.vramTotalMb, 81920u);
  ASSERT_EQ(importedRun.deviceProfile.subgroupSize, 32u);

  // 4. Verify reconstructed results & telemetry preservation
  ASSERT_EQ(importedRun.results.size(), 4u);
  ASSERT_EQ(importedRun.results[0].benchmarkName, "FP32");
  ASSERT_EQ(importedRun.results[0].backendName, "Vulkan");
  ASSERT_NEAR(importedRun.results[0].time_ms, 10.15, 1e-4);
  ASSERT_EQ(importedRun.results[0].operations, 200000000000ULL);
  ASSERT_EQ(importedRun.results[0].sample_count, 5u);
  ASSERT_NEAR(importedRun.results[0].min_time_ms, 9.95, 1e-4);
  ASSERT_TRUE(importedRun.results[0].hasPowerTelemetry);
  ASSERT_NEAR(importedRun.results[0].powerWatts, 53.4f, 1e-2f);
  ASSERT_TRUE(importedRun.results[0].hasRegisterTelemetry);
  ASSERT_EQ(importedRun.results[0].vgprCount, 48u);
  ASSERT_TRUE(importedRun.results[0].hasPerfQueryTelemetry);
  ASSERT_EQ(importedRun.results[0].perfCounters.waves, 16384u);

  ASSERT_EQ(importedRun.results[1].configIndex, 18u);
  ASSERT_EQ(importedRun.results[1].baselineConfigIndex, 17);
  ASSERT_EQ(importedRun.results[2].metric, "TB/s");
  ASSERT_EQ(importedRun.results[3].metric, "MItems/s");
}

TEST_CASE(JsonSerialization, MultiConfigDisambiguationRoundTrip) {
  // Verify that Read, Write, and R/W VRAM Bandwidth configs and RayASBuild configs
  // preserve distinct configIndex values upon import and do not overwrite each other.
  std::vector<ResultData> results;

  const char *modes[3] = {"Read 128 threads/group", "Write 128 threads/group", "R/W 128 threads/group"};
  for (int i = 0; i < 3; ++i) {
    ResultData r{};
    r.backendName = "Vulkan";
    r.deviceName = "AMD Radeon AI PRO R9700";
    r.benchmarkName = modes[i];
    r.component = "Memory";
    r.subcategory = "VRAM Bandwidth";
    r.metric = "GB/s";
    r.operations = 134217728ULL;
    r.time_ms = 0.25 + 0.05 * i;
    r.configIndex = i;
    r.sortWeight = 130;
    r.isValid = true;
    results.push_back(r);
  }

  std::string jsonPayload = resultsToJson(results);
  ImportedRun importedRun;
  std::string err;
  ASSERT_TRUE(ResultImporter::loadFromString(jsonPayload, importedRun, err));
  ASSERT_EQ(importedRun.results.size(), 3u);
  ASSERT_EQ(importedRun.results[0].configIndex, 0u);
  ASSERT_EQ(importedRun.results[1].configIndex, 1u);
  ASSERT_EQ(importedRun.results[2].configIndex, 2u);
}

TEST_CASE(JsonSerialization, MalformedJsonRejected) {
  ImportedRun run;
  std::string error;

  // Empty string
  ASSERT_FALSE(ResultImporter::loadFromString("", run, error));
  ASSERT_FALSE(error.empty());

  // Truncated JSON
  ASSERT_FALSE(ResultImporter::loadFromString("{\"version\": \"1.0.0\", \"results\": [", run, error));

  // Missing results array
  ASSERT_FALSE(ResultImporter::loadFromString("{\"version\": \"1.0.0\"}", run, error));
}

TEST_CASE(JsonSerialization, RayScheduling16SppFpsDividedBySamplesPerPixel) {
  // 4K resolution (8,294,400 pixels) at 16 SPP (configs 23/24):
  // 16 passes/frame * 8,294,400 rays / 0.1 s = 1,327.104 MRays/s => true FPS must be 10.0, not 160.0.
  ResultData pt16{};
  pt16.backendName = "Vulkan";
  pt16.deviceName = "Test GPU";
  pt16.benchmarkName = "RayScheduling (Open-World Forest) (Full Scene Path Tracing (16 SPP) - DGC)";
  pt16.component = "Ray Tracing";
  pt16.subcategory = "Full Scene Path Tracing";
  pt16.metric = "MRays/s";
  pt16.operations = 8294400ULL * 16ULL;
  pt16.time_ms = 100.0;
  pt16.width = 3840;
  pt16.height = 2160;
  pt16.samplesPerFrame = 16;
  pt16.configIndex = 3;
  pt16.isValid = true;

  std::string jsonPayload = resultsToJson({pt16});
  ASSERT_TRUE(jsonPayload.find("\"samples_per_pixel\": 16") != std::string::npos);

  yyjson_doc *doc = yyjson_read(jsonPayload.data(), jsonPayload.size(), 0);
  ASSERT_TRUE(doc != nullptr);
  yyjson_val *root = yyjson_doc_get_root(doc);
  yyjson_val *resultsArr = yyjson_obj_get(root, "results");
  ASSERT_TRUE(yyjson_is_arr(resultsArr) && yyjson_arr_size(resultsArr) == 1);
  yyjson_val *entry = yyjson_arr_get_first(resultsArr);
  ASSERT_NEAR(yyjson_get_num(yyjson_obj_get(entry, "value")), 1327.104, 1e-3);
  ASSERT_NEAR(yyjson_get_num(yyjson_obj_get(entry, "fps")), 10.0, 1e-4);
  yyjson_doc_free(doc);

  ImportedRun imported;
  std::string err;
  ASSERT_TRUE(ResultImporter::loadFromString(jsonPayload, imported, err));
  ASSERT_EQ(imported.results.size(), 1u);
  ASSERT_EQ(imported.results[0].samplesPerFrame, 16u);
}

TEST_CASE(JsonSerialization, RaySchedulingMultiLightBaselineFormatting) {
  // 1. Verify RaySchedulingBench::GetBaselineConfigIndex maps Total Scene Render (8->6),
  //    Directional Shadows (16->15), and Multi-Light (23->22)
  RaySchedulingBench bench(RaySchedulingBench::SceneType::ReactorBay);
  ASSERT_EQ(bench.GetBaselineConfigIndex(6), -1);
  ASSERT_EQ(bench.GetBaselineConfigIndex(8), 6);
  ASSERT_EQ(bench.GetBaselineConfigIndex(15), -1);
  ASSERT_EQ(bench.GetBaselineConfigIndex(16), 15);
  ASSERT_EQ(bench.GetBaselineConfigIndex(22), -1);
  ASSERT_EQ(bench.GetBaselineConfigIndex(23), 22);

  // 2. Verify ResultFormatter::print() marks both MK configs as [Baseline] and compares 23->22 and 8->6
  auto makeMultiLightResult = [&](uint32_t cfg, int32_t baseCfg, double timeMs, int sortWeight,
                                  const std::string &subcategory, const std::string &name) {
    ResultData r{};
    r.backendName = "Vulkan";
    r.deviceName = "Test GPU";
    r.deviceIndex = 0;
    r.benchmarkName = "RayScheduling (Zero-Day Reactor Bay) (" + name + ")";
    r.component = "Ray Tracing";
    r.subcategory = subcategory;
    r.metric = "MRays/s";
    r.operations = 8294400ULL;
    r.time_ms = timeMs;
    r.width = 3840;
    r.height = 2160;
    r.configIndex = cfg;
    r.baselineConfigIndex = baseCfg;
    r.sortWeight = sortWeight;
    r.isValid = true;
    return r;
  };

  ResultFormatter fmt;
  fmt.addResult(makeMultiLightResult(6, -1, 20.0, 632, "Total Scene Render",
                                     "Scene Render: Full Frame (Megakernel)"));
  fmt.addResult(makeMultiLightResult(8, 6, 16.0, 634, "Total Scene Render",
                                     "Scene Render: Full Frame (DGC)"));
  fmt.addResult(makeMultiLightResult(22, -1, 8.0, 639, "Multi-Light Evaluation",
                                     "Stage: Multi-Light Evaluation - 128 Lights (Megakernel)"));
  fmt.addResult(makeMultiLightResult(23, 22, 6.4, 640, "Multi-Light Evaluation",
                                     "Stage: Multi-Light Evaluation - 128 Lights (DGC Light Binning)"));

  std::ostringstream captured;
  struct CoutRedirectGuard {
    std::streambuf *saved = nullptr;
    explicit CoutRedirectGuard(std::streambuf *buf) : saved(std::cout.rdbuf(buf)) {}
    ~CoutRedirectGuard() {
      if (saved) std::cout.rdbuf(saved);
    }
  };
  {
    CoutRedirectGuard redirect(captured.rdbuf());
    fmt.print();
  }

  auto stripAnsi = [](const std::string &in) {
    std::string out;
    out.reserve(in.size());
    bool inEsc = false;
    for (char ch : in) {
      if (ch == '\033') {
        inEsc = true;
      } else if (inEsc) {
        if (ch == 'm') inEsc = false;
      } else {
        out.push_back(ch);
      }
    }
    return out;
  };

  std::string lineFullMk, lineFullDgc, line128Mk, line128Dgc;
  std::istringstream lines(stripAnsi(captured.str()));
  std::string line;
  while (std::getline(lines, line)) {
    if (line.find("Vulkan") == std::string::npos) continue;
    if (lineFullMk.empty() && line.find("Full Frame (Megakernel)") != std::string::npos) lineFullMk = line;
    else if (lineFullDgc.empty() && line.find("Full Frame (DGC)") != std::string::npos) lineFullDgc = line;
    else if (line128Mk.empty() && line.find("128 Lights (Megakernel)") != std::string::npos) line128Mk = line;
    else if (line128Dgc.empty() && line.find("128 Lights (DGC Light Binning)") != std::string::npos) line128Dgc = line;
  }

  ASSERT_TRUE(!lineFullMk.empty());
  ASSERT_TRUE(lineFullMk.find("[Baseline]") != std::string::npos);
  ASSERT_TRUE(lineFullMk.find("└──>") == std::string::npos);

  ASSERT_TRUE(!lineFullDgc.empty());
  ASSERT_TRUE(lineFullDgc.find("└──>") != std::string::npos);
  ASSERT_TRUE(lineFullDgc.find("1.25x (+25.0%)") != std::string::npos);

  ASSERT_TRUE(!line128Mk.empty());
  ASSERT_TRUE(line128Mk.find("[Baseline]") != std::string::npos);
  ASSERT_TRUE(line128Mk.find("└──>") == std::string::npos);

  ASSERT_TRUE(!line128Dgc.empty());
  ASSERT_TRUE(line128Dgc.find("└──>") != std::string::npos);
  ASSERT_TRUE(line128Dgc.find("1.25x (+25.0%)") != std::string::npos);

  // 3. Verify legacy GUI schema import assigns matching configIndex & baselineConfigIndex
  //    for current Reactor Bay IDs and archival volumetric IDs
  const std::string guiJson = R"json({
    "schema_version": "1.0",
    "backend": "Vulkan",
    "resolution": "4K (UHD) (3840x2160)",
    "results": [{
      "device_id": 0,
      "device_name": "0: Test GPU",
      "benchmarks": [
        {"id": "rt_sched_multilight_128_trad", "unit": "MRays/s", "status": "ok", "raw_operations": 8294400, "raw_time_ms": 8.0},
        {"id": "rt_sched_multilight_128_dgc",  "unit": "MRays/s", "status": "ok", "raw_operations": 8294400, "raw_time_ms": 6.4},
        {"id": "rt_sched_volumetric_trad",     "unit": "MRays/s", "status": "ok", "raw_operations": 8294400, "raw_time_ms": 20.0},
        {"id": "rt_sched_volumetric_dgc",      "unit": "MRays/s", "status": "ok", "raw_operations": 8294400, "raw_time_ms": 16.0},
        {"id": "rt_sched_full_reactor_trad",   "unit": "MRays/s", "status": "ok", "raw_operations": 8294400, "raw_time_ms": 20.0},
        {"id": "rt_sched_full_reactor_wl",     "unit": "MRays/s", "status": "ok", "raw_operations": 8294400, "raw_time_ms": 16.0}
      ]
    }]
  })json";

  ImportedRun guiImported;
  std::string guiErr;
  ASSERT_TRUE(ResultImporter::loadFromString(guiJson, guiImported, guiErr));
  ASSERT_EQ(guiImported.results.size(), 6u);
  ASSERT_EQ(guiImported.results[0].configIndex, 22u);
  ASSERT_EQ(guiImported.results[0].baselineConfigIndex, -1);
  ASSERT_EQ(guiImported.results[1].configIndex, 23u);
  ASSERT_EQ(guiImported.results[1].baselineConfigIndex, 22);
  ASSERT_EQ(guiImported.results[2].configIndex, 27u);
  ASSERT_EQ(guiImported.results[2].baselineConfigIndex, -1);
  ASSERT_EQ(guiImported.results[3].configIndex, 28u);
  ASSERT_EQ(guiImported.results[3].baselineConfigIndex, 27);
  ASSERT_EQ(guiImported.results[4].configIndex, 6u);
  ASSERT_EQ(guiImported.results[4].baselineConfigIndex, -1);
  ASSERT_EQ(guiImported.results[5].configIndex, 8u);
  ASSERT_EQ(guiImported.results[5].baselineConfigIndex, 6);
}

TEST_CASE(JsonSerialization, GigaUnitAutoScalingAndSpeedupParity) {
  ASSERT_EQ(formatScaledMetricString(999.99, "MRays/s"), std::string("999.99 MRays/s"));
  ASSERT_EQ(formatScaledMetricString(999.996, "MRays/s"), std::string("1.00 GRays/s"));
  ASSERT_EQ(formatScaledMetricString(2621.45, "MRays/s"), std::string("2.62 GRays/s"));
  ASSERT_EQ(formatScaledMetricString(14765.8, "MTris/s"), std::string("14.77 GTris/s"));
  ASSERT_EQ(formatScaledMetricString(8502.3, "MInst/s"), std::string("8.50 GInst/s"));
  ASSERT_EQ(formatScaledMetricString(1540.0, "MHits/s"), std::string("1.54 GHits/s"));
  // Non-RT units must remain untouched
  ASSERT_EQ(formatScaledMetricString(1250.0, "GB/s"), std::string("1,250.00 GB/s"));

  // Verify ResultFormatter table scales >=1000 MRays/s to 2-decimal GRays/s while computing exact 2.50x speedup
  auto makeRt = [](uint32_t cfg, int32_t baseCfg, double mrays, const std::string &name) {
    ResultData r{};
    r.backendName = "Vulkan";
    r.deviceName = "Test GPU";
    r.deviceIndex = 0;
    r.benchmarkName = "RayScheduling (Showroom Studio) (" + name + ")";
    r.component = "Ray Tracing";
    r.subcategory = "Material Shading";
    r.metric = "MRays/s";
    r.operations = static_cast<uint64_t>(mrays * 1e6);
    r.time_ms = 1000.0;
    r.width = 3840;
    r.height = 2160;
    r.configIndex = cfg;
    r.baselineConfigIndex = baseCfg;
    r.sortWeight = 600 + static_cast<int>(cfg);
    r.isValid = true;
    return r;
  };

  ResultFormatter fmt;
  fmt.addResult(makeRt(0, -1, 500.0, "Material Shading (Megakernel)"));
  fmt.addResult(makeRt(2, 0, 1250.0, "Material Shading (DGC)"));

  std::ostringstream captured;
  {
    std::streambuf *saved = std::cout.rdbuf(captured.rdbuf());
    fmt.print();
    std::cout.rdbuf(saved);
  }
  std::string out = captured.str();
  ASSERT_TRUE(out.find("500.00 MRays/s") != std::string::npos);
  ASSERT_TRUE(out.find("1.25 GRays/s") != std::string::npos);
  ASSERT_TRUE(out.find("2.50x") != std::string::npos);
}
