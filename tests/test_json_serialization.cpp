#include "test_harness.h"
#include "core/ResultFormatter.h"
#include "core/ResultImporter.h"
#include "core/RunnerAPI.h"

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
