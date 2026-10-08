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

  // Construct synthetic benchmark results
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
  r2.baselineConfigIndex = 0;
  results.push_back(r2);

  // 1. Export to JSON
  std::string jsonPayload = resultsToJson(results, profiles);
  ASSERT_FALSE(jsonPayload.empty());
  ASSERT_TRUE(jsonPayload.find("\"device_profiles\"") != std::string::npos);
  ASSERT_TRUE(jsonPayload.find("\"results\"") != std::string::npos);
  ASSERT_TRUE(jsonPayload.find("AMD Radeon 8060S Graphics") != std::string::npos);
  ASSERT_TRUE(jsonPayload.find("0x1586") != std::string::npos);
  ASSERT_TRUE(jsonPayload.find("TFLOPS") != std::string::npos);
  ASSERT_TRUE(jsonPayload.find("power_telemetry") != std::string::npos);

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

  // 4. Verify reconstructed results
  ASSERT_EQ(importedRun.results.size(), 2u);
  ASSERT_EQ(importedRun.results[0].benchmarkName, "FP32");
  ASSERT_EQ(importedRun.results[0].backendName, "Vulkan");
  ASSERT_NEAR(importedRun.results[0].time_ms, 10.15, 1e-4);
  ASSERT_EQ(importedRun.results[0].operations, 200000000000ULL);
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
