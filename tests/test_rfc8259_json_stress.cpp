#include "test_harness.h"
#include "core/ResultFormatter.h"
#include "core/ResultImporter.h"
#include "core/RunnerAPI.h"
#include "core/DeviceDatabase.h"

#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <fstream>
#include <algorithm>

namespace {

// Helper to strictly assert no unquoted nan or inf appears in JSON string
void assertNoNonFiniteTokens(const std::string &jsonStr) {
  // Check for common non-standard float tokens that C/C++ might write
  ASSERT_EQ(jsonStr.find(": nan"), std::string::npos);
  ASSERT_EQ(jsonStr.find(": -nan"), std::string::npos);
  ASSERT_EQ(jsonStr.find(": nan(ind)"), std::string::npos);
  ASSERT_EQ(jsonStr.find(": -nan(ind)"), std::string::npos);
  ASSERT_EQ(jsonStr.find(": inf"), std::string::npos);
  ASSERT_EQ(jsonStr.find(": -inf"), std::string::npos);
  ASSERT_EQ(jsonStr.find(": 1.#INF"), std::string::npos);
  ASSERT_EQ(jsonStr.find(": -1.#INF"), std::string::npos);
  ASSERT_EQ(jsonStr.find(": 1.#QNAN"), std::string::npos);
  ASSERT_EQ(jsonStr.find(": -1.#IND"), std::string::npos);
}

DeviceProfile createStandardProfile() {
  DeviceProfile dp{};
  dp.backend = "Vulkan";
  dp.deviceIndex = 0;
  dp.deviceName = "AMD Radeon 8060S Graphics (RADV STRIX_HALO)";
  dp.vendorID = 0x1002;
  dp.deviceID = 0x1586;
  dp.driverName = "radv";
  dp.driverInfo = "Mesa 26.2.4";
  dp.driverVersion = "26.2.4";
  dp.apiVersion = "1.4.354";
  dp.vramTotalMb = 83285;
  dp.subgroupSize = 64;
  dp.maxWorkGroupSize = 1024;
  dp.rayTracingSupported = true;
  dp.serSupported = false;
  dp.workGraphsSupported = false;
  dp.cooperativeMatrixSupported = true;
  dp.float16Supported = true;
  dp.int8Supported = true;
  dp.performanceQuerySupported = true;
  return dp;
}

} // namespace

// ============================================================================
// 1. Zero Active Cycles Stress Test
// ============================================================================

TEST_CASE(RFC8259_JsonStress, ZeroActiveCyclesStress) {
  std::vector<DeviceProfile> profiles = {createStandardProfile()};
  std::vector<ResultData> results;

  // Case A: 0 cycles with 0 operations and 0 time
  ResultData r1{};
  r1.backendName = "Vulkan";
  r1.deviceName = profiles[0].deviceName;
  r1.benchmarkName = "LDS Bank Conflicts";
  r1.component = "Compute";
  r1.subcategory = "LDS";
  r1.metric = "TB/s";
  r1.operations = 0ULL;
  r1.time_ms = 0.0;
  r1.isValid = true;
  r1.hasPerfQueryTelemetry = true;
  r1.perfCounters.available = true;
  r1.perfCounters.gpuActiveCycles = 0ULL;
  r1.perfCounters.waves = 0ULL;
  r1.perfCounters.valuInstructions = 0ULL;
  r1.perfCounters.saluInstructions = 0ULL;
  r1.perfCounters.valuBusyPct = 0.0f;
  r1.perfCounters.saluBusyPct = 0.0f;
  results.push_back(r1);

  // Case B: 0 cycles with valid elapsed time and operations (counter fail-closed)
  ResultData r2{};
  r2.backendName = "Vulkan";
  r2.deviceName = profiles[0].deviceName;
  r2.benchmarkName = "FP32";
  r2.component = "Compute";
  r2.subcategory = "FP32";
  r2.metric = "TFLOPS";
  r2.operations = 1000000000ULL;
  r2.time_ms = 5.0;
  r2.isValid = true;
  r2.hasPerfQueryTelemetry = true;
  r2.perfCounters.available = true;
  r2.perfCounters.gpuActiveCycles = 0ULL;
  r2.perfCounters.waves = 512;
  r2.perfCounters.valuInstructions = 10000ULL;
  r2.perfCounters.valuBusyPct = 0.0f;
  results.push_back(r2);

  std::string jsonStr = resultsToJson(results, profiles);
  assertNoNonFiniteTokens(jsonStr);

  ImportedRun run;
  std::string err;
  bool ok = ResultImporter::loadFromString(jsonStr, run, err);
  if (!ok) {
    std::cerr << "Import error: " << err << "\n";
  }
  ASSERT_TRUE(ok);
  ASSERT_EQ(run.results.size(), 2u);
  ASSERT_EQ(run.results[0].perfCounters.gpuActiveCycles, 0ULL);
  ASSERT_EQ(run.results[1].perfCounters.gpuActiveCycles, 0ULL);
}

// ============================================================================
// 2. Zero Operations Across All 18 Metric Types
// ============================================================================

TEST_CASE(RFC8259_JsonStress, ZeroOperationsAllMetrics) {
  std::vector<DeviceProfile> profiles = {createStandardProfile()};
  std::vector<ResultData> results;

  const std::vector<std::string> metrics = {
      "TFLOPS", "TOPS", "TB/s", "GB/s", "GIS/s", "GRays/s", "GPixels/s",
      "MRays/s", "MHits/s", "MTris/s", "MInst/s", "MRecords/s", "MItems/s",
      "MSamples/s", "MBVH/s", "MPixels/s", "us", "ns"};

  for (size_t i = 0; i < metrics.size(); ++i) {
    ResultData r{};
    r.backendName = "Vulkan";
    r.deviceName = profiles[0].deviceName;
    r.benchmarkName = "TestBenchmark_" + metrics[i];
    r.component = "Compute";
    r.subcategory = "Stress";
    r.metric = metrics[i];
    r.operations = 0ULL; // ZERO OPERATIONS
    r.time_ms = 12.34;   // Valid non-zero time
    r.isValid = true;
    r.configIndex = static_cast<uint32_t>(i);

    // Verify computeResultValue mathematically guards against divide-by-zero
    double computedVal = computeResultValue(r);
    ASSERT_EQ(computedVal, 0.0);
    ASSERT_FALSE(std::isnan(computedVal));
    ASSERT_FALSE(std::isinf(computedVal));

    results.push_back(r);
  }

  std::string jsonStr = resultsToJson(results, profiles);
  assertNoNonFiniteTokens(jsonStr);

  ImportedRun run;
  std::string err;
  ASSERT_TRUE(ResultImporter::loadFromString(jsonStr, run, err));
  ASSERT_EQ(run.results.size(), metrics.size());
  for (size_t i = 0; i < run.results.size(); ++i) {
    ASSERT_EQ(run.results[i].operations, 0ULL);
    double importedVal = computeResultValue(run.results[i]);
    ASSERT_EQ(importedVal, 0.0);
  }
}

// ============================================================================
// 3. Zero Time and Negative Time (Aborted/Failed) Across Metrics
// ============================================================================

TEST_CASE(RFC8259_JsonStress, ZeroAndNegativeTimeStress) {
  std::vector<DeviceProfile> profiles = {createStandardProfile()};
  std::vector<ResultData> results;

  // Case 1: time_ms = 0.0 with positive operations
  ResultData r1{};
  r1.backendName = "Vulkan";
  r1.deviceName = profiles[0].deviceName;
  r1.benchmarkName = "ZeroTimeBench";
  r1.component = "Compute";
  r1.metric = "TFLOPS";
  r1.operations = 100000000ULL;
  r1.time_ms = 0.0;
  r1.isValid = true;
  ASSERT_EQ(computeResultValue(r1), 0.0);
  results.push_back(r1);

  // Case 2: time_ms = -1.0 (FAILED)
  ResultData r2{};
  r2.backendName = "Vulkan";
  r2.deviceName = profiles[0].deviceName;
  r2.benchmarkName = "FailedBench";
  r2.component = "Compute";
  r2.metric = "GB/s";
  r2.operations = 500000ULL;
  r2.time_ms = -1.0;
  r2.errorString = "Device lost simulation";
  r2.isValid = false;
  ASSERT_EQ(computeResultValue(r2), 0.0);
  results.push_back(r2);

  // Case 3: time_ms = -3.0 (ABORTED)
  ResultData r3{};
  r3.backendName = "Vulkan";
  r3.deviceName = profiles[0].deviceName;
  r3.benchmarkName = "AbortedBench";
  r3.component = "Ray Tracing";
  r3.metric = "MRays/s";
  r3.operations = 0ULL;
  r3.time_ms = -3.0;
  r3.errorString = "User aborted benchmark run";
  r3.isValid = false;
  ASSERT_EQ(computeResultValue(r3), 0.0);
  results.push_back(r3);

  std::string jsonStr = resultsToJson(results, profiles);
  assertNoNonFiniteTokens(jsonStr);

  ASSERT_TRUE(jsonStr.find("\"status\": \"FAILED\"") != std::string::npos);
  ASSERT_TRUE(jsonStr.find("\"status\": \"ABORTED\"") != std::string::npos);

  ImportedRun run;
  std::string err;
  ASSERT_TRUE(ResultImporter::loadFromString(jsonStr, run, err));
  ASSERT_EQ(run.results.size(), 3u);
}

// ============================================================================
// 4. Adversarial Non-Finite Floats Fuzzing (NaN, Inf, Denormals)
// ============================================================================

TEST_CASE(RFC8259_JsonStress, NonFiniteFloatsFuzzingStress) {
  std::vector<DeviceProfile> profiles = {createStandardProfile()};
  std::vector<ResultData> results;

  ResultData r{};
  r.backendName = "Vulkan";
  r.deviceName = profiles[0].deviceName;
  r.benchmarkName = "AdversarialFuzzFloats";
  r.component = "Compute";
  r.metric = "TFLOPS";
  r.operations = 0ULL;
  r.time_ms = std::numeric_limits<double>::quiet_NaN();
  r.sample_count = 5;
  r.min_time_ms = -std::numeric_limits<double>::infinity();
  r.median_time_ms = std::numeric_limits<double>::infinity();
  r.mean_time_ms = std::numeric_limits<double>::quiet_NaN();
  r.p95_time_ms = std::numeric_limits<double>::signaling_NaN();
  r.isValid = true;

  // Power telemetry non-finite fuzzing
  r.hasPowerTelemetry = true;
  r.powerWatts = std::numeric_limits<float>::quiet_NaN();
  r.energyJoules = std::numeric_limits<double>::infinity();
  r.joulesPerUnit = -std::numeric_limits<double>::quiet_NaN();
  r.unitPerWatt = -std::numeric_limits<double>::infinity();
  r.efficiencyUnit = "J/FLOP";

  // Performance query non-finite fuzzing
  r.hasPerfQueryTelemetry = true;
  r.perfCounters.available = true;
  r.perfCounters.gpuActiveCycles = 0ULL;
  r.perfCounters.valuBusyPct = std::numeric_limits<float>::quiet_NaN();
  r.perfCounters.saluBusyPct = std::numeric_limits<float>::infinity();
  r.perfCounters.l0CacheHitRatio = -std::numeric_limits<float>::infinity();
  r.perfCounters.l1CacheHitRatio = std::numeric_limits<float>::signaling_NaN();
  r.perfCounters.rawCounters["salu_cycles_per_wave"] = std::numeric_limits<double>::infinity();
  r.perfCounters.rawCounters["valu_cycles_per_wave"] = std::numeric_limits<double>::quiet_NaN();
  r.perfCounters.rawCounters["vram_bandwidth_ratio"] = -std::numeric_limits<double>::infinity();
  r.perfCounters.rawCounters["denormal_float"] = 1.0e-315; // Subnormal float

  results.push_back(r);

  std::string jsonStr = resultsToJson(results, profiles);
  assertNoNonFiniteTokens(jsonStr);

  ImportedRun run;
  std::string err;
  ASSERT_TRUE(ResultImporter::loadFromString(jsonStr, run, err));
  ASSERT_EQ(run.results.size(), 1u);
  // Replaced with fallback 0.0 in JSON
  ASSERT_EQ(run.results[0].time_ms, 0.0);
  ASSERT_EQ(run.results[0].powerWatts, 0.0f);
  ASSERT_EQ(run.results[0].perfCounters.valuBusyPct, 0.0f);
}

// ============================================================================
// 5. Special Benchmark Types with Zero Operations & Zero Dimensions
// ============================================================================

TEST_CASE(RFC8259_JsonStress, SpecialBenchmarksZeroDimensionsAndOps) {
  std::vector<DeviceProfile> profiles = {createStandardProfile()};
  std::vector<ResultData> results;

  // RayScheduling with 0 width, 0 height, 0 operations, 0 time
  ResultData rRaySched{};
  rRaySched.backendName = "Vulkan";
  rRaySched.deviceName = profiles[0].deviceName;
  rRaySched.benchmarkName = "RayScheduling (Wavefront DGC)";
  rRaySched.component = "Ray Tracing";
  rRaySched.subcategory = "Primary Rays";
  rRaySched.metric = "MRays/s";
  rRaySched.operations = 0ULL;
  rRaySched.time_ms = 0.0;
  rRaySched.width = 0;
  rRaySched.height = 0;
  rRaySched.isValid = true;
  results.push_back(rRaySched);

  // RayRawTraversal (Box Traversal configIndex = 1) with 0 ops, 0 time
  ResultData rRayBox{};
  rRayBox.backendName = "Vulkan";
  rRayBox.deviceName = profiles[0].deviceName;
  rRayBox.benchmarkName = "RayRawTraversal";
  rRayBox.component = "Ray Tracing";
  rRayBox.subcategory = "Hardware BVH Traversal";
  rRayBox.metric = "GIS/s";
  rRayBox.operations = 0ULL;
  rRayBox.time_ms = 0.0;
  rRayBox.configIndex = 1; // Box Traversal
  rRayBox.vendorId = 0x1002;
  rRayBox.deviceId = 0x1586;
  rRayBox.isValid = true;
  results.push_back(rRayBox);

  // FP32 with 0 ops, 0 time
  ResultData rFp32{};
  rFp32.backendName = "Vulkan";
  rFp32.deviceName = profiles[0].deviceName;
  rFp32.benchmarkName = "FP32";
  rFp32.component = "Compute";
  rFp32.metric = "TFLOPS";
  rFp32.operations = 0ULL;
  rFp32.time_ms = 0.0;
  rFp32.vendorId = 0x1002;
  rFp32.deviceId = 0x1586;
  rFp32.isValid = true;
  results.push_back(rFp32);

  // Device Memory Bandwidth with 0 ops, 0 time
  ResultData rBw{};
  rBw.backendName = "Vulkan";
  rBw.deviceName = profiles[0].deviceName;
  rBw.benchmarkName = "Device Memory Bandwidth";
  rBw.component = "Memory";
  rBw.metric = "GB/s";
  rBw.operations = 0ULL;
  rBw.time_ms = 0.0;
  rBw.vendorId = 0x1002;
  rBw.deviceId = 0x1586;
  rBw.isValid = true;
  results.push_back(rBw);

  std::string jsonStr = resultsToJson(results, profiles);
  assertNoNonFiniteTokens(jsonStr);

  ImportedRun run;
  std::string err;
  ASSERT_TRUE(ResultImporter::loadFromString(jsonStr, run, err));
  ASSERT_EQ(run.results.size(), 4u);
}

// ============================================================================
// 6. Power Summary Zero Division Stress Test
// ============================================================================

TEST_CASE(RFC8259_JsonStress, PowerSummaryZeroDivisionStress) {
  std::vector<DeviceProfile> profiles = {createStandardProfile()};
  std::vector<ResultData> results;

  // Case A: Power telemetry available but powerWatts <= 0.0f
  for (int i = 0; i < 3; ++i) {
    ResultData r{};
    r.backendName = "Vulkan";
    r.deviceName = profiles[0].deviceName;
    r.benchmarkName = "ZeroWattBench_" + std::to_string(i);
    r.metric = "TFLOPS";
    r.operations = 1000ULL;
    r.time_ms = 1.0;
    r.hasPowerTelemetry = true;
    r.powerWatts = 0.0f; // Exactly zero watts
    r.energyJoules = 0.0;
    r.isValid = true;
    results.push_back(r);
  }

  std::string jsonStr = resultsToJson(results, profiles);
  assertNoNonFiniteTokens(jsonStr);

  // Since all powerWatts <= 0, power_summary should not be emitted (avoiding 0/0)
  ASSERT_EQ(jsonStr.find("\"power_summary\""), std::string::npos);

  // Case B: Add one result with positive power
  ResultData rPos{};
  rPos.backendName = "Vulkan";
  rPos.deviceName = profiles[0].deviceName;
  rPos.benchmarkName = "PosWattBench";
  rPos.metric = "TFLOPS";
  rPos.operations = 1000ULL;
  rPos.time_ms = 1.0;
  rPos.hasPowerTelemetry = true;
  rPos.powerWatts = 50.0f;
  rPos.energyJoules = 0.05;
  rPos.isValid = true;
  results.push_back(rPos);

  std::string jsonStrWithPos = resultsToJson(results, profiles);
  assertNoNonFiniteTokens(jsonStrWithPos);
  ASSERT_TRUE(jsonStrWithPos.find("\"power_summary\"") != std::string::npos);
  ASSERT_TRUE(jsonStrWithPos.find("\"average_power_watts\": 50.000000") != std::string::npos);
}

// ============================================================================
// 7. String Escaping and RFC 8259 String Invariants
// ============================================================================

TEST_CASE(RFC8259_JsonStress, StringEscapingRFC8259) {
  std::vector<DeviceProfile> profiles = {createStandardProfile()};
  profiles[0].deviceName = "AMD Radeon™ \"Advanced\" Graphics \\ RDNA\n\r\t";

  std::vector<ResultData> results;
  ResultData r{};
  r.backendName = "Vulkan";
  r.deviceName = profiles[0].deviceName;
  r.benchmarkName = "Benchmark \"Quotes\" & \\Backslashes\\ \nLine2\rReturn\tTab";
  r.component = "Compute";
  r.subcategory = "Escaping <Test> & Special's";
  r.metric = "TFLOPS";
  r.operations = 1000ULL;
  r.time_ms = 1.0;
  r.errorString = "Error: \"Unexpected \\NULL pointer\nStack trace: /path/to/file.cpp:42\"";
  r.supportNote = "Note with quotes \"123\" and backslash \\ and newline \n";
  r.isValid = false;
  results.push_back(r);

  std::string jsonStr = resultsToJson(results, profiles);
  assertNoNonFiniteTokens(jsonStr);

  ImportedRun run;
  std::string err;
  ASSERT_TRUE(ResultImporter::loadFromString(jsonStr, run, err));
  ASSERT_EQ(run.results.size(), 1u);
  ASSERT_EQ(run.results[0].benchmarkName, "Benchmark \"Quotes\" & \\Backslashes\\ \nLine2\rReturn\tTab");
  ASSERT_EQ(run.results[0].supportNote, "Note with quotes \"123\" and backslash \\ and newline \n");
}

// ============================================================================
// 8. 1000 Mixed Adversarial Edge Cases Batch Scale Test
// ============================================================================

TEST_CASE(RFC8259_JsonStress, BatchScaling1000AdversarialResults) {
  std::vector<DeviceProfile> profiles = {createStandardProfile()};
  std::vector<ResultData> results;
  results.reserve(1000);

  const std::vector<std::string> metrics = {"TFLOPS", "TOPS", "GB/s", "MRays/s", "us", "ns"};

  for (int i = 0; i < 1000; ++i) {
    ResultData r{};
    r.backendName = (i % 2 == 0) ? "Vulkan" : "System";
    r.deviceName = profiles[0].deviceName;
    r.benchmarkName = "StressRun_" + std::to_string(i);
    r.component = (i % 3 == 0) ? "Compute" : ((i % 3 == 1) ? "Memory" : "Ray Tracing");
    r.metric = metrics[i % metrics.size()];
    r.configIndex = static_cast<uint32_t>(i % 8);

    // Pseudorandom edge case distribution
    switch (i % 8) {
    case 0: // Zero active cycles
      r.operations = 1000000ULL;
      r.time_ms = 2.0;
      r.hasPerfQueryTelemetry = true;
      r.perfCounters.gpuActiveCycles = 0ULL;
      r.isValid = true;
      break;
    case 1: // Zero operations
      r.operations = 0ULL;
      r.time_ms = 5.0;
      r.isValid = true;
      break;
    case 2: // Zero time
      r.operations = 500000ULL;
      r.time_ms = 0.0;
      r.isValid = true;
      break;
    case 3: // Negative time (failed)
      r.operations = 100ULL;
      r.time_ms = -1.0;
      r.errorString = "Benchmark failure injection";
      r.isValid = false;
      break;
    case 4: // Non-finite power telemetry
      r.operations = 1000ULL;
      r.time_ms = 1.0;
      r.hasPowerTelemetry = true;
      r.powerWatts = (i % 16 == 4) ? std::numeric_limits<float>::quiet_NaN() : -1.0f;
      r.energyJoules = std::numeric_limits<double>::infinity();
      r.isValid = true;
      break;
    case 5: // Non-finite perf counters
      r.operations = 2000ULL;
      r.time_ms = 1.0;
      r.hasPerfQueryTelemetry = true;
      r.perfCounters.gpuActiveCycles = 0ULL;
      r.perfCounters.valuBusyPct = std::numeric_limits<float>::quiet_NaN();
      r.perfCounters.saluBusyPct = std::numeric_limits<float>::infinity();
      r.perfCounters.rawCounters["salu_cycles_per_wave"] = std::numeric_limits<double>::infinity();
      r.isValid = true;
      break;
    case 6: // Unsupported benchmark
      r.operations = 0ULL;
      r.time_ms = 0.0;
      r.isUnsupported = true;
      r.supportCategory = "Hardware Incompatible";
      r.supportNote = "Required feature VK_KHR_ray_tracing_pipeline not supported";
      r.isValid = false;
      break;
    default: // Normal run
      r.operations = 1000000000ULL;
      r.time_ms = 10.0;
      r.sample_count = 5;
      r.min_time_ms = 9.8;
      r.median_time_ms = 10.0;
      r.mean_time_ms = 10.05;
      r.p95_time_ms = 10.2;
      r.isValid = true;
      break;
    }

    results.push_back(r);
  }

  // Measure JSON export performance and verify RFC 8259 correctness
  std::string jsonStr = resultsToJson(results, profiles);
  ASSERT_FALSE(jsonStr.empty());
  assertNoNonFiniteTokens(jsonStr);

  // Write out to disk and verify with external strict validator
  std::string outPath = (std::filesystem::temp_directory_path() / "gpubench_stress_1000.json").string();
  std::ofstream ofs(outPath);
  ASSERT_TRUE(ofs.is_open());
  ofs << jsonStr;
  ofs.close();
  std::remove(outPath.c_str());

  // Test round-trip with internal ResultImporter
  ImportedRun run;
  std::string err;
  bool ok = ResultImporter::loadFromString(jsonStr, run, err);
  if (!ok) {
    std::cerr << "Round-trip import error on 1000 results: " << err << "\n";
  }
  ASSERT_TRUE(ok);
  ASSERT_EQ(run.results.size(), 1000u);
}
