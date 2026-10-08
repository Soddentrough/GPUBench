#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct ResultData {
  std::string backendName;
  std::string deviceName;
  std::string benchmarkName;
  std::string component;   // e.g., "Compute", "Memory"
  std::string subcategory; // e.g., "Bandwidth", "Latency", "FP32"
  std::string metric;
  uint64_t operations;
  double time_ms;
  // Multi-sample statistical distribution metrics (Optimization O-3)
  double min_time_ms = 0.0;
  double median_time_ms = 0.0;
  double mean_time_ms = 0.0;
  double p95_time_ms = 0.0;
  uint32_t sample_count = 0;
  std::vector<double> sample_durations_ms;
  bool isEmulated;
  // True when the benchmark was selected but is not supported on this
  // device/backend (e.g. missing hardware capability). Such entries carry
  // no measurement and are displayed as "UNSUPPORTED" in the human report.
  bool isUnsupported = false;
  // Human-readable explanation when isUnsupported is true (may be empty).
  std::string supportNote;
  // Limitation category when isUnsupported is true: "hardware", "api",
  // "toolchain", or "" (see IBenchmark::SupportLimitation).
  std::string supportCategory;
  uint32_t maxWorkGroupSize;
  uint32_t deviceIndex;
  uint32_t configIndex;
  uint32_t vendorId = 0;
  uint32_t deviceId = 0;
  int sortWeight;
  uint32_t width = 0;
  uint32_t height = 0;
  std::string errorString;
  bool isValid = true;
  int32_t baselineConfigIndex = -1;
  // Compiler / Kernel Resource Telemetry (Optimization O-4)
  bool hasRegisterTelemetry = false;
  uint32_t vgprCount = 0;
  uint32_t sgprCount = 0;
  uint32_t ldsSizeBytes = 0;
  uint32_t scratchSizeBytes = 0;
  uint32_t codeSizeBytes = 0;
  uint32_t maxWavesPerSimd = 0;
  std::string compilerTarget;
  // Power & Energy Efficiency Telemetry (Optimization O-3 / O-8)
  bool hasPowerTelemetry = false;
  float powerWatts = 0.0f;
  double energyJoules = 0.0;
  double joulesPerUnit = 0.0;
  double unitPerWatt = 0.0;
  std::string efficiencyUnit;
};

struct ImportedRun;

class ResultFormatter {
public:
  ResultFormatter();
  ~ResultFormatter();

  void addResult(const ResultData &result);
  void print();
  const std::vector<ResultData>& getResults() const { return results; }

  static void printComparison(const ImportedRun &runA, const ImportedRun &runB);
  static void printComparison(const std::vector<ImportedRun> &runs);

  static std::string formatNumber(uint64_t n);
  static std::string formatDouble(double value, int precision);

private:
  std::vector<ResultData> results;
};

double computeResultValue(const ResultData &r);
std::string cleanWorkloadName(const std::string &rawName, const std::string &subcat);
std::string extractSceneName(const std::string &rawName);
std::string getDefaultJsonFilename();
std::string resultsToJson(const std::vector<ResultData> &results);
