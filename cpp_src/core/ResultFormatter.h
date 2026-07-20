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
  int sortWeight;
};

class ResultFormatter {
public:
  ResultFormatter();
  ~ResultFormatter();

  void addResult(const ResultData &result);
  void print();
  const std::vector<ResultData>& getResults() const { return results; }

private:
  std::string formatNumber(uint64_t n);
  std::string formatDouble(double value, int precision);
  std::vector<ResultData> results;
};
