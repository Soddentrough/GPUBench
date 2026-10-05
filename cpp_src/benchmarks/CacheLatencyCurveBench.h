#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"
#include <cstdint>
#include <string>
#include <vector>

struct CacheCurveConfig {
  std::string name;
  uint64_t workingSetBytes;
  std::string cacheLevelHint;
  uint32_t startWordIndex;
};

class CacheLatencyCurveBench : public IBenchmark {
public:
  CacheLatencyCurveBench();
  ~CacheLatencyCurveBench() override;

  const char *GetName() const override { return "Cache Latency Curve"; }
  std::vector<std::string> GetAliases() const override {
    return {"cache-curve", "cachelatencycurve", "cache_latency_curve", "cache-latency-curve", "cachecurve"};
  }
  const char *GetMetric() const override { return "ns"; }
  const char *GetMetric(uint32_t config_idx) const override { return "ns"; }
  bool IsSupported(const DeviceInfo &info, IComputeContext *context = nullptr) const override { return true; }

  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;

  const char *GetComponent(uint32_t config_idx = 0) const override { return "Memory"; }
  const char *GetSubCategory(uint32_t config_idx = 0) const override { return "Cache Latency Curve"; }
  int GetSortWeight() const override { return 15; }

  uint32_t GetNumConfigs() const override { return static_cast<uint32_t>(configs.size()); }
  uint32_t GetExpectedKernelCount() const override { return 1; }
  std::string GetConfigName(uint32_t config_idx) const override;
  std::string GetConfigSupportNote(uint32_t config_idx) const override;

private:
  IComputeContext *context = nullptr;
  std::vector<CacheCurveConfig> configs;
  ComputeKernel kernel = nullptr;
  ComputeBuffer buffer = nullptr;
  uint64_t totalBufferBytes = 0;
  static constexpr uint32_t kIterations = 1000000;
};
