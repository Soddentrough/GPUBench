#pragma once

#include "benchmarks/IBenchmark.h"
#include <string>
#include <vector>

class SysMemLatencyBench : public IBenchmark {
public:
  SysMemLatencyBench();
  virtual ~SysMemLatencyBench();

  const char *GetName() const override;
  std::vector<std::string> GetAliases() const override {
    return {"sysmem_latency", "ram_latency", "sl"};
  }
  const char *GetMetric() const override;
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;

  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  uint32_t GetNumConfigs() const override;
  std::string GetConfigName(uint32_t config_idx) const override;
  const char *GetComponent(uint32_t config_idx = 0) const override {
    return "Memory";
  }
  const char *GetSubCategory(uint32_t config_idx = 0) const override {
    return "Latency";
  }

  bool IsDeviceDependent() const override { return false; }
  bool IsEmulated() const override { return false; }

private:
  void *buffer = nullptr;
  size_t bufferSize = 0;
  double lastRunTimeMs = 0.0;
  uint64_t lastRunOps = 0;
};
