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
    return {"sysmem_latency", "ram_latency", "sysmem_lat", "sl"};
  }
  const char *GetMetric() const override;
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;

  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;
  uint32_t GetNumConfigs() const override;
  std::string GetConfigName(uint32_t config_idx) const override;
  const char *GetDescription() const override { return GetDescription(0); }
  const char *GetDescription(uint32_t config_idx) const override;
  std::string GetConfigCaveat(uint32_t config_idx = 0) const override;
  const char *GetComponent(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return "Memory";
  }
  const char *GetSubCategory(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return "Latency";
  }
  int GetSortWeight() const override { return 500; }

  bool IsDeviceDependent() const override { return false; }
  bool IsEmulated(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return false;
  }

private:
  template <typename Tag> friend struct ::gpubench::test::TestBody;
  void *buffer = nullptr;
  size_t bufferSize = 0;
  uint32_t currentIndex = 0;
  double lastRunTimeMs = 0.0;
  uint64_t lastRunOps = 0;
  bool usedVirtualAlloc = false;
  bool usingLargePages = false;
  std::string pageCaveat;
};
