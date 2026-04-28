#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"

class Fp4Bench : public IBenchmark {
public:
  const char *GetName() const override { return "FP4"; }
  std::vector<std::string> GetAliases() const override {
    return {"f4", "performance"};
  }
  const char *GetMetric() const override { return "TOPS"; }

  uint32_t GetNumConfigs() const override;
  std::string GetConfigName(uint32_t config_idx) const override;

  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  const char *GetComponent(uint32_t config_idx = 0) const override {
    return "Compute";
  }
  const char *GetSubCategory(uint32_t config_idx = 0) const override {
    return "FP4";
  }
  int GetSortWeight() const override { return 60; }
  bool IsEmulated() const override { return is_emulated; }

private:
  IComputeContext *context = nullptr;
  ComputeKernel kernel = nullptr;
  ComputeBuffer buffer = nullptr;
  bool is_emulated = true;
  mutable std::string name = "FP4";
};
