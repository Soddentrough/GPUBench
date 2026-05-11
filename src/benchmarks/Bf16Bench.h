#pragma once

#include "benchmarks/IBenchmark.h"

class Bf16Bench : public IBenchmark {
public:
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context) const override;
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx) override;
  void Teardown() override;

  BenchmarkResult GetResult(uint32_t config_idx) const override;
  uint32_t GetNumConfigs() const override;
  std::string GetConfigName(uint32_t config_idx) const override;
  const char *GetName() const override { return "BF16"; }
  std::vector<std::string> GetAliases() const override {
    return {"bf16", "performance"};
  }
  const char *GetComponent(uint32_t config_idx = 0) const override {
    return "Compute";
  }
  const char *GetSubCategory(uint32_t config_idx = 0) const override {
    return "BF16";
  }
  int GetSortWeight() const override { return 35; }
  uint32_t GetExpectedKernelCount() const override { return 2; }

private:
  IComputeContext *context = nullptr;
  ComputeKernel vectorKernel = nullptr;
  ComputeKernel matrixKernel = nullptr;
  ComputeBuffer buffer = nullptr;
};
