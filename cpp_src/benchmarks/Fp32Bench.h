#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"
#include <cstdint>

class Fp32Bench : public IBenchmark {
public:
  const char *GetName() const override { return "FP32"; }
  std::vector<std::string> GetAliases() const override {
    return {"f32", "sp"};
  }
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  bool SupportsWorkScaling(uint32_t config_idx = 0) const override { (void)config_idx; return true; }
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;
  const char *GetComponent(uint32_t config_idx = 0) const override {
    return "Compute";
  }
  const char *GetSubCategory(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return "FP32";
  }
  const char *GetDescription() const override {
    return "Single-precision IEEE 754 binary32 FMA throughput across GPU vector ALUs.";
  }
  double GetOperationalIntensity(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return 524288.0;
  }
  int GetSortWeight() const override { return 20; }

private:
  template <typename Tag> friend struct ::gpubench::test::TestBody;
  IComputeContext *context = nullptr;
  ComputeKernel kernel = nullptr;
  ComputeBuffer buffer = nullptr;
  uint32_t numElements = 0;
};
