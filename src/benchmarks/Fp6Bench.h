#pragma once

#include "benchmarks/IBenchmark.h"

class Fp6Bench : public IBenchmark {
public:
  const char *GetName() const override { return "Performance"; }
  bool IsSupported(const DeviceInfo &device,
                   IComputeContext *context = nullptr) const override;
  void Setup(IComputeContext &context, const std::string &build_dir) override;
  void Run(uint32_t config_idx = 0) override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  const char *GetComponent(uint32_t config_idx = 0) const override {
    return "Compute";
  }
  const char *GetSubCategory(uint32_t config_idx = 0) const override {
    return "FP6";
  }
  void Teardown() override;
  bool IsEmulated() const override { return is_emulated; }

private:
  IComputeContext *context = nullptr;
  ComputeKernel kernel = nullptr;
  ComputeBuffer buffer = nullptr;
  uint64_t operations = 0;
  bool is_emulated = true;
  mutable std::string name = "FP6";
};
