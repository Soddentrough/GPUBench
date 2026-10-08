#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"
#include <cstdint>
#include <string>
#include <vector>

struct DualIssueConfig {
  std::string name;
  std::string vulkanShader;
  std::string kernelName;
  uint64_t opsPerThread;
  std::string metric;
  std::string hint;
};

class DualIssueBench : public IBenchmark {
public:
  DualIssueBench();
  ~DualIssueBench() override = default;

  const char *GetName() const override { return "ILP & Dual-Issue"; }
  std::vector<std::string> GetAliases() const override {
    return {"dualissue", "dual-issue", "ilp", "concurrency", "vopd", "dual"};
  }
  const char *GetMetric() const override { return "TFLOPS"; }
  const char *GetMetric(uint32_t config_idx) const override;
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;

  const char *GetComponent(uint32_t /*config_idx*/ = 0) const override {
    return "Compute";
  }
  const char *GetSubCategory(uint32_t /*config_idx*/ = 0) const override {
    return "ILP & Concurrency";
  }
  int GetSortWeight() const override { return 25; }

  uint32_t GetNumConfigs() const override {
    return static_cast<uint32_t>(configs.size());
  }
  uint32_t GetExpectedKernelCount() const override {
    return static_cast<uint32_t>(configs.size());
  }
  std::string GetConfigName(uint32_t config_idx) const override;
  std::string GetConfigSupportNote(uint32_t config_idx) const override;

private:
  IComputeContext *context = nullptr;
  std::vector<DualIssueConfig> configs;
  std::vector<ComputeKernel> kernels;
  ComputeBuffer buffer = nullptr;
  uint32_t numElements = 0;
};
