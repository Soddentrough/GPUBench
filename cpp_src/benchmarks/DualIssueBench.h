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
  bool SupportsWorkScaling(uint32_t config_idx = 0) const override { (void)config_idx; return true; }
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;

  const char *GetComponent(uint32_t /*config_idx*/ = 0) const override {
    return "Compute";
  }
  const char *GetSubCategory(uint32_t /*config_idx*/ = 0) const override {
    return "ILP & Concurrency";
  }
  using IBenchmark::GetDescription;
  const char *GetDescription() const override {
    return "Instruction-level parallelism, VOPD dual-issue, and FP32 + INT32 VALU co-execution efficiency.";
  }
  const char *GetDescription(uint32_t config_idx) const override;
  double GetOperationalIntensity(uint32_t config_idx = 0) const override {
    // Each dual_issue_*.comp shader reads 4 B + writes 4 B = 8 B per thread
    switch (config_idx) {
      case 0: // 524,288 ops / 8 B
      case 3:
        return 65536.0;
      case 1: // 1,048,576 ops / 8 B
      case 4:
        return 131072.0;
      case 2: // 2,097,152 ops / 8 B
      case 5:
        return 262144.0;
      case 6: // 3,145,728 ops / 8 B
        return 393216.0;
      default:
        return 65536.0;
    }
  }
  int GetSortWeight() const override { return 25; }

  uint32_t GetNumConfigs() const override {
    return static_cast<uint32_t>(configs.size());
  }
  uint32_t GetExpectedKernelCount() const override {
    return static_cast<uint32_t>(configs.size());
  }
  using IBenchmark::GetConfigName;
  std::string GetConfigName(uint32_t config_idx) const override;
  using IBenchmark::GetConfigSupportNote;
  std::string GetConfigSupportNote(uint32_t config_idx) const override;

private:
  template <typename Tag> friend struct ::gpubench::test::TestBody;
  IComputeContext *context = nullptr;
  std::vector<DualIssueConfig> configs;
  std::vector<ComputeKernel> kernels;
  std::vector<ComputeBuffer> buffers;
  ComputeBuffer buffer = nullptr;
  uint32_t numElements = 0;
};
