#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"
#include <cstdint>
#include <string>
#include <vector>

struct LdsConflictConfig {
  std::string name;
  uint32_t stride;
  std::string hint;
};

class LdsBankConflictBench : public IBenchmark {
public:
  LdsBankConflictBench();
  ~LdsBankConflictBench() override;

  const char *GetName() const override { return "LDS Bank Conflicts"; }
  std::vector<std::string> GetAliases() const override {
    return {"lds", "lds-bank-conflicts", "lds_bank_conflicts", "ldsbankconflict", "ldsconflicts"};
  }
  const char *GetMetric() const override { return "TB/s"; }
  const char *GetMetric(uint32_t config_idx) const override { return "TB/s"; }
  bool IsSupported(const DeviceInfo &info, IComputeContext *context = nullptr) const override { return true; }

  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;

  const char *GetComponent(uint32_t config_idx = 0) const override { return "Compute"; }
  const char *GetSubCategory(uint32_t config_idx = 0) const override { return "LDS Bank Conflicts"; }
  int GetSortWeight() const override { return 26; }

  uint32_t GetNumConfigs() const override { return static_cast<uint32_t>(configs.size()); }
  uint32_t GetExpectedKernelCount() const override { return 1; }
  std::string GetConfigName(uint32_t config_idx) const override;
  std::string GetConfigSupportNote(uint32_t config_idx) const override;

private:
  IComputeContext *context = nullptr;
  std::vector<LdsConflictConfig> configs;
  ComputeKernel kernel = nullptr;
  ComputeBuffer buffer = nullptr;

  static constexpr uint32_t kWorkgroups = 2048;
  static constexpr uint32_t kThreadsPerWg = 32;
  static constexpr uint32_t kLoadsPerIter = 8;
  static constexpr uint32_t kIterations = 50000;
  static constexpr uint64_t kTotalBytes = static_cast<uint64_t>(kWorkgroups) *
                                          kThreadsPerWg * kLoadsPerIter *
                                          kIterations * sizeof(uint32_t);
};
