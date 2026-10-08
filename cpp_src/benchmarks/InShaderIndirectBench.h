#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"
#include <cstdint>
#include <string>
#include <vector>

enum class IndirectBenchMode {
  CPU_Direct,
  GPU_Indirect
};

struct IndirectBenchConfig {
  std::string name;
  IndirectBenchMode mode;
  uint32_t selectivity; // 0..100%
  std::string hint;
};

class InShaderIndirectBench : public IBenchmark {
public:
  InShaderIndirectBench();
  ~InShaderIndirectBench() override;

  const char *GetName() const override { return "In-Shader Indirect Synthesis"; }
  std::vector<std::string> GetAliases() const override {
    return {"indirect", "indirect-dispatch", "inshaderindirect", "in-shader-indirect", "indirect_synth", "command_synth"};
  }
  const char *GetMetric() const override { return "us"; }
  const char *GetMetric(uint32_t /*config_idx*/) const override { return "us"; }
  int32_t GetBaselineConfigIndex(uint32_t config_idx) const override {
    return (config_idx == 0) ? -1 : 0;
  }
  bool IsSupported(const DeviceInfo &info, IComputeContext *context = nullptr) const override {
    if (context && context->getBackend() != ComputeBackend::Vulkan) {
      return false;
    }
    return true;
  }
  std::string GetSupportNote() const override {
    return "Requires Vulkan 1.3+ indirect dispatch (vkCmdDispatchIndirect)";
  }

  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;

  const char *GetComponent(uint32_t config_idx = 0) const override { return "Compute"; }
  const char *GetSubCategory(uint32_t config_idx = 0) const override { return "Indirect Command Synthesis"; }
  int GetSortWeight() const override { return 28; }

  uint32_t GetNumConfigs() const override { return static_cast<uint32_t>(configs.size()); }
  uint32_t GetExpectedKernelCount() const override { return 3; }
  std::string GetConfigName(uint32_t config_idx) const override;
  std::string GetConfigSupportNote(uint32_t config_idx) const override;

private:
  IComputeContext *context = nullptr;
  std::vector<IndirectBenchConfig> configs;

  ComputeKernel kernelClassify = nullptr;
  ComputeKernel kernelIndirectWorker = nullptr;
  ComputeKernel kernelDirectWorker = nullptr;

  ComputeBuffer inputBuffer = nullptr;
  ComputeBuffer workListBuffer = nullptr;
  ComputeBuffer indirectBuffer = nullptr;
  ComputeBuffer outputBuffer = nullptr;

  static constexpr uint32_t kTotalItems = 4194304; // 4M items
  static constexpr uint32_t kWorkgroupSize = 64;
  static constexpr uint32_t kTotalWorkgroups = kTotalItems / kWorkgroupSize;
};
