#pragma once

#include "IBenchmark.h"
#include <cstdint>
#include <string>
#include <vector>

class CoopVectorBench : public IBenchmark {
public:
  const char *GetName() const override {
    return "Cooperative Vector (Neural Shading)";
  }
  std::vector<std::string> GetAliases() const override {
    return {"coopvector", "neural", "mlp"};
  }
  const char *GetMetric() const override { return "TFLOPS"; }
  const char *GetMetric(uint32_t config_idx) const override {
    return (config_idx == 2u || config_idx == 3u) ? "TOPS" : "TFLOPS";
  }
  const char *GetComponent(uint32_t /*config_idx*/ = 0) const override {
    return "Compute";
  }
  const char *GetSubCategory(uint32_t /*config_idx*/ = 0) const override {
    return "Neural Shading";
  }
  int GetSortWeight() const override { return 109; }
  uint32_t GetExpectedKernelCount() const override { return 4; }

  uint32_t GetNumConfigs() const override { return 4; }
  std::string GetConfigName(uint32_t config_idx) const override;
  const char *GetDescription() const override { return GetDescription(0); }
  const char *GetDescription(uint32_t config_idx) const override;
  int32_t GetBaselineConfigIndex(uint32_t config_idx) const override {
    if (config_idx == 1u) return 0; // FP16 CoopVector vs FP16 Shader Baseline
    if (config_idx == 3u) return 2; // INT8 CoopVector vs INT8 Shader Baseline
    return -1;
  }
  double GetOperationalIntensity(uint32_t config_idx = 0) const override {
    return (config_idx >= 2u) ? 512.0 : 256.0;
  }

  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  bool IsConfigSupported(uint32_t config_idx, const DeviceInfo &info,
                         IComputeContext *context = nullptr) const override;
  SupportLimitation GetSupportLimitation(
      const DeviceInfo &info,
      IComputeContext *context = nullptr) const override;
  SupportLimitation GetConfigSupportLimitation(
      uint32_t config_idx, const DeviceInfo &info,
      IComputeContext *context = nullptr) const override;
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override;
  std::string GetConfigSupportNote(
      uint32_t config_idx, const DeviceInfo &info,
      IComputeContext *context = nullptr) const override;

  void Setup(IComputeContext &context,
             const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;

private:
  static constexpr uint32_t kNumThreads = 262144u;
  static constexpr uint32_t kIterations = 256u;

  IComputeContext *m_context = nullptr;
  ComputeKernel m_fp16BaselineKernel = nullptr;
  ComputeKernel m_fp16Kernel = nullptr;
  ComputeKernel m_int8BaselineKernel = nullptr;
  ComputeKernel m_int8Kernel = nullptr;
  ComputeBuffer m_fp16Weights = nullptr;
  ComputeBuffer m_fp16Biases = nullptr;
  ComputeBuffer m_fp16BaselineOut = nullptr;
  ComputeBuffer m_fp16Out = nullptr;
  ComputeBuffer m_int8Weights = nullptr;
  ComputeBuffer m_int8Biases = nullptr;
  ComputeBuffer m_int8BaselineOut = nullptr;
  ComputeBuffer m_int8Out = nullptr;
};
