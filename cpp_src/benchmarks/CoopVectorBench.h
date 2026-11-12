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
    return (config_idx == 1u) ? "TOPS" : "TFLOPS";
  }
  const char *GetComponent(uint32_t /*config_idx*/ = 0) const override {
    return "Compute";
  }
  const char *GetSubCategory(uint32_t /*config_idx*/ = 0) const override {
    return "Neural Shading";
  }
  int GetSortWeight() const override { return 109; }
  uint32_t GetExpectedKernelCount() const override { return 2; }

  uint32_t GetNumConfigs() const override { return 2; }
  std::string GetConfigName(uint32_t config_idx) const override;
  const char *GetDescription() const override { return GetDescription(0); }
  const char *GetDescription(uint32_t config_idx) const override;
  double GetOperationalIntensity(uint32_t config_idx = 0) const override {
    return (config_idx == 0u) ? 256.0 : 512.0;
  }

  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  bool IsConfigSupported(uint32_t config_idx, const DeviceInfo &info,
                         IComputeContext *context = nullptr) const override;
  SupportLimitation GetSupportLimitation(
      const DeviceInfo &info,
      IComputeContext *context = nullptr) const override;
  std::string GetSupportNote(const DeviceInfo &info,
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
  ComputeKernel m_fp16Kernel = nullptr;
  ComputeKernel m_int8Kernel = nullptr;
  ComputeBuffer m_fp16Weights = nullptr;
  ComputeBuffer m_fp16Biases = nullptr;
  ComputeBuffer m_fp16Out = nullptr;
  ComputeBuffer m_int8Weights = nullptr;
  ComputeBuffer m_int8Biases = nullptr;
  ComputeBuffer m_int8Out = nullptr;
};
