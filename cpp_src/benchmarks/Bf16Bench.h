#pragma once

#include "benchmarks/IBenchmark.h"

class Bf16Bench : public IBenchmark {
public:
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context) const override;
  std::string GetSupportNote() const override {
    if (lastCheckedContext && lastCheckedContext->getBackend() == ComputeBackend::OpenCL) {
      return "OpenCL standard does not define native BFloat16 floating-point arithmetic (cl_khr_bfloat16 missing)";
    }
    return "HIP toolchain clang emulates bf16 via FP32 (no native "
           "hip_bfloat162/__hfma2 in headers, scalar-unit codegen); "
           "Vulkan measures the native rate";
  }
  SupportLimitation GetSupportLimitation() const override {
    if (lastCheckedContext && lastCheckedContext->getBackend() == ComputeBackend::OpenCL) {
      return SupportLimitation::kApi;
    }
    return SupportLimitation::kToolchain;
  }
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx) override;
  void Teardown() override;

  BenchmarkResult GetResult(uint32_t config_idx) const override;
  uint32_t GetNumConfigs() const override;
  std::string GetConfigName(uint32_t config_idx) const override;
  const char *GetName() const override { return "BF16"; }
  std::vector<std::string> GetAliases() const override {
    return {"bf16", "bfloat16"};
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
  mutable IComputeContext *lastCheckedContext = nullptr;
  ComputeKernel vectorKernel = nullptr;
  ComputeKernel matrixKernel = nullptr;
  ComputeBuffer buffer = nullptr;
};
