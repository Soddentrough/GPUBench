#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"
#include <cstdint>

class Fp16Bench : public IBenchmark {
public:
  const char *GetName() const override { return "FP16"; }
  std::vector<std::string> GetAliases() const override {
    return {"f16", "half"};
  }
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  std::string GetSupportNote() const override {
    return "shaderFloat16 hardware bit not set (device does not support native 16-bit floating point)";
  }
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    (void)info;
    (void)context;
    return "shaderFloat16 hardware bit not set (device does not support native 16-bit floating point)";
  }
  SupportLimitation GetSupportLimitation() const override {
    return SupportLimitation::kHardware;
  }
  SupportLimitation GetSupportLimitation(const DeviceInfo &info,
                                         IComputeContext *context = nullptr) const override {
    (void)info;
    (void)context;
    return SupportLimitation::kHardware;
  }
  std::string GetConfigSupportNote(uint32_t config_idx,
                                   const DeviceInfo &info,
                                   IComputeContext *context = nullptr) const override {
    if (config_idx == 1 && !info.cooperativeMatrixSupport) {
      return "extension VK_KHR_cooperative_matrix missing or ROCm WMMA not supported";
    }
    return GetSupportNote(info, context);
  }
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  const char *GetComponent(uint32_t config_idx = 0) const override {
    return "Compute";
  }
  const char *GetSubCategory(uint32_t config_idx = 0) const override {
    return "FP16";
  }
  int GetSortWeight() const override { return 30; }
  uint32_t GetNumConfigs() const override;
  virtual uint32_t GetExpectedKernelCount() const override { return 2; }
  std::string GetConfigName(uint32_t config_idx) const override;

private:
  IComputeContext *context = nullptr;
  ComputeKernel vectorKernel = nullptr;
  ComputeKernel matrixKernel = nullptr;
  ComputeBuffer buffer = nullptr;
};
