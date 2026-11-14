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
    if (config_idx == 1) {
      if (context && context->getBackend() == ComputeBackend::OpenCL) {
        return "Cooperative matrix / WMMA not supported in OpenCL API";
      }
      if (!info.cooperativeMatrixSupport) {
        return "extension VK_KHR_cooperative_matrix missing or ROCm WMMA not supported";
      }
    }
    return GetSupportNote(info, context);
  }
  SupportLimitation GetConfigSupportLimitation(uint32_t config_idx) const override {
    return GetSupportLimitation();
  }
  SupportLimitation GetConfigSupportLimitation(uint32_t config_idx,
                                               const DeviceInfo &info,
                                               IComputeContext *context = nullptr) const override {
    if (config_idx == 1) {
      if (context && context->getBackend() == ComputeBackend::OpenCL) return SupportLimitation::kApi;
      if (!info.cooperativeMatrixSupport) return SupportLimitation::kHardware;
    }
    return GetSupportLimitation(info, context);
  }
  bool IsConfigSupported(uint32_t config_idx) const override {
    if (config_idx == 0) return true;
    return matrixKernel != nullptr;
  }
  bool IsConfigSupported(uint32_t config_idx, const DeviceInfo &info,
                         IComputeContext *context = nullptr) const override {
    if (!IsSupported(info, context)) return false;
    if (config_idx == 0) return true;
    if (context && context->getBackend() == ComputeBackend::OpenCL) return false;
    return info.cooperativeMatrixSupport;
  }
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
    return "FP16";
  }
  int GetSortWeight() const override { return 30; }
  uint32_t GetNumConfigs() const override { return 2; }
  virtual uint32_t GetExpectedKernelCount() const override { return 2; }
  std::string GetConfigName(uint32_t config_idx) const override {
    return config_idx == 0 ? "Vector" : "Matrix";
  }
  const char *GetDescription() const override {
    return "Half-precision IEEE 754 binary16 throughput across packed f16vec2/f16vec4 VALU and Cooperative Matrix / WMMA / MFMA cores.";
  }
  double GetOperationalIntensity(uint32_t config_idx = 0) const override {
    return config_idx == 0 ? 1048576.0 : 4096.0;
  }

private:
  template <typename Tag> friend struct ::gpubench::test::TestBody;
  IComputeContext *context = nullptr;
  ComputeKernel vectorKernel = nullptr;
  ComputeKernel matrixKernel = nullptr;
  ComputeBuffer buffer = nullptr;
  ComputeBuffer matrixInputBuffer = nullptr;
  ComputeBuffer matrixBuffer = nullptr;
  uint32_t tileM = 16;
  uint32_t tileN = 16;
  uint32_t tileK = 16;
  uint32_t matrixBlockX = 32;
};
