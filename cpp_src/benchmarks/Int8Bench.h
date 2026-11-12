#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"
#include <string>
#include <vector>

class Int8Bench : public IBenchmark {
public:
  const char *GetName() const override { return "INT8"; }
  std::vector<std::string> GetAliases() const override {
    return {"int8", "int8b"};
  }
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  std::string GetSupportNote() const override {
    return "shaderInt8 hardware bit not set";
  }
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    (void)info;
    (void)context;
    return "shaderInt8 hardware bit not set";
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
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;
  const char *GetComponent(uint32_t config_idx = 0) const override {
    return "Compute";
  }
  const char *GetSubCategory(uint32_t config_idx = 0) const override {
    return "INT8";
  }
  int GetSortWeight() const override { return 70; }
  uint32_t GetNumConfigs() const override { return 2; }
  virtual uint32_t GetExpectedKernelCount() const override { return 2; }
  std::string GetConfigName(uint32_t config_idx) const override {
    return config_idx == 0 ? "Vector" : "Matrix";
  }
  const char *GetMetric() const override { return "TOPS"; }
  const char *GetDescription() const override {
    return "8-bit integer (INT8) packed dot-product VALU and Cooperative Matrix / WMMA / MFMA throughput.";
  }
  double GetOperationalIntensity(uint32_t config_idx = 0) const override {
    return config_idx == 0 ? 524288.0 : 8192.0;
  }

private:
  template <typename Tag> friend struct ::gpubench::test::TestBody;
  IComputeContext *context = nullptr;
  ComputeKernel vectorKernel = nullptr;
  ComputeKernel matrixKernel = nullptr;
  ComputeBuffer buffer = nullptr;
  ComputeBuffer matrixInputBuffer = nullptr;
  ComputeBuffer outputBuffer = nullptr;
  uint32_t tileM = 16;
  uint32_t tileN = 16;
  uint32_t tileK = 16;
  uint32_t matrixBlockX = 32;
};
