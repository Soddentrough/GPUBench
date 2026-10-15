#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"

class Int4Bench : public IBenchmark {
public:
  const char *GetName() const override { return "INT4"; }
  std::vector<std::string> GetAliases() const override {
    return {"int4"};
  }
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  std::string GetSupportNote() const override {
    return "No support for 4-bit integer types in Vulkan API (no 4-bit component type in VK_KHR_cooperative_matrix or SPIR-V)";
  }
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return "No support for 4-bit integer types in OpenCL API";
    }
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      if (!info.int4Support) {
        return "INT4 WMMA hardware support not available on this GPU";
      }
      return "";
    }
    return "No support for 4-bit integer types in Vulkan API (no 4-bit component type in VK_KHR_cooperative_matrix or SPIR-V)";
  }
  SupportLimitation GetSupportLimitation() const override {
    return SupportLimitation::kApi;
  }
  SupportLimitation GetSupportLimitation(const DeviceInfo &info,
                                         IComputeContext *context = nullptr) const override {
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      if (!info.int4Support) {
        return SupportLimitation::kHardware;
      }
      return SupportLimitation::kNone;
    }
    return SupportLimitation::kApi;
  }
  std::string GetConfigSupportNote(uint32_t config_idx,
                                   const DeviceInfo &info,
                                   IComputeContext *context = nullptr) const override {
    (void)config_idx;
    return GetSupportNote(info, context);
  }
  SupportLimitation GetConfigSupportLimitation(uint32_t config_idx,
                                               const DeviceInfo &info,
                                               IComputeContext *context = nullptr) const override {
    (void)config_idx;
    return GetSupportLimitation(info, context);
  }
  bool IsConfigSupported(uint32_t config_idx) const override {
    (void)config_idx;
    return matrixKernel != nullptr;
  }
  bool IsConfigSupported(uint32_t config_idx, const DeviceInfo &info,
                         IComputeContext *context = nullptr) const override {
    (void)config_idx;
    return IsSupported(info, context);
  }
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  const char *GetComponent(uint32_t config_idx = 0) const override {
    return "Compute";
  }
  const char *GetSubCategory(uint32_t config_idx = 0) const override {
    return "INT4";
  }
  int GetSortWeight() const override { return 80; }

  uint32_t GetNumConfigs() const override { return 1; }
  virtual uint32_t GetExpectedKernelCount() const override { return 1; }
  std::string GetConfigName(uint32_t config_idx) const override {
    (void)config_idx;
    return "Matrix";
  }
  const char *GetMetric() const override { return "TOPS"; }

  bool IsEmulated(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return !is_native_matrix;
  }

  bool ValidateResults(uint32_t config_idx = 0) const override;

private:
  IComputeContext *context = nullptr;
  ComputeKernel matrixKernel = nullptr;
  ComputeBuffer buffer = nullptr;
  bool is_native_matrix = false;
};
