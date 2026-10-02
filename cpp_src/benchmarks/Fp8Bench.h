#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"

class Fp8Bench : public IBenchmark {
public:
  const char *GetName() const override { return "FP8"; }
  std::vector<std::string> GetAliases() const override {
    return {"fp8", "f8"};
  }
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  SupportLimitation GetSupportLimitation() const override {
    return SupportLimitation::kHardware;
  }
  SupportLimitation GetSupportLimitation(const DeviceInfo &info,
                                         IComputeContext *context = nullptr) const override {
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return SupportLimitation::kApi;
    }
    if (!info.fp8Support) {
      return SupportLimitation::kHardware;
    }
    if (context && context->getBackend() == ComputeBackend::Vulkan && !info.cooperativeMatrixSupport) {
      return SupportLimitation::kHardware;
    }
    return SupportLimitation::kNone;
  }
  std::string GetSupportNote() const override {
    return "";
  }
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return "No support for 8-bit floating point in OpenCL API (extension cl_khr_fp8 missing)";
    }
    if (!info.fp8Support) {
      return "shaderFloat8 hardware bit not set (no native FP8 support on GPU)";
    }
    if (context && context->getBackend() == ComputeBackend::Vulkan && !info.cooperativeMatrixSupport) {
      return "VK_KHR_cooperative_matrix not supported on device";
    }
    return "";
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
    return "FP8";
  }
  int GetSortWeight() const override { return 40; }

  uint32_t GetNumConfigs() const override { return 1; }
  virtual uint32_t GetExpectedKernelCount() const override { return 1; }
  std::string GetConfigName(uint32_t config_idx) const override {
    (void)config_idx;
    return "Matrix";
  }
  const char *GetMetric() const override { return "TFLOPS"; }

  bool IsEmulated(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return !is_native_matrix;
  }

private:
  IComputeContext *context = nullptr;
  mutable IComputeContext *lastCheckedContext = nullptr;
  ComputeKernel matrixKernel = nullptr;
  ComputeBuffer buffer = nullptr;
  bool is_native_matrix = false;
  mutable std::string name = "FP8";
};
