#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"

class Fp4Bench : public IBenchmark {
public:
  const char *GetName() const override { return "FP4"; }
  std::vector<std::string> GetAliases() const override {
    return {"fp4", "f4", "mxfp4"};
  }
  const char *GetMetric() const override { return "TFLOPS"; }

  uint32_t GetNumConfigs() const override;
  uint32_t GetExpectedKernelCount() const override { return 1; }
  std::string GetConfigName(uint32_t config_idx) const override;

  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  std::string GetSupportNote() const override {
    return "No native 16x16x16 Float4E2M1EXT cooperative matrix or CDNA 4 gfx950 MXFP4 MFMA support (requires VK_EXT_shader_ocp_microscaling_types::shaderFloat4 + VK_KHR_cooperative_matrix or AMD CDNA 4 gfx950)";
  }
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return "No support for 4-bit microscaling floating point in OpenCL API";
    }
    if (!info.fp4Support) {
      return "No native OCP MXFP4 (Float4E2M1EXT) 16x16x16 matrix units exposed on GPU";
    }
    if (context && context->getBackend() == ComputeBackend::Vulkan &&
        !info.cooperativeMatrixSupport) {
      return "VK_KHR_cooperative_matrix not supported on device";
    }
    return "";
  }
  std::string GetConfigSupportNote(uint32_t config_idx, const DeviceInfo &info,
                                   IComputeContext *context = nullptr) const override {
    (void)config_idx;
    return GetSupportNote(info, context);
  }
  SupportLimitation GetSupportLimitation() const override {
    return SupportLimitation::kHardware;
  }
  SupportLimitation GetSupportLimitation(const DeviceInfo &info,
                                         IComputeContext *context = nullptr) const override {
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return SupportLimitation::kApi;
    }
    if (!info.fp4Support) {
      return SupportLimitation::kHardware;
    }
    if (context && context->getBackend() == ComputeBackend::Vulkan &&
        !info.cooperativeMatrixSupport) {
      return SupportLimitation::kHardware;
    }
    return SupportLimitation::kNone;
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
  bool ValidateResults(uint32_t config_idx = 0) const override;
  const char *GetComponent(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return "Compute";
  }
  const char *GetSubCategory(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return "FP4";
  }
  int GetSortWeight() const override { return 60; }
  const char *GetDescription() const override {
    return "OCP Microscaling 4-bit floating-point (MXFP4 E2M1) Cooperative Matrix / MFMA throughput.";
  }
  double GetOperationalIntensity(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return 16384.0;
  }
  bool IsEmulated(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return false;
  }

private:
  template <typename Tag> friend struct ::gpubench::test::TestBody;
  IComputeContext *context = nullptr;
  ComputeKernel matrixKernel = nullptr;
  ComputeBuffer buffer = nullptr;
  ComputeBuffer outputBuffer = nullptr;
  uint32_t matrixBlockX = 32;
};
