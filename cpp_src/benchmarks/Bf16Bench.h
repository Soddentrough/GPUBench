#pragma once

#include "benchmarks/IBenchmark.h"

class Bf16Bench : public IBenchmark {
public:
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context) const override;
  std::string GetSupportNote() const override {
    return "Requires VK_KHR_shader_bfloat16 or ROCm BF16 WMMA support";
  }
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return "No support for native BFloat16 floating-point arithmetic in OpenCL API (extension cl_khr_bfloat16 missing)";
    }
    if (!info.bf16Support) {
      if (info.archName.find("RDNA 3") != std::string::npos ||
          info.archName.find("gfx11") != std::string::npos) {
        return "RDNA 3 lacks native BF16 VALU dot-product hardware (no native BF16 support on Vulkan)";
      }
      return "shaderBfloat16 / BF16 hardware support bit not set";
    }
    return "";
  }
  SupportLimitation GetSupportLimitation() const override {
    return SupportLimitation::kHardware;
  }
  SupportLimitation GetSupportLimitation(const DeviceInfo &info,
                                         IComputeContext *context = nullptr) const override {
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return SupportLimitation::kApi;
    }
    if (!info.bf16Support) {
      return SupportLimitation::kHardware;
    }
    return SupportLimitation::kToolchain;
  }
  bool IsConfigSupported(uint32_t config_idx) const override {
    if (config_idx == 0)
      return vectorKernel != nullptr;
    if (config_idx == 1)
      return matrixKernel != nullptr;
    return false;
  }
  bool IsConfigSupported(uint32_t config_idx, const DeviceInfo &info,
                         IComputeContext *context = nullptr) const override {
    if (!IsSupported(info, context))
      return false;
    if (config_idx == 0) {
      if (!info.shaderBfloat16DotProductSupport) {
        return false;
      }
      if (this->context != nullptr &&
          (context == nullptr || context == this->context)) {
        return vectorKernel != nullptr;
      }
      return true;
    }
    if (config_idx == 1) {
      if (!info.cooperativeMatrixSupport)
        return false;
      if (context && context->getBackend() == ComputeBackend::Vulkan &&
          !info.shaderBfloat16CoopMatrixSupport) {
        return false;
      }
      if (this->context != nullptr &&
          (context == nullptr || context == this->context)) {
        return matrixKernel != nullptr;
      }
      return true;
    }
    return false;
  }
  std::string GetConfigSupportNote(uint32_t config_idx,
                                   const DeviceInfo &info,
                                   IComputeContext *context = nullptr) const override {
    if (config_idx == 0 && !info.shaderBfloat16DotProductSupport) {
      return "Packed BFloat16 vector dot2 instruction not supported by device/driver";
    }
    if (config_idx == 1) {
      if (!info.cooperativeMatrixSupport) {
        return "VK_KHR_cooperative_matrix or ROCm WMMA/MFMA not supported";
      }
      if (context && context->getBackend() == ComputeBackend::Vulkan &&
          !info.shaderBfloat16CoopMatrixSupport) {
        return "shaderBFloat16CooperativeMatrix not supported by active Vulkan driver";
      }
    }
    return GetSupportNote(info, context);
  }
  SupportLimitation GetConfigSupportLimitation(uint32_t config_idx,
                                               const DeviceInfo &info,
                                               IComputeContext *context = nullptr) const override {
    if (config_idx == 0 && !info.shaderBfloat16DotProductSupport) {
      return (context && context->getBackend() == ComputeBackend::Vulkan)
                 ? SupportLimitation::kApi
                 : SupportLimitation::kHardware;
    }
    if (config_idx == 1) {
      if (!info.cooperativeMatrixSupport) {
        return SupportLimitation::kHardware;
      }
      if (context && context->getBackend() == ComputeBackend::Vulkan &&
          !info.shaderBfloat16CoopMatrixSupport) {
        return SupportLimitation::kApi;
      }
    }
    return GetSupportLimitation(info, context);
  }
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx) override;
  void Teardown() override;

  bool SupportsWorkScaling(uint32_t config_idx = 0) const override { (void)config_idx; return true; }
  BenchmarkResult GetResult(uint32_t config_idx) const override;
  uint32_t GetNumConfigs() const override { return 2; }
  std::string GetConfigName(uint32_t config_idx) const override {
    return config_idx == 0 ? "Vector" : "Matrix";
  }
  const char *GetName() const override { return "BF16"; }
  std::vector<std::string> GetAliases() const override {
    return {"bf16", "bfloat16"};
  }
  const char *GetComponent(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return "Compute";
  }
  const char *GetSubCategory(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return "BF16";
  }
  const char *GetDescription() const override {
    return "Brain Floating-Point (bfloat16) packed dot-product VALU and Cooperative Matrix / WMMA / MFMA throughput.";
  }
  double GetOperationalIntensity(uint32_t config_idx = 0) const override {
    return config_idx == 0 ? 524288.0 : 4096.0;
  }
  int GetSortWeight() const override { return 35; }
  uint32_t GetExpectedKernelCount() const override { return 2; }

  bool ValidateResults(uint32_t config_idx = 0) const override;
  bool IsEmulated(uint32_t config_idx = 0) const override {
    return config_idx == 0 ? vectorEmulated : false;
  }
  std::string GetConfigCaveat(uint32_t config_idx = 0) const override {
    if (config_idx == 0 && vectorEmulated) {
      return "Driver lowered BF16 vector arithmetic to FP32 VALU instructions (no native BF16 vector instructions in compiled ISA)";
    }
    return "";
  }

private:
  IComputeContext *context = nullptr;
  mutable IComputeContext *lastCheckedContext = nullptr;
  ComputeKernel vectorKernel = nullptr;
  ComputeKernel matrixKernel = nullptr;
  ComputeBuffer buffer = nullptr;
  ComputeBuffer matrixInputBuffer = nullptr;
  ComputeBuffer matrixBuffer = nullptr;
  bool vectorEmulated = false;
  uint32_t tileM = 16;
  uint32_t tileN = 16;
  uint32_t tileK = 16;
  uint32_t matrixBlockX = 32;

  template <typename Tag> friend struct ::gpubench::test::TestBody;
};
