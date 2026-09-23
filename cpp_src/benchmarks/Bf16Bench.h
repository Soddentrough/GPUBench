#pragma once

#include "benchmarks/IBenchmark.h"

class Bf16Bench : public IBenchmark {
public:
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context) const override;
  std::string GetSupportNote() const override {
    return "No native BFloat16 arithmetic types available in current Vulkan/GLSL or HIP toolchains (shaders would fall back to FP16)";
  }
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return "No support for native BFloat16 floating-point arithmetic in OpenCL API (extension cl_khr_bfloat16 missing)";
    }
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      return "HIP toolchain clang emulates bf16 via FP32 (no native hip_bfloat162/__hfma2 in headers)";
    }
    if (context && context->getBackend() == ComputeBackend::Vulkan) {
      return "No native BFloat16 arithmetic shader types available in glslang/Vulkan toolchain (shaders would fall back to FP16)";
    }
    if (!info.bf16Support) {
      return "shaderBfloat16 hardware bit not set";
    }
    return "No native BFloat16 arithmetic types available in current toolchain";
  }
  SupportLimitation GetSupportLimitation() const override {
    return SupportLimitation::kToolchain;
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
  bool IsConfigSupported(uint32_t config_idx) const override { return false; }
  bool IsConfigSupported(uint32_t config_idx, const DeviceInfo &info,
                         IComputeContext *context = nullptr) const override {
    (void)config_idx;
    return IsSupported(info, context);
  }
  std::string GetConfigSupportNote(uint32_t config_idx,
                                   const DeviceInfo &info,
                                   IComputeContext *context = nullptr) const override {
    if (config_idx == 1 && !info.cooperativeMatrixSupport) {
      return "VK_KHR_cooperative_matrix or ROCm WMMA not supported";
    }
    return GetSupportNote(info, context);
  }
  SupportLimitation GetConfigSupportLimitation(uint32_t config_idx,
                                               const DeviceInfo &info,
                                               IComputeContext *context = nullptr) const override {
    if (config_idx == 1 && !info.cooperativeMatrixSupport) {
      return SupportLimitation::kHardware;
    }
    return GetSupportLimitation(info, context);
  }
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx) override;
  void Teardown() override;

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
