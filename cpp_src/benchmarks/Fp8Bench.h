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
    return SupportLimitation::kToolchain;
  }
  SupportLimitation GetSupportLimitation(const DeviceInfo &info,
                                         IComputeContext *context = nullptr) const override {
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return SupportLimitation::kApi;
    }
    if (!info.fp8Support) {
      return SupportLimitation::kHardware;
    }
    return SupportLimitation::kToolchain;
  }
  std::string GetSupportNote() const override {
    return "extension GL_EXT_shader_explicit_arithmetic_types_float8 missing in glslang toolchain";
  }
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return "No support for 8-bit floating point in OpenCL API (extension cl_khr_fp8 missing)";
    }
    if (!info.fp8Support) {
      return "shaderFloat8 hardware bit not set (no native FP8 support on GPU)";
    }
    if (context && context->getBackend() == ComputeBackend::Vulkan) {
      return "extension GL_EXT_shader_explicit_arithmetic_types_float8 missing in glslang toolchain";
    }
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      return "ROCm HIP compiler lacks native FP8 vector arithmetic types on gfx1201";
    }
    return "extension GL_EXT_shader_explicit_arithmetic_types_float8 missing in glslang toolchain";
  }
  std::string GetConfigSupportNote(uint32_t config_idx,
                                   const DeviceInfo &info,
                                   IComputeContext *context = nullptr) const override {
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      if (config_idx == 0) {
        return "ROCm HIP compiler lacks native FP8 vector arithmetic types on gfx1201";
      }
    }
    return GetSupportNote(info, context);
  }
  SupportLimitation GetConfigSupportLimitation(uint32_t config_idx,
                                               const DeviceInfo &info,
                                               IComputeContext *context = nullptr) const override {
    return GetSupportLimitation(info, context);
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

  uint32_t GetNumConfigs() const override;
  virtual uint32_t GetExpectedKernelCount() const override { return 2; }
  std::string GetConfigName(uint32_t config_idx) const override;
  const char *GetMetric() const override { return "TFLOPS"; }

  bool IsEmulated(uint32_t config_idx = 0) const override {
    if (config_idx == 0) return is_emulated_vector;
    return !is_native_matrix;
  }

private:
  IComputeContext *context = nullptr;
  mutable IComputeContext *lastCheckedContext = nullptr;
  ComputeKernel vectorKernel = nullptr;
  ComputeKernel matrixKernel = nullptr;
  ComputeBuffer buffer = nullptr;
  bool is_emulated_vector = false;
  bool is_native_vector = false;
  bool is_native_matrix = false;
  mutable std::string name = "FP8";
};
