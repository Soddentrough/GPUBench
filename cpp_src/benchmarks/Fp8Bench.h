#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"

class Fp8Bench : public IBenchmark {
public:
  const char *GetName() const override { return "FP8"; }
  std::vector<std::string> GetAliases() const override {
    return {"f8", "performance"};
  }
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  std::string GetSupportNote() const override {
    return "GPU supports FP8 natively, but no shader toolchain on any "
           "backend can emit it yet: glslang lacks FP8 GLSL for Vulkan "
           "(no GL_EXT_shader_explicit_arithmetic_types_float8) and the "
           "HIP FP8 path is emulated (disabled as inaccurate)";
  }
  SupportLimitation GetSupportLimitation() const override {
    return SupportLimitation::kToolchain;
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
  ComputeKernel vectorKernel = nullptr;
  ComputeKernel matrixKernel = nullptr;
  ComputeBuffer buffer = nullptr;
  bool is_emulated_vector = false;
  bool is_native_vector = false;
  bool is_native_matrix = false;
  mutable std::string name = "FP8";
};
