#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"

class Fp4Bench : public IBenchmark {
public:
  const char *GetName() const override { return "FP4"; }
  std::vector<std::string> GetAliases() const override {
    return {"fp4", "f4"};
  }
  const char *GetMetric() const override { return "TOPS"; }

  uint32_t GetNumConfigs() const override;
  std::string GetConfigName(uint32_t config_idx) const override;

  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  std::string GetSupportNote() const override {
    return "shaderFloat4 hardware bit not set (no native FP4 compute units; CDNA4/gfx950+ only)";
  }
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    (void)info;
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return "No support for 4-bit floating point in OpenCL API";
    }
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      return "shaderFloat4 hardware bit not set (no native FP4 units on RDNA4; CDNA4/gfx950+ only)";
    }
    return "shaderFloat4 hardware bit not set (no native FP4 units on RDNA4; CDNA4/gfx950+ only)";
  }
  SupportLimitation GetSupportLimitation() const override {
    return SupportLimitation::kHardware;
  }
  SupportLimitation GetSupportLimitation(const DeviceInfo &info,
                                         IComputeContext *context = nullptr) const override {
    (void)info;
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return SupportLimitation::kApi;
    }
    return SupportLimitation::kHardware;
  }
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  const char *GetComponent(uint32_t config_idx = 0) const override {
    return "Compute";
  }
  const char *GetSubCategory(uint32_t config_idx = 0) const override {
    return "FP4";
  }
  int GetSortWeight() const override { return 60; }
  bool IsEmulated(uint32_t config_idx = 0) const override { return is_emulated; }

private:
  IComputeContext *context = nullptr;
  ComputeKernel kernel = nullptr;
  ComputeBuffer buffer = nullptr;
  bool is_emulated = true;
  mutable std::string name = "FP4";
};
