#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"
#include <cstdint>

class Fp64Bench : public IBenchmark {
public:
  const char *GetName() const override { return "FP64"; }
  std::vector<std::string> GetAliases() const override {
    return {"f64", "dp"};
  }
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  std::string GetSupportNote() const override {
    return "shaderFloat64 hardware feature not enabled (device does not support 64-bit floating point)";
  }
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    (void)info;
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      return "FP64 (double-precision) compute not supported on this ROCm device";
    }
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return "cl_khr_fp64 double-precision extension not supported on this OpenCL device";
    }
    return "shaderFloat64 hardware feature not enabled (device does not support 64-bit floating point)";
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
    (void)config_idx;
    return "FP64";
  }
  using IBenchmark::GetDescription;
  const char *GetDescription() const override {
    return "Double-precision IEEE 754 binary64 FMA throughput across hardware FP64 ALUs.";
  }
  using IBenchmark::GetExplanation;
  BenchmarkExplanation GetExplanation(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return {
      "Peak 64-bit double-precision (FP64) floating-point math throughput across GPU shader ALUs.",
      "Used by scientific simulation, CAD/CAE solvers, and double-precision physics/orbital math—rarely used by games (consumer gaming GPUs intentionally keep FP64 at 1:16 to 1:64 of FP32).",
      "Register-resident compute loop executing 2,048 FP64 FMA operations (4,096 FLOPs) per thread."
    };
  }
  double GetOperationalIntensity(uint32_t config_idx = 0) const override {
    (void)config_idx;
    // fp64.comp: 4096 FP64 FLOPs per thread / (8 B seed read + 8 B output write = 16 B) = 256.0 FLOP/B
    return 256.0;
  }
  bool RequiresInterIterationBarrier(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return true;
  }
  int GetSortWeight() const override { return 10; }

private:
  template <typename Tag> friend struct ::gpubench::test::TestBody;
  IComputeContext *context = nullptr;
  ComputeKernel kernel = nullptr;
  ComputeBuffer buffer = nullptr;
};
