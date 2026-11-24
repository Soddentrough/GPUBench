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
      if (context && context->getBackend() == ComputeBackend::ROCm) {
        return "Native FP8 (E4M3) matrix hardware (RDNA 4 WMMA / CDNA 3+ MFMA) not supported on this GPU";
      }
      return "VK_EXT_shader_float8 (shaderFloat8) not supported on this GPU/driver";
    }
    if (context && context->getBackend() == ComputeBackend::Vulkan && !info.cooperativeMatrixSupport) {
      return "VK_KHR_cooperative_matrix not supported on device";
    }
    return "";
  }
  using IBenchmark::GetConfigSupportNote;
  std::string GetConfigSupportNote(uint32_t config_idx,
                                   const DeviceInfo &info,
                                   IComputeContext *context = nullptr) const override {
    (void)config_idx;
    return GetSupportNote(info, context);
  }
  using IBenchmark::GetConfigSupportLimitation;
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
  using IBenchmark::GetConfigName;
  std::string GetConfigName(uint32_t config_idx) const override {
    (void)config_idx;
    return "Matrix";
  }
  const char *GetMetric() const override { return "TFLOPS"; }
  using IBenchmark::GetDescription;
  const char *GetDescription() const override {
    return "8-bit floating-point (FP8 E4M3) Cooperative Matrix / WMMA / MFMA tensor throughput.";
  }
  const char *GetDescription(uint32_t config_idx, const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    (void)config_idx;
    (void)info;
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      return "8-bit floating-point (FP8 E4M3) hardware WMMA / MFMA matrix throughput.";
    }
    if (context && context->getBackend() == ComputeBackend::Vulkan) {
      return "8-bit floating-point (FP8 E4M3) Cooperative Matrix throughput via VK_EXT_shader_float8 and VK_KHR_cooperative_matrix.";
    }
    return GetDescription();
  }
  double GetOperationalIntensity(uint32_t config_idx = 0) const override {
    (void)config_idx;
    // coop_matrix_fp8.comp: 32768 * 8192 FLOPs / (256 B A + 256 B B + 1024 B C = 1536 B) = 174,762.67 FLOP/B
    return (32768.0 * 8192.0) / 1536.0;
  }

  bool IsEmulated(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return !is_native_matrix;
  }

  bool ValidateResults(uint32_t config_idx = 0) const override;

private:
  template <typename Tag> friend struct ::gpubench::test::TestBody;
  IComputeContext *context = nullptr;
  mutable IComputeContext *lastCheckedContext = nullptr;
  ComputeKernel matrixKernel = nullptr;
  ComputeBuffer buffer = nullptr;
  ComputeBuffer outputBuffer = nullptr;
  bool is_native_matrix = false;
  uint32_t tileM = 16;
  uint32_t tileN = 16;
  uint32_t tileK = 16;
  uint32_t matrixBlockX = 32;
  mutable std::string name = "FP8";
};
