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
  using IBenchmark::GetConfigName;
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
      if (context && context->getBackend() == ComputeBackend::ROCm) {
        return "Requires AMD CDNA 4 (gfx950) OCP MXFP4 (E2M1) 16x16x128 MFMA hardware";
      }
      return "No native OCP MXFP4 (VK_EXT_shader_ocp_microscaling_types::shaderFloat4 / Float4E2M1EXT) 16x16x16 cooperative matrix support exposed on GPU/driver";
    }
    if (context && context->getBackend() == ComputeBackend::Vulkan &&
        !info.cooperativeMatrixSupport) {
      return "VK_KHR_cooperative_matrix not supported on device";
    }
    return "";
  }
  using IBenchmark::GetConfigSupportNote;
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
  using IBenchmark::GetDescription;
  const char *GetDescription() const override {
    return "OCP Microscaling 4-bit floating-point (MXFP4 E2M1) Cooperative Matrix / MFMA throughput.";
  }
  const char *GetDescription(uint32_t config_idx, const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    (void)config_idx;
    (void)info;
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      return "OCP Microscaling 4-bit floating-point (MXFP4 E2M1) 16x16x128 MFMA matrix throughput on CDNA 4.";
    }
    if (context && context->getBackend() == ComputeBackend::Vulkan) {
      return "OCP Microscaling 4-bit floating-point (MXFP4 E2M1) 16x16x16 Cooperative Matrix throughput via VK_EXT_shader_ocp_microscaling_types.";
    }
    return GetDescription();
  }
  double GetOperationalIntensity(uint32_t config_idx = 0) const override {
    (void)config_idx;
    // coop_matrix_fp4.comp: 32768 * 8192 FLOPs / (128 B A + 128 B B + 1024 B C = 1280 B) = 209,715.20 FLOP/B
    return (32768.0 * 8192.0) / 1280.0;
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
