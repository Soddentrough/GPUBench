#pragma once

#include "benchmarks/IBenchmark.h"

class Fp6Bench : public IBenchmark {
public:
  const char *GetName() const override { return "FP6"; }
  std::vector<std::string> GetAliases() const override {
    return {"fp6", "f6", "mxfp6"};
  }
  const char *GetMetric() const override { return "TFLOPS"; }
  uint32_t GetNumConfigs() const override { return 1; }
  uint32_t GetExpectedKernelCount() const override { return 1; }
  using IBenchmark::GetConfigName;
  std::string GetConfigName(uint32_t config_idx) const override {
    (void)config_idx;
    return "Matrix";
  }

  bool IsSupported(const DeviceInfo &device,
                   IComputeContext *context = nullptr) const override;
  std::string GetSupportNote() const override {
    return "No native 16x16x16 Float6E3M2EXT cooperative matrix or CDNA 4 gfx950 MXFP6 MFMA support (requires VK_EXT_shader_ocp_microscaling_types::shaderFloat6 + VK_KHR_cooperative_matrix or AMD CDNA 4 gfx950)";
  }
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return "No support for 6-bit microscaling floating point in OpenCL API";
    }
    if (!info.fp6Support) {
      if (context && context->getBackend() == ComputeBackend::ROCm) {
        return "Requires AMD CDNA 4 (gfx950) OCP MXFP6 (E3M2) 16x16x128 MFMA hardware";
      }
      return "No native OCP MXFP6 (VK_EXT_shader_ocp_microscaling_types::shaderFloat6 / Float6E3M2EXT) 16x16x16 cooperative matrix support exposed on GPU/driver";
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
    if (!info.fp6Support) {
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
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;
  const char *GetComponent(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return "Compute";
  }
  const char *GetSubCategory(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return "FP6";
  }
  int GetSortWeight() const override { return 50; }
  using IBenchmark::GetDescription;
  const char *GetDescription() const override {
    return "OCP Microscaling 6-bit floating-point (MXFP6 E3M2) Cooperative Matrix / MFMA throughput.";
  }
  const char *GetDescription(uint32_t config_idx, const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    (void)config_idx;
    (void)info;
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      return "OCP Microscaling 6-bit floating-point (MXFP6 E3M2) 16x16x128 MFMA matrix throughput on CDNA 4.";
    }
    if (context && context->getBackend() == ComputeBackend::Vulkan) {
      return "OCP Microscaling 6-bit floating-point (MXFP6 E3M2) 16x16x16 Cooperative Matrix throughput via VK_EXT_shader_ocp_microscaling_types.";
    }
    return GetDescription();
  }
  using IBenchmark::GetExplanation;
  BenchmarkExplanation GetExplanation(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return {
      "OCP Microscaling 6-bit floating-point (MXFP6 E3M2) hardware matrix-core throughput.",
      "Next-generation sub-byte AI format preserving higher accuracy than FP4/INT4 for quantized neural rendering models and local LLM inference.",
      "Hardware MXFP6 matrix multiply-accumulate tile loop (16x16x16 Cooperative Matrix or CDNA 4 MFMA)."
    };
  }
  BenchmarkExplanation GetExplanation(uint32_t config_idx, const DeviceInfo &info,
                                      IComputeContext *context = nullptr) const override {
    (void)config_idx;
    (void)info;
    BenchmarkExplanation e = GetExplanation(0);
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      e.method = "HIP kernel executing native CDNA 4 MXFP6 (16x16x128) MFMA matrix instructions.";
    } else if (context && context->getBackend() == ComputeBackend::Vulkan) {
      e.method = "Vulkan compute shader executing 16x16x16 MXFP6 Cooperative Matrix tiles via VK_EXT_shader_ocp_microscaling_types.";
    }
    return e;
  }
  double GetOperationalIntensity(uint32_t config_idx = 0) const override {
    (void)config_idx;
    // coop_matrix_fp6.comp: 32768 * 8192 FLOPs / (192 B A + 192 B B + 1024 B C = 1408 B) = 190,650.18 FLOP/B
    return (32768.0 * 8192.0) / 1408.0;
  }
  void Teardown() override;
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
