#pragma once

#include "benchmarks/IBenchmark.h"

class Bf16Bench : public IBenchmark {
public:
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context) const override;
  std::string GetSupportNote() const override {
    return "Requires VK_KHR_shader_bfloat16 or ROCm BF16 WMMA / MFMA support";
  }
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return "No support for native BFloat16 floating-point arithmetic in OpenCL API (extension cl_khr_bfloat16 missing)";
    }
    if (!info.bf16Support) {
      if (context && context->getBackend() == ComputeBackend::ROCm) {
        return "Native BFloat16 (BF16) compute / WMMA / MFMA hardware not supported on this ROCm GPU";
      }
      if (info.archName.find("RDNA 3") != std::string::npos ||
          info.archName.find("gfx11") != std::string::npos) {
        return "RDNA 3 lacks BF16-return v_dot2_bf16_bf16 VALU dot-product hardware (Vulkan RADV lowers to FP32; ROCm uses native FP32-accum v_dot2_f32_bf16)";
      }
      return "VK_KHR_shader_bfloat16 (shaderBFloat16) hardware/driver support not available";
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
  using IBenchmark::GetConfigSupportNote;
  std::string GetConfigSupportNote(uint32_t config_idx,
                                   const DeviceInfo &info,
                                   IComputeContext *context = nullptr) const override {
    if (config_idx == 0 && !info.shaderBfloat16DotProductSupport) {
      if (context && context->getBackend() == ComputeBackend::ROCm) {
        return "Packed BFloat16 vector dot2 (v_dot2_f32_bf16) instruction not supported on this ROCm GPU";
      }
      return "VK_KHR_shader_bfloat16::shaderBFloat16DotProduct not supported by device/driver";
    }
    if (config_idx == 1) {
      if (!info.cooperativeMatrixSupport) {
        if (context && context->getBackend() == ComputeBackend::ROCm) {
          return "ROCm BF16 WMMA / MFMA matrix instructions not supported on this GPU";
        }
        return "VK_KHR_cooperative_matrix extension not supported on this GPU/driver";
      }
      if (context && context->getBackend() == ComputeBackend::Vulkan &&
          !info.shaderBfloat16CoopMatrixSupport) {
        return "VK_KHR_shader_bfloat16::shaderBFloat16CooperativeMatrix not supported by active Vulkan driver";
      }
    }
    return GetSupportNote(info, context);
  }
  using IBenchmark::GetConfigSupportLimitation;
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
  SupportDiscrepancy GetConfigSupportDiscrepancy(
      uint32_t config_idx, const DeviceInfo &info,
      IComputeContext *context = nullptr) const override {
    SupportDiscrepancy d{};
    if (context && context->getBackend() == ComputeBackend::Vulkan) {
      if (config_idx == 0 && info.shaderBfloat16DotProductSupport &&
          !info.bf16DotProductHardwareSupported) {
        d.differs = true;
        d.hardwareStatus = "Unsupported natively (architecture lacks native BF16 VALU dot2; uses FP32 lowering)";
        d.driverStatus = "Supported (VK_KHR_shader_bfloat16::shaderBFloat16DotProduct = true; allowed to run)";
        d.summary = "Driver exposes BF16 dot2 via FP32 lowering (no native BF16 VALU hardware)";
      } else if (config_idx == 1 && info.cooperativeMatrixSupport &&
                 !info.shaderBfloat16CoopMatrixSupport &&
                 info.bf16CoopMatrixHardwareSupported) {
        d.differs = true;
        d.hardwareStatus = "Supported in hardware (architecture includes BF16 WMMA/Matrix units)";
        d.driverStatus = "Unsupported (shaderBFloat16CooperativeMatrix = false in active Vulkan driver)";
        d.summary = "Hardware includes BF16 WMMA units, but active Vulkan driver does not expose shaderBFloat16CooperativeMatrix";
      }
    }
    return d;
  }
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx) override;
  void Teardown() override;

  bool SupportsWorkScaling(uint32_t config_idx = 0) const override { (void)config_idx; return true; }
  BenchmarkResult GetResult(uint32_t config_idx) const override;
  uint32_t GetNumConfigs() const override { return 2; }
  using IBenchmark::GetConfigName;
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
  using IBenchmark::GetDescription;
  const char *GetDescription() const override {
    return "Brain Floating-Point (bfloat16) packed dot-product VALU and Cooperative Matrix / WMMA / MFMA throughput.";
  }
  const char *GetDescription(uint32_t config_idx) const override {
    return config_idx == 0
               ? "Brain Floating-Point (bfloat16) packed 2-element dot-product VALU throughput."
               : "Brain Floating-Point (bfloat16) matrix multiply-accumulate throughput across Cooperative Matrix / WMMA / MFMA cores.";
  }
  const char *GetDescription(uint32_t config_idx, const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    (void)info;
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      return config_idx == 0
                 ? "Brain Floating-Point (bfloat16) packed dot-product VALU throughput via native FP32-accumulating v_dot2_f32_bf16."
                 : "Brain Floating-Point (bfloat16) matrix multiply-accumulate throughput across WMMA / MFMA matrix cores.";
    }
    if (context && context->getBackend() == ComputeBackend::Vulkan) {
      return config_idx == 0
                 ? "Brain Floating-Point (bfloat16) packed dot-product VALU throughput via VK_KHR_shader_bfloat16."
                 : "Brain Floating-Point (bfloat16) matrix multiply-accumulate throughput via VK_KHR_cooperative_matrix and VK_KHR_shader_bfloat16.";
    }
    return GetDescription(config_idx);
  }
  double GetOperationalIntensity(uint32_t config_idx = 0) const override {
    // Config 0 (bf16.comp): 32768 * 256 ops (8,388,608 FLOPs) / (8 B read + 8 B write = 16 B) = 524,288.0 FLOP/B
    // Config 1 (coop_matrix_bf16.comp): 32768 * 8192 FLOPs / (512 B A + 512 B B + 1024 B C = 2048 B) = 131,072.0 FLOP/B
    return (config_idx == 0u) ? 524288.0 : 131072.0;
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
  std::string GetConfigCaveat(uint32_t config_idx,
                              const DeviceInfo &info,
                              IComputeContext *context = nullptr) const override {
    if (config_idx == 0 &&
        (vectorEmulated ||
         (context && context->getBackend() == ComputeBackend::Vulkan &&
          info.shaderBfloat16DotProductSupport &&
          !info.bf16DotProductHardwareSupported))) {
      return "Driver lowered BF16 vector arithmetic to FP32 VALU instructions (no native BF16 VALU dot2 hardware)";
    }
    return GetConfigCaveat(config_idx);
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
