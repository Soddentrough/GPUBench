#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"
#include <string>
#include <vector>

class Int8Bench : public IBenchmark {
public:
  const char *GetName() const override { return "INT8"; }
  std::vector<std::string> GetAliases() const override {
    return {"int8", "int8b"};
  }
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  std::string GetSupportNote() const override {
    return "shaderInt8 hardware feature not enabled";
  }
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    (void)info;
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      return "Native INT8 (8-bit integer) compute not supported on this ROCm device";
    }
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return "8-bit integer compute not supported on this OpenCL device";
    }
    return "shaderInt8 hardware feature not enabled on this Vulkan device";
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
  using IBenchmark::GetConfigSupportNote;
  std::string GetConfigSupportNote(uint32_t config_idx,
                                   const DeviceInfo &info,
                                   IComputeContext *context = nullptr) const override {
    if (config_idx == 1) {
      if (context && context->getBackend() == ComputeBackend::OpenCL) {
        return "Cooperative matrix / WMMA not supported in OpenCL API";
      }
      if (!info.cooperativeMatrixSupport) {
        if (context && context->getBackend() == ComputeBackend::ROCm) {
          return "ROCm INT8 WMMA / MFMA matrix instructions not supported on this GPU";
        }
        return "VK_KHR_cooperative_matrix extension not supported on this GPU/driver";
      }
    }
    return GetSupportNote(info, context);
  }
  SupportLimitation GetConfigSupportLimitation(uint32_t config_idx) const override {
    return GetSupportLimitation();
  }
  SupportLimitation GetConfigSupportLimitation(uint32_t config_idx,
                                               const DeviceInfo &info,
                                               IComputeContext *context = nullptr) const override {
    if (config_idx == 1) {
      if (context && context->getBackend() == ComputeBackend::OpenCL) return SupportLimitation::kApi;
      if (!info.cooperativeMatrixSupport) return SupportLimitation::kHardware;
    }
    return GetSupportLimitation(info, context);
  }
  bool IsConfigSupported(uint32_t config_idx) const override {
    if (config_idx == 0) return true;
    return matrixKernel != nullptr;
  }
  bool IsConfigSupported(uint32_t config_idx, const DeviceInfo &info,
                         IComputeContext *context = nullptr) const override {
    if (!IsSupported(info, context)) return false;
    if (config_idx == 0) return true;
    if (context && context->getBackend() == ComputeBackend::OpenCL) return false;
    return info.cooperativeMatrixSupport;
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
    return "INT8";
  }
  int GetSortWeight() const override { return 70; }
  uint32_t GetNumConfigs() const override { return 2; }
  virtual uint32_t GetExpectedKernelCount() const override { return 2; }
  using IBenchmark::GetConfigName;
  std::string GetConfigName(uint32_t config_idx) const override {
    return config_idx == 0 ? "Vector" : "Matrix";
  }
  const char *GetMetric() const override { return "TOPS"; }
  using IBenchmark::GetDescription;
  const char *GetDescription() const override {
    return "8-bit integer (INT8) packed dot-product VALU and Cooperative Matrix / WMMA / MFMA throughput.";
  }
  const char *GetDescription(uint32_t config_idx) const override {
    return config_idx == 0
               ? "8-bit integer (INT8) packed 4-element dot-product VALU throughput."
               : "8-bit integer (INT8) matrix multiply-accumulate throughput across Cooperative Matrix / WMMA / MFMA cores.";
  }
  const char *GetDescription(uint32_t config_idx, const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    (void)info;
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      return config_idx == 0
                 ? "8-bit integer (INT8) packed 4-element dot-product VALU throughput via v_dot4_i32_iu8."
                 : "8-bit integer (INT8) matrix multiply-accumulate throughput across WMMA / MFMA matrix cores.";
    }
    if (context && context->getBackend() == ComputeBackend::Vulkan) {
      return config_idx == 0
                 ? "8-bit integer (INT8) packed 4-element dot-product VALU throughput via GL_EXT_integer_dot_product / SPIR-V SDot."
                 : "8-bit integer (INT8) matrix multiply-accumulate throughput via VK_KHR_cooperative_matrix.";
    }
    return GetDescription(config_idx);
  }
  using IBenchmark::GetExplanation;
  BenchmarkExplanation GetExplanation(uint32_t config_idx = 0) const override {
    if (config_idx == 0u) {
      return {
        "Packed 8-bit integer 4-element dot-product (DP4a / SDOT) throughput inside standard shader vector ALUs.",
        "Used by fallback/shader-based AI upscalers (e.g., FSR / XeSS DP4a paths), quantized image filters, and integer compute without dedicated matrix cores.",
        "Register-resident compute shader executing packed 4x INT8 dot-product-accumulate chains."
      };
    }
    return {
      "Hardware 8-bit integer (INT8) matrix-core throughput (WMMA / MFMA / Cooperative Matrix) with INT32 accumulation.",
      "Powers quantized real-time AI upscaling (XeSS / FSR / DLSS paths), neural denoisers, and high-throughput quantized ML inference.",
      "Subgroup-cooperative 16x16x16 INT8 matrix multiply-accumulate tile loop in registers."
    };
  }
  BenchmarkExplanation GetExplanation(uint32_t config_idx, const DeviceInfo &info,
                                      IComputeContext *context = nullptr) const override {
    (void)info;
    BenchmarkExplanation e = GetExplanation(config_idx);
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      e.method = (config_idx == 0u)
          ? "HIP kernel executing native packed INT8 dot-product instructions (v_dot4_i32_iu8)."
          : "HIP kernel executing native INT8 WMMA / MFMA matrix-core tiles.";
    } else if (context && context->getBackend() == ComputeBackend::Vulkan) {
      e.method = (config_idx == 0u)
          ? "Vulkan compute shader using GL_EXT_integer_dot_product / SPIR-V SDot instructions."
          : "Vulkan compute shader using VK_KHR_cooperative_matrix INT8 tensor tiles.";
    }
    return e;
  }
  double GetOperationalIntensity(uint32_t config_idx = 0) const override {
    // Config 0 (int8.comp): 16384 * 64 ops (1,048,576 IOPs) / (4 B read + 4 B write = 8 B) = 131,072.0 OP/B
    // Config 1 (coop_matrix_int8.comp): 32768 * 8192 IOPs / (256 B A + 256 B B + 1024 B C = 1536 B) = 174,762.67 OP/B
    return (config_idx == 0u) ? 131072.0 : ((32768.0 * 8192.0) / 1536.0);
  }

private:
  template <typename Tag> friend struct ::gpubench::test::TestBody;
  IComputeContext *context = nullptr;
  ComputeKernel vectorKernel = nullptr;
  ComputeKernel matrixKernel = nullptr;
  ComputeBuffer buffer = nullptr;
  ComputeBuffer matrixInputBuffer = nullptr;
  ComputeBuffer outputBuffer = nullptr;
  uint32_t tileM = 16;
  uint32_t tileN = 16;
  uint32_t tileK = 16;
  uint32_t matrixBlockX = 32;
};
