#pragma once

#include "IBenchmark.h"
#include "core/VulkanContext.h"
#include <cstdint>
#include <string>
#include <vector>

class RayOMMBench : public IBenchmark {
public:
  const char *GetName() const override {
    return "Ray Tracing Opacity Micromap";
  }
  std::vector<std::string> GetAliases() const override {
    return {"rayomm", "omm", "micromap"};
  }
  const char *GetMetric() const override { return "MRays/s"; }
  const char *GetComponent(uint32_t /*config_idx*/ = 0) const override {
    return "Ray Tracing";
  }
  const char *GetSubCategory(uint32_t /*config_idx*/ = 0) const override {
    return "Alpha-Cutout Traversal";
  }
  int GetSortWeight() const override { return 325; }
  uint32_t GetExpectedKernelCount() const override { return 1; }

  uint32_t GetNumConfigs() const override { return 3; }
  std::string GetConfigName(uint32_t config_idx) const override;
  const char *GetDescription() const override { return GetDescription(0); }
  const char *GetDescription(uint32_t config_idx) const override;
  int32_t GetBaselineConfigIndex(uint32_t config_idx) const override {
    return (config_idx == 1u || config_idx == 2u) ? 0 : -1;
  }

  void SetResolution(uint32_t w, uint32_t h) override {
    if (w > 0 && h > 0) {
      m_width = w;
      m_height = h;
    }
  }

  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  bool IsConfigSupported(uint32_t config_idx, const DeviceInfo &info,
                         IComputeContext *context = nullptr) const override;
  SupportLimitation GetSupportLimitation(
      const DeviceInfo &info,
      IComputeContext *context = nullptr) const override;
  SupportLimitation GetConfigSupportLimitation(
      uint32_t config_idx, const DeviceInfo &info,
      IComputeContext *context = nullptr) const override;
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override;
  std::string GetConfigSupportNote(
      uint32_t config_idx, const DeviceInfo &info,
      IComputeContext *context = nullptr) const override;

  void Setup(IComputeContext &context,
             const std::string &kernel_dir) override;
  void PrepareConfig(uint32_t config_idx) override;
  void ReleaseConfigResources(uint32_t config_idx) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;

private:
  static constexpr uint32_t kGridDim = 64u;
  static constexpr uint32_t kNumLayers = 16u;
  static constexpr uint32_t kNumPrimitives =
      kGridDim * kGridDim * 2u * kNumLayers; // 131,072 triangles

  VulkanContext *m_vk = nullptr;
  std::string m_kernelDir;
  uint32_t m_width = 1920u;
  uint32_t m_height = 1080u;

  ComputeBuffer m_vertexBuf = nullptr;
  ComputeBuffer m_outBuf = nullptr;
  ComputeKernel m_rtKernel = nullptr;

  VkAccelerationStructureKHR m_blas = VK_NULL_HANDLE;
  ComputeBuffer m_blasBuf = nullptr;
  VkAccelerationStructureKHR m_tlas = VK_NULL_HANDLE;
  ComputeBuffer m_tlasBuf = nullptr;

  VkMicromapEXT m_micromap = VK_NULL_HANDLE;
  VkBuffer m_ommStorageBuf = VK_NULL_HANDLE;
  VmaAllocation m_ommStorageAlloc = VK_NULL_HANDLE;

  PFN_vkGetAccelerationStructureBuildSizesKHR m_pfnGetASBuildSizes = nullptr;
  PFN_vkCreateAccelerationStructureKHR m_pfnCreateAS = nullptr;
  PFN_vkDestroyAccelerationStructureKHR m_pfnDestroyAS = nullptr;
  PFN_vkCmdBuildAccelerationStructuresKHR m_pfnCmdBuildAS = nullptr;
  PFN_vkGetAccelerationStructureDeviceAddressKHR m_pfnGetASAddress = nullptr;

  PFN_vkCreateMicromapEXT m_pfnCreateMicromap = nullptr;
  PFN_vkDestroyMicromapEXT m_pfnDestroyMicromap = nullptr;
  PFN_vkGetMicromapBuildSizesEXT m_pfnGetMicromapBuildSizes = nullptr;
  PFN_vkCmdBuildMicromapsEXT m_pfnCmdBuildMicromaps = nullptr;

  void buildFoliageAS(uint32_t config_idx);
};
