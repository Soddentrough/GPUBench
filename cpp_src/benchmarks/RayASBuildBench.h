#pragma once

#include "IBenchmark.h"
#include <map>
#include <string>
#include <vulkan/vulkan.h>

class RayASBuildBench : public IBenchmark {
public:
  RayASBuildBench() = default;
  ~RayASBuildBench() override = default;

  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context) const override;
  SupportLimitation GetSupportLimitation() const override {
    return SupportLimitation::kApi;
  }
  std::string GetSupportNote() const override {
    return "Ray tracing benchmark requires Vulkan hardware ray tracing pipelines and acceleration structures (VK_KHR_ray_tracing_pipeline)";
  }
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx) override;
  void Teardown() override;

  BenchmarkResult GetResult(uint32_t config_idx) const override;
  const char *GetName() const override;
  const char *GetComponent(uint32_t config_idx) const override;
  const char *GetMetric(uint32_t config_idx) const override;
  const char *GetSubCategory(uint32_t config_idx) const override;
  std::string GetConfigName(uint32_t config_idx) const override;

  uint32_t GetNumConfigs() const override { return 8; } // 1M/5M/10M BLAS Build/Update + 3 Real-World Branched TLAS scenes
  int GetSortWeight(uint32_t config_idx = 0) const override {
    return 600 + static_cast<int>(config_idx);
  }

  struct BlasInfo {
    uint32_t numPrimitives = 0;
    ComputeBuffer buffer = nullptr;
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
  };

  struct TlasInfo {
    std::string name;
    uint32_t numInstances = 0;
    ComputeBuffer instanceBuffer = nullptr;
    ComputeBuffer buffer = nullptr;
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
  };

private:
  void loadRTProcs(VkDevice device);

  IComputeContext *context = nullptr;

  // Single-BLAS benchmark resources (1M, 5M, 10M)
  ComputeBuffer vertexBuffer = nullptr;
  ComputeBuffer scratchBuffer = nullptr;
  ComputeBuffer updateScratchBuffer = nullptr;
  std::vector<BlasInfo> blases;

  // Multi-BLAS scene library (5,000 distinct geometries)
  ComputeBuffer blasLibVertexBuffer = nullptr;
  std::vector<ComputeBuffer> blasLibBuffers;
  std::vector<VkAccelerationStructureKHR> blasLibHandles;
  std::vector<VkDeviceAddress> blasLibAddrs;

  // Real-world branched TLAS scenes
  std::vector<TlasInfo> tlases;

  std::map<uint32_t, double> buildTimes;
  uint32_t iterations = 0;

  // Function pointers
  PFN_vkGetAccelerationStructureBuildSizesKHR
      vkGetAccelerationStructureBuildSizesKHR_ptr;
  PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR_ptr;
  PFN_vkCmdBuildAccelerationStructuresKHR
      vkCmdBuildAccelerationStructuresKHR_ptr;
  PFN_vkGetAccelerationStructureDeviceAddressKHR
      vkGetAccelerationStructureDeviceAddressKHR_ptr;
  PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR_ptr;
};
