#pragma once

#include "IBenchmark.h"
#include <map>
#include <vector>
#include <vulkan/vulkan.h>

class RayAnyHitBench : public IBenchmark {
public:
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx) override;
  void Teardown() override;

  BenchmarkResult GetResult(uint32_t config_idx) const override;
  uint32_t GetNumConfigs() const override { return 5; }
  const char *GetName() const override;
  const char *GetComponent(uint32_t config_idx) const override;
  const char *GetMetric() const override;
  const char *GetSubCategory(uint32_t config_idx) const override;
  std::string GetConfigName(uint32_t config_idx) const override;

  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context) const override;

private:
  void buildAS();

  IComputeContext *context = nullptr;
  ComputeKernel kernel = nullptr;
  ComputeBuffer resultBuffer = nullptr;
  ComputeBuffer vertexBuffer = nullptr;
  ComputeBuffer instanceBuffer = nullptr;
  ComputeBuffer triangleBlasBuffer = nullptr;
  ComputeBuffer triangleTlasBuffer = nullptr;
  ComputeBuffer scratchBuffer = nullptr;

  VkAccelerationStructureKHR triangleBlas = VK_NULL_HANDLE;
  VkAccelerationStructureKHR triangleTlas = VK_NULL_HANDLE;

  uint32_t rayCount;
  uint32_t numPrimitives;
  std::map<uint32_t, double> rtResults;

  // Function pointers
  PFN_vkGetAccelerationStructureBuildSizesKHR
      vkGetAccelerationStructureBuildSizesKHR_ptr;
  PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR_ptr;
  PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR_ptr;
  PFN_vkGetAccelerationStructureDeviceAddressKHR
      vkGetAccelerationStructureDeviceAddressKHR_ptr;
  PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR_ptr;

  void loadRTProcs(VkDevice device);
};
