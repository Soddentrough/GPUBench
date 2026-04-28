#pragma once

#include "IBenchmark.h"
#include <map>
#include <string>
#include <vulkan/vulkan.h>

class RayPayloadBench : public IBenchmark {
public:
  RayPayloadBench() = default;
  ~RayPayloadBench() override = default;

  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context) const override;
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx) override;
  void Teardown() override;

  BenchmarkResult GetResult(uint32_t config_idx) const override;
  const char *GetName() const override;
  const char *GetComponent(uint32_t config_idx) const override;
  const char *GetMetric() const override;
  const char *GetSubCategory(uint32_t config_idx) const override;
  std::string GetConfigName(uint32_t config_idx) const override;

  uint32_t GetNumConfigs() const override { return 3; } // 16B, 128B, 256B

private:
  void loadRTProcs(VkDevice device);
  void buildAS();

  IComputeContext *context = nullptr;
  ComputeKernel kernel16 = nullptr;
  ComputeKernel kernel128 = nullptr;
  ComputeKernel kernel256 = nullptr;

  ComputeBuffer vertexBuffer = nullptr;
  ComputeBuffer instanceBuffer = nullptr;
  ComputeBuffer scratchBuffer = nullptr;
  ComputeBuffer resultBuffer = nullptr;

  ComputeBuffer triangleBlasBuffer = nullptr;
  ComputeBuffer triangleTlasBuffer = nullptr;

  VkAccelerationStructureKHR triangleBlas = VK_NULL_HANDLE;
  VkAccelerationStructureKHR triangleTlas = VK_NULL_HANDLE;

  uint32_t rayCount = 0;
  uint32_t numPrimitives = 0;
  std::map<uint32_t, double> rtResults;

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
