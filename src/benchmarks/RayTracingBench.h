#pragma once

#include "IBenchmark.h"
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

class RayTracingBench : public IBenchmark {
public:
  const char *GetName() const override;
  const char *GetMetric() const override;

  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;

  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;

  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;

  uint32_t GetNumConfigs() const override { return 2; }
  std::vector<std::string> GetAliases() const override {
    return {"rt", "raytracing"};
  }
  std::string GetConfigName(uint32_t config_idx) const override;
  const char *GetComponent(uint32_t config_idx = 0) const override;
  const char *GetSubCategory(uint32_t config_idx = 0) const override;

private:
  IComputeContext *context = nullptr;
  ComputeKernel kernel = nullptr;
  ComputeBuffer resultBuffer = nullptr;

  // Acceleration Structures
  VkAccelerationStructureKHR triangleBlas = VK_NULL_HANDLE;
  VkAccelerationStructureKHR boxBlas = VK_NULL_HANDLE;
  VkAccelerationStructureKHR triangleTlas = VK_NULL_HANDLE;
  VkAccelerationStructureKHR boxTlas = VK_NULL_HANDLE;

  ComputeBuffer vertexBuffer = nullptr;
  ComputeBuffer aabbBuffer = nullptr;
  ComputeBuffer instanceBuffer = nullptr;
  ComputeBuffer triangleBlasBuffer = nullptr;
  ComputeBuffer boxBlasBuffer = nullptr;
  ComputeBuffer triangleTlasBuffer = nullptr;
  ComputeBuffer boxTlasBuffer = nullptr;
  ComputeBuffer scratchBuffer = nullptr;

  // RT function pointers
  PFN_vkGetAccelerationStructureBuildSizesKHR
      vkGetAccelerationStructureBuildSizesKHR_ptr = nullptr;
  PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR_ptr =
      nullptr;
  PFN_vkCmdBuildAccelerationStructuresKHR
      vkCmdBuildAccelerationStructuresKHR_ptr = nullptr;
  PFN_vkGetAccelerationStructureDeviceAddressKHR
      vkGetAccelerationStructureDeviceAddressKHR_ptr = nullptr;
  PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR_ptr =
      nullptr;

  uint32_t rayCount = 4000000;
  uint32_t numPrimitives = 4096;
  uint32_t iterations = 100;
  double rtResults[2] = {0.0, 0.0};

  void loadRTProcs(VkDevice device);
  void buildAS();
};
