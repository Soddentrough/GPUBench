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
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx) override;
  void Teardown() override;

  BenchmarkResult GetResult(uint32_t config_idx) const override;
  const char *GetName() const override;
  const char *GetComponent(uint32_t config_idx) const override;
  const char *GetMetric() const override;
  const char *GetSubCategory(uint32_t config_idx) const override;
  std::string GetConfigName(uint32_t config_idx) const override;

  uint32_t GetNumConfigs() const override { return 3; } // BLAS Build, TLAS Build, BLAS Update

private:
  void loadRTProcs(VkDevice device);

  IComputeContext *context = nullptr;

  ComputeBuffer vertexBuffer = nullptr;
  ComputeBuffer instanceBuffer = nullptr;
  ComputeBuffer scratchBuffer = nullptr;
  ComputeBuffer updateScratchBuffer = nullptr;

  ComputeBuffer blasBuffer = nullptr;
  ComputeBuffer tlasBuffer = nullptr;

  VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
  VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;

  uint32_t numPrimitives = 0;
  uint32_t numInstances = 0;
  std::map<uint32_t, double> buildTimes;
  uint32_t iterations = 0;

  VkAccelerationStructureBuildSizesInfoKHR blasSizes{};
  VkAccelerationStructureBuildSizesInfoKHR tlasSizes{};

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
