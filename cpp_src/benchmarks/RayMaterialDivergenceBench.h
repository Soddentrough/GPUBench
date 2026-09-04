#pragma once

#include "IBenchmark.h"
#include <map>
#include <string>
#include <vulkan/vulkan.h>

class RayMaterialDivergenceBench : public IBenchmark {
public:
  RayMaterialDivergenceBench() = default;
  ~RayMaterialDivergenceBench() override = default;

  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context) const override;
  SupportLimitation GetSupportLimitation() const override {
    return SupportLimitation::kApi;
  }
  SupportLimitation GetSupportLimitation(const DeviceInfo &info,
                                         IComputeContext *context = nullptr) const override {
    (void)info;
    if (context && (context->getBackend() == ComputeBackend::OpenCL ||
                    context->getBackend() == ComputeBackend::ROCm)) {
      return SupportLimitation::kApi;
    }
    return SupportLimitation::kHardware;
  }
  std::string GetSupportNote() const override {
    return "Ray tracing requires Vulkan ray query acceleration structures";
  }
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return "No support for ray tracing acceleration structures in OpenCL API";
    }
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      return "No support for ray tracing acceleration structures in ROCm/HIP API";
    }
    if (!info.rayTracingSupport) {
      return "extension VK_KHR_acceleration_structure or VK_KHR_ray_query missing";
    }
    return "Ray tracing requires Vulkan ray query acceleration structures";
  }
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx) override;
  void Teardown() override;

  BenchmarkResult GetResult(uint32_t config_idx) const override;
  const char *GetName() const override;
  const char *GetComponent(uint32_t config_idx) const override;
  const char *GetMetric() const override;
  const char *GetSubCategory(uint32_t config_idx) const override;
  std::string GetConfigName(uint32_t config_idx) const override;

  uint32_t GetNumConfigs() const override { return 2; } // Coherent, Divergent
  int GetSortWeight(uint32_t = 0) const override { return 635; }
  std::vector<std::string> GetAliases() const override {
    return {"raymatdiv", "materialdivergence", "raydivergence"};
  }

private:
  void loadRTProcs(VkDevice device);
  void buildAS(uint32_t config_idx);

  IComputeContext *context = nullptr;
  ComputeKernel kernel = nullptr;

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
  uint32_t numInstances = 0;
  int builtConfigIdx = -1; // Which config the current AS was built for
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
